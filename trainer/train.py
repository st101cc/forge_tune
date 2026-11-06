"""Single-run SFT entry point. The API invokes this in WSL after queuing a run."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from advisor import classification_label_field, classification_prediction, classification_report, detect_schema, input_key, normalize_columns


def final_solution(answer: str) -> str:
    """A reasoning model's answer without its <think> section: the worked solution written after </think>."""
    _, closed, after = answer.rpartition("</think>")
    return after.strip() if closed and after.strip() else answer


def training_text_dataset(dataset, tokenizer, max_length=None, keep_reasoning=False):
    """Tokenize once and supply explicit, validated completion masks to TRL.

    Unless keep_reasoning, targets are trained without their <think> reasoning, and examples longer than max_length
    are skipped rather than cut, since a cut example teaches the model to stop before its answer."""
    target = (lambda answer: answer) if keep_reasoning else final_solution

    def format_example(example):
        prompt = ""
        if "messages" in example:
            messages = example["messages"]
            if not messages or messages[-1].get("role") != "assistant" or not str(messages[-1].get("content", "")).strip():
                raise ValueError("Chat training requires a nonempty final assistant target.")
            # ponytail: supervise the final assistant turn; expand conversations into turns if all-turn training is needed.
            messages = [*messages[:-1], {**messages[-1], "content": target(str(messages[-1]["content"]))}]
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            if not text.startswith(prompt):
                raise ValueError("Chat template cannot preserve the response boundary; use a compatible template.")
        elif "prompt" in example and "completion" in example:
            prompt = str(example["prompt"])
            if not str(example["completion"]).strip():
                raise ValueError("Completion target is empty.")
            text = prompt + target(str(example["completion"]))
        elif "text" in example:
            label_field = classification_label_field(list(example))
            if label_field:
                prompt = "Text: " + str(example["text"]) + "\nLabel:"
                text = prompt + " " + str(example[label_field])
            else:
                text = str(example["text"])
        else:
            raise ValueError("Dataset needs messages, prompt/completion, or text columns.")
        ids = tokenizer.encode(text, add_special_tokens=False)
        prefix = tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
        if ids[:len(prefix)] != prefix:
            raise ValueError("Tokenizer merges across the response boundary; add a separator to the prompt.")
        if len(ids) <= len(prefix):
            raise ValueError("No response target tokens are available.")
        if tokenizer.eos_token_id is not None and ids[-1] != tokenizer.eos_token_id:
            ids.append(tokenizer.eos_token_id)
        count = len(ids)
        mask = [0] * len(prefix) + [1] * (count - len(prefix))
        return {"input_ids": ids, "completion_mask": mask, "token_count": count}

    formatted = dataset.map(format_example, remove_columns=dataset.column_names)
    if max_length:
        formatted = formatted.filter(lambda row: row["token_count"] <= max_length)
        if not len(formatted):
            raise ValueError(f"No example fits the {max_length}-token training window, so no complete response target remains; increase the window or shorten the data.")
    return formatted


def chunked_next_token_loss(model, inputs, num_items_in_batch=None, chunk=1024):
    """Next-token loss scored 1,024 positions at a time, each chunk recomputed in the backward pass.

    The full sequence x vocabulary score table (about 2.5 GB per 4,096 tokens with Qwen's 152k vocabulary, plus its
    gradient) never exists at once, so long reasoning traces fit a 15 GB T4. Measured with Qwen2.5-3B QLoRA: 8,192
    tokens ran out of memory with the standard loss; with this one, real training steps at 12,288 tokens peaked at 11.3 GiB."""
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    model = getattr(model, "module", model)  # Unwrap DataParallel/DistributedDataParallel.
    causal = model.get_base_model() if hasattr(model, "get_base_model") else model
    forward = {key: value for key, value in inputs.items() if key in ("input_ids", "attention_mask", "position_ids")}
    if "attention_mask" in forward and bool(forward["attention_mask"].all()):
        # Nothing is padded. An explicit mask makes PyTorch's attention on a T4 fall back to building the whole
        # sequence x sequence matrix (15.7 GiB at 16k tokens); without one it uses the memory-efficient kernel.
        del forward["attention_mask"]
    hidden = causal.model(**forward).last_hidden_state[:, :-1]
    targets = inputs["labels"][:, 1:]

    def score(states, labels):
        return F.cross_entropy(causal.lm_head(states).float().flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")

    total = sum(checkpoint(score, hidden[:, start:start + chunk], targets[:, start:start + chunk], use_reentrant=False) for start in range(0, hidden.shape[1], chunk))
    return total / (num_items_in_batch if num_items_in_batch is not None else (targets != -100).sum())


def manifest_training_split(dataset, manifest):
    train = manifest.get("trainIndices", [])
    development = manifest.get("developmentIndices", [])
    if not train or not development:
        raise ValueError("Training requires separate nonempty train and development splits.")
    if set(train) & set(development):
        raise ValueError("Train and development splits overlap.")
    if manifest.get("trainDataset") == manifest.get("evalDataset") and (set(train) | set(development)) & set(manifest.get("evalIndices", [])):
        raise ValueError("Final test examples overlap with training or development.")
    keys = [{input_key(dataset[i]) for i in indices} for indices in (train, development)]
    if manifest.get("trainDataset") == manifest.get("evalDataset"):
        keys.append({input_key(dataset[i]) for i in manifest.get("evalIndices", [])})
    if any(keys[i] & keys[j] for i in range(len(keys)) for j in range(i)):
        raise ValueError("Repeated inputs overlap across splits; create a new grouped split before training.")
    return {"train": dataset.select(train), "test": dataset.select(development)}


def main() -> None:
    # Trust the OS certificate store so model downloads work behind TLS-inspecting proxies and antivirus.
    import truststore
    truststore.inject_into_ssl()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = run["config"]
    method = config["parameter_method"]
    if method not in {"lora", "qlora"}:
        raise ValueError("Unsupported parameter method")

    # Bind the training subprocess to one free GPU before CUDA initializes.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()
        chosen = max((line.split(",") for line in devices), key=lambda parts: int(parts[1]))
        os.environ["CUDA_VISIBLE_DEVICES"] = chosen[0].strip()

    # Long training sequences fragment the CUDA cache; expandable segments let freed blocks be reused (set before torch loads).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Imports are intentionally inside the entry point so the API can start before
    # large ML dependencies have completed installation.
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig
    import torch

    hp = config["hyperparameters"]
    def load_ref(dataset_info):
        # Normalise exactly like advisor.load_source so training sees the rows that were profiled and evaluated.
        if dataset_info["source"] == "huggingface":
            return normalize_columns(load_dataset(dataset_info["name"], split=dataset_info.get("split", "train")))
        path = dataset_info["name"]
        extension = Path(path).suffix.lower()
        loader = {".csv": "csv", ".jsonl": "json", ".parquet": "parquet"}.get(extension)
        if not loader:
            raise RuntimeError("Uploaded dataset must be CSV, JSONL, or Parquet.")
        return normalize_columns(load_dataset(loader, data_files=path, split="train"))

    dataset_info = config["dataset"]
    dataset = load_ref(dataset_info)
    manifest_path = config.get("split_manifest_path")
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        split = manifest_training_split(dataset, manifest)
    elif config.get("test_dataset"):
        split = {"train": dataset, "test": load_ref(config["test_dataset"])}
    else:
        split = dataset.train_test_split(test_size=0.1, seed=config["split_seed"])
    if {input_key(row) for row in split["train"]} & {input_key(row) for row in split["test"]}:
        raise ValueError("Training and validation inputs overlap; create a grouped workflow split.")
    adapter = LoraConfig(r=hp["rank"], lora_alpha=hp["alpha"], lora_dropout=0.05, task_type=TaskType.CAUSAL_LM, target_modules="all-linear")
    framework = "Transformers + TRL + PEFT"
    trainer_adapter = None
    try:
        if method != "qlora":
            raise ImportError("Unsloth fast path is only used for QLoRA.")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["base_model"], max_seq_length=hp["max_sequence_length"], load_in_4bit=True
        )
        model = FastLanguageModel.get_peft_model(
            model, r=hp["rank"], lora_alpha=hp["alpha"], lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        framework = "Unsloth + TRL + PEFT"
    except (ImportError, ModuleNotFoundError):
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if method == "qlora" else None
        if not torch.cuda.is_available():
            raise RuntimeError("Training requires a CUDA GPU.")
        device = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
        torch.cuda.set_device(device)
        base = AutoModelForCausalLM.from_pretrained(config["base_model"], quantization_config=quantization, device_map={"": device}, torch_dtype=torch.float16)
        base.config.use_cache = False
        model = base
        trainer_adapter = adapter
    output = run_dir / "adapter"

    formatted = {key: training_text_dataset(value, tokenizer, hp["max_sequence_length"], hp.get("train_on_reasoning", False)) for key, value in split.items()}
    lengths = list(formatted["train"]["token_count"])
    formatted = {key: value.remove_columns("token_count") for key, value in formatted.items()}
    context = {"rows":len(lengths), "skippedRows":len(split["train"]) - len(lengths), "window":hp["max_sequence_length"], "reasoning":bool(hp.get("train_on_reasoning")), "maxTokens":max(lengths), "truncatedRows":0, "retainedTokenFraction":1.0}
    (run_dir / "training-context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
    print("Training context: " + json.dumps(context), flush=True)
    class ChunkedLossSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = chunked_next_token_loss(model, inputs, num_items_in_batch)
            return (loss, {"loss": loss}) if return_outputs else loss

    trainer = ChunkedLossSFTTrainer(
        model=model, peft_config=trainer_adapter, processing_class=tokenizer, train_dataset=formatted["train"], eval_dataset=formatted["test"],
        args=SFTConfig(output_dir=str(output), completion_only_loss=True, prediction_loss_only=True, max_length=hp["max_sequence_length"], per_device_eval_batch_size=1, fp16=True, bf16=False, gradient_checkpointing=True, optim="adamw_torch", seed=config["split_seed"], num_train_epochs=hp["epochs"], learning_rate=hp["learning_rate"], per_device_train_batch_size=hp["batch_size"], gradient_accumulation_steps=hp["gradient_accumulation"], eval_strategy="epoch", logging_steps=1, save_strategy="epoch", report_to=[]),
    )
    model = trainer.model
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable_count:
        raise RuntimeError("Trainer preparation froze all parameters; no adapter can be trained.")
    print(f"Trainable adapter parameters: {trainable_count}", flush=True)
    trained = trainer.train()
    metrics = trainer.evaluate()
    metrics.update({"trainingContext": context, "validationSplit": "development" if manifest_path else "validation", "train_loss": trained.training_loss, "optimizerSteps": trained.global_step, "trainRows": len(formatted["train"]), "evalRows": len(formatted["test"]), "maxSequenceLength": hp["max_sequence_length"], "trainedOnReasoning": bool(hp.get("train_on_reasoning")), "trainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    label_field = classification_label_field(list(split["test"].column_names))
    if detect_schema(list(split["test"].column_names)) == "classification" and label_field:
        labels = sorted({str(value).strip() for part in split.values() for value in part[label_field] if str(value).strip()})
        predictions, report_rows = [], []
        for index, row in enumerate(split["test"]):
            prompt = "Text: " + str(row["text"])[:4000] + "\nLabel:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated_tokens = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(generated_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            predicted = classification_prediction(generated, labels)
            predictions.append(predicted)
            report_rows.append({**row, "_row_id": index})
        metrics["classification"] = classification_report(report_rows, label_field, predictions)
    model.save_pretrained(output, safe_serialization=True)
    metrics["framework"] = framework
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    exports = run_dir / "exports"
    exports.mkdir(exist_ok=True)
    (exports / "Modelfile").write_text("# Convert merged weights to GGUF first\\nFROM ./model.gguf\\n", encoding="utf-8")


if __name__ == "__main__":
    main()
