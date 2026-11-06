const state = {
  source: "upload", dataset: null, model: "Qwen/Qwen2.5-7B-Instruct", method: "qlora",
  goal: "balanced", analysis: null, cloudStatus: null, cloudAssist: false,
  profile: {task:"assistant", language:"multilingual", examples:"small", priority:"balanced"},
  advisorStatus: "idle", datasetError: null, isAnalyzing: false, analysisStage: 0, analysisToken: null,
  runs: JSON.parse(localStorage.getItem("forge-tune-runs") || "[]")
};

const models = [
  ["Qwen/Qwen2.5-3B-Instruct", "Qwen 2.5", "3B", "8 GB", "Fast", "Qwen Research", "Quick and capable for first experiments."],
  ["Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5", "7B", "13 GB", "Balanced", "Apache 2.0", "Best default for useful local fine-tunes."],
  ["meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2", "3B", "8 GB", "Fast", "Llama community", "Compact instruction model with broad tooling."],
  ["mistralai/Mistral-7B-Instruct-v0.3", "Mistral", "7B", "13 GB", "Balanced", "Apache 2.0", "A dependable general-purpose 7B model."],
  ["Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen 2.5 Coder", "7B", "13 GB", "Balanced", "Apache 2.0", "Specialized for code generation and explanation."],
  ["Qwen/Qwen2.5-14B-Instruct", "Qwen 2.5", "14B", "22 GB", "Slow", "Apache 2.0", "Higher capacity; needs QLoRA on this GPU."]
].map(([id,name,size,vram,speed,license,note]) => ({id,name,size,vram,speed,license,note}));

const app = document.querySelector("#app");
const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[char]));
const route = () => location.hash.replace(/^#\/?/, "") || "workspace";
const save = () => localStorage.setItem("forge-tune-runs", JSON.stringify(state.runs));
const runDatasetLabel = run => run.dataset || run.datasetName || "Dataset not recorded";
const runModelLabel = run => {
  const item = models.find(candidate => candidate.id === run.model);
  return item ? item.name + " " + item.size : (run.model || "Base model not recorded");
};
const runTitle = run => runDatasetLabel(run) + " — " + runModelLabel(run);
const runStatusClass = run => run.status === "complete" ? "complete" : run.status === "failed" || run.status === "cancelled" ? "danger" : "";
const percentage = value => Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "—";
const datasetBlockers = () => state.analysis ? (state.analysis.blockers || state.analysis.findings?.filter(item => item.severity === "blocker") || []) : [];
const datasetReady = () => Boolean(state.analysis && !state.datasetError && !state.isAnalyzing && !datasetBlockers().length);
const model = () => models.find(item => item.id === state.model) || models[1];

function recommendation() {
  if (state.analysis?.recommendation) {
    const choice = state.analysis.recommendation;
    const selected = models.find(item => item.id === choice.model.id) || models[1];
    return {model:selected, method:choice.method, epochs:state.analysis.facts.validRows < 500 ? 2 : state.analysis.facts.validRows > 50000 ? 1 : 3};
  }
  const profile = state.profile;
  let modelId = "Qwen/Qwen2.5-7B-Instruct";
  if (profile.task === "code") modelId = "Qwen/Qwen2.5-Coder-7B-Instruct";
  else if (profile.priority === "speed") modelId = "Qwen/Qwen2.5-3B-Instruct";
  else if (profile.language === "english" && profile.task === "writing") modelId = "mistralai/Mistral-7B-Instruct-v0.3";
  const selected = models.find(item => item.id === modelId);
  const method = selected.size === "3B" && profile.priority === "quality" ? "lora" : "qlora";
  const epochs = profile.examples === "tiny" ? 2 : profile.examples === "large" ? 1 : 3;
  return {model:selected, method, epochs};
}

function toast(title, message) {
  const node = document.createElement("div");
  node.className = "toast";
  node.innerHTML = "<strong>" + escapeHtml(title) + "</strong><span>" + escapeHtml(message) + "</span>";
  document.querySelector("#toast-region").append(node);
  setTimeout(() => node.remove(), 4200);
}

function backDestination() {
  const [page, id] = route().split("/");
  if (page === "configure" || page === "models" || page === "runs") return "#/workspace";
  if (page === "workflow") return "#/workspace";
  if (page === "train") return "#/runs";
  if (page === "evaluate") return "#/train/" + id;
  return "#/workspace";
}

function shell(kicker, title, lead, actions = "") {
  const needsBack = route() !== "workspace" && !actions.includes("id='back-");
  const back = needsBack ? "<a class='button page-back' href='" + backDestination() + "'>← Back</a>" : "";
  const controls = back || actions ? "<div class='page-actions'>" + back + actions + "</div>" : "";
  return "<div class='page'><div class='topbar'><div><div class='kicker'>" + kicker + "</div><h1>" + title + "</h1><p class='lead'>" + lead + "</p></div>" + controls + "</div>";
}

function steps(active, skipped = []) {
  return "<div class='steps'>" + ["Analyze dataset", "Test base & prompts", "Review & train", "Compare & decide"].map((name, index) =>
    skipped.includes(index) ? "<div class='step skipped'><b>–</b>" + name + " (skipped)</div>" :
    "<div class='step " + (index === active ? "active" : index < active ? "done" : "") + "'><b>" + (index < active ? "✓" : index + 1) + "</b>" + name + "</div>"
  ).join("") + "</div>";
}

// A complete run may have ended before testing or training (e.g. the base model was kept).
function skippedSteps(workflow) {
  if (workflow.status !== "complete") return [];
  const current = workflow.state || workflow;
  return [current.baseline_results?.length || current.prompt_trials?.length ? null : 1, Object.keys(current.training_runs || {}).length ? null : 2].filter(index => index !== null);
}

function setNav() {
  const page = route().split("/")[0];
  document.querySelectorAll("[data-route]").forEach(link => link.classList.toggle("active", link.dataset.route === page));
}

function datasetInput() {
  const labels = ["Connecting to dataset…", "Inspecting schema and valid rows…", "Profiling quality and GPU fit…"];
  const widths = [28, 58, 82];
  const checking = state.isAnalyzing ? "<div class='check-progress' role='status' aria-live='polite'><div><span>" + labels[state.analysisStage] + "</span><strong>Checking</strong></div><div class='progress'><i style='width:" + widths[state.analysisStage] + "%'></i></div></div>" : "";
  if (state.source === "huggingface") {
    return "<div class='form-row'><label class='field'>Dataset ID<input id='hf-id' placeholder='e.g. tatsu-lab/alpaca' value='" +
      (state.dataset?.source === "huggingface" ? escapeHtml(state.dataset.name) : "") +
      "'></label><label class='field'>Configuration (optional)<input id='hf-config' placeholder='default'></label></div><button class='button primary' id='validate-hf' style='margin-top:12px' " + (state.isAnalyzing ? "disabled" : "") + ">" + (state.isAnalyzing ? "Checking dataset…" : "Check dataset") + "</button>" + checking;
  }
  const selected = state.dataset?.source === "upload" ? "<div class='notice'>✓ " + escapeHtml(state.dataset.name) + " is ready. Detected " + escapeHtml(state.dataset.format) + " with " + state.dataset.rows + " examples.</div>" : "";
  return "<div class='dropzone' style='margin-top:13px'><strong>Choose a data file</strong><p>JSONL, CSV, or Parquet with chat messages or prompt/completion columns.</p><input id='file-input' type='file' accept='.jsonl,.csv,.parquet' " + (state.isAnalyzing ? "disabled" : "") + "><button class='button' id='sample-data' style='margin-left:8px' " + (state.isAnalyzing ? "disabled" : "") + ">Use a small sample instead</button></div>" + checking + selected;
}

function datasetCheckReport() {
  if (!state.analysis || state.isAnalyzing) return "";
  const analysis = state.analysis;
  const blockers = datasetBlockers();
  const findings = analysis.findings || [];
  const facts = analysis.facts || {};
  const items = findings.map(item => "<li class='" + escapeHtml(item.severity) + "'><strong>" + escapeHtml(item.code.replaceAll("_", " ")) + "</strong><span>" + escapeHtml(item.message) + "</span></li>").join("");
  return "<section class='dataset-check " + (blockers.length ? "blocked" : "ready") + "'><div class='section-head'><div><h2>Dataset check</h2><p>Schema, row validity, size, duplicates, and language are checked before model selection.</p></div><span class='pill " + (blockers.length ? "danger" : "complete") + "'>" + (blockers.length ? "action needed" : "passed") + "</span></div><div class='dataset-facts'><div><span>Schema</span><strong>" + escapeHtml(facts.schema || "unknown") + "</strong></div><div><span>Valid rows</span><strong>" + (facts.validRows ?? "—") + " / " + (facts.rows ?? "—") + "</strong></div><div><span>Invalid rows</span><strong>" + (facts.rejectedRows ?? "—") + "</strong></div><div><span>Duplicates</span><strong>" + (Number.isFinite(facts.duplicateRate) ? Math.round(facts.duplicateRate * 100) + "%" : "—") + "</strong></div></div>" + (items ? "<ul class='dataset-findings'>" + items + "</ul>" : "") + "</section>";
}

function datasetCheckRequired() {
  if (!state.dataset || state.analysis || state.isAnalyzing || state.datasetError) return "";
  return "<div class='notice dataset-check-needed'><strong>Dataset check required</strong><br>Check schema, size, invalid rows, and duplicates before selecting a model.</div>";
}

function datasetWorkspace() {
  const selected = model();
  app.innerHTML = shell("Fine-tuning workspace", "Train a model that knows your work.", "Bring a dataset, choose a safe preset, and ForgeTune handles the local training workflow.", "<button class='button' id='open-runs'>View run history</button>") +
    steps(datasetReady() ? 1 : 0) +
    "<div class='grid columns'><section class='card'><div class='section-head'><div><h2>1. Add your training data</h2><p>Use files you own or import a public Hugging Face dataset.</p></div></div><div class='body-pad'><div class='choice-list'>" +
    sourceChoice("upload", "Upload a dataset", "JSONL, CSV, or Parquet. We check the format before training.") +
    sourceChoice("huggingface", "Import from Hugging Face", "Use a public dataset ID and optionally choose a configuration.") +
    "</div><div id='dataset-input'>" + datasetInput() + datasetCheckRequired() + (state.datasetError ? "<div class='notice dataset-error'><strong>Dataset could not be imported</strong><br>" + escapeHtml(state.datasetError) + "</div>" : "") + datasetCheckReport() + "</div></div></section><aside class='card'><div class='section-head'><div><h2>Recommended starting point</h2><p>Based on your visible 24 GB GPU.</p></div></div><div class='summary'>" +
    summary("Base model", selected.name + " " + selected.size) + summary("Training method", "4-bit QLoRA") + summary("Expected VRAM", selected.vram) + summary("Held-out evaluation", "10% split") +
    "<button class='button primary' id='continue-model' " + (datasetReady() ? "" : "disabled") + ">" + (datasetBlockers().length ? "Resolve dataset checks first" : state.dataset && !state.analysis ? "Check dataset first" : "Continue to model setup →") + "</button></div></aside></div></div>";
  document.querySelectorAll("input[name=source]").forEach(input => input.addEventListener("change", () => { state.source = input.value; datasetWorkspace(); }));
  document.querySelector("#open-runs").onclick = () => location.hash = "#/runs";
  document.querySelector("#continue-model")?.addEventListener("click", () => location.hash = "#/configure");
  document.querySelector("#sample-data")?.addEventListener("click", () => {
    state.file = null;
    state.dataset = {name:"Customer support replies (sample)", format:"prompt / completion", rows:240, source:"upload"};
    state.analysis = demoAnalysis();
    datasetWorkspace(); toast("Sample dataset ready", "240 examples will be split 90/10 for training and evaluation.");
  });
  document.querySelector("#file-input")?.addEventListener("change", async event => {
    const file = event.target.files[0]; if (!file) return;
    state.file = file; toast("Analyzing dataset", "Checking format, language, quality, and GPU fit locally.");
    await uploadAndAnalyze(file);
  });
  document.querySelector("#validate-hf")?.addEventListener("click", async () => {
    const id = document.querySelector("#hf-id").value.trim();
    if (!id) return toast("Dataset ID needed", "Enter a Hugging Face dataset ID first.");
    state.datasetError = null;
    state.dataset = {name:id, format:"auto-detect", rows:"pending import", source:"huggingface"};
    await analyzeCurrentDataset();
  });
}

function demoAnalysis() {
  return {facts:{schema:"prompt_completion",rows:240,validRows:240,rejectedRows:0,languages:{"English / Latin":210,"Chinese / CJK":30},codeRatio:0.01,duplicateRate:0.01,lengthCharacters:{p50:186,p95:620},estimatedTokenLength:{p50:54,p95:177}},classification:{task:"assistant",confidence:.92,source:"local"},findings:[{severity:"warning",code:"small_dataset",message:"Fewer than 500 examples; use low rank and review overfitting carefully."}],redaction:{sampleRows:20,fieldsRedacted:0,sentToCloud:false},blockers:[],recommendation:{model:{id:"Qwen/Qwen2.5-7B-Instruct"},method:"qlora"},candidates:[]};
}

async function uploadAndAnalyze(file) {
  try {
    const form = new FormData(); form.append("file", file);
    const upload = await fetch("/api/datasets/upload", {method:"POST", body:form});
    if (!upload.ok) throw new Error();
    state.dataset = await upload.json();
    state.dataset.rows = "analyzed";
    await analyzeCurrentDataset();
  } catch {
    const extension = file.name.split(".").pop().toUpperCase();
    state.dataset = {name:file.name, format:extension, rows:"preview only", source:"upload"};
    state.analysis = demoAnalysis(); datasetWorkspace();
    toast("Preview analysis shown", "Start the WSL trainer for a real local profile.");
  }
}

async function analyzeCurrentDataset(cloudAssist = false) {
  const token = crypto.randomUUID();
  state.isAnalyzing = true; state.analysisStage = 0; state.analysisToken = token; state.datasetError = null;
  render();
  const timers = [
    setTimeout(() => { if (state.analysisToken === token) { state.analysisStage = 1; render(); } }, 450),
    setTimeout(() => { if (state.analysisToken === token) { state.analysisStage = 2; render(); } }, 1200)
  ];
  try {
    const response = await fetch("/api/datasets/analyze", {method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify({dataset:state.dataset,goal:state.goal,cloud_assist:cloudAssist})});
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "The dataset could not be loaded.");
    }
    state.analysis = await response.json();
    state.datasetError = null;
    state.cloudAssist = cloudAssist;
    state.dataset.rows = state.analysis.facts.rows;
    toast(state.analysis.blockers?.length ? "Dataset needs attention" : "Dataset analysis complete", state.analysis.blockers?.length ? state.analysis.blockers[0].message : "One goal choice remains before training.");
  } catch (error) {
    state.analysis = null;
    state.datasetError = error?.message || "The trainer could not analyze this dataset.";
    toast("Dataset import failed", state.datasetError);
  } finally {
    timers.forEach(clearTimeout);
    if (state.analysisToken === token) {
      state.isAnalyzing = false; state.analysisToken = null; render();
    }
  }
}

function sourceChoice(id, title, note) {
  return "<label class='choice " + (state.source === id ? "selected" : "") + "'><input type='radio' name='source' value='" + id + "' " + (state.source === id ? "checked" : "") + "><div><strong>" + title + "</strong><span>" + note + "</span></div></label>";
}

function summary(label, value) {
  return "<div class='summary-row'><span>" + label + "</span><strong>" + value + "</strong></div>";
}

function methodChoice(id, title, note, disabled) {
  return "<label class='choice " + (state.method === id ? "selected" : "") + "'><input type='radio' name='method' value='" + id + "' " + (state.method === id ? "checked" : "") + (disabled ? " disabled" : "") + "><div><strong>" + title + (disabled ? " · unavailable for 14B" : "") + "</strong><span>" + note + "</span></div></label>";
}

function agentCheck(name, verdict, detail) {
  const tone = verdict === "Pass" ? "pass" : verdict === "Caution" ? "warn" : "info";
  return "<div class='agent-check'><span class='agent-icon'>✦</span><div><strong>" + name + "</strong><p>" + detail + "</p></div><em class='" + tone + "'>" + verdict + "</em></div>";
}

function configure() {
  const analysis = state.analysis || demoAnalysis(), selected = model(), needsQ = selected.size === "14B", rec = recommendation();
  const blockers = analysis.blockers || analysis.findings.filter(item => item.severity === "blocker");
  if (blockers.length) { location.hash = "#/workspace"; return; }
  const findings = analysis.findings.map(item => agentCheck(item.code.replaceAll("_", " "), item.severity === "blocker" ? "Blocked" : "Caution", item.message)).join("");
  const facts = analysis.facts;
  app.innerHTML = shell("Automatic training plan", blockers.length ? "Your dataset needs attention." : "Your dataset is profiled.", blockers.length ? "Training is paused until the dataset auditor’s blocker is resolved." : "The advisor detected the dataset profile. Choose only the result you value most.", "<button class='button' id='back-data'>← Dataset</button>") +
    steps(1) + "<section class='card advisor-card'><div class='section-head'><div><h2>Dataset auditor</h2><p>These are measured locally from your dataset; no cloud model was needed.</p></div><span class='pill " + (blockers.length ? "" : "complete") + "'>" + (blockers.length ? "blocked" : "ready") + "</span></div><div class='advisor-layout'><div class='fact-grid'>" +
    "<div><span>Valid examples</span><strong>" + facts.validRows + " / " + facts.rows + "</strong></div><div><span>Detected task</span><strong>" + analysis.classification.task + "</strong></div><div><span>Language mix</span><strong>" + Object.keys(facts.languages).join(" · ") + "</strong></div><div><span>Duplicate rate</span><strong>" + Math.round(facts.duplicateRate * 100) + "%</strong></div><div><span>Typical length</span><strong>" + facts.estimatedTokenLength.p50 + " tokens</strong></div><div><span>Long examples</span><strong>p95 " + facts.estimatedTokenLength.p95 + " tokens</strong></div></div><div><div class='advisor-consensus'><span class='eyebrow'>One decision from you</span><h2>What matters most?</h2><div class='goal-options'>" + ["speed","balanced","quality"].map(goal => "<label><input type='radio' name='goal' value='" + goal + "' " + (state.goal === goal ? "checked" : "") + "> " + (goal === "speed" ? "Fastest iteration" : goal === "quality" ? "Highest quality" : "Balanced") + "</label>").join("") + "</div><button class='button' id='cloud-test'>Test cloud assist</button>" + (state.cloudStatus?.configured && analysis.classification.confidence < .8 ? "<button class='button' id='cloud-clarify'>Clarify ambiguous task</button>" : "") + "<p id='cloud-note'>Cloud assist: " + (state.cloudStatus?.configured ? "configured; used only for ambiguous classifications" : "not configured; local analysis is complete") + "</p></div><div class='agent-checks'>" + findings + agentCheck("Model & method planner", blockers.length ? "Blocked" : "Pass", blockers.length ? "No plan is produced until data blockers are fixed." : rec.model.name + " " + rec.model.size + " + " + rec.method.toUpperCase() + " fits the local GPU plan.") + agentCheck("Evaluation auditor", "Required", "Use the same held-out slice for base and tuned comparison; training loss alone is insufficient.") + "</div></div></div></section>" +
    "<div class='grid columns' style='margin-top:16px'><section class='card'><div class='section-head'><div><h2>Recommended base model</h2><p>Automatically ranked from dataset fit, local GPU, and your goal.</p></div></div><div class='body-pad'><label class='field'>Model<select id='model-select'>" +
    models.map(item => "<option value='" + item.id + "' " + (item.id === selected.id ? "selected" : "") + ">" + item.name + " " + item.size + " · " + item.vram + " VRAM · " + item.speed + "</option>").join("") +
    "</select></label><div class='notice'>" + (needsQ ? "14B needs QLoRA to fit in the detected 24 GB GPU. It will train more slowly." : selected.size + " is expected to fit comfortably with the recommended QLoRA preset.") +
    "</div><div class='peft-note'><strong>PEFT is the category—not a third method.</strong><span>LoRA and QLoRA both train small adapters instead of changing every base-model weight.</span></div><h3 style='margin:18px 0 9px'>Fine-tuning method</h3><div class='choice-list'>" +
    methodChoice("lora", "LoRA · BF16 base", "Best fidelity when the model fits comfortably in GPU memory.", needsQ) +
    methodChoice("qlora", "QLoRA · 4-bit base", "Safest default on 24 GB; required for 14B models.", false) +
    "</div></div></section><aside class='card'><div class='section-head'><div><h2>Safe preset</h2><p>Optimized with gradient descent for a first useful run.</p></div></div><div class='summary'>" +
    summary("Rank / alpha", facts.validRows < 500 ? "8 / 16" : "16 / 32") + summary("Learning rate", "2e-4") + summary("Epochs", String(rec.epochs)) + summary("Batch / accumulation", "1 / 8") + summary("Context length", Math.min(4096, Math.max(1024, facts.estimatedTokenLength.p95 * 2)) + " tokens") +
    "<button class='button primary' id='start-train' " + (blockers.length ? "disabled" : "") + ">Start local training →</button></div></aside></div></div>";
  document.querySelector("#back-data").onclick = () => location.hash = "#/dataset";
  document.querySelectorAll("input[name=goal]").forEach(input => input.onchange = async () => { state.goal = input.value; await analyzeCurrentDataset(); });
  document.querySelector("#cloud-test").onclick = async () => { const response = await fetch("/api/advisor/connection-test", {method:"POST"}); const result = await response.json(); toast(result.ok ? "Cloud assist connected" : "Cloud assist unavailable", result.message); };
  document.querySelector("#cloud-clarify")?.addEventListener("click", () => analyzeCurrentDataset(true));
  document.querySelector("#model-select").onchange = event => { state.model = event.target.value; if (model().size === "14B") state.method = "qlora"; configure(); };
  document.querySelectorAll("input[name=method]").forEach(input => input.onchange = () => { state.method = input.value; configure(); });
  document.querySelector("#start-train").onclick = startWorkflow;
}

async function startWorkflow() {
  if (!state.dataset) return location.hash = "#/workspace";
  try {
    const dataset = {
      source: state.dataset.source,
      name: state.dataset.name,
      extension: state.dataset.extension || state.dataset.format,
      split: state.dataset.split || "train"
    };
    const response = await fetch("/api/workflow/runs", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({dataset, goal:state.goal, cloud_assist:Boolean(state.cloudAssist), candidate_limit:1, max_retries:2, split_seed:42})
    });
    if (!response.ok) throw new Error("Workflow could not be started.");
    const workflow = await response.json();
    location.hash = "#/workflow/" + workflow.id;
    pollWorkflow(workflow.id);
    toast("Guided workflow started", "ForgeTune is profiling the dataset before any GPU training begins.");
  } catch {
    toast("Workflow unavailable", "Start the local trainer and try again. No training run was created.");
  }
}

const workflowPollers = new Map();

function pollWorkflow(id) {
  if (workflowPollers.has(id)) return;
  const interval = setInterval(async () => {
    try {
      const response = await fetch("/api/workflow/runs/" + id);
      if (!response.ok) return;
      const workflow = await response.json();
      // Rebuilding unchanged data collapses open sections and moves the scroll position.
      if (route() === "workflow/" + id && JSON.stringify(workflow) !== JSON.stringify(window.currentWorkflow)) renderWorkflow(workflow);
      if (["waiting", "complete", "blocked", "cancelled", "failed", "interrupted"].includes(workflow.status)) {
        clearInterval(interval); workflowPollers.delete(id);
      }
    } catch { /* The trainer may be restarting; keep polling. */ }
  }, 2500);
  workflowPollers.set(id, interval);
}

// The section the user just answered stays on screen, locked, until the workflow reaches its next step.
let submittedStep = null;

function lockedCopy(section) {
  const copy = section.cloneNode(true);  // cloneNode keeps typed values and checkboxes; outerHTML would not.
  [copy, ...copy.querySelectorAll("[id]")].forEach(node => node.removeAttribute("id"));
  copy.querySelectorAll("input, textarea, select, button").forEach(control => control.disabled = true);
  copy.inert = true;
  copy.classList.add("submitted-step");
  copy.insertAdjacentHTML("afterbegin", "<span class='pill complete submitted-badge'>Submitted</span>");
  return copy;
}

function workflowWorking(workflow) {
  const progress = workflow.progress;
  const since = progress?.startedAt ? "since " + new Date(progress.startedAt * 1000).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"}) : "working";
  // Steps that can't count their work (e.g. waiting on Compass) report no percent and get a moving bar.
  const percent = Number.isFinite(progress?.percent) ? progress.percent : null;
  const bar = percent === null ? "<div class='progress indeterminate'><i></i></div>" : "<div class='progress' role='progressbar' aria-valuemin='0' aria-valuemax='100' aria-valuenow='" + percent + "'><i style='width:" + percent + "%'></i></div>";
  return "<div class='check-progress workflow-working' role='status' aria-live='polite'><div><span>" + escapeHtml(progress?.message || "Starting the next step…") + "</span><strong>" + (percent === null ? since : percent + "% · " + since) + "</strong></div>" + bar + "</div>";
}

function workflowAction(id, response, message) {
  const source = document.activeElement?.closest?.("#app form, #app section");
  const buttons = source ? [...source.querySelectorAll("button:not(:disabled)")] : [];
  buttons.forEach(button => button.disabled = true);
  fetch("/api/workflow/runs/" + id + "/resume", {
    method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify({response})
  }).then(result => {
    if (!result.ok) throw new Error();
    submittedStep = source ? {workflowId:id, node:lockedCopy(source)} : null;
    pollWorkflow(id); renderWorkflow({...window.currentWorkflow, status:"queued", pendingAction:null, progress:null});
    toast("Workflow resumed", message);
  }).catch(() => {
    buttons.forEach(button => button.disabled = false);
    toast("Could not resume workflow", "The trainer may be busy or offline.");
  });
}

function gradeDescription(grade) {
  return grade ? `<p><strong>${escapeHtml(grade.task)} · ${escapeHtml(grade.verdict)}</strong> ${escapeHtml(grade.reason)}</p>` : "";
}

function gradingSummary(result) {
  const q = result.qualityEvidence || {};
  if (!q.estimated) return "";
  return `<p>Local judge: ${escapeHtml(q.judgeModel)} · ${escapeHtml(q.rubricVersion)} · ${q.evaluatedExamples || 0}/${q.totalExamples || 0} answers graded · ${q.passedExamples || 0} pass · ${q.failedExamples || 0} fail · ${q.reviewExamples ?? Math.max(0, (q.totalExamples || 0) - (q.evaluatedExamples || 0))} need review. ${escapeHtml(q.message || "")}</p>`;
}

function recordedSamples(base, tuned = {}) {
  const samples = base.samples || [];
  if (!samples.length) return "<p>No recorded baseline responses are available.</p>";
  return samples.map(sample => {
    const after = (tuned.samples || []).find(item => item.rowId === sample.rowId);
    return "<details class='experiment-example'><summary>Evaluation row " + escapeHtml(sample.rowId) + "</summary><strong>Input</strong><pre class='code'>" + escapeHtml(sample.input || "Input not retained") + "</pre>" + (sample.retrieved?.length ? "<p>Retrieved sources: " + sample.retrieved.map(item => escapeHtml(item.title || item.id)).join(", ") + "</p>" : "") + "<strong>Base response</strong><pre class='code'>" + escapeHtml(sample.baseOutput || "") + "</pre>" + gradeDescription(sample.grade) + (sample.referenceAvailable ? "<strong>Reference answer</strong><pre class='code'>" + escapeHtml(sample.reference || "") + "</pre>" : "") + (after ? "<strong>Tuned response</strong><pre class='code'>" + escapeHtml(after.tunedOutput || "") + "</pre>" + gradeDescription(after.grade) : "") + "</details>";
  }).join("");
}

function workflowComparison(comparison) {
  if (comparison?.decision === "keep_baseline" && !comparison.pairs?.length) return "<div class='notice'><strong>Base model retained</strong> " + escapeHtml(comparison.decisionReason || "") + "</div><pre class='code'>" + escapeHtml(comparison.instruction || "Base prompt, no extra instructions") + "</pre>";
  const pairs = comparison?.pairs || [];
  if (!pairs.length) return "<div class='empty'><strong>No comparable candidates</strong><span>The workflow finished without a valid tuned result.</span></div>";
  const verdict = comparison.winner ? "A candidate improved — review before deployment" : comparison.decision === "keep_baseline" ? "Keep the base model" : "Improvement not established";
  return `<div class="notice"><strong>${escapeHtml(verdict)}</strong><p>${comparison.winner ? "Inspect the measured gains and any regressions below." : "Training completed, but these results do not justify deploying the fine-tune as an improvement."}</p></div><p>${comparison.evaluationIds?.length || 0} unseen questions · Same questions and prompt for both models</p>` + pairs.map(pair => {
    const base = pair.base || {}, tuned = pair.tuned || {};
    const cutoffs = result => (result.samples || []).filter(sample => sample.hitTokenLimit || sample.stoppedRepeating).length;
    const a = base.qualityEvidence || {}, b = tuned.qualityEvidence || {};
    const measured = a.metric && base.evaluationIds?.length > 0 && a.coverage === 1 && b.coverage === 1 && a.metric === b.metric && a.evaluatorId === b.evaluatorId && a.rubric === b.rubric && a.evaluatedExamples === base.evaluationIds?.length && b.evaluatedExamples === tuned.evaluationIds?.length && JSON.stringify(base.evaluationIds) === JSON.stringify(tuned.evaluationIds) && Number.isFinite(a.score) && Number.isFinite(b.score);
    const estimated = a.estimated || b.estimated;
    return `<h3>${escapeHtml(pair.modelId)}</h3><div class="metric-grid grid"><div class="metric"><span>${estimated ? "Estimated correctness" : "Answer accuracy"} · base → tuned</span><strong>${measured ? percentage(a.score) + " → " + percentage(b.score) : "Needs review"}</strong></div><div class="metric"><span>Cut-off answers · base → tuned</span><strong>${cutoffs(base)} → ${cutoffs(tuned)}</strong></div><div class="metric"><span>Training artifact</span><strong>${pair.passed ? "Verified" : "Needs review"}</strong></div></div><p>${estimated ? "Task-specific local model grades are estimates, not verified ground-truth accuracy. Review uncertain grades before deployment." : measured ? "Accuracy uses the task metric recorded for this run." : "No complete, comparable correctness score was recorded. Run automatic evaluation or review the unresolved grades."}</p><strong>Base</strong>${gradingSummary(base)}<strong>Tuned</strong>${gradingSummary(tuned)}<details class="experiment-example"><summary>View detailed answers and diagnostics (${Math.min(5, base.samples?.length || 0)} of ${base.samples?.length || 0} samples)</summary><p>Word overlap: ${base.meanTokenOverlap ?? "n/a"} → ${tuned.meanTokenOverlap ?? "n/a"} (diagnostic only).</p><p>${escapeHtml(comparison.decisionReason || "")}</p>${recordedSamples({...base, samples:(base.samples || []).slice(0, 5)}, tuned)}</details>`;
  }).join("");
}

// The final page: every model the run measured, and the fine-tuned adapter to take away.
function finalSummary(workflow) {
  const current = workflow.state || workflow, comparison = workflow.comparison || {};
  const pairs = comparison.pairs || [];
  const benchmark = (current.model_benchmark?.results || []).filter(result => result.status === "complete");
  const score = result => {
    const q = result?.qualityEvidence || {};
    if (q.coverage !== 1 || !Number.isFinite(q.score) || !q.evaluatedExamples) return "—";
    return Math.round(q.score * q.evaluatedExamples) + " of " + q.evaluatedExamples + " correct (" + Math.round(q.score * 100) + "%)" + (q.estimated ? "<br><small>graded by the local judge</small>" : "");
  };
  const rows = [];
  const tunedRow = pair => rows.push(["<strong>" + escapeHtml(pair.modelId) + " — fine-tuned</strong>", "Your VM", "—", "<strong>" + score(pair.tuned) + "</strong>"]);
  for (const result of benchmark) {
    const pair = pairs.find(item => item.modelId === result.modelId);
    rows.push([escapeHtml(result.modelId), result.provider === "local" ? "Your VM" : "Hosted, comparison only", score(result), pair ? score(pair.base) : "—"]);
    if (pair) tunedRow(pair);
  }
  for (const pair of pairs.filter(item => !benchmark.some(result => result.modelId === item.modelId))) {
    rows.push([escapeHtml(pair.modelId), "Your VM", "—", score(pair.base)]);
    tunedRow(pair);
  }
  const verdict = comparison.winner ? "The fine-tune improved on unseen questions — review before deployment." : pairs.length ? "Training finished, but the fine-tune did not clearly beat the base model." : "No model was fine-tuned in this run, so there is no fine-tuned model to export." + (Number.isInteger(comparison.selectedTrial) ? " You chose to keep the base model at prompt review." : "");
  const trained = pairs.find(pair => pair.modelId === comparison.winner) || pairs.find(pair => pair.passed);
  // Without a fine-tune there are no unseen-test scores, so that column is left out.
  const columns = pairs.length ? 4 : 3;
  const head = ["Model", "Runs on", "Benchmark: correct answers", "Unseen test: correct answers"].slice(0, columns);
  const table = rows.length ? "<div class='benchmark-table'><table><thead><tr><th>" + head.join("</th><th>") + "</th></tr></thead><tbody>" + rows.map(row => "<tr><td>" + row.slice(0, columns).join("</td><td>") + "</td></tr>").join("") + "</tbody></table></div><p>Each model answered the same " + (current.model_benchmark?.developmentIds?.length || 0) + " benchmark questions; the counts are answers graded correct against the reference. The benchmark picked the model to fine-tune." + (pairs.length ? " The " + (comparison.evaluationIds?.length || 0) + " unseen test questions were never used for picking or training." : "") + "</p>" : "";
  const exporter = trained?.passed ? "<p><a class='button primary' href='/api/workflow/runs/" + encodeURIComponent(workflow.id) + "/export' download>Export fine-tuned model (.zip)</a> LoRA adapter for " + escapeHtml(trained.modelId) + "; load it on top of that base model with PEFT.</p>" : "";
  return "<div class='notice'><strong>" + escapeHtml(verdict) + "</strong></div>" + table + exporter + (pairs.length ? "<details class='experiment-example'><summary>Detailed answers and diagnostics</summary>" + workflowComparison(comparison) + "</details>" : "");
}

function modelPlanRow(id, provider, reason, checked) {
  return `<label class="check-row"><input type="checkbox" data-model="${escapeHtml(id)}" data-provider="${provider === "compass" ? "compass" : "local"}"${checked ? " checked" : ""}><span><strong>${escapeHtml(id)}</strong> · ${provider === "compass" ? "Compass" : "Your VM"}${reason ? "<br>" + escapeHtml(reason) : ""}</span></label>`;
}

function modelPlanReview(pending) {
  const proposal = pending.proposal || {}, chosen = proposal.models || [];
  const proposed = new Set(chosen.map(item => item.id));
  const rows = chosen.map(item => modelPlanRow(item.id, item.provider, item.reason, true)).join("") +
    (pending.localOptions || []).filter(id => !proposed.has(id)).map(id => modelPlanRow(id, "local", "", false)).join("");
  const excluded = (proposal.excluded || []).map(item => "<li><strong>" + escapeHtml(item.id) + "</strong> · " + escapeHtml(item.reason) + "</li>").join("");
  return `<form id="model-plan" class="card body-pad experiment-form" data-compass="${escapeHtml(JSON.stringify(pending.compassOptions || []))}"><h2>Approve models to benchmark</h2><p>${escapeHtml(pending.message || "Approve or change the models to benchmark.")}</p>` +
    (proposal.note ? `<div class="notice">${escapeHtml(proposal.note)}</div>` : "") +
    (proposal.recommendation ? `<p>${escapeHtml(proposal.recommendation)}</p>` : "") +
    `<div id="model-plan-rows">${rows}</div><div id="compass-add-section"><label>Add a Compass model<select id="compass-add" disabled><option value="">Checking which Compass models respond…</option></select></label><p id="compass-add-status" role="status"></p><div><button class="button" type="button" id="add-compass" disabled>Add</button> <button class="button" type="button" id="recheck-compass">Recheck all</button></div></div>` +
    (excluded ? `<h3>Not suggested</h3><ul class="resolution-list">${excluded}</ul>` : "") +
    (pending.privacyFlagged ? `<div class="notice" id="model-plan-privacy">This dataset was flagged for possible private data. Approving a list with Compass models confirms you are happy to send those questions to Compass.</div>` : "") +
    `<p id="model-plan-limit" role="status"></p><button class="button primary" type="button" id="approve-models">Approve and run</button> <button class="button" type="button" id="abort-workflow">Cancel</button></form>`;
}

function bindModelPlan(workflow) {
  const form = document.querySelector("#model-plan");
  if (!form) return;
  const boxes = () => [...form.querySelectorAll("input[data-model]")].filter(box => box.checked);
  const sync = () => {
    const picked = boxes(), compass = picked.filter(box => box.dataset.provider === "compass").length, local = picked.length - compass;
    const problem = picked.length < 2 ? "Choose at least 2 models." : picked.length > 7 ? "Choose at most 7 models." : local > 4 ? "Choose at most 4 of your own models." : compass > 3 ? "Choose at most 3 Compass models." : "";
    form.querySelector("#model-plan-limit").textContent = problem;
    form.querySelector("#approve-models").disabled = Boolean(problem);
    const privacy = form.querySelector("#model-plan-privacy");
    if (privacy) privacy.style.display = compass ? "" : "none";
  };
  form.addEventListener("change", sync);
  const select = form.querySelector("#compass-add"), status = form.querySelector("#compass-add-status");
  const fillCompassOptions = models => {
    const listed = new Set([...form.querySelectorAll("input[data-model]")].map(box => box.dataset.model));
    const options = models.filter(id => JSON.parse(form.dataset.compass).includes(id) && !listed.has(id));
    select.innerHTML = options.length ? '<option value="">Choose a model</option>' + options.map(id => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join("") : '<option value="">No other Compass models responded</option>';
    select.disabled = !options.length;
    form.querySelector("#add-compass").disabled = true;
  };
  const loadCompassModels = async () => {
    if (!document.body.contains(form)) return;
    try {
      const response = await fetch("/api/compass/available");
      const result = await response.json().catch(() => ({}));
      if (response.status === 503) { form.querySelector("#compass-add-section").innerHTML = "<p>Compass isn't configured on the trainer, so only your own models can be benchmarked.</p>"; return; }
      if (!response.ok || result.status === "failed") throw new Error(result.error || result.detail || "The trainer could not reach Compass.");
      if (result.status === "checking") {
        status.textContent = result.total ? `Checking which Compass models respond: ${result.checked} of ${result.total}` : "Loading the Compass catalog…";
        return setTimeout(loadCompassModels, 2000);
      }
      status.textContent = `Only models that answered a quick check are listed (checked ${new Date(result.checkedAt).toLocaleString()}; models a run finds refused are removed). Rechecked models show from the next run.` + (result.error ? ` The last recheck failed: ${result.error}` : "");
      fillCompassOptions(result.models);
    } catch (error) {
      select.innerHTML = '<option value="">Compass models unavailable</option>';
      status.textContent = error.message;
    }
  };
  loadCompassModels();
  form.querySelector("#recheck-compass").onclick = async () => {
    await fetch("/api/compass/available?refresh=true").catch(() => {});
    loadCompassModels();
  };
  select.addEventListener("change", () => form.querySelector("#add-compass").disabled = !select.value);
  form.querySelector("#add-compass").onclick = async () => {
    const input = select, add = form.querySelector("#add-compass"), id = input.value;
    if (!id) return;
    // Recheck: the saved list can be old, and a refusal here takes the model off it.
    add.disabled = true; add.textContent = "Checking…";
    try {
      const response = await fetch("/api/compass/check?model=" + encodeURIComponent(id));
      const result = await response.json().catch(() => ({}));
      if (!response.ok) toast("Could not check the model", result.detail || "The trainer could not reach Compass.");
      else if (!result.ok) toast("Compass can't run this model", result.message);
      else {
        form.querySelector("#model-plan-rows").insertAdjacentHTML("beforeend", modelPlanRow(id, "compass", "Added by you", true));
        input.selectedOptions[0].remove();
        input.value = "";
        sync();
      }
    } catch {
      toast("Could not check the model", "The trainer may be busy or offline.");
    } finally {
      add.disabled = !input.value; add.textContent = "Add";
    }
  };
  // No submit button, so Enter (on a checkbox or anywhere) can never approve; only a click on the button does.
  form.onsubmit = event => event.preventDefault();
  form.querySelector("#approve-models").onclick = () => workflowAction(workflow.id, {action:"approve", models:boxes().map(box => ({id:box.dataset.model, provider:box.dataset.provider}))}, "Models approved. Benchmarking is starting.");
  sync();
}

function renderWorkflow(workflow) {
  window.currentWorkflow = workflow;
  if (workflow.status === "failed" && workflow.error && !workflow.state?.split_manifest_path) {
    const dataset = workflow.dataset?.name || workflow.state?.dataset?.name || "the selected dataset";
    app.innerHTML = shell("Dataset import failed", "ForgeTune could not read " + escapeHtml(dataset) + ".", "No model baseline or GPU training was started.", "<button class='button primary' id='replace-dataset'>Choose another dataset</button>") +
      steps(0) + "<section class='card body-pad'><h2>Fix this in step 1: Dataset</h2><div class='notice dataset-error'><strong>Import error</strong><br>" + escapeHtml(workflow.error) + "</div><p style='color:var(--muted);font-size:11px;line-height:1.65'>Use a current Hugging Face dataset that loads without a legacy script, or upload a CSV, JSONL, or Parquet file.</p></section></div>";
    document.querySelector("#replace-dataset").onclick = () => updateDataset(workflow);
    return;
  }
  const pending = workflow.pendingAction;
  let action = "";
  if (workflow.status === "waiting" && pending?.type === "plan_approval") {
    const candidates = pending.plan?.candidates || [];
    action = "<section class='card body-pad' style='margin-top:16px'><h2>Approve training plan</h2><p>" + escapeHtml(pending.message || "Review the proposed candidates before GPU training.") + "</p><div class='run-list'>" + candidates.map(item => "<div class='run'><div><strong>" + escapeHtml(item.model.id) + "</strong><span>" + escapeHtml(String(item.method).toUpperCase()) + " · " + item.estimatedVramGb + " GB VRAM · " + escapeHtml(item.estimatedDuration || "unknown") + " training</span></div></div>").join("") + "</div><button class='button primary' id='approve-workflow'>Approve and train candidates →</button></section>";
  } else if (workflow.status === "waiting" && pending?.type === "blocker_resolution") {
    const assessment = pending.assessment || {};
    const operations = (assessment.operations || []).map(item => "<li>" + escapeHtml(item.replaceAll("_", " ")) + "</li>").join("");
    action = "<section class='card body-pad' style='margin-top:16px'><h2>Blocker resolution review</h2><p>" + escapeHtml(pending.message || "Review the proposed data remediation.") + "</p><div class='notice dataset-error'>" + (assessment.blockers || []).map(item => escapeHtml(item.message)).join("<br>") + "</div>" + (operations ? "<h3 style='margin:16px 0 8px'>Proposed approved-only operations</h3><ul class='resolution-list'>" + operations + "</ul>" : "<p style='color:var(--muted);font-size:10px'>No safe automatic remediation is available for this blocker.</p>") + (assessment.canMaterializeAfterApproval ? "<button class='button primary' id='approve-remediation'>Approve cleanup and re-check →</button> " : "") + "<button class='button' id='replace-blocked-dataset'>Replace dataset</button> <button class='button danger' id='abort-workflow'>Abort</button></section>";
  } else if (workflow.status === "waiting" && pending?.type === "task_review") {
    action = `<form id="task-brief" class="card body-pad experiment-form"><h2>Define the task</h2><p>State what the model should do and how you will judge success. Small evaluation sets are exploratory.</p><label>Task description<textarea name="task_description" required maxlength="2000" placeholder="Classify support tickets into billing, delivery, or account"></textarea></label><label>Success metric / rubric<textarea name="success_metric" required maxlength="2000" placeholder="Label accuracy on unseen tickets; review errors per category"></textarea></label><label>Evaluation examples per phase (up to available rows)<input name="analysis_limit" type="number" min="1" max="200" value="${pending.analysisLimit || 50}" required></label><label class="check-row"><input name="ai_model_pick" type="checkbox" checked> Let AI pick models, including Compass (sends the task brief and a data summary to Compass)</label><button class="button primary">Confirm task & test models</button> <button class="button" type="button" id="abort-workflow">Cancel</button></form>`;
  } else if (workflow.status === "waiting" && pending?.type === "model_plan") {
    action = modelPlanReview(pending);
  } else if (workflow.status === "waiting" && pending?.type === "data_review") {
    action = datasetClarification(pending);
  } else if (workflow.status === "waiting" && pending?.type === "prompt_review") {
    action = workflowPromptReview(pending);
  } else if (workflow.status === "waiting" && pending?.type === "dataset_ready") {
    action = "<section class='card body-pad experiment-form'><h2>Dataset ready</h2><p>Analysis is saved. You can update the data or continue to baseline evaluation and a separate training approval.</p><button class='button primary' id='continue-dataset'>Test base model & prompts</button></section>";
  } else if (workflow.status === "waiting" && pending?.type === "retry_review") {
    action = "<section class='card body-pad' style='margin-top:16px'><h2>Retry decision</h2><p>" + escapeHtml(pending.reason || pending.message || "A candidate needs review.") + "</p><button class='button primary' id='retry-workflow'>Retry candidate</button> <button class='button' id='skip-workflow'>Skip candidate</button> <button class='button danger' id='abort-workflow'>Abort</button></section>";
  }
  if (pending?.type === "plan_approval") {
    action += "<section class='card body-pad'><h2>Baseline responses before training</h2>" + (pending.plan?.baselineResults || []).map(result => "<h3>" + escapeHtml(result.modelId || "Baseline unavailable") + "</h3>" + recordedSamples(result)).join("") + "</section>";
  }
  if (workflow.dataset?.sampleOnly) action = "<div class='notice'>Hugging Face sample: " + escapeHtml(workflow.dataset.importedRows) + " rows · subset " + escapeHtml(workflow.dataset.hubConfig || "default") + " · split " + escapeHtml(workflow.dataset.hubSplit || "train") + ". Analysis and training use this local sample, not the full Hub dataset.</div>" + action;
  else if (workflow.dataset?.hubRevision) action = "<div class='notice'>Full Hugging Face split: " + escapeHtml(workflow.dataset.importedRows) + " rows · subset " + escapeHtml(workflow.dataset.hubConfig || "default") + " · split " + escapeHtml(workflow.dataset.hubSplit || "train") + ". All imported rows are available to the training/evaluation split. The quality audit samples up to 600 rows.</div>" + action;
  action = datasetProfile(workflow.profile || workflow.state?.profile || pending?.profile) + action;
  // The final page is only the summary; the dataset audit and earlier checks belong to the earlier steps.
  if (workflow.status === "complete" && workflow.comparison) action = "";
  const isDone = ["complete", "blocked", "cancelled", "failed", "interrupted"].includes(workflow.status);
  const isRunning = workflow.status !== "waiting" && !isDone;
  if (!isRunning || submittedStep?.workflowId !== workflow.id) submittedStep = null;
  if (isRunning) action += (submittedStep ? "<div id='submitted-step-slot'></div>" : "") + workflowWorking(workflow);
  if (workflow.resumable) action += "<section class='card body-pad'><h2>Resume this run</h2><p>Finished steps are kept. The step that stopped starts again from its beginning, so a model benchmark reruns every model.</p><button class='button primary' id='continue-workflow'>Resume</button></section>";
  const comparison = workflow.comparison ? "<section class='card body-pad' style='margin-top:16px'><h2>Summary</h2>" + finalSummary(workflow) + (workflow.status === "complete" && workflow.comparison.pairs?.length ? "<button class='button' id='evaluate-accuracy'>Run automatic evaluation</button><p>Grade saved answers on all recorded unseen questions. Training is not repeated.</p>" : "") + (workflow.accuracyError ? "<p role='alert'>" + escapeHtml(workflow.accuracyError) + "</p>" : "") + "</section>" : "";
  app.innerHTML = shell("LangGraph workflow", workflow.status === "waiting" ? "Your review is needed." : workflow.status === "interrupted" ? "This run stopped before it finished." : isDone ? "Workflow results are ready." : workflow.candidate_status === "verified" ? "Evaluating your fine-tune." : "ForgeTune is working.", workflow.status === "waiting" ? "The graph is checkpointed and waiting for your decision." : "Dataset analysis, prompt experiments, and optional fine-tuning are recorded in one resumable workflow.", "<button class='button' id='back-config'>← Datasets</button> <button class='button' id='update-dataset'>Update dataset</button>") +
    steps(workflow.status === "complete" ? 3 : ["dataset_ready", "data_review", "blocker_resolution"].includes(pending?.type) ? 0 : pending?.type === "plan_approval" || Object.keys(workflow.training_runs || {}).length ? 2 : 1, skippedSteps(workflow)) + "<section class='card body-pad'><div class='section-head' style='padding:0 0 15px'><div><h2>" + escapeHtml(workflow.dataset?.displayName || workflow.state?.dataset?.displayName || "Dataset workflow") + "</h2><p>Task: " + escapeHtml(datasetTasks[workflow.profile?.classification?.task || workflow.state?.profile?.classification?.task] || "Dataset analysis") + " · Candidates: " + (workflow.approved_plan?.candidateCount || workflow.state?.max_candidates || 3) + "</p></div><span class='pill " + (workflow.status === "complete" ? "complete" : "") + "'>" + escapeHtml(workflow.status) + "</span></div><p style='color:var(--muted);font-size:11px;margin-top:12px'>" + (workflow.error ? escapeHtml(workflow.error) : workflow.status === "waiting" ? "No GPU work will start until this checkpoint is resolved." : isDone ? "All available workflow results have been persisted locally." : "The trainer will update this page automatically.") + "</p></section>" + action + comparison + (workflow.status === "complete" && workflow.comparison ? "" : workflowTools(workflow)) + "</div>";
  if (submittedStep) document.querySelector("#submitted-step-slot").replaceWith(submittedStep.node);
  bindWorkflowTools(workflow);
  bindModelPlan(workflow);
  document.querySelector("#evaluate-accuracy")?.addEventListener("click", async event => {
    event.currentTarget.disabled = true;
    try {
      const response = await fetch("/api/workflow/runs/" + workflow.id + "/evaluate", {method:"POST"});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Automatic evaluator is unavailable.");
      renderWorkflow({...workflow, status:"evaluating_accuracy", comparison:null});
      pollWorkflow(workflow.id);
      toast("Automatic evaluation started", "The local judge is grading saved base and tuned answers.");
    } catch (error) { toast("Evaluation unavailable", error.message); renderWorkflow(workflow); }
  });
  document.querySelector("#continue-workflow")?.addEventListener("click", async event => {
    event.currentTarget.disabled = true;
    try {
      const response = await fetch("/api/workflow/runs/" + workflow.id + "/continue", {method:"POST"});
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.detail || "The trainer may be busy or offline.");
      renderWorkflow({...workflow, status:"queued", error:null, resumable:false, progress:null});
      pollWorkflow(workflow.id);
      toast("Workflow resumed", "Continuing from the last saved checkpoint.");
    } catch (error) { toast("Could not resume workflow", error.message); renderWorkflow(workflow); }
  });
  document.querySelector("#back-config").onclick = () => location.hash = "#/workspace";
  document.querySelector("#update-dataset").onclick = () => updateDataset(workflow);
  document.querySelector("#continue-dataset")?.addEventListener("click", () => workflowAction(workflow.id, {action:"continue_training"}, "Starting model evaluation; training will require approval."));
  document.querySelector("#task-brief")?.addEventListener("submit", event => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    workflowAction(workflow.id, {task_description:form.get("task_description"), success_metric:form.get("success_metric"), analysis_limit:Number(form.get("analysis_limit")), ai_model_pick:form.get("ai_model_pick") === "on"}, "Task saved. Model evaluation is starting.");
  });
  document.querySelector("#auto-analyze")?.addEventListener("click", () => workflowAction(workflow.id, {action:"auto_analyze"}, "Testing prompts and diagnosing the next step automatically."));
  const promptForm = document.querySelector("#workflow-prompt-form");
  if (promptForm) promptForm.onsubmit = event => {
    event.preventDefault();
    workflowAction(workflow.id, {action:"try_prompt", instruction:document.querySelector("#workflow-instruction").value, context:document.querySelector("#workflow-context").value}, "Running on the same dataset development examples.");
  };
  const decisionForm = document.querySelector("#workflow-prompt-decision");
  if (decisionForm) decisionForm.onsubmit = event => {
    event.preventDefault();
    workflowAction(workflow.id, {action:event.submitter.value, trial_index:Number(document.querySelector("#selected-trial").value)}, "Your prompt decision was submitted.");
  };
  const clarificationForm = document.querySelector("#dataset-clarification");
  if (clarificationForm) clarificationForm.onsubmit = event => {
    event.preventDefault();
    workflowAction(workflow.id, {action:"accept_classification", task:document.querySelector("#confirmed-task").value, notes:document.querySelector("#clarification-notes").value}, "Your clarification was submitted.");
  };
  document.querySelector("#approve-workflow")?.addEventListener("click", () => workflowAction(workflow.id, {action:"approve"}, "Training candidates have been queued."));
  document.querySelector("#approve-remediation")?.addEventListener("click", () => workflowAction(workflow.id, {action:"approve_remediation"}, "Approved cleanup is running; the dataset will be profiled again."));
  document.querySelector("#replace-blocked-dataset")?.addEventListener("click", () => updateDataset(workflow));
  document.querySelector("#retry-workflow")?.addEventListener("click", () => workflowAction(workflow.id, {action:"retry"}, "The candidate retry has been queued."));
  document.querySelector("#skip-workflow")?.addEventListener("click", () => workflowAction(workflow.id, {action:"skip_candidate"}, "Continuing with the remaining candidates."));
  document.querySelector("#abort-workflow")?.addEventListener("click", () => workflowAction(workflow.id, {action:"abort"}, "The workflow was stopped."));
}

async function workflowPage(id) {
  try {
    const response = await fetch("/api/workflow/runs/" + id);
    if (!response.ok) throw new Error();
    const workflow = await response.json();
    renderWorkflow(workflow);
    if (!["waiting", "complete", "blocked", "cancelled", "failed", "interrupted"].includes(workflow.status)) pollWorkflow(id);
  } catch {
    app.innerHTML = shell("LangGraph workflow", "Workflow unavailable.", "Start the WSL trainer and open this workflow again.") + "</div>";
  }
}

async function startRun() {
  if (!state.dataset) return location.hash = "#/workspace";
  const run = {id:crypto.randomUUID(), model:state.model, method:state.method, dataset:state.dataset.name, status:"running", progress:8, loss:2.14, elapsed:0, framework:"Transformers + PEFT", startedAt:new Date().toLocaleString()};
  try {
    let dataset = {source:state.dataset.source, name:state.dataset.name, extension:state.dataset.format};
    if (state.dataset.source === "upload") {
      if (!state.file) throw new Error("Sample data runs in preview mode.");
      const form = new FormData(); form.append("file", state.file);
      const upload = await fetch("/api/datasets/upload", {method:"POST", body:form});
      if (!upload.ok) throw new Error("Dataset upload could not be validated.");
      dataset = await upload.json();
    }
    const response = await fetch("/api/runs", {method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify({
      dataset, base_model:state.model, parameter_method:state.method,
      hyperparameters:{rank:(state.analysis?.facts.validRows || 999) < 500 ? 8 : 16,alpha:(state.analysis?.facts.validRows || 999) < 500 ? 16 : 32,learning_rate:0.0002,epochs:recommendation().epochs,batch_size:1,gradient_accumulation:8,max_sequence_length:Math.min(4096, Math.max(1024, (state.analysis?.facts.estimatedTokenLength?.p95 || 1024) * 2))},
      split_seed:42
    })});
    if (!response.ok) throw new Error("The local trainer rejected this run.");
    const remote = await response.json();
    run.id = remote.id; run.status = remote.status; run.remote = true; run.progress = 2;
    state.runs.unshift(run); save(); location.hash = "#/train/" + run.id; pollTrainerRun(run.id);
    toast("Local training queued", "The WSL trainer has accepted this run.");
  } catch {
    state.runs.unshift(run); save(); location.hash = "#/train/" + run.id; simulateRun(run.id);
    toast("Preview training started", "Start the WSL trainer to run this configuration on your GPU.");
  }
}

function pollTrainerRun(id) {
  const interval = setInterval(async () => {
    const run = state.runs.find(item => item.id === id);
    if (!run || run.status === "cancelled") return clearInterval(interval);
    try {
      const response = await fetch("/api/runs/" + id);
      if (!response.ok) return;
      const remote = await response.json();
      run.status = remote.status;
      run.progress = remote.status === "queued" ? 2 : remote.status === "running" ? Math.min(92, run.progress + 3) : 100;
      if (remote.status === "complete") { run.loss = remote.metrics?.eval_loss ?? run.loss; run.perplexity = remote.metrics?.perplexity ?? run.perplexity; run.evaluation = remote.metrics || {}; run.framework = remote.framework || "Transformers + TRL + PEFT"; }
      save(); if (route() === "train/" + id) train(id);
      if (["complete", "failed", "cancelled"].includes(remote.status)) clearInterval(interval);
    } catch { /* The trainer may be restarting; keep polling. */ }
  }, 3000);
}

function simulateRun(id) {
  const interval = setInterval(() => {
    const run = state.runs.find(item => item.id === id);
    if (!run || run.status !== "running") return clearInterval(interval);
    run.progress = Math.min(100, run.progress + Math.ceil(Math.random() * 9));
    run.loss = Math.max(.42, run.loss - Math.random() * .11); run.elapsed += 18;
    if (run.progress >= 100) { run.status = "complete"; run.loss = .48; run.perplexity = 1.62; run.framework = "Unsloth (auto-selected)"; toast("Training complete", "Your adapter and evaluation report are ready."); }
    save(); if (route() === "train/" + id) train(id);
  }, 1100);
}

function train(id) {
  const run = state.runs.find(item => item.id === id); if (!run) return location.hash = "#/runs";
  const complete = run.status === "complete";
  app.innerHTML = shell("Local training run", complete ? "Your fine-tune is ready." : "Training on your GPU.", complete ? "The held-out evaluation is complete. Compare answers and prepare exports." : "Gradient descent is updating the adapter while the base model stays unchanged.", "<button class='button " + (complete ? "primary" : "danger") + "' id='run-action'>" + (complete ? "Open evaluation →" : "Cancel run") + "</button>") +
    steps(complete ? 3 : 2) + "<section class='card body-pad'><div class='section-head' style='padding:0 0 15px'><div><h2>" + escapeHtml(runTitle(run)) + "</h2><p>Dataset: " + escapeHtml(runDatasetLabel(run)) + " · " + escapeHtml(run.framework) + "</p></div><span class='pill " + (complete ? "complete" : "") + "'>" + (complete ? "complete" : run.status) + "</span></div><div style='margin:20px 0 8px;display:flex;justify-content:space-between'><strong>" + run.progress + "% complete</strong><span style='color:var(--muted);font-size:10px'>" + Math.floor(run.elapsed / 60) + "m " + run.elapsed % 60 + "s elapsed · " + (complete ? "evaluation finished" : "about 6m remaining") + "</span></div><div class='progress'><i style='width:" + run.progress + "%'></i></div><div class='metric-grid grid'><div class='metric'><span>Current training loss</span><strong>" + run.loss.toFixed(2) + "</strong></div><div class='metric'><span>Method</span><strong style='font-size:14px'>" + run.method.toUpperCase() + "</strong></div><div class='metric'><span>GPU memory plan</span><strong style='font-size:14px'>" + model().vram + "</strong></div></div></section></div>";
  document.querySelector("#run-action").onclick = () => {
    if (complete) return location.hash = "#/evaluate/" + id;
    run.status = "cancelled"; save(); train(id); toast("Run cancelled", "No base-model weights were changed.");
  };
}

function classificationEvaluation(run) {
  const report = run.evaluation?.classification || run.classification;
  if (!report?.supported) {
    return "<section class='card body-pad'><h2>Classification error analysis is unavailable</h2><p style='color:var(--muted);font-size:10px;line-height:1.65'>This run does not have an explicit ground-truth label column. Add <code>label</code>, <code>class</code>, or <code>target</code> beside <code>text</code> to calculate TP, FP, FN, TN, precision, recall, and a held-out error queue. Generative quality metrics are shown below instead.</p></section>";
  }
  const confusion = report.confusion || {};
  const errors = (report.errors || []).map(item => "<article class='error-row " + escapeHtml(item.category) + "'><div><span class='error-kind'>" + escapeHtml(item.category.replace("_", " ")) + "</span><strong>Expected: " + escapeHtml(item.expected) + " · Predicted: " + escapeHtml(item.predicted) + "</strong><p>" + escapeHtml(item.input || "Input was not retained") + "</p></div><span>Row " + escapeHtml(item.rowId ?? "—") + "</span></article>").join("") || "<div class='empty'><strong>No held-out errors</strong>Every evaluated example matched its ground-truth label.</div>";
  return "<section class='card'><div class='section-head'><div><h2>Classification performance</h2><p>Positive class: " + escapeHtml(report.positiveLabel || "not applicable") + " · Label field: " + escapeHtml(report.labelField) + " · " + report.total + " held-out examples</p></div></div><div class='metric-grid grid'><div class='metric'><span>Precision</span><strong>" + percentage(report.precision) + "</strong></div><div class='metric'><span>Recall</span><strong>" + percentage(report.recall) + "</strong></div><div class='metric'><span>F1 score</span><strong>" + percentage(report.f1) + "</strong></div></div><div class='confusion-grid'><div><span>True positives</span><strong>" + (confusion.tp ?? "—") + "</strong></div><div><span>False positives</span><strong>" + (confusion.fp ?? "—") + "</strong></div><div><span>False negatives</span><strong>" + (confusion.fn ?? "—") + "</strong></div><div><span>True negatives</span><strong>" + (confusion.tn ?? "—") + "</strong></div></div></section><section class='card error-analysis'><div class='section-head'><div><h2>Incorrect or not caught</h2><p>False negatives are items the model failed to catch; false positives are items it flagged incorrectly. Review these before exporting.</p></div><span class='pill " + (report.errorCount ? "danger" : "complete") + "'>" + report.errorCount + " errors</span></div><div class='error-list'>" + errors + "</div></section>";
}

function evaluate(id) {
  const run = state.runs.find(item => item.id === id); if (!run || run.status !== "complete") return location.hash = "#/runs";
  app.innerHTML = shell("Evaluation & export", "See what changed, then take it with you.", "Results use the held-out 10% of your data. Review examples before deployment.", "<button class='button' id='back-run'>← Training run</button>") +
    steps(3) + classificationEvaluation(run) + "<section class='card' style='margin-top:16px'><div class='metric-grid grid'><div class='metric'><span>Evaluation loss</span><strong>" + (Number.isFinite(run.loss) ? run.loss.toFixed(2) : "—") + "</strong></div><div class='metric'><span>Perplexity</span><strong>" + (Number.isFinite(run.perplexity) ? run.perplexity.toFixed(2) : "—") + "</strong></div><div class='metric'><span>Run dataset</span><strong style='font-size:14px'>" + escapeHtml(runDatasetLabel(run)) + "</strong></div></div></section><div class='grid columns' style='margin-top:16px'><section class='card comparison'><h2>Response comparison</h2><p>Open the guided workflow comparison for recorded model outputs. No response samples were recorded for this legacy run.</p></section><aside class='card body-pad'><h2>Export for Ollama</h2><p style='color:var(--muted);font-size:10px;line-height:1.6'>Download the adapter for reuse, or convert the merged model to GGUF before importing it into Ollama.</p><div class='export-actions'><p>Browser downloads and GGUF conversion are not connected yet. Retrieve verified artifacts from the local training run directory.</p></div><pre class='code'>ollama create forge-support -f Modelfile\nollama run forge-support</pre></aside></div></div>";
  document.querySelector("#back-run").onclick = () => location.hash = "#/train/" + id;
  document.querySelectorAll("[data-export]").forEach(button => button.onclick = () => toast(button.dataset.export === "gguf" ? "GGUF conversion queued" : "Adapter package ready", button.dataset.export === "gguf" ? "The local trainer will merge and convert weights when it is online." : "Includes adapter_model.safetensors and adapter_config.json."));
}

// Correct/incorrect answers on the run's unseen test questions, recorded by its workflow; loss is not answer accuracy.
function correctnessTable(run) {
  const result = run.correctness;
  if (!result) return "<p>" + (run.status === "complete" ? "No answer correctness was recorded for this run. Runs started from a workflow are graded on unseen test questions." : "Correct and incorrect answers appear after training, once the fine-tuned model has answered the unseen test questions.") + "</p>";
  const row = (name, item) => "<tr><td>" + name + "</td><td>" + item.correct + "</td><td>" + item.incorrect + "</td><td>" + item.needGrade + "</td><td>" + item.loopsStopped + "</td><td>" + (Number.isFinite(item.answerSeconds) ? Math.round(item.answerSeconds).toLocaleString() + " s" : "—") + "</td></tr>";
  return "<div class='benchmark-table'><table><thead><tr><th>Model</th><th>Correct</th><th>Incorrect</th><th>Need your grade</th><th>Loops stopped</th><th>Answering time</th></tr></thead><tbody>" +
    row("Base " + escapeHtml(runModelLabel(run)), result.base) + row("<strong>Fine-tuned</strong>", result.tuned) + "</tbody></table></div><p>" + result.questions + " unseen test questions, never used for training; the same questions and grading for both models.</p>";
}

function correctnessLine(run) {
  const result = run.correctness;
  return result ? "fine-tuned " + result.tuned.correct + "/" + result.questions + " correct (base " + result.base.correct + "/" + result.questions + ")" : "correctness not recorded";
}

function runDetails(id) {
  const run = state.runs.find(item => item.id === id);
  if (!run) return syncServerRuns().catch(() => false).then(() => state.runs.some(item => item.id === id) ? runDetails(id) : location.hash = "#/runs");
  const isComplete = run.status === "complete";
  const artifacts = isComplete ? "Adapter, metrics, and evaluation report verified" : run.status === "failed" ? "No verified artifacts" : "Artifacts will appear when the run completes";
  app.innerHTML = shell("Training run details", runTitle(run), "A durable record of the dataset, configuration, progress, metrics, and local artifacts for this run.", "<button class='button' id='back-history'>← Training history</button>") +
    "<section class='card details-card'><div class='section-head'><div><h2>" + escapeHtml(runTitle(run)) + "</h2><p>Run ID: " + escapeHtml(run.id) + "</p></div><span class='pill " + runStatusClass(run) + "'>" + escapeHtml(run.status) + "</span></div><div class='details-grid'><div><span>Dataset used</span><strong>" + escapeHtml(runDatasetLabel(run)) + "</strong></div><div><span>Base model</span><strong>" + escapeHtml(runModelLabel(run)) + "</strong></div><div><span>Fine-tuning method</span><strong>" + escapeHtml((run.method || "unknown").toUpperCase()) + "</strong></div><div><span>Started</span><strong>" + escapeHtml(run.startedAt || "Not recorded") + "</strong></div><div><span>Training framework</span><strong>" + escapeHtml(run.framework || "Not recorded") + "</strong></div><div><span>Elapsed time</span><strong>" + Math.floor((run.elapsed || 0) / 60) + "m " + ((run.elapsed || 0) % 60) + "s</strong></div></div></section>" +
    "<section class='card body-pad' style='margin-top:16px'><h2>Answer correctness</h2>" + correctnessTable(run) + (isComplete ? "" : "<p>Progress: " + (run.progress || 0) + "%</p>") + "</section>" +
    "<section class='card history-artifacts' style='margin-top:16px'><h2>Local artifacts</h2><p>" + escapeHtml(artifacts) + "</p><div class='progress'><i style='width:" + (isComplete ? 100 : run.progress || 0) + "%'></i></div></section></div>";
  document.querySelector("#back-history").onclick = () => location.hash = "#/runs";
}

async function syncServerRuns() {
  // Runs started by workflows or other browsers exist only on the trainer; merge them by ID.
  const response = await fetch("/api/runs");
  if (!response.ok) return false;
  for (const remote of await response.json()) {
    const config = remote.config || {};
    const fields = {id:remote.id, remote:true, status:remote.status, model:config.base_model, method:config.parameter_method,
      dataset:config.dataset?.displayName || config.dataset?.name, startedAt:remote.startedAt || remote.createdAt, framework:remote.framework,
      loss:remote.metrics?.eval_loss, perplexity:remote.metrics?.perplexity, progress:remote.status === "complete" ? 100 : undefined, workflowId:remote.workflowId, correctness:remote.correctness,
      elapsed:remote.startedAt && remote.completedAt ? Math.round((Date.parse(remote.completedAt) - Date.parse(remote.startedAt)) / 1000) : undefined};
    Object.keys(fields).forEach(key => fields[key] === undefined && delete fields[key]);
    const local = state.runs.find(run => run.id === remote.id);
    if (local) Object.assign(local, fields); else state.runs.push(fields);
  }
  state.runs.sort((a, b) => String(b.startedAt || "").localeCompare(String(a.startedAt || "")));
  save();
  return true;
}

function runs() {
  if (!runs.synced) { runs.synced = true; syncServerRuns().then(ok => ok && route() === "runs" && runs()).catch(() => {}).finally(() => { runs.synced = false; }); }
  const body = state.runs.length ? state.runs.map(run => "<div class='run'><div><strong>" + escapeHtml(runTitle(run)) + "</strong><span><b>Dataset:</b> " + escapeHtml(runDatasetLabel(run)) + " · " + escapeHtml(run.startedAt || "date not recorded") + " · " + correctnessLine(run) + "</span></div><div class='run-actions'><span class='pill " + runStatusClass(run) + "'>" + escapeHtml(run.status) + "</span><button class='button' data-open='" + run.id + "'>View details</button></div></div>").join("") : "<div class='empty'><strong>No training runs yet</strong>Start with a dataset and safe QLoRA preset.</div>";
  app.innerHTML = shell("Training runs", "Your local training history.", "Each run keeps its dataset reference, method, metrics, and artifacts on this computer.", "<button class='button primary' id='new-run'>+ New fine-tune</button>") + "<section class='card run-list'>" + body + "</section></div>";
  document.querySelector("#new-run").onclick = () => location.hash = "#/workspace";
  document.querySelectorAll("[data-open]").forEach(button => button.onclick = () => {
    const item = state.runs.find(run => run.id === button.dataset.open);
    location.hash = "#/runs/" + item.id;
  });
}

function catalog() {
  app.innerHTML = shell("Model catalog", "Find the right balance of quality and speed.", "Memory estimates assume the recommended QLoRA preset on the detected 24 GB GPU.") + "<div class='catalog'>" + models.map(item =>
    "<article class='card model'><span class='size'>" + item.size + "</span><h2 style='margin-top:7px'>" + item.name + "</h2><p>" + item.note + "</p><dl><div><dt>Expected VRAM</dt><dd>" + item.vram + "</dd></div><div><dt>Training pace</dt><dd>" + item.speed + "</dd></div><div><dt>License</dt><dd>" + item.license + "</dd></div><div><dt>Method</dt><dd>" + (item.size === "14B" ? "QLoRA only" : "LoRA / QLoRA") + "</dd></div></dl><button class='button' data-pick='" + item.id + "'>Use this model</button></article>"
  ).join("") + "</div></div>";
  document.querySelectorAll("[data-pick]").forEach(button => button.onclick = () => { state.model = button.dataset.pick; if (model().size === "14B") state.method = "qlora"; location.hash = "#/configure"; });
}

function methodsPage() {
  const methods = [
    ["LoRA", "Implemented", "Trains a small adapter while the base model stays frozen. Use when the model fits without 4-bit loading.", "LoRAConfig + PEFT adapter"],
    ["QLoRA", "Implemented", "Trains a small adapter over a 4-bit quantized base model. This is the default for 14B models and limited VRAM.", "BitsAndBytesConfig + PEFT adapter"],
    ["Prompt + RAG", "Implemented", "Tests improved instructions and retrieved reference documents before training so the workflow can avoid unnecessary fine-tunes.", "Automatic prompt trials + local retrieval"],
    ["Full fine-tuning", "Not implemented", "The trainer does not update every base-model weight. Use LoRA or QLoRA for the supported local workflow.", "Unavailable"],
  ];
  app.innerHTML = shell("Fine-tuning methods", "What ForgeTune can run", "The final workflow uses measured prompt, retrieval, and adapter results before recommending deployment.", "<button class='button' id='methods-back'>← Workspace</button>") +
    "<section class='card body-pad'><div class='section-head'><div><h2>Implemented methods</h2><p>These labels reflect the actual local trainer and workflow code.</p></div><span class='pill complete'>Current</span></div><div class='catalog'>" +
    methods.map(([name,status,description,detail]) => "<article class='card model'><span class='size'>" + status + "</span><h2 style='margin-top:7px'>" + escapeHtml(name) + "</h2><p>" + escapeHtml(description) + "</p><dl><div><dt>Implementation</dt><dd>" + escapeHtml(detail) + "</dd></div></dl></article>").join("") +
    "</div></section></div>";
  document.querySelector("#methods-back").onclick = () => location.hash = "#/workspace";
}
function render() {
  setNav(); const [page, id] = route().split("/");
  if (page === "configure") configure();
  else if (page === "workflow") workflowPage(id);
  else if (page === "train") train(id);
  else if (page === "evaluate") evaluate(id);
  else if (page === "runs" && id) runDetails(id);
  else if (page === "runs") runs();
  else if (page === "models") catalog();
  else if (page === "methods") methodsPage();
  else datasetStudio();
  window.scrollTo({top:0, behavior:"smooth"});
}

async function loadHardware() {
  try {
    const response = await fetch("/api/hardware"), hardware = await response.json();
    document.querySelector("#gpu-name").textContent = hardware.name || "NVIDIA GPU";
    document.querySelector("#gpu-memory").textContent = hardware.memoryGb ? hardware.memoryGb + " GB VRAM detected" : "GPU memory unavailable";
    document.querySelector("#trainer-status").textContent = hardware.trainerOnline ? "Trainer connected" : "Trainer not connected";
    document.querySelector(".status-line i").style.background = hardware.trainerOnline ? "var(--cyan)" : "var(--amber)";
  } catch { document.querySelector("#gpu-name").textContent = "NVIDIA GPU"; }
}

async function loadAdvisorStatus() {
  try {
    const response = await fetch("/api/advisor/status");
    if (response.ok) state.cloudStatus = await response.json();
  } catch { state.cloudStatus = {configured:false}; }
}

window.addEventListener("hashchange", render);
loadHardware(); loadAdvisorStatus(); render();
