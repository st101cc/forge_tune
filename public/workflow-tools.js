const isGraded=sample=>["pass","fail"].includes(sample.grade?.verdict);

// Graded answers keep their select in the form so their pre-filled verdict is still submitted.
function gradeItem(r,sample) {
  return `<div class="experiment-example"><strong>Question ${escapeHtml(sample.rowId)}</strong><pre class="code">${escapeHtml(sample.input || "")}</pre><strong>Reference — check it against your rubric</strong><pre class="code">${escapeHtml(sample.reference || "")}</pre><strong>Model answer</strong><pre class="code">${escapeHtml(sample.baseOutput || "")}</pre><label class="field">Correctness<select data-grade-model="${escapeHtml(r.modelId)}" data-grade-row="${sample.rowId}"><option value="">Not graded</option><option value="1"${sample.grade?.verdict==="pass" ? " selected" : ""}>Correct</option><option value="0"${sample.grade?.verdict==="fail" ? " selected" : ""}>Incorrect / incomplete</option></select></label></div>`;
}

function modelBenchmark(workflow) {
  const benchmark=(workflow.state || workflow).model_benchmark;
  if(!benchmark) return workflow.status==="complete" ? `<section class="card body-pad workflow-tools"><h2>Benchmark model choices</h2><p>This saved run used the previous selection rules. Compare eligible models in a new workflow using this dataset and its existing split. Existing final results remain historical evidence.</p><button class="button" id="start-model-benchmark">Benchmark models on this dataset</button></section>` : "";
  const reviewing=workflow.pendingAction?.type==="model_selection";
  const results=benchmark.results || [];
  const needsGrades=results.filter(r=>r.status==="complete").some(r=>r.qualityEvidence?.coverage!==1 || !Number.isFinite(r.qualityEvidence?.score));
  if (benchmark.sampleWarning && !String(benchmark.reason).includes(benchmark.sampleWarning)) benchmark.reason = (benchmark.reason || "") + " " + benchmark.sampleWarning;
  return `<section class="card body-pad workflow-tools" id="model-benchmark"><h2>Model selection benchmark</h2><div class="notice"><strong>${benchmark.winner ? escapeHtml(benchmark.winner) : "More evidence needed"}</strong><p>${escapeHtml(benchmark.error || benchmark.reason || "")}</p></div><p>${benchmark.developmentIds?.length || 0} development questions · ${benchmark.maxOutputTokens || 256} output tokens per answer · Final test questions excluded${benchmark.tooLongQuestions ? ` · ${benchmark.tooLongQuestions} question(s) too long for the local models left out` : ""}</p><div class="benchmark-table"><table><thead><tr><th>Model</th><th>Correctness</th><th>USD / test set</th><th>Seconds / answer</th></tr></thead><tbody>${results.map(r=>`<tr><td>${escapeHtml(r.modelId)}<br><small>${escapeHtml(r.provider)} · ${escapeHtml(r.status)}</small></td><td>${r.qualityEvidence?.coverage===1 && Number.isFinite(r.qualityEvidence?.score) ? percentage(r.qualityEvidence.score) + (r.qualityEvidence.estimated ? " (estimated)" : "") : "Needs grading"}</td><td>${r.provider==="local" && !(r.costUsd>0) ? "Free (your VM)" : Number.isFinite(r.costUsd) ? "$"+r.costUsd.toFixed(6) : "Not reported"}</td><td>${Number.isFinite(r.meanLatencySeconds) ? r.meanLatencySeconds.toFixed(2) : "Unavailable"}</td></tr>`).join("")}</tbody></table></div>${reviewing ? `<form id="benchmark-review">${needsGrades ? `<p><button class="button" type="button" id="benchmark-regrade">Regrade automatically</button> Grades the saved answers again with the current judge; no model answers again.</p><p>Automatic grading could not resolve every answer. The local judge's verdicts are pre-filled; check them and grade the answers it skipped, using a shared correctness rubric. A correct answer must satisfy the rubric fully; mark partial, invented, or incomplete answers incorrect.</p><label class="field">Correctness rubric<textarea id="benchmark-rubric" maxlength="2000" required>Answers the question correctly, follows its requirements, and contains no unsupported claims.</textarea></label>${results.filter(r=>r.status==="complete").map(r=>`<details class="experiment-example"><summary>Grade ${escapeHtml(r.modelId)} (${r.samples?.length || 0} answers, ${(r.samples || []).filter(s=>!["pass","fail"].includes(s.grade?.verdict)).length || "none"} ${(r.samples || []).filter(s=>!["pass","fail"].includes(s.grade?.verdict)).length===1 ? "needs" : "need"} your grade)</summary>${(r.samples || []).filter(s=>!isGraded(s)).map(s=>gradeItem(r,s)).join("")}${(r.samples || []).some(isGraded) ? `<details class="experiment-example"><summary>Show ${(r.samples || []).filter(isGraded).length} already graded answers</summary>${(r.samples || []).filter(isGraded).map(s=>gradeItem(r,s)).join("")}</details>` : ""}</details>`).join("")}` : ""}<p>Picks the model on your VM with the highest correctness score to fine-tune; within 2 percentage points, the faster one. Hosted models are for comparison only. This small benchmark gives a provisional choice.</p><button class="button primary" type="submit">Choose best from results</button></form>` : ""}</section>`;
}

function workflowTools(workflow) {
  const current = workflow.state || workflow;
  const runs = workflow.trainingRecords || [];
  const training = runs.length ? `<section class="card body-pad workflow-tools"><h2>Training progress</h2>${runs.map(run=>{
    const metrics=run.metrics || {}, context=run.trainingContext || metrics.trainingContext;
    const hp=run.config?.hyperparameters || {}, steps=context ? Math.ceil(context.rows/((hp.batch_size || 1)*(hp.gradient_accumulation || 1)))*(hp.epochs || 1) : 0;
    return `<div class="training-record"><h3>${escapeHtml(run.config?.base_model || "Training run")}</h3><p><span class="pill">${escapeHtml(run.status)}</span> ${escapeHtml(run.config?.hyperparameters?.epochs ?? "—")} epochs · ${escapeHtml(run.config?.hyperparameters?.max_sequence_length ?? "—")} token window</p>${context ? `<p>${Math.round(context.retainedTokenFraction*100)}% of training tokens retained · ${escapeHtml(context.truncatedRows)} conversations truncated</p>` : "<p>Preparing training data…</p>"}${metrics.optimizerSteps ? `<p>${escapeHtml(metrics.optimizerSteps)} optimizer steps · ${escapeHtml(metrics.trainRows)} training rows · ${escapeHtml(metrics.evalRows)} development rows</p><p>Development loss: ${escapeHtml(metrics.eval_loss)}. Loss measures prediction fit; it is not answer accuracy.</p>` : ""}${run.status==="running" ? `<p data-training-progress="${escapeHtml(run.id)}">Waiting for the latest training step…</p><details><summary>Live training log</summary><pre class="code" data-training-steps="${steps}" data-training-log="${escapeHtml(run.id)}">Loading progress…</pre></details>` : ""}</div>`;
  }).join("")}</section>` : "";
  const trials=current.prompt_trials || [];
  const ragSamples=trials.filter(trial=>trial.usesRag).flatMap(trial=>trial.results || []).flatMap(result=>result.samples || []);
  const coverage=ragSamples.length ? `<p><strong>Relevant sources found for ${ragSamples.filter(sample=>sample.retrieved?.length).length} of ${ragSamples.length} RAG cases.</strong></p>` : "";
  const history=workflow.pendingAction?.type!=="prompt_review" && trials.length ? `<section class="card body-pad workflow-tools"><h2>Prompt and RAG results</h2><p>Recorded development checks from before training. Final test examples are evaluated separately.</p>${trials.map((trial,index)=>`<details class="experiment-example"><summary>Test ${index+1} · ${trial.usesRag ? "Prompt + retrieved references" : trial.instruction ? "Improved instructions" : "Baseline"}</summary><pre class="code">${escapeHtml(trial.instruction || "No extra instructions")}</pre>${(trial.results || []).map(result=>`<h3>${escapeHtml(result.modelId || "Local model")}</h3><p>${escapeHtml(result.status)} · ${result.qualityEvidence?.estimated ? "estimated correctness" : "task correctness"} ${result.qualityEvidence?.score==null ? "needs review" : percentage(result.qualityEvidence.score)}</p>${gradingSummary(result)}${recordedSamples({...result, samples:(result.samples || []).slice(0, 5)})}`).join("")}</details>`).join("")}</section>` : "";
  const references = `<section class="card body-pad workflow-tools"><h2>Check external knowledge</h2><p>${current.knowledge_documents?.length ? `${current.knowledge_documents.length} reference documents connected. Each RAG trial retrieves relevant documents separately for each question.` : "Connect independent reference documents to test whether retrieval helps before fine-tuning."}</p>${coverage}${workflow.pendingAction?.type === "prompt_review" ? `<form id="workflow-rag-form"><label class="field" for="workflow-reference-files">Independent reference files</label><input id="workflow-reference-files" type="file" accept=".txt,.md" multiple required><p>Use source documents, not the evaluation answers. Up to 50 text or Markdown files; 20,000 characters per file.</p><button class="button" type="submit">Test with these references</button></form>` : ""}<p>Missing or irrelevant sources do not prove that RAG is unnecessary.</p></section>`;
  const brief = current.task_description ? `<section class="card body-pad"><h2>Task and success criteria</h2><p>${escapeHtml(current.task_description)}</p><p>${escapeHtml(current.success_metric)}</p><p>Up to ${current.analysis_limit || 50} examples per phase, limited by available rows. Fewer than 30 examples is exploratory; larger counts alone do not establish representativeness.</p></section>` : "";
  return brief + modelBenchmark(workflow) + training + references + history;
}

function bindWorkflowTools(workflow) {
  document.querySelector("#start-model-benchmark")?.addEventListener("click",async event=>{
    event.target.disabled=true;
    try {
      const state=workflow.state || workflow;
      const response=await fetch("/api/workflow/runs",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({dataset:state.dataset,previous_workflow_id:workflow.id,split_seed:state.split_seed ?? 42,test_dataset:state.test_dataset || null,auto_analysis:true,goal:state.goal || "balanced"})});
      const created=await response.json();
      if(!response.ok) throw new Error(typeof created.detail==="string" ? created.detail : "Could not start the model benchmark.");
      location.hash="#/workflow/"+created.id;
    }catch(error){toast("Benchmark unavailable",error.message);event.target.disabled=false;}
  });
  document.querySelector("#benchmark-regrade")?.addEventListener("click",()=>{
    workflowAction(workflow.id,{action:"regrade"},"Regrading the saved answers.");
  });
  document.querySelector("#benchmark-review")?.addEventListener("submit",event=>{
    event.preventDefault();
    const grades={};
    for(const input of document.querySelectorAll("[data-grade-model]")) {
      if(!input.value) {toast("Grade every answer","Apply the same rubric to every completed model before choosing.");return;}
      (grades[input.dataset.gradeModel] ||= {})[input.dataset.gradeRow]=Number(input.value);
    }
    workflowAction(workflow.id,{action:"select_best",grades,rubric:document.querySelector("#benchmark-rubric")?.value || ""},"Selecting from measured development results.");
  });
  document.querySelector("#workflow-rag-form")?.addEventListener("submit",async event=>{
    event.preventDefault();const button=event.submitter;button.disabled=true;
    try {
      const files=[...document.querySelector("#workflow-reference-files").files];
      if(!files.length || files.length>50 || files.some(file=>file.size>80000)) throw new Error("Choose 1–50 small text or Markdown files.");
      const documents=await Promise.all(files.map(async(file,index)=>({id:String(index+1),title:file.name,text:await file.text(),source:file.name})));
      if(documents.some(doc=>!doc.text.trim() || doc.text.length>20000) || documents.reduce((sum,doc)=>sum+doc.text.length,0)>200000) throw new Error("Keep each reference within 20,000 characters and the total within 200,000.");
      workflowAction(workflow.id,{action:"test_rag",documents},"Testing prompts with retrieved source passages.");
    } catch(error) {toast("Reference files need attention",error.message);button.disabled=false;}
  });
  for(const element of document.querySelectorAll("[data-training-log]")) fetch("/api/runs/"+encodeURIComponent(element.dataset.trainingLog)+"/logs").then(r=>r.json()).then(data=>{
    if(!element.isConnected) return;
    const lines=data.lines || [];element.textContent=lines.slice(-15).join("\n");
    const total=Number(element.dataset.trainingSteps);
    const line=[...lines].reverse().find(text=>new RegExp("\\b[0-9]+/"+total+"\\s*\\[").test(text));
    const step=line?.match(new RegExp("\\b([0-9]+)/"+total+"\\s*\\["))?.[1];
    const progress=document.querySelector('[data-training-progress="'+element.dataset.trainingLog+'"]');
    if(progress)progress.textContent=step ? `${step} / ${total} optimizer steps (${Math.round(Number(step)/total*100)}%)` : "Training is running; waiting for the first logged step.";
  }).catch(()=>{if(element.isConnected)element.textContent="Progress log temporarily unavailable.";});
}
