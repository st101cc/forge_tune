let datasetParent = null;
let datasetImport = null;
try { datasetImport = JSON.parse(localStorage.getItem("forgetune-import") || "null"); } catch {}
if (!/^[0-9a-f-]{36}$/i.test(datasetImport?.id || "")) datasetImport = null;
let datasetImportTimer;
let datasetImportPolling = false;
let datasetSource = datasetImport ? "huggingface" : "file";
let datasetUploadBusy = false;
let datasetUploadError = "";
let hfOverview = null;
const datasetTasks = {assistant:"Assistant / question answering", writing:"Writing / completion", code:"Code generation", structured:"Structured output / JSON", classification:"Classification / labels"};
const hfOverviewKey = () => (document.querySelector("#studio-hf")?.value.trim() || "") + "|" + (document.querySelector("#hf-split")?.value.trim() || "");
const hfOverviewReady = () => !!hfOverview && hfOverview.key === hfOverviewKey();
function formatBytes(bytes) {
  if (bytes == null) return "unknown";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let index = 0;
  while (bytes >= 1000 && index < units.length - 1) { bytes /= 1000; index++; }
  return (index && bytes < 10 ? bytes.toFixed(1) : Math.round(bytes)) + " " + units[index];
}
function formatRows(rows) {
  if (rows == null) return "unknown";
  return rows >= 1e6 ? (rows / 1e6).toFixed(rows >= 1e7 ? 0 : 2) + " M" : rows >= 1e4 ? Math.round(rows / 1e3) + " K" : rows.toLocaleString();
}
function hfRecommendationReason(data) {
  const rec = data.recommendation;
  if (rec.mode === "all") return "Small enough to import in full: every row is saved and each subset is audited before you choose training subsets.";
  const count = data.subsets.length;
  const size = data.sizesKnown ? "A full import would download " + formatBytes(data.totalBytes) + " (" + formatRows(data.totalRows) + " rows)" + (data.totalBytes > data.freeBytes ? ", more than the trainer's free disk space" : data.totalBytes > data.freeBytes / 2 ? ", over half of the trainer's free disk space" : "") + ". " : "Hugging Face does not report the full size of every subset. ";
  return size + "Audits read at most 600 rows per subset, so a sample gives the same analysis: about " + (count * rec.maxRows).toLocaleString() + " rows across " + count + " subset" + (count === 1 ? "" : "s") + ". Training uses the sampled rows; raise the row count (up to 10,000 per subset) for a larger training set.";
}
function renderHfOverview() {
  const box = document.querySelector("#hf-overview");
  if (!box) return;
  box.hidden = !hfOverviewReady();
  if (hfOverviewReady()) {
    const data = hfOverview.data, rec = data.recommendation;
    const license = [].concat(data.license || []).join(", ");
    box.innerHTML = '<h4>Dataset overview</h4><p>' + escapeHtml(data.datasetId) + ' · ' + escapeHtml(data.split) + ' split' + (license ? ' · license: ' + escapeHtml(license) : '') + '</p>' +
      '<div class="hf-overview-table"><table><thead><tr><th scope="col">Subset</th><th scope="col">Rows</th><th scope="col">Download</th></tr></thead><tbody>' +
      data.subsets.map(item => '<tr><th scope="row">' + escapeHtml(item.configuration || "default") + '</th><td>' + (item.rowsEstimated ? "≈ " : "") + formatRows(item.rows) + '</td><td>' + formatBytes(item.downloadBytes) + '</td></tr>').join("") +
      '</tbody><tfoot><tr><th scope="row">Total</th><td>' + (data.sizesKnown ? "≈ " + formatRows(data.totalRows) : "unknown") + '</td><td>' + (data.sizesKnown ? formatBytes(data.totalBytes) : "unknown") + '</td></tr></tfoot></table></div>' +
      '<p class="hf-overview-free">' + formatBytes(data.freeBytes) + ' free on the trainer · ≈ marks estimated row counts</p>' +
      '<div class="hf-recommendation"><strong>Recommended: ' + (rec.mode === "all" ? "Full dataset" : "Small sample, " + rec.maxRows.toLocaleString() + " rows per subset") + '</strong><p>' + escapeHtml(hfRecommendationReason(data)) + '</p></div>';
  }
  const submit = document.querySelector(".import-submit");
  if (submit && !datasetUploadBusy) submit.textContent = datasetSubmitLabel();
}
function datasetSubmitLabel() {
  if (datasetSource === "huggingface" && !hfOverviewReady()) return "Check dataset";
  if (datasetSource === "huggingface" && document.querySelector("#hf-mode")?.value === "all") return "Import Full Data";
  return datasetParent ? "Upload new version & analyze" : datasetSource === "huggingface" ? "Import Hugging Face sample & analyze" : datasetSource === "github" ? "Import from GitHub & analyze" : "Upload & auto-analyze";
}
function setImportScope() {
  const sample = document.querySelector("#hf-mode")?.value === "sample";
  const fields = document.querySelector("#hf-sample-fields");
  if (!fields) return;
  fields.hidden = !sample;
  document.querySelector("#hf-rows").disabled = !sample;
  document.querySelector("#hf-scope-note").textContent = sample ? "First rows of each subset · up to 50 MB each · choose training subsets after analysis" :"All subsets of the chosen split · audit up to 600 rows per subset · choose training subsets after analysis";
  document.querySelector(".import-submit").textContent = datasetSubmitLabel();
}
function saveDatasetImport() {
  if (datasetImport) {
    const {id, request, parent, cloudAssist, selected} = datasetImport;
    localStorage.setItem("forgetune-import", JSON.stringify({id, request, parent, cloudAssist, selected}));
  }
  else localStorage.removeItem("forgetune-import");
}
function clearDatasetImport() {
  datasetImport = null;
  clearTimeout(datasetImportTimer);
  saveDatasetImport();
  renderDatasetImport();
}
async function analyzeImportedDataset(dataset, parent, cloudAssist) {
  const response = await fetch("/api/workflow/runs", {method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify({dataset, analysis_only:true, previous_workflow_id:parent?.id || null, cloud_assist:cloudAssist, candidate_limit:1})});
  const workflow = await response.json();
  if (!response.ok) throw new Error(typeof workflow.detail === "string" ? workflow.detail : "Dataset analysis could not start.");
  if (datasetImport) clearDatasetImport();
  datasetParent = null;
  location.hash = "#/workflow/" + workflow.id;
}
function subsetReview(job) {
  const collection = job.result;
  if (!collection?.allSubsets) return "";
  return '<section><h3>Choose subsets for training</h3><p>Every subset has been checked for the chosen split. Audits sample up to 600 rows per subset; failed imports are shown below. Select one or more subsets to prepare separate workflows. Training still needs approval.</p>' +
    '<form id="subset-selection">' + collection.subsets.map((item, index) => '<div class="subset-review"><label><input type="checkbox" data-subset-index="' + index + '" aria-label="Use ' + escapeHtml(item.configuration || "default") + ' for training"' + (item.status !== "analyzed" ? ' disabled' : '') + ((datasetImport.selected || []).includes(item.configuration) ? ' checked' : '') + '> ' + escapeHtml(item.configuration || "default") + '</label><p>' + escapeHtml(item.status) + ' · ' + Number(item.dataset?.importedRows || 0).toLocaleString() + ' rows</p>' + (item.error ? '<p role="alert">' + escapeHtml(item.error) + '</p>' : '') + (item.profile ? '<details><summary>View analysis for ' + escapeHtml(item.configuration || "default") + '</summary>' + datasetProfile(item.profile) + '</details>' : '') + '</div>').join("") +
    '<button class="button primary" type="submit">Continue with selected subsets</button></form>' +
    Object.entries(job.selectedWorkflows || {}).map(([key,id]) => '<p><a class="button" href="#/workflow/' + encodeURIComponent(id) + '">Open ' + escapeHtml(JSON.parse(key) || "default") + ' workflow</a></p>').join("") + '</section>';
}
function renderDatasetImport() {
  const panel = document.querySelector("#dataset-import");
  if (!panel) return;
  document.querySelector("#dataset-upload-form fieldset").disabled = datasetUploadBusy || !!datasetImport;
  document.querySelectorAll("[data-source], #new-dataset").forEach(button => button.disabled = datasetUploadBusy || !!datasetImport);
  panel.hidden = !datasetImport;
  if (!datasetImport) return;
  const job = datasetImport.job || {status:"running", importedRows:0, importedBytes:0};
  if (job.status === "needs_configuration") {
    document.querySelector("#dataset-upload-error").textContent = "This dataset has multiple subsets. Import it again to analyze each one and choose afterwards.";
    clearDatasetImport();
    return;
  }
  const running = ["running", "cancelling"].includes(job.status);
  const count = Number(job.importedRows || 0).toLocaleString();
  const size = (Number(job.importedBytes || 0) / 1048576).toFixed(1);
  panel.innerHTML = '<p role="status" aria-live="polite"><strong>' + escapeHtml(job.status === "complete" ? job.result?.allSubsets ? "All subset audits finished" : "Full split imported" : job.status === "cancelling" ? "Cancelling import…" : running ? job.phase === "analyzing" ? "Analyzing subsets…" : "Importing dataset…" : "Import " + job.status) + '</strong><br>' + count + ' rows · ' + size + ' MiB saved' + (job.totalSubsets ? ' · ' + Number(job.completedSubsets || 0) + '/' + Number(job.totalSubsets) + ' subsets checked · ' + escapeHtml(job.currentSubset || "") : '') + (job.totalRows ? ' · ' + Number(job.totalRows).toLocaleString() + ' total rows' : '') + '</p>' +
    (running ? '<progress aria-label="Dataset import progress"' + (job.totalRows ? ' max="' + Number(job.totalRows) + '" value="' + Number(job.importedRows || 0) + '"' : '') + '></progress><p>The download continues if you leave this page. Cancellation waits for the current network read to finish.</p><button type="button" class="button" id="cancel-dataset-import"' + (job.status === "cancelling" ? ' disabled' : '') + '>Cancel import</button>' : '') +
    (job.status === "complete" ? job.result?.allSubsets ? subsetReview(job) : '<p>The complete selected split is saved. Analysis audits up to 600 rows; training can use the full saved split after approval.</p><button type="button" class="button primary" id="analyze-dataset-import">Analyze imported dataset</button> ' : '') +
    (!running ? '<button type="button" class="button" id="clear-dataset-import">Import another dataset</button>' : '') +
    (job.error ? '<p role="alert">' + escapeHtml(job.error) + '</p>' : '');
  document.querySelector("#clear-dataset-import")?.addEventListener("click", clearDatasetImport);
  const selectionForm = document.querySelector("#subset-selection");
  if (selectionForm) {
    const selected = () => [...selectionForm.querySelectorAll("[data-subset-index]:checked")].map(input => job.result.subsets[Number(input.dataset.subsetIndex)].configuration);
    const updateSelection = () => {
      datasetImport.selected = selected(); saveDatasetImport();
      selectionForm.querySelector("button").disabled = !datasetImport.selected.length;
    };
    selectionForm.addEventListener("change", updateSelection);
    updateSelection();
    selectionForm.onsubmit = async event => {
      event.preventDefault();
      selectionForm.querySelector("button").disabled = true;
      try {
        const response = await fetch("/api/datasets/imports/" + datasetImport.id + "/select", {method:"POST", headers:{"content-type":"application/json"},
          body:JSON.stringify({configurations:selected(), previous_workflow_id:datasetImport.parent?.id || null, cloud_assist:!!datasetImport.cloudAssist})});
        const result = await response.json();
        if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Could not prepare selected subsets.");
        job.selectedWorkflows = {...job.selectedWorkflows, ...Object.fromEntries(result.workflows.map(item => [JSON.stringify(item.configuration), item.id]))};
        renderDatasetImport();
      } catch (error) { document.querySelector("#dataset-upload-error").textContent = error.message; updateSelection(); }
    };
  }
  document.querySelector("#cancel-dataset-import")?.addEventListener("click", async event => {
    event.currentTarget.disabled = true;
    try {
      const response = await fetch("/api/datasets/imports/" + datasetImport.id + "/cancel", {method:"POST"});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Cancellation failed.");
      datasetImport.job = result; saveDatasetImport(); renderDatasetImport();
    } catch (error) { document.querySelector("#dataset-upload-error").textContent = error.message; renderDatasetImport(); }
  });
  document.querySelector("#analyze-dataset-import")?.addEventListener("click", async event => {
    event.currentTarget.disabled = true;
    try { await analyzeImportedDataset(job.result, datasetImport.parent, datasetImport.cloudAssist); }
    catch (error) { document.querySelector("#dataset-upload-error").textContent = error.message; renderDatasetImport(); }
  });
}
async function pollDatasetImport() {
  if (!datasetImport || datasetImportPolling) return;
  const id = datasetImport.id;
  datasetImportPolling = true;
  try {
    const response = await fetch("/api/datasets/imports/" + id);
    const job = await response.json();
    // 404 means the trainer will never know this import (data reset, other backend); retrying would lock the form forever.
    if (response.status === 404 && datasetImport?.id === id) {
      clearDatasetImport();
      const box = document.querySelector("#dataset-upload-error");
      if (box) box.textContent = datasetUploadError = "That import is no longer on the trainer. Start the import again.";
      return;
    }
    if (!response.ok) throw new Error(typeof job.detail === "string" ? job.detail : "Could not read import status.");
    if (datasetImport?.id === id) {
      datasetImport.job = job; saveDatasetImport(); renderDatasetImport();
    }
  } catch (error) {
    const box = document.querySelector("#dataset-upload-error");
    if (box) box.textContent = error.message + " Reconnecting to the import…";
  } finally {
    datasetImportPolling = false;
    clearTimeout(datasetImportTimer);
    if (datasetImport && (!datasetImport.job || ["running", "cancelling"].includes(datasetImport.job.status))) datasetImportTimer = setTimeout(pollDatasetImport, 1500);
  }
}
function updateDataset(workflow) {
  datasetParent = {id:workflow.id, name:workflow.dataset?.displayName || workflow.state?.dataset?.displayName || workflow.dataset?.name || workflow.state?.dataset?.name || "dataset"};
  location.hash = "#/workspace";
  datasetStudio();
}
async function datasetStudio() {
  const github = datasetSource === "github", huggingface = datasetSource === "huggingface";
  app.innerHTML = shell("Your data, ready for what’s next", "Better data.<br><span class='accent-title'>Better models.</span>", "Bring your dataset. We’ll check its quality, understand its purpose, and ask you about anything that needs a human decision.", "").replace("class='page'", "class='page dataset-page'") +
    `<div class="dataset-layout"><section class="card import-card"><div class="import-heading"><span class="section-number">01</span><div><h2>${datasetParent ? "Update dataset" : "Upload a dataset"}</h2><p>A new starting point for your next model.</p></div><span class="local-badge">LOCAL FIRST</span></div>${datasetParent ? `<div class="notice">New version of ${escapeHtml(datasetParent.name)}. The previous file, answers, and analysis will remain available.</div><button class="button" id="new-dataset">Start a separate dataset instead</button>` : ""}<div class="source-switch" role="group" aria-label="Dataset source"><button type="button" data-source="file" aria-pressed="${datasetSource === "file"}" ${datasetUploadBusy ? "disabled" : ""}>↑ Local file</button><button type="button" data-source="github" aria-pressed="${github}" ${datasetUploadBusy ? "disabled" : ""}>GitHub</button><button type="button" data-source="huggingface" aria-pressed="${huggingface}" ${datasetUploadBusy ? "disabled" : ""}>Hugging Face</button></div><form id="dataset-upload-form"><fieldset ${datasetUploadBusy ? "disabled" : ""}>${huggingface ? `<div class="github-entry"><span class="import-symbol">↗</span><h3>Import a Hugging Face dataset</h3><p>Import and analyze every subset first, then choose your training data. Each subset stays separate.</p><label class="field" for="studio-hf">Dataset URL or ID</label><input id="studio-hf" required maxlength="2048" placeholder="nvidia/Nemotron-Cascade-2-SFT-Data"><div class="form-row"><div><label class="field" for="hf-split">Split</label><input id="hf-split" required value="train" maxlength="100"></div></div><div id="hf-overview" class="hf-overview" aria-live="polite" hidden></div><label class="field" for="hf-mode">Import scope</label><select id="hf-mode"><option value="all">Full dataset (all subsets)</option><option value="sample">Small sample</option></select><div id="hf-sample-fields" hidden><label class="field" for="hf-rows">Maximum rows to import</label><input id="hf-rows" type="number" min="30" max="10000" value="600" required disabled></div><small id="hf-scope-note">All subsets of the chosen split · audit up to 600 rows per subset · choose training subsets after analysis</small></div>` : github ? `<div class="github-entry"><span class="import-symbol">↗</span><h3>Bring a dataset from GitHub</h3><p>Paste the link to a public data file. We’ll save a local copy and start the analysis.</p><label class="field" for="studio-github">GitHub file URL</label><input id="studio-github" type="url" required maxlength="2048" placeholder="https://github.com/owner/repo/blob/main/data.jsonl"><small>File links or raw.githubusercontent.com · up to 50 MB</small></div>` : `<div class="file-entry"><span class="import-symbol">↑</span><h3>Your dataset starts here</h3><p>Choose a file from your computer to get started.</p><label class="field" for="studio-file">Dataset file</label><input type="file" id="studio-file" accept=".csv,.jsonl,.parquet" required><div class="format-chips"><span>CSV</span><span>JSONL</span><span>PARQUET</span></div></div>`}<div class="cloud-choice"><label><input type="checkbox" id="dataset-cloud"><span><strong>Ask AI for a second opinion</strong><small>Optional cloud help for ambiguous data. Sends up to 20 locally redacted rows; you confirm the interpretation.</small></span></label></div><button class="button primary import-submit" type="submit">${datasetUploadBusy ? "Importing and starting analysis…" : datasetParent ? "Upload new version & analyze" : huggingface ? "Import Full Data" : github ? "Import from GitHub & analyze" : "Upload & auto-analyze"}<span aria-hidden="true">→</span></button><p class="import-footnote">Analysis first. Training only when you’re ready.</p></fieldset></form><div id="dataset-import" class="notice" hidden></div><div role="alert" id="dataset-upload-error">${escapeHtml(datasetUploadError)}</div></section><aside class="analysis-guide"><span class="eyebrow">A little guidance. A lot less guesswork.</span><h2>From raw data<br>to a clear next step.</h2><ol><li><span>01</span><div><h3>Inspect the essentials</h3><p>Schema, sampled row quality, duplicates, and language.</p></div></li><li><span>02</span><div><h3>Understand the task</h3><p>Suggest the likely purpose. Ask you when intent is unclear.</p></div></li><li><span>03</span><div><h3>Keep you in control</h3><p>Review the findings, update the data, or continue to model evaluation.</p></div></li></ol><div class="version-note"><span>↺</span><p><strong>Every version has a story.</strong>Updates preserve earlier files, findings, and your decisions.</p></div></aside></div><section class="card dataset-library"><div class="library-heading"><div><span class="eyebrow">Your workspace</span><h2>Dataset versions and analysis</h2></div><span class="library-count" id="dataset-count">— versions</span></div><div id="dataset-history" aria-live="polite"><p class="history-loading">Loading saved versions…</p></div></section></div>`;
  document.querySelectorAll("[data-source]").forEach(button => button.onclick = () => { datasetSource = button.dataset.source; datasetUploadError = ""; datasetStudio(); });
  document.querySelector("#new-dataset")?.addEventListener("click", () => {datasetParent = null; datasetStudio();});
  document.querySelector("#hf-mode")?.addEventListener("change", setImportScope);
  document.querySelectorAll("#studio-hf, #hf-split").forEach(input => input.addEventListener("input", renderHfOverview));
  if (huggingface && datasetImport?.request) {
    document.querySelector("#studio-hf").value = datasetImport.request.url;
    document.querySelector("#hf-split").value = datasetImport.request.split;
  }
  setImportScope();
  renderHfOverview();
  renderDatasetImport();
  if (datasetImport) pollDatasetImport();
  document.querySelector("#dataset-upload-form").onsubmit = async event => {
    event.preventDefault();
    if (datasetUploadBusy || datasetImport) return;
    const file = document.querySelector("#studio-file")?.files[0];
    const hfUrl = document.querySelector("#studio-hf")?.value.trim();
    const githubUrl = document.querySelector("#studio-github")?.value.trim();
    const cloudAssist = document.querySelector("#dataset-cloud").checked;
    if (!file && !githubUrl && !hfUrl) return;
    const parent = datasetParent;
    datasetUploadBusy = true; datasetUploadError = "";
    event.target.querySelector("fieldset").disabled = true;
    const checking = !!hfUrl && !hfOverviewReady();
    event.target.querySelector("button[type=submit]").textContent = checking ? "Checking dataset…" : "Uploading and starting analysis…";
    document.querySelector("#dataset-upload-error").textContent = "";
    try {
      let upload;
      if (checking) {
        // Overview first: sizes come from Hub metadata, so the user picks a scope before anything downloads.
        const key = hfOverviewKey();
        const response = await fetch("/api/datasets/huggingface/overview?url=" + encodeURIComponent(hfUrl) + "&split=" + encodeURIComponent(document.querySelector("#hf-split").value.trim()));
        const data = await response.json();
        if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Could not check this dataset.");
        hfOverview = {key, data};
        document.querySelector("#hf-mode").value = data.recommendation.mode;
        if (data.recommendation.maxRows) document.querySelector("#hf-rows").value = data.recommendation.maxRows;
        return;
      }
      if (hfUrl) {
        const mode = document.querySelector("#hf-mode").value;
        const request = {url:hfUrl, configuration:null, split:document.querySelector("#hf-split").value.trim(), mode};
        if (request.mode === "sample") request.max_rows = Number(document.querySelector("#hf-rows").value);
        upload = await fetch("/api/datasets/huggingface", {method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify(request)});
        if (upload.status === 202) {
          const job = await upload.json();
          datasetImport = {id:job.id, job, request, parent, cloudAssist};
          saveDatasetImport(); renderDatasetImport(); pollDatasetImport();
          return;
        }
        if (upload.ok && request.mode !== "sample") throw new Error("The trainer does not support full imports yet. Update and restart its backend, then try again.");
      } else if (githubUrl) {
        upload = await fetch("/api/datasets/github", {method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify({url:githubUrl})});
      } else {
        const form = new FormData(); form.append("file", file);
        upload = await fetch("/api/datasets/upload", {method:"POST", body:form});
      }
      const dataset = await upload.json();
      if (!upload.ok) throw new Error(typeof dataset.detail === "string" ? dataset.detail : "Dataset upload failed.");
      await analyzeImportedDataset(dataset, parent, cloudAssist);
    } catch (error) {
      datasetUploadError = error.message || "Start the trainer and try again.";
      const errorBox = document.querySelector("#dataset-upload-error");
      if (errorBox) errorBox.textContent = datasetUploadError;
    } finally {
      datasetUploadBusy = false;
      const fieldset = document.querySelector("#dataset-upload-form fieldset");
      if (fieldset) { fieldset.disabled = !!datasetImport; fieldset.querySelector("button[type=submit]").textContent = datasetSubmitLabel(); }
      setImportScope();
      renderHfOverview();
      renderDatasetImport();
    }
  };
  try {
    const response = await fetch("/api/workflow/runs");
    if (!response.ok) throw new Error();
    const workflows = await response.json();
    const history = document.querySelector("#dataset-history");
    if (!history) return;
    document.querySelector("#dataset-count").textContent = workflows.length + (workflows.length === 1 ? " version" : " versions");
    history.innerHTML = workflows.sort((a,b) => (b.createdAt || "").localeCompare(a.createdAt || "")).map(workflow => {
      const data = workflow.state?.dataset || {};
      const status = workflow.pendingAction?.type === "dataset_ready" ? "Ready for review" : workflow.pendingAction?.type === "data_review" ? "Your answer needed" : workflow.status;
      return `<div class="run dataset-version"><div><strong>${escapeHtml(data.displayName || data.name || "Dataset")}</strong><span>${escapeHtml(workflow.createdAt ? new Date(workflow.createdAt).toLocaleString() : "")} · ${data.hubRevision ? (data.sampleOnly ? "Hugging Face sample · " : "Hugging Face full split · ") : data.sourceUrl ? "GitHub · " : ""}${escapeHtml(status)}${workflow.state?.previous_workflow_id ? " · updated version" : ""}</span></div><a class="button" href="#/workflow/${encodeURIComponent(workflow.id)}">Open version</a></div>`;
    }).join("") || "<div class='dataset-empty'><span>◇</span><div><h3>A fresh start.</h3><p>Your datasets and their analysis will appear here after your first import.</p></div></div>";
  } catch {
    const history = document.querySelector("#dataset-history");
    if (history) history.textContent = "Trainer offline. Start the local trainer to upload data and resume saved analysis.";
  }
}
function datasetProfile(profile) {
  const facts = profile?.facts;
  if (!facts) return "";
  return `<section class="card body-pad experiment-form"><h2>Dataset audit</h2><div class="metric-grid grid"><div class="metric"><span>Total rows</span><strong>${escapeHtml(facts.rows)}</strong></div><div class="metric"><span>Rows sampled</span><strong>${escapeHtml(facts.sampledRows)}</strong></div><div class="metric"><span>Valid in sample</span><strong>${escapeHtml(facts.validRows)}</strong></div><div class="metric"><span>Duplicate rate in sample</span><strong>${Math.round((facts.duplicateRate || 0)*100)}%</strong></div></div><p>Schema: ${escapeHtml(facts.schema)} · Suggested task: ${escapeHtml(datasetTasks[profile.classification?.task] || profile.classification?.task || "Unknown")} · Source: ${escapeHtml(profile.classification?.source || "local")}</p>${profile.classification?.explanation ? `<p>${escapeHtml(profile.classification.explanation)}</p>` : ""}<ul>${(profile.findings || []).map(item => `<li><strong>${escapeHtml(item.severity)}</strong>: ${escapeHtml(item.message)}</li>`).join("")}</ul><details><summary>Inspect redacted sample rows</summary>${(profile._samples || []).slice(0,5).map(row => `<pre class="code">${escapeHtml(row.text)}</pre>`).join("")}</details></section>`;
}
function datasetClarification(pending) {
  return `<section class="card body-pad experiment-form"><h2>Clarify the intended task</h2><p>${escapeHtml(pending.message)}</p><form id="dataset-clarification"><label class="field" for="confirmed-task">What should the model do?</label><select id="confirmed-task" required><option value="">Choose the intended task</option>${Object.entries(datasetTasks).map(([value,label]) => `<option value="${value}">${label}</option>`).join("")}</select><label class="field">Describe the expected result<textarea id="clarification-notes" required maxlength="2000" placeholder="For example: answer customer questions using the support articles."></textarea></label><button class="button primary" type="submit">Confirm task & continue analysis</button></form></section>`;
}

function workflowPromptReview(pending) {
  const trials = pending.trials || [];
  const recommendation = pending.recommendation;
  return `<section class="card body-pad experiment-form"><h2>Automatic model analysis</h2>${recommendation ? `<div class="notice"><h3>${escapeHtml(recommendation.title)}</h3><p>${escapeHtml(recommendation.reason)}</p><p>${escapeHtml(recommendation.rag)}</p></div>` : "<p>Let the workflow test prompts and assess the next step. You do not need to write instructions.</p>"}<button class="button primary" id="auto-analyze">${recommendation ? "Retry automatic analysis" : "Run automatic analysis"}</button><p>${escapeHtml(pending.message)}</p><p>These development examples were selected automatically. Final evaluation examples are kept separate. Token overlap is a rough diagnostic; review the actual responses. This local pilot uses up to 4096 input tokens and generates up to 512 tokens per answer.</p>${trials.map((trial, index) => `<details ${index === trials.length - 1 ? "open" : ""}><summary>Test ${index + 1} · ${trial.usesRag ? "Prompt + retrieved references" : trial.instruction ? "Improved instructions" : "Baseline"}</summary><pre class="code">${escapeHtml(trial.instruction || "No extra instructions")}</pre>${trial.results.map(result => `<h3>${escapeHtml(result.modelId || "Local inference unavailable")}</h3><p>${escapeHtml(result.status)} · task correctness ${result.qualityEvidence?.score == null ? "unmeasured" : percentage(result.qualityEvidence.score)} · word overlap ${escapeHtml(result.meanTokenOverlap ?? "—")}</p>${result.message ? `<div class="notice">${escapeHtml(result.message)}</div>` : ""}${recordedSamples(result)}`).join("")}</details>`).join("")}<details><summary>Optional: edit a prompt manually</summary><form id="workflow-prompt-form"><label class="field">Improved instructions<textarea id="workflow-instruction" maxlength="4000" placeholder="Describe the output format and rules."></textarea></label><label class="field">Reference context (optional)<textarea id="workflow-context" maxlength="8000" placeholder="Supply missing facts to test whether external knowledge helps."></textarea></label><button class="button primary" type="submit">Run prompt trial</button><p>Leave both fields blank to retry the base prompt after fixing the local runtime.</p></form></details><form id="workflow-prompt-decision"><label class="field" for="selected-trial">Selected test</label><select id="selected-trial" required><option value="">No successful test selected</option>${trials.map((trial, index) => `<option value="${index}" ${recommendation?.selectedTrial === index ? "selected" : ""} ${trial.results.some(result => result.status === "complete") ? "" : "disabled"}>Test ${index + 1}</option>`).join("")}</select><p>The workflow selects the strongest measured test automatically. Training still requires approval.</p><button class="button" type="submit" value="keep_baseline">Keep base model</button> <button class="button primary" type="submit" value="fine_tune">Review a small fine-tune</button></form></section>`;
}
