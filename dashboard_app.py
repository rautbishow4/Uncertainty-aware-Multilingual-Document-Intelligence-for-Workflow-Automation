"""
dashboard_app.py
----------------
FastAPI dashboard for uploading documents and running the trained document
intelligence pipeline.

Run:
    python -m uvicorn dashboard_app:app --host 127.0.0.1 --port 8000
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse


SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_CHECKPOINT = Path("checkpoints_layoutlmv3") / "best_model.pt"

app = FastAPI(title="Document Intelligence Dashboard")
_PIPELINE_CACHE: Dict[str, Any] = {}


def _cache_key(
    checkpoint_path: str,
    threshold: float,
    mc_passes: int,
    no_gpu: bool,
) -> str:
    return json.dumps(
        {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "ocr_engine": "easyocr",
            "threshold": threshold,
            "mc_passes": mc_passes,
            "device": "cpu" if no_gpu else "auto",
        },
        sort_keys=True,
    )


def get_pipeline(
    checkpoint_path: str,
    threshold: float,
    mc_passes: int,
    no_gpu: bool,
) -> Any:
    path = Path(checkpoint_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Checkpoint not found: {checkpoint_path}")

    key = _cache_key(checkpoint_path, threshold, mc_passes, no_gpu)
    if key not in _PIPELINE_CACHE:
        from inference.pipeline import DocumentIntelligencePipeline

        device = torch.device("cpu" if no_gpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        _PIPELINE_CACHE[key] = DocumentIntelligencePipeline.from_checkpoint(
            checkpoint_path=str(path),
            confidence_threshold=threshold,
            mc_passes=mc_passes,
            device=device,
        )
    return _PIPELINE_CACHE[key]


def result_to_payload(result) -> Dict[str, Any]:
    fields = [
        {
            "field_name": field.field_name,
            "label": field.label,
            "value": field.value,
            "confidence": round(float(field.confidence), 4),
            "uncertainty": round(float(field.uncertainty), 4) if field.uncertainty is not None else None,
            "bbox": field.bbox,
        }
        for field in result.fields
    ]
    relations = [
        {
            "head": relation.head_text,
            "tail": relation.tail_text,
            "confidence": round(float(relation.confidence), 4),
        }
        for relation in result.relations
    ]
    return {
        "doc_id": result.doc_id,
        "language": result.language,
        "overall_confidence": round(float(result.overall_confidence), 4),
        "overall_confidence_percent": round(float(result.overall_confidence) * 100, 2),
        "needs_human_review": result.needs_human_review,
        "review_reason": result.review_reason,
        "processing_time_ms": round(float(result.processing_time_ms), 1),
        "mc_passes_used": result.mc_passes_used,
        "metadata": result.metadata,
        "raw_ocr_text": result.raw_ocr_text,
        "fields": fields,
        "relations": relations,
        "json_result": json.loads(result.to_json(indent=2)),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return DASHBOARD_HTML


@app.get("/api/status")
def status() -> Dict[str, Any]:
    return {
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "default_checkpoint": str(DEFAULT_CHECKPOINT),
        "default_checkpoint_exists": DEFAULT_CHECKPOINT.exists(),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }


@app.post("/api/extract")
async def extract_document(
    file: UploadFile = File(...),
    checkpoint_path: str = Form(str(DEFAULT_CHECKPOINT)),
    threshold: float = Form(0.70),
    mc_passes: int = Form(20),
    language: str = Form("auto"),
    no_gpu: bool = Form(False),
) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use PDF, PNG, JPG, TIFF, or BMP.",
        )
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")
    if mc_passes < 1 or mc_passes > 100:
        raise HTTPException(status_code=400, detail="MC passes must be between 1 and 100.")
    pipeline = get_pipeline(checkpoint_path, threshold, mc_passes, no_gpu)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        result = pipeline.process(
            input_path=str(tmp_path),
            doc_id=Path(file.filename or tmp_path.name).stem,
            language=language,
            uncertainty_threshold=threshold,
        )
        return JSONResponse(result_to_payload(result))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Document Intelligence Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #eef2f5;
      --text: #17202a;
      --muted: #5e6b78;
      --line: #d7dde4;
      --accent: #176f5b;
      --accent-2: #305f91;
      --warn: #a45b12;
      --bad: #a43838;
      --good: #1d785f;
      --shadow: 0 8px 24px rgba(31, 42, 55, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    header {
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }

    .topbar {
      max-width: 1440px;
      margin: 0 auto;
      min-height: 64px;
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 700;
    }

    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--line);
    }

    .dot.ready { background: var(--good); }

    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 24px;
    }

    section {
      min-width: 0;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .panel-header {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .panel-title {
      font-size: 14px;
      font-weight: 700;
      color: var(--text);
    }

    .panel-body {
      padding: 18px;
    }

    form {
      display: grid;
      gap: 16px;
    }

    label {
      display: grid;
      gap: 7px;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
    }

    input, select, button {
      font: inherit;
    }

    input[type="text"],
    input[type="number"],
    select {
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 10px 11px;
      color: var(--text);
    }

    .dropzone {
      border: 1.5px dashed #9aa8b6;
      background: var(--surface-2);
      border-radius: 8px;
      min-height: 142px;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 18px;
      cursor: pointer;
      transition: border-color 0.15s ease, background 0.15s ease;
    }

    .dropzone.dragover {
      border-color: var(--accent);
      background: #e8f4ef;
    }

    .dropzone strong {
      display: block;
      margin-bottom: 6px;
      font-size: 15px;
    }

    .dropzone span {
      color: var(--muted);
      font-size: 13px;
    }

    #fileInput {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      opacity: 0;
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .toggle-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      color: var(--muted);
      font-size: 13px;
    }

    .toggle-row input {
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }

    button {
      min-height: 42px;
      border: 1px solid transparent;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
      padding: 10px 14px;
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.65;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }

    .metric {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 90px;
    }

    .metric .label {
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
    }

    .metric .value {
      margin-top: 8px;
      font-size: 24px;
      font-weight: 800;
      line-height: 1.1;
    }

    .metric.good .value { color: var(--good); }
    .metric.warn .value { color: var(--warn); }
    .metric.bad .value { color: var(--bad); }

    .tabs {
      display: flex;
      gap: 6px;
      padding: 6px;
      background: var(--surface-2);
      border-radius: 8px;
      margin-bottom: 14px;
      width: fit-content;
    }

    .tab {
      background: transparent;
      color: var(--muted);
      border: 1px solid transparent;
      min-height: 34px;
      padding: 7px 12px;
    }

    .tab.active {
      color: var(--text);
      background: var(--surface);
      border-color: var(--line);
    }

    .result-view {
      display: none;
    }

    .result-view.active {
      display: block;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }

    th, td {
      padding: 10px 11px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      font-size: 13px;
    }

    th {
      background: var(--surface-2);
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }

    td.value-cell {
      max-width: 420px;
      overflow-wrap: anywhere;
    }

    pre {
      margin: 0;
      background: #101820;
      color: #e8edf2;
      border-radius: 8px;
      padding: 16px;
      overflow: auto;
      max-height: 620px;
      font-size: 13px;
      line-height: 1.5;
    }

    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      min-height: 360px;
      display: grid;
      place-items: center;
      color: var(--muted);
      text-align: center;
      padding: 24px;
      background: var(--surface);
    }

    .message {
      margin-top: 14px;
      font-size: 13px;
      color: var(--muted);
      min-height: 20px;
      overflow-wrap: anywhere;
    }

    .message.error { color: var(--bad); }

    .preview {
      display: none;
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: var(--surface);
    }

    .preview img {
      width: 100%;
      max-height: 360px;
      object-fit: contain;
      display: block;
      background: #f0f2f4;
    }

    .ocr-text {
      white-space: pre-wrap;
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--line);
    }

    @media (max-width: 980px) {
      main {
        grid-template-columns: 1fr;
      }
      .metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 560px) {
      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }
      main {
        padding: 14px;
      }
      .row, .metrics {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <h1>Document Intelligence Dashboard</h1>
      <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">Checking runtime</span></div>
    </div>
  </header>

  <main>
    <section class="panel">
      <div class="panel-header">
        <div class="panel-title">Document Upload</div>
      </div>
      <div class="panel-body">
        <form id="uploadForm">
          <label>
            File
            <div id="dropzone" class="dropzone">
              <div>
                <strong id="fileName">Choose a PDF or image</strong>
                <span>PNG, JPG, JPEG, TIFF, BMP, or PDF</span>
              </div>
            </div>
            <input id="fileInput" type="file" accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp,application/pdf" />
          </label>

          <label>
            Checkpoint
            <input id="checkpoint" type="text" value="checkpoints_layoutlmv3\best_model.pt" />
          </label>

          <label>
            Language
            <input id="language" type="text" value="auto" />
          </label>

          <div class="row">
            <label>
              Confidence Threshold
              <input id="threshold" type="number" min="0" max="1" step="0.05" value="0.70" />
            </label>
            <label>
              MC Passes
              <input id="mcPasses" type="number" min="1" max="100" step="1" value="20" />
            </label>
          </div>

          <div class="toggle-row">
            <span>Force CPU</span>
            <input id="noGpu" type="checkbox" />
          </div>

          <button id="submitButton" type="submit">Extract Data</button>
        </form>
        <div id="message" class="message"></div>
        <div id="preview" class="preview"><img id="previewImage" alt="Document preview" /></div>
      </div>
    </section>

    <section>
      <div id="emptyState" class="empty">
        <div>Upload a document to view confidence, extracted fields, relations, OCR text, and JSON output.</div>
      </div>

      <div id="results" style="display:none;">
        <div class="metrics">
          <div id="confidenceMetric" class="metric">
            <div class="label">Confidence</div>
            <div id="confidenceValue" class="value">0%</div>
          </div>
          <div id="reviewMetric" class="metric">
            <div class="label">Review</div>
            <div id="reviewValue" class="value">-</div>
          </div>
          <div class="metric">
            <div class="label">Fields</div>
            <div id="fieldCount" class="value">0</div>
          </div>
          <div class="metric">
            <div class="label">Time</div>
            <div id="timeValue" class="value">0 ms</div>
          </div>
        </div>

        <div class="tabs">
          <button class="tab active" type="button" data-tab="fieldsView">Fields</button>
          <button class="tab" type="button" data-tab="relationsView">Relations</button>
          <button class="tab" type="button" data-tab="jsonView">JSON</button>
          <button class="tab" type="button" data-tab="ocrView">OCR Text</button>
        </div>

        <div id="fieldsView" class="result-view active">
          <table>
            <thead><tr><th>Type</th><th>Label</th><th>Value</th><th>Confidence</th><th>Uncertainty</th><th>BBox</th></tr></thead>
            <tbody id="fieldsBody"></tbody>
          </table>
        </div>

        <div id="relationsView" class="result-view">
          <table>
            <thead><tr><th>Head</th><th>Tail</th><th>Confidence</th></tr></thead>
            <tbody id="relationsBody"></tbody>
          </table>
        </div>

        <div id="jsonView" class="result-view">
          <pre id="jsonOutput"></pre>
        </div>

        <div id="ocrView" class="result-view">
          <pre id="ocrOutput" class="ocr-text"></pre>
        </div>
      </div>
    </section>
  </main>

  <script>
    const fileInput = document.getElementById("fileInput");
    const dropzone = document.getElementById("dropzone");
    const fileName = document.getElementById("fileName");
    const form = document.getElementById("uploadForm");
    const submitButton = document.getElementById("submitButton");
    const message = document.getElementById("message");
    const preview = document.getElementById("preview");
    const previewImage = document.getElementById("previewImage");

    function setMessage(text, isError = false) {
      message.textContent = text;
      message.className = isError ? "message error" : "message";
    }

    async function loadStatus() {
      try {
        const response = await fetch("/api/status");
        const data = await response.json();
        const gpu = data.cuda_available ? data.gpu : "CPU";
        document.getElementById("statusText").textContent = `Ready · ${gpu}`;
        document.getElementById("statusDot").classList.add("ready");
      } catch {
        document.getElementById("statusText").textContent = "Runtime unavailable";
      }
    }

    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", event => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", event => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
      if (event.dataTransfer.files.length) {
        fileInput.files = event.dataTransfer.files;
        handleFileSelection();
      }
    });
    fileInput.addEventListener("change", handleFileSelection);

    function handleFileSelection() {
      const file = fileInput.files[0];
      if (!file) return;
      fileName.textContent = file.name;
      if (file.type.startsWith("image/")) {
        previewImage.src = URL.createObjectURL(file);
        preview.style.display = "block";
      } else {
        preview.style.display = "none";
      }
    }

    document.querySelectorAll(".tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".result-view").forEach(v => v.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(tab.dataset.tab).classList.add("active");
      });
    });

    function formatPercent(value) {
      return `${(Number(value) * 100).toFixed(2)}%`;
    }

    function cell(text, className = "") {
      const td = document.createElement("td");
      td.textContent = text ?? "";
      if (className) td.className = className;
      return td;
    }

    function renderResults(data) {
      document.getElementById("emptyState").style.display = "none";
      document.getElementById("results").style.display = "block";

      const confidenceMetric = document.getElementById("confidenceMetric");
      confidenceMetric.className = `metric ${data.needs_human_review ? "warn" : "good"}`;
      document.getElementById("confidenceValue").textContent = `${data.overall_confidence_percent.toFixed(2)}%`;

      const reviewMetric = document.getElementById("reviewMetric");
      reviewMetric.className = `metric ${data.needs_human_review ? "warn" : "good"}`;
      document.getElementById("reviewValue").textContent = data.needs_human_review ? "Needed" : "Clear";
      document.getElementById("fieldCount").textContent = data.fields.length;
      document.getElementById("timeValue").textContent = `${Math.round(data.processing_time_ms)} ms`;

      const fieldsBody = document.getElementById("fieldsBody");
      fieldsBody.innerHTML = "";
      if (data.fields.length === 0) {
        const tr = document.createElement("tr");
        const td = cell("No fields extracted", "value-cell");
        td.colSpan = 6;
        tr.appendChild(td);
        fieldsBody.appendChild(tr);
      } else {
        data.fields.forEach(field => {
          const tr = document.createElement("tr");
          tr.appendChild(cell(field.field_name));
          tr.appendChild(cell(field.label));
          tr.appendChild(cell(field.value, "value-cell"));
          tr.appendChild(cell(formatPercent(field.confidence)));
          tr.appendChild(cell(field.uncertainty === null ? "" : field.uncertainty.toFixed(4)));
          tr.appendChild(cell(field.bbox ? JSON.stringify(field.bbox) : ""));
          fieldsBody.appendChild(tr);
        });
      }

      const relationsBody = document.getElementById("relationsBody");
      relationsBody.innerHTML = "";
      if (data.relations.length === 0) {
        const tr = document.createElement("tr");
        const td = cell("No relations extracted", "value-cell");
        td.colSpan = 3;
        tr.appendChild(td);
        relationsBody.appendChild(tr);
      } else {
        data.relations.forEach(relation => {
          const tr = document.createElement("tr");
          tr.appendChild(cell(relation.head, "value-cell"));
          tr.appendChild(cell(relation.tail, "value-cell"));
          tr.appendChild(cell(formatPercent(relation.confidence)));
          relationsBody.appendChild(tr);
        });
      }

      document.getElementById("jsonOutput").textContent = JSON.stringify(data.json_result, null, 2);
      document.getElementById("ocrOutput").textContent = data.raw_ocr_text || "";
    }

    form.addEventListener("submit", async event => {
      event.preventDefault();
      const file = fileInput.files[0];
      if (!file) {
        setMessage("Choose a document first.", true);
        return;
      }

      const payload = new FormData();
      payload.append("file", file);
      payload.append("checkpoint_path", document.getElementById("checkpoint").value);
      payload.append("threshold", document.getElementById("threshold").value);
      payload.append("mc_passes", document.getElementById("mcPasses").value);
      payload.append("language", document.getElementById("language").value);
      payload.append("no_gpu", document.getElementById("noGpu").checked ? "true" : "false");

      submitButton.disabled = true;
      setMessage("Loading model and extracting document data...");

      try {
        const response = await fetch("/api/extract", { method: "POST", body: payload });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Extraction failed");
        }
        renderResults(data);
        setMessage(`Completed. Confidence ${data.overall_confidence_percent.toFixed(2)}%.`);
      } catch (error) {
        setMessage(error.message, true);
      } finally {
        submitButton.disabled = false;
      }
    });

    loadStatus();
  </script>
</body>
</html>
"""
