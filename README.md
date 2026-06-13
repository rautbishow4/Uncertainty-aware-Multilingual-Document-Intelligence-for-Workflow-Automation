# Uncertainty-Aware Multilingual Document Intelligence for Workflow Automation

An end-to-end document AI system for extracting structured information from visually rich multilingual documents. The project combines **EasyOCR**, **LayoutLMv3**, **semantic entity recognition**, **relation extraction**, **MC-Dropout uncertainty estimation**, and a **FastAPI dashboard** for uploading documents and viewing results.

The system was trained on the [XFUND](https://github.com/doc-analysis/XFUND) multilingual form understanding dataset across seven languages:

| Code | Language |
|---|---|
| `zh` | Chinese |
| `ja` | Japanese |
| `es` | Spanish |
| `fr` | French |
| `it` | Italian |
| `de` | German |
| `pt` | Portuguese |

## Highlights

- Uses **EasyOCR only** for text and bounding-box extraction.
- Uses **LayoutLMv3** for text, layout, and image-aware document understanding.
- Supports PDF and image inputs: PNG, JPG, JPEG, TIFF, BMP.
- Extracts semantic fields using BIO labels.
- Predicts entity relations between fields.
- Estimates prediction confidence using MC-Dropout.
- Flags low-confidence documents for human review.
- Provides both command-line inference and a FastAPI dashboard.
- Displays extracted results in JSON and table formats.

## Architecture

```text
Document Image/PDF
        |
        v
EasyOCR Text Extraction
        |
        v
Words + Bounding Boxes + Raw OCR Text
        |
        v
LayoutLMv3 Tokenization + Image Processing
        |
        v
LayoutLMv3 Encoder
        |
        +--> SER Head: Semantic Entity Recognition
        |
        +--> RE Head: Relation Extraction
        |
        v
MC-Dropout Confidence Estimation
        |
        v
JSON Output + Dashboard Tables + Human Review Flag
```

## Repository Structure

```text
configs/
  base_config.yaml              Training and inference configuration

data/
  download_xfund.py             XFUND dataset downloader/index builder
  xfund_dataset.py              PyTorch dataset and collate function

evaluation/
  metrics.py                    SER, RE, and calibration metrics

inference/
  ocr_engine.py                 EasyOCR text extraction wrapper
  pipeline.py                   End-to-end inference pipeline

models/
  layout_xlm_uncertainty.py     Main model wrapper using LayoutLMv3
  ser_head.py                   Semantic entity recognition head
  re_head.py                    Relation extraction head
  uncertainty_module.py         MC-Dropout uncertainty utilities

training/
  trainer.py                    Training loop, evaluation, checkpointing

tests/
  test_model.py                 Unit tests

dashboard_app.py                FastAPI dashboard
infer.py                        Command-line inference
train.py                        Command-line training
requirements.txt                Python dependencies
```

## Model Choice

The final system uses:

```text
microsoft/layoutlmv3-base
```

LayoutLMv3 was selected because it can use text, bounding boxes, and image features without requiring Detectron2. The original LayoutXLM plan was not used in the final version because LayoutXLM/ LayoutLMv2 require Detectron2, which is difficult to install reliably on native Windows.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Check GPU availability:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Dataset

Download or prepare XFUND:

```powershell
python data\download_xfund.py --output_dir .\xfund_data --languages zh ja es fr it de pt
```

The downloaded dataset is intentionally ignored by Git because it is a generated external artifact.

## Training

Train LayoutLMv3 on all seven XFUND languages:

```powershell
python train.py --config configs\base_config.yaml --mode multitask --checkpoint_dir checkpoints_layoutlmv3 --no_wandb
```

Train on a single language:

```powershell
python train.py --config configs\base_config.yaml --mode language_specific --lang zh --checkpoint_dir checkpoints_zh --no_wandb
```

If GPU memory is limited, reduce the batch size in `configs/base_config.yaml`:

```yaml
batch_size: 1
gradient_accumulation_steps: 8
```

## Command-Line Inference

Run inference on an image:

```powershell
python infer.py --input documents\invoice.jpg --checkpoint checkpoints_layoutlmv3\best_model.pt --output results_invoice --pretty
```

Run inference on a PDF:

```powershell
python infer.py --input documents\sample.pdf --checkpoint checkpoints_layoutlmv3\best_model.pt --output results_pdf --pretty
```

EasyOCR is used automatically.

## FastAPI Dashboard

Start the dashboard:

```powershell
python -m uvicorn dashboard_app:app --host 127.0.0.1 --port 8000
```

Open in a browser:

```text
http://127.0.0.1:8000
```

The dashboard allows users to:

- upload a PDF or image document
- choose a checkpoint
- set the confidence threshold
- set the number of MC-Dropout passes
- force CPU mode if needed
- view overall confidence
- view human-review status
- inspect extracted fields in a table
- inspect extracted relations in a table
- view raw OCR text
- view the full JSON output

## Output Example

```json
{
  "doc_id": "invoice",
  "language": "auto",
  "overall_confidence": 0.85,
  "needs_human_review": false,
  "review_reason": null,
  "fields": {
    "key": {
      "value": "Invoice Number",
      "confidence": 0.91,
      "uncertainty": 0.12,
      "bbox": [120, 90, 260, 110]
    }
  },
  "relations": [],
  "processing_time_ms": 2150.4,
  "mc_passes_used": 20
}
```

## Results

### BERT Baseline

The first working version used `bert-base-multilingual-cased` as a fallback because LayoutXLM required Detectron2 on native Windows.

```text
Best SER F1: 0.6308
Best RE F1:  0.2689
Best Avg F1: 0.4499
```

### LayoutLMv3 Final Model

The LayoutLMv3 version improved performance because it uses text, layout, and image information.

```text
Best SER F1: 0.8083
Best RE F1:  0.2905
Best Avg F1: 0.5494
```

Final per-language SER F1:

| Language | SER F1 |
|---|---:|
| DE | 0.8189 |
| ES | 0.7779 |
| FR | 0.8329 |
| IT | 0.8389 |
| JA | 0.6813 |
| PT | 0.8260 |
| ZH | 0.7934 |
| Average | 0.7956 |

## Tests

Run the unit tests:

```powershell
python -m pytest tests\test_model.py -q
```

Expected result:

```text
19 passed
```


## Limitations

- The dashboard does not include a separate language detector. Multilingual extraction is supported through multitask training on XFUND.
- Relation extraction remains weaker than semantic entity recognition.
- Real-world invoices and receipts may differ from XFUND forms, so domain-specific fine-tuning can improve results.
- PDF inference currently processes the first page.
- OCR quality strongly affects downstream extraction quality.

## Future Work

- Add explicit language detection.
- Improve relation extraction with better entity span alignment.
- Add multi-page PDF support.
- Fine-tune on invoice-specific or receipt-specific datasets.
- Add visual highlighting of extracted fields on the original document.
- Add export options for CSV and Excel.

## Acknowledge
Dataset and pretrained model licenses belong to their respective owners, including XFUND, Hugging Face, Microsoft LayoutLMv3, PyTorch, Transformers, and EasyOCR.
