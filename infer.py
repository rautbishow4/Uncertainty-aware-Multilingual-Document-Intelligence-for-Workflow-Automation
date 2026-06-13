"""
infer.py
--------
CLI for running document intelligence inference on a file or folder.

Usage:
    # Single document
    python infer.py --input invoice.pdf --checkpoint checkpoints/best_model.pt

    # Batch folder
    python infer.py --input ./documents/ --checkpoint checkpoints/best_model.pt --output ./results/

    # With custom confidence threshold
    python infer.py --input form.png --checkpoint checkpoints/best_model.pt --threshold 0.75

    # With more MC passes for higher accuracy
    python infer.py --input contract.pdf --checkpoint checkpoints/best_model.pt --mc_passes 50
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


def collect_inputs(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        files = [
            str(f) for f in sorted(p.iterdir())
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        logger.info(f"Found {len(files)} document(s) in {p}")
        return files
    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Document Intelligence Inference")
    parser.add_argument("--input",       required=True, help="Input file or folder")
    parser.add_argument("--checkpoint",  required=True, help="Model checkpoint path")
    parser.add_argument("--output",      default="./results", help="Output folder for JSON results")
    parser.add_argument("--threshold",   type=float, default=0.70,
                        help="Confidence threshold for human review trigger (default: 0.70)")
    parser.add_argument("--mc_passes",   type=int, default=20,
                        help="Number of MC-Dropout forward passes (default: 20)")
    parser.add_argument("--lang",        default="auto", help="Language hint (auto = detect)")
    parser.add_argument("--no_gpu",      action="store_true")
    parser.add_argument("--pretty",      action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    import torch
    device = torch.device("cpu" if args.no_gpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # ── Load pipeline ────────────────────────────────────────────
    logger.info(f"Loading model from: {args.checkpoint}")
    from inference.pipeline import DocumentIntelligencePipeline
    pipeline = DocumentIntelligencePipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        confidence_threshold=args.threshold,
        mc_passes=args.mc_passes,
        device=device,
    )
    logger.info(f"Model loaded. Confidence threshold: {args.threshold:.0%} | MC passes: {args.mc_passes}")

    # ── Collect inputs ────────────────────────────────────────────
    input_files = collect_inputs(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference summary ─────────────────────────────────────────
    summary = {
        "total":          len(input_files),
        "high_confidence": 0,
        "flagged_review":  0,
        "failed":          0,
        "results":         [],
    }

    for i, input_path in enumerate(input_files, 1):
        doc_name = Path(input_path).stem
        logger.info(f"\n[{i}/{len(input_files)}] Processing: {input_path}")

        try:
            result = pipeline.process(
                input_path=input_path,
                doc_id=doc_name,
                language=args.lang,
                uncertainty_threshold=args.threshold,
            )

            # Save JSON
            output_path = output_dir / f"{doc_name}_result.json"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.to_json(indent=2 if args.pretty else None))

            # Console summary
            status = "⚠  REVIEW" if result.needs_human_review else "✓  OK"
            logger.info(f"  {status} | Confidence: {result.overall_confidence:.2%} | "
                        f"Fields: {len(result.fields)} | Relations: {len(result.relations)} | "
                        f"Time: {result.processing_time_ms:.0f}ms")

            if result.needs_human_review:
                summary["flagged_review"] += 1
                logger.warning(f"  Reason: {result.review_reason}")
            else:
                summary["high_confidence"] += 1

            summary["results"].append({
                "doc_id":     result.doc_id,
                "confidence": round(result.overall_confidence, 4),
                "review":     result.needs_human_review,
                "fields":     len(result.fields),
                "output":     str(output_path),
            })

        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)
            summary["failed"] += 1
            summary["results"].append({"doc_id": doc_name, "error": str(e)})

    # ── Write summary ─────────────────────────────────────────────
    summary_path = output_dir / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print final report ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"  Total processed   : {summary['total']}")
    print(f"  High confidence ✓ : {summary['high_confidence']}")
    print(f"  Flagged for review: {summary['flagged_review']}")
    print(f"  Failed            : {summary['failed']}")
    print(f"  Output directory  : {output_dir.resolve()}")
    print(f"  Summary file      : {summary_path}")
    print("=" * 60)

    if summary["flagged_review"] > 0:
        print(f"\n⚠  {summary['flagged_review']} document(s) need human review (confidence < {args.threshold:.0%})")
        flagged = [r for r in summary["results"] if r.get("review")]
        for r in flagged:
            print(f"   - {r['doc_id']} (confidence: {r['confidence']:.2%})")


if __name__ == "__main__":
    main()
