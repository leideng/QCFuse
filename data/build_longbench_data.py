#!/usr/bin/env python3
"""Build Blend-format LongBench JSONL files for musique, 2wikimqa, hotpotqa."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "qcfuse-longbench-v1-top20-200"
DATASETS = ("musique", "2wikimqa", "hotpotqa")


def load_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def normalize_context(value: Any) -> str:
    if isinstance(value, list):
        return "\n\n".join(str(item) for item in value)
    return "" if value is None else str(value)


def normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def make_splitter(tokenizer: Any, chunk_size: int, chunk_overlap: int) -> Any:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:  # pragma: no cover - compatibility with older langchain.
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    def token_len(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_len,
    )


def split_context(splitter: Any, context: str) -> list[str]:
    if not context.strip():
        return [""]
    chunks = splitter.split_text(context)
    return chunks or [""]


def sort_by_similarity(
    model: Any,
    query: str,
    chunks: list[str],
    batch_size: int,
) -> tuple[list[str], list[float]]:
    import numpy as np

    if not chunks:
        return [], []
    if not query.strip():
        return chunks, [0.0] * len(chunks)

    query_embedding = model.encode(query, normalize_embeddings=True, show_progress_bar=False)
    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    scores = np.dot(np.asarray(chunk_embeddings), np.asarray(query_embedding))
    order = np.argsort(scores)[::-1]
    return [chunks[i] for i in order], [float(scores[i]) for i in order]


def limit_context(
    chunks: list[str],
    scores: list[float],
    context_topk: int,
) -> tuple[list[str], list[float]]:
    if context_topk <= 0:
        return chunks, scores
    limit = min(len(chunks), context_topk)
    return chunks[:limit], scores[:limit]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_file(
    in_path: Path,
    out_path: Path,
    tokenizer: Any,
    embedding_model: Any,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    max_samples: int,
    context_topk: int,
) -> int:
    splitter = make_splitter(tokenizer, chunk_size, chunk_overlap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if count >= max_samples:
                break
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON in {in_path}:{line_no}: {exc}") from exc

            query = str(raw.get("input", ""))
            chunks = split_context(splitter, normalize_context(raw.get("context", "")))
            chunks, scores = sort_by_similarity(embedding_model, query, chunks, batch_size)
            chunks, scores = limit_context(chunks, scores, context_topk)
            item = {
                "input": query,
                "context": chunks,
                "answers": normalize_answers(raw.get("answers", [])),
                "num_chunks": len(chunks),
                "similarity_scores": scores,
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    return count


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    datasets: list[str],
    output_counts: dict[str, int],
    input_hashes: dict[str, str],
) -> None:
    metadata = {
        "script_version": SCRIPT_VERSION,
        "datasets": datasets,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "tokenizer_path": args.tokenizer_path,
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "context_topk": args.context_topk,
        "max_samples_per_dataset": args.max_samples,
        "batch_size": args.batch_size,
        "device": args.device,
        "input_sha256": input_hashes,
        "output_counts": output_counts,
    }
    path = output_dir / "longbench_blend_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[meta] wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=Path("raw_longbench"))
    parser.add_argument("--output_dir", type=Path, default=Path("final_data"))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--context_topk", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda device id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Unsupported datasets: {', '.join(unknown)}")
    if args.chunk_size <= 0 or args.chunk_overlap < 0 or args.context_topk <= 0 or args.max_samples <= 0:
        raise SystemExit(
            "--chunk_size, --context_topk, and --max_samples must be positive; "
            "--chunk_overlap must be non-negative"
        )

    tokenizer = load_tokenizer(args.tokenizer_path)
    from sentence_transformers import SentenceTransformer

    device = None if args.device == "auto" else args.device
    embedding_model = SentenceTransformer(args.embedding_model, device=device)

    processed = 0
    output_counts: dict[str, int] = {}
    input_hashes: dict[str, str] = {}
    for dataset in datasets:
        in_path = args.input_dir / f"{dataset}.jsonl"
        if not in_path.exists():
            print(f"[skip] missing {in_path}")
            continue
        input_hashes[dataset] = file_sha256(in_path)
        out_path = args.output_dir / f"{dataset}.jsonl"
        count = convert_file(
            in_path=in_path,
            out_path=out_path,
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            context_topk=args.context_topk,
        )
        processed += count
        output_counts[dataset] = count
        print(f"[ok] {dataset}: {count} samples -> {out_path}")

    if processed == 0:
        raise SystemExit("No samples were processed. Check --input_dir and --datasets.")
    write_metadata(args.output_dir, args, datasets, output_counts, input_hashes)


if __name__ == "__main__":
    main()
