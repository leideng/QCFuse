#!/usr/bin/env python3
"""Build Blend-format RULER MV/MQ/VT JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


SCRIPT_VERSION = "qcfuse-ruler-v1-10k-cache-count"
TASKS = {
    "vt": {
        "output": "ruler_vt.jsonl",
        "template": "Find all variables that are assigned the value {query} in the text above.",
    },
    "niah_multiquery": {
        "output": "ruler_mq.jsonl",
        "template": "What are all the special magic numbers for {query} mentioned in the provided text?",
    },
    "niah_multivalue": {
        "output": "ruler_mv.jsonl",
        "template": "What are all the special magic numbers for {query} mentioned in the provided text?",
    },
}
RAW_FIELDS = {"context_raw", "query", "evidence", "outputs"}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[run]", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def replace_once(
    text: str,
    pattern: str,
    repl: str | Callable[[re.Match[str]], str],
    label: str,
) -> str:
    new_text, count = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not patch {label}")
    return new_text


def replace_all(
    text: str,
    pattern: str,
    repl: str | Callable[[re.Match[str]], str],
    label: str,
) -> str:
    new_text, count = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if count == 0:
        raise RuntimeError(f"Could not patch {label}")
    return new_text


def restore_if_old_patch(path: Path, markers: list[str]) -> tuple[str, bool]:
    text = path.read_text(encoding="utf-8")
    if all(marker in text for marker in markers):
        return text, True
    if '"context_raw": context' in text or '"query": query' in text or '"query": value' in text:
        backup = path.with_name(path.name + ".blend_backup")
        if not backup.exists():
            raise RuntimeError(f"{path} is already patched, but {backup} is missing")
        text = backup.read_text(encoding="utf-8")
        path.write_text(text, encoding="utf-8")
    return text, False


def write_patch(path: Path, old_text: str, new_text: str) -> None:
    if old_text == new_text:
        return
    backup = path.with_name(path.name + ".blend_backup")
    if not backup.exists():
        backup.write_text(old_text, encoding="utf-8")
    path.write_text(new_text, encoding="utf-8")
    print(f"[patch] {path}")


def patch_niah(path: Path) -> None:
    markers = ['"context_raw": context', '"query": query', '"evidence": evidence', '"evidence": sample["evidence"]']
    original, done = restore_if_old_patch(path, markers)
    if done:
        return

    def patch_return(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}evidence = [\n"
            f"{indent}    needle.format(\n"
            f"{indent}        type_needle_v=args.type_needle_v,\n"
            f"{indent}        key=keys[i],\n"
            f"{indent}        value=v,\n"
            f"{indent}    )\n"
            f"{indent}    for i in indices\n"
            f"{indent}    for v in values[i]\n"
            f"{indent}]\n"
            f"{indent}return {{\n"
            f"{indent}    \"input_text\": input_text,\n"
            f"{indent}    \"answers\": answers,\n"
            f"{indent}    \"context_raw\": context,\n"
            f"{indent}    \"query\": query,\n"
            f"{indent}    \"evidence\": evidence,\n"
            f"{indent}}}"
        )

    text = replace_once(original, r"^(?P<indent>\s*)return\s+input_text\s*,\s*answers\s*$", patch_return, "niah return")
    text = replace_once(
        text,
        r"^(?P<indent>\s*)sample_input_text\s*,\s*_\s*=\s*generate_input_output\(incremental\)\s*$",
        lambda m: f"{m.group('indent')}sample = generate_input_output(incremental)\n{m.group('indent')}sample_input_text = sample[\"input_text\"]",
        "niah estimate",
    )
    text = replace_once(
        text,
        r"^(?P<indent>\s*)input_text\s*,\s*answer\s*=\s*generate_input_output\(mid\)\s*$",
        lambda m: f"{m.group('indent')}sample = generate_input_output(mid)\n{m.group('indent')}input_text = sample[\"input_text\"]\n{m.group('indent')}answer = sample[\"answers\"]",
        "niah search",
    )
    text = replace_once(
        text,
        r"^(?P<indent>\s*)input_text\s*,\s*answer\s*=\s*generate_input_output\(used_haystack\)\s*$",
        lambda m: f"{m.group('indent')}sample = generate_input_output(used_haystack)\n{m.group('indent')}input_text = sample[\"input_text\"]\n{m.group('indent')}answer = sample[\"answers\"]",
        "niah final",
    )
    text = replace_once(
        text,
        r'^(?P<indent>\s*)"outputs"\s*:\s*answer\s*,\s*$',
        lambda m: (
            f'{m.group("indent")}"outputs": answer,\n'
            f'{m.group("indent")}"context_raw": sample["context_raw"],\n'
            f'{m.group("indent")}"query": sample["query"],\n'
            f'{m.group("indent")}"evidence": sample["evidence"],'
        ),
        "niah output",
    )
    write_patch(path, original, text)


def patch_variable_tracking(path: Path) -> None:
    markers = ['"context_raw": context', '"query": value', '"evidence": chains[0]', '"evidence": sample["evidence"]']
    original, done = restore_if_old_patch(path, markers)
    if done:
        return

    def patch_return(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}return {{\n"
            f"{indent}    \"input_text\": input_text,\n"
            f"{indent}    \"answers\": vars[0],\n"
            f"{indent}    \"context_raw\": context,\n"
            f"{indent}    \"query\": value,\n"
            f"{indent}    \"evidence\": chains[0],\n"
            f"{indent}}}"
        )

    text = replace_once(original, r"^(?P<indent>\s*)return\s+input_text\s*,\s*vars\[0\]\s*$", patch_return, "vt return")
    text = replace_once(
        text,
        r"^(?P<indent>\s*)sample_input_text\s*,\s*_\s*=\s*(?P<call>generate_input_output\(incremental,.*\))\s*$",
        lambda m: f"{m.group('indent')}sample = {m.group('call')}\n{m.group('indent')}sample_input_text = sample[\"input_text\"]",
        "vt estimate",
    )
    text = replace_once(
        text,
        r"^(?P<indent>\s*)input_text\s*,\s*answer\s*=\s*(?P<call>generate_input_output\(mid,.*\))\s*$",
        lambda m: f"{m.group('indent')}sample = {m.group('call')}\n{m.group('indent')}input_text = sample[\"input_text\"]\n{m.group('indent')}answer = sample[\"answers\"]",
        "vt search",
    )
    text = replace_once(
        text,
        r"^(?P<indent>\s*)input_text\s*,\s*answer\s*=\s*(?P<call>generate_input_output\(used_noises,.*\))\s*$",
        lambda m: f"{m.group('indent')}sample = {m.group('call')}\n{m.group('indent')}input_text = sample[\"input_text\"]\n{m.group('indent')}answer = sample[\"answers\"]",
        "vt final",
    )
    text = replace_all(
        text,
        r'^(?P<indent>\s*)"outputs"\s*:\s*answer\s*,\s*$',
        lambda m: (
            f'{m.group("indent")}"outputs": answer,\n'
            f'{m.group("indent")}"context_raw": sample["context_raw"],\n'
            f'{m.group("indent")}"query": sample["query"],\n'
            f'{m.group("indent")}"evidence": sample["evidence"],'
        ),
        "vt output",
    )
    write_patch(path, original, text)


def patch_ruler(ruler_dir: Path) -> None:
    synthetic = ruler_dir / "scripts" / "data" / "synthetic"
    patch_niah(synthetic / "niah.py")
    patch_variable_tracking(synthetic / "variable_tracking.py")


def ensure_essay_data(ruler_dir: Path) -> None:
    essay = ruler_dir / "scripts" / "data" / "synthetic" / "json" / "PaulGrahamEssays.json"
    if essay.exists():
        return
    run([sys.executable, "download_paulgraham_essay.py"], cwd=essay.parent)
    if not essay.exists():
        raise RuntimeError(f"Essay file was not created: {essay}")


def git_revision(ruler_dir: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ruler_dir),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc.stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only.
        return f"unknown ({exc})"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generator_source_hashes(ruler_dir: Path) -> dict[str, str]:
    synthetic = ruler_dir / "scripts" / "data" / "synthetic"
    return {
        "niah.py": file_sha256(synthetic / "niah.py"),
        "variable_tracking.py": file_sha256(synthetic / "variable_tracking.py"),
    }


def build_raw_cache_config(
    args: argparse.Namespace,
    raw_samples: int,
    ruler_git_rev: str,
) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "ruler_git_rev": ruler_git_rev,
        "generator_source_hashes": generator_source_hashes(args.ruler_dir),
        "subset": args.subset,
        "tokenizer_path": args.tokenizer_path,
        "tokenizer_type": args.tokenizer_type,
        "model_template_type": args.model_template_type,
        "ruler_max_seq_length": args.ruler_max_seq_length,
        "raw_num_samples": raw_samples,
        "seed": args.seed,
    }


def make_cache_id(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def prepare_raw_cache(
    args: argparse.Namespace,
    raw_samples: int,
    ruler_git_rev: str,
) -> tuple[Path, dict[str, Any]]:
    config = build_raw_cache_config(args, raw_samples, ruler_git_rev)
    cache_id = make_cache_id(config)
    raw_save_dir = args.raw_dir / cache_id

    if args.force_raw and raw_save_dir.exists():
        shutil.rmtree(raw_save_dir)
        print(f"[raw] removed cache because --force_raw was set: {raw_save_dir}")

    raw_save_dir.mkdir(parents=True, exist_ok=True)
    meta_path = raw_save_dir / "_blend_raw_meta.json"
    if meta_path.exists():
        old_config = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_config != config:
            raise RuntimeError(
                f"Raw cache metadata mismatch at {meta_path}. "
                "Use --force_raw or a different --raw_dir."
            )
    else:
        meta_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[raw] cache_id={cache_id}")
    print(f"[raw] save_dir={raw_save_dir}")
    return raw_save_dir, config


def raw_ready(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            if count == 1 and not RAW_FIELDS.issubset(json.loads(line)):
                return False
    return count >= expected


def run_prepare(args: argparse.Namespace, task: str, raw_save_dir: Path, raw_samples: int) -> Path:
    raw_path = raw_save_dir / task / f"{args.subset}.jsonl"
    if raw_ready(raw_path, raw_samples):
        print(f"[raw] reuse {raw_path}")
        return raw_path
    if raw_path.exists():
        raw_path.unlink()
        print(f"[raw] removed incomplete or stale raw file: {raw_path}")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    prepare = args.ruler_dir / "scripts" / "data" / "prepare.py"
    run(
        [
            sys.executable,
            str(prepare),
            "--save_dir",
            str(raw_save_dir),
            "--benchmark",
            "synthetic",
            "--task",
            task,
            "--subset",
            args.subset,
            "--tokenizer_path",
            args.tokenizer_path,
            "--tokenizer_type",
            args.tokenizer_type,
            "--max_seq_length",
            str(args.ruler_max_seq_length),
            "--model_template_type",
            args.model_template_type,
            "--num_samples",
            str(raw_samples),
            "--random_seed",
            str(args.seed),
        ],
        cwd=args.ruler_dir / "scripts" / "data",
    )
    if not raw_ready(raw_path, raw_samples):
        raise RuntimeError(f"Raw RULER output is incomplete or missing required fields: {raw_path}")
    return raw_path


def load_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def encode(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def token_offset(tokenizer: Any, text: str, char_offset: int) -> int:
    return len(encode(tokenizer, text[: max(0, char_offset)]))


def anchor_span(tokenizer: Any, context: str, anchors: list[str]) -> tuple[int, int] | None:
    spans = []
    for anchor in anchors:
        if not anchor:
            continue
        start = context.find(anchor)
        if start < 0:
            continue
        end = start + len(anchor)
        tok_start = token_offset(tokenizer, context, start)
        tok_end = token_offset(tokenizer, context, end)
        spans.append((tok_start, max(tok_start + 1, tok_end)))
    if not spans:
        return None
    return min(start for start, _ in spans), max(end for _, end in spans)


def choose_start(total_tokens: int, target_tokens: int, span: tuple[int, int] | None) -> int | None:
    if total_tokens < target_tokens:
        return None
    if span is None:
        return 0
    start, end = span
    if end - start > target_tokens:
        return None
    lower = max(0, end - target_tokens)
    upper = min(start, total_tokens - target_tokens)
    return lower if lower <= upper else None


def chunk_context(
    tokenizer: Any,
    context: str,
    chunk_size: int,
    target_num_chunks: int,
    anchors: list[str],
) -> list[str]:
    token_ids = encode(tokenizer, context)
    target_tokens = chunk_size * target_num_chunks
    start = choose_start(len(token_ids), target_tokens, anchor_span(tokenizer, context, anchors))
    if start is None:
        return []
    window = token_ids[start : start + target_tokens]
    return [decode(tokenizer, window[i : i + chunk_size]) for i in range(0, target_tokens, chunk_size)]


def contains_text(text: str, needle: str) -> bool:
    if needle in text:
        return True
    return " ".join(needle.split()) in " ".join(text.split())


def evidence_anchors(task: str, raw: dict[str, Any], answers: list[str]) -> list[str]:
    if task == "vt":
        return [str(raw["query"])]
    evidence = raw.get("evidence")
    if isinstance(evidence, list) and evidence:
        return [str(item).strip() for item in evidence if str(item).strip()]
    return answers


def evidence_ok(task: str, raw: dict[str, Any], answers: list[str], chunks: list[str]) -> bool:
    joined = "".join(chunks)
    if task == "vt":
        return contains_text(joined.casefold(), str(raw["query"]).casefold())
    evidence = raw.get("evidence")
    if isinstance(evidence, list) and evidence:
        if not all(contains_text(joined, str(item).strip()) for item in evidence):
            return False
    return all(contains_text(joined, answer) for answer in answers)


def convert_raw(
    raw_path: Path,
    out_path: Path,
    task: str,
    tokenizer: Any,
    chunk_size: int,
    target_num_chunks: int,
    target_samples: int,
) -> dict[str, int]:
    stats = {"kept": 0, "skipped_short": 0, "skipped_missing_evidence": 0, "skipped_bad_fields": 0}
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if stats["kept"] >= target_samples:
                break
            if not line.strip():
                continue
            raw = json.loads(line)
            context = raw.get("context_raw")
            query = raw.get("query")
            answers = raw.get("outputs", raw.get("answers"))
            if not isinstance(context, str) or query is None or answers is None:
                stats["skipped_bad_fields"] += 1
                continue
            if isinstance(answers, str):
                answers = [answers]
            answers = [str(answer) for answer in answers]

            chunks = chunk_context(tokenizer, context, chunk_size, target_num_chunks, evidence_anchors(task, raw, answers))
            if len(chunks) != target_num_chunks:
                stats["skipped_short"] += 1
                continue
            if not evidence_ok(task, raw, answers, chunks):
                stats["skipped_missing_evidence"] += 1
                continue

            item = {
                "input": TASKS[task]["template"].format(query=str(query)),
                "query": str(query),
                "answers": answers,
                "context": chunks,
                "num_chunks": len(chunks),
                "similarity_scores": None,
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    if stats["kept"] != target_samples:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{task}: kept {stats['kept']} valid samples, expected {target_samples}. "
            f"Increase --raw_num_samples or --ruler_max_seq_length. Stats: {stats}"
        )
    tmp_path.replace(out_path)
    return stats


def write_final_metadata(
    args: argparse.Namespace,
    raw_save_dir: Path,
    raw_config: dict[str, Any],
    stats_by_task: dict[str, dict[str, int]],
    raw_samples: int,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "script_version": SCRIPT_VERSION,
        "tasks": TASKS,
        "raw_save_dir": str(raw_save_dir),
        "raw_config": raw_config,
        "final_num_samples_per_task": args.num_samples,
        "raw_num_samples_per_task": raw_samples,
        "chunk_size": args.chunk_size,
        "target_num_chunks": args.target_num_chunks,
        "target_context_tokens": args.chunk_size * args.target_num_chunks,
        "ruler_max_seq_length": args.ruler_max_seq_length,
        "stats_by_task": stats_by_task,
    }
    meta_path = args.output_dir / "ruler_blend_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[meta] wrote {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler_dir", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, default=Path("ruler_raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("final_data"))
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--tokenizer_type", default="hf")
    parser.add_argument("--model_template_type", default="base")
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--raw_num_samples", type=int)
    parser.add_argument("--raw_sample_multiplier", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ruler_max_seq_length", type=int, default=11264)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--target_num_chunks", type=int, default=20)
    parser.add_argument("--force_raw", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> int:
    if args.tokenizer_type != "hf":
        raise SystemExit("Only --tokenizer_type hf is supported")
    if not args.ruler_dir.exists():
        raise SystemExit(f"RULER repo not found: {args.ruler_dir}")
    if (
        args.num_samples <= 0
        or args.chunk_size <= 0
        or args.target_num_chunks <= 0
        or args.raw_sample_multiplier <= 0
        or args.ruler_max_seq_length <= 0
    ):
        raise SystemExit(
            "--num_samples, --chunk_size, --target_num_chunks, "
            "--raw_sample_multiplier, and --ruler_max_seq_length must be positive"
        )
    raw_samples = args.raw_num_samples or max(args.num_samples * args.raw_sample_multiplier, args.num_samples)
    if raw_samples < args.num_samples:
        raise SystemExit("--raw_num_samples must be >= --num_samples")
    return raw_samples


def main() -> None:
    args = parse_args()
    raw_samples = validate_args(args)

    ruler_git_rev = git_revision(args.ruler_dir)
    print(f"[ruler] git_rev={ruler_git_rev}")
    patch_ruler(args.ruler_dir)
    ensure_essay_data(args.ruler_dir)
    raw_save_dir, raw_config = prepare_raw_cache(args, raw_samples, ruler_git_rev)
    tokenizer = load_tokenizer(args.tokenizer_path)

    stats_by_task: dict[str, dict[str, int]] = {}
    for task, spec in TASKS.items():
        raw_path = run_prepare(args, task, raw_save_dir, raw_samples)
        out_path = args.output_dir / spec["output"]
        stats = convert_raw(raw_path, out_path, task, tokenizer, args.chunk_size, args.target_num_chunks, args.num_samples)
        stats_by_task[task] = stats
        print(f"[ok] {task}: {stats} -> {out_path}")

    write_final_metadata(args, raw_save_dir, raw_config, stats_by_task, raw_samples)


if __name__ == "__main__":
    main()
