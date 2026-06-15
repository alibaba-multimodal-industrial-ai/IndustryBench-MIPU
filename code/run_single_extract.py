"""
Single-image attribute extraction runner.

Reads image-level benchmark (single_image_level.jsonl), sends each image
to an MLLM, and extracts structured property-value pairs (CPV results).

Output format is compatible with run_eval.py (uses record_id as item_id).

Usage:
    python run_single_extract.py \
        --input ../data/single_image_level.jsonl \
        --output results/single_extract_results.jsonl \
        --provider openai --model qwen-plus \
        --api-key $API_KEY --api-base $API_BASE_URL \
        --workers 10 --request-workers 30 --retry 3
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from string import Template
from typing import Any

from tqdm import tqdm

from model_client import create_client, ModelClient
from utils import parse_json_response


PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_done_ids(output_path: str, only_success: bool = False) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if only_success and rec.get("status") != "success":
                continue
            rid = _safe_str(rec.get("item_id")).strip()
            if rid:
                done.add(rid)
    return done


def _strip_failed_rows(output_path: str) -> int:
    if not os.path.exists(output_path):
        return 0
    kept: list[str] = []
    stripped = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                kept.append(line if line.endswith("\n") else line + "\n")
                continue
            if rec.get("status") == "success":
                kept.append(line if line.endswith("\n") else line + "\n")
            else:
                stripped += 1
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(kept)
    return stripped


def _load_prompt_template() -> str:
    path = os.path.join(PROMPT_DIR, "extraction_prompt_single.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _build_prompt(template: str, record: dict) -> str:
    return Template(template).safe_substitute(
        image_count="1",
        image_source=_safe_str(record.get("image_source", "商品图片")),
        main_entity=_safe_str(record.get("main_entity")).strip(),
        cate1_name=_safe_str(record.get("cate1_name")),
        cate_name=_safe_str(record.get("cate_name")),
        cpv_schema=_safe_str(record.get("cpv_schema")),
    )


def _clean_cpv_results(cpv_results: Any) -> list[dict]:
    cleaned: list[dict] = []
    if not isinstance(cpv_results, list):
        return cleaned
    for item in cpv_results:
        if not isinstance(item, dict):
            continue
        pn = _safe_str(item.get("property_name") or item.get("attribute")).strip()
        pv = item.get("property_value") or item.get("final_value")
        if not pn or pv in (None, "", []):
            continue
        entry: dict = {"property_name": pn, "property_value": pv}
        conf = item.get("confidence")
        if conf is not None:
            try:
                entry["confidence"] = float(conf)
            except (ValueError, TypeError):
                pass
        cleaned.append(entry)
    return cleaned


async def _call_model(
    client: ModelClient,
    request_semaphore: asyncio.Semaphore,
    prompt: str,
    image_path: Path,
    retry: int,
) -> dict:
    attempts = 0
    for attempt in range(retry):
        attempts += 1
        try:
            async with request_semaphore:
                messages = [{"role": "user", "content": prompt}]
                response_text, finish_reason = await client.chat_with_finish(messages, images=[image_path])

            parsed = parse_json_response(response_text)
            if parsed is not None and "cpv_results" in parsed:
                return {
                    "status": "success",
                    "cpv_results": _clean_cpv_results(parsed["cpv_results"]),
                    "error_type": None,
                    "attempts": attempts,
                    "finish_reason": finish_reason,
                }

            if attempt < retry - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            err_type = "output_truncated" if finish_reason == "length" else "output_parse_failed"
            return {
                "status": "failed",
                "cpv_results": [],
                "error_type": err_type,
                "attempts": attempts,
                "finish_reason": finish_reason,
            }
        except Exception as exc:
            if attempt < retry - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return {
                "status": "failed",
                "cpv_results": [],
                "error_type": "request_failed",
                "attempts": attempts,
                "finish_reason": "error",
            }

    return {"status": "failed", "cpv_results": [], "error_type": "unexpected_exit", "attempts": attempts}


async def _process_one(
    client: ModelClient,
    item_semaphore: asyncio.Semaphore,
    request_semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    output_path: str,
    record: dict,
    prompt_template: str,
    data_dir: str,
    retry: int,
) -> str:
    async with item_semaphore:
        started_at = time.time()
        image_rel = record.get("image_path", "").strip()
        image_abs = Path(data_dir) / image_rel
        prompt = _build_prompt(prompt_template, record)
        result = await _call_model(client, request_semaphore, prompt, image_abs, retry)

        record_id = _safe_str(record.get("record_id")).strip()
        out = {
            "item_id": record_id,
            "record_id": record_id,
            "original_item_id": _safe_str(record.get("item_id")).strip(),
            "title": record.get("title", ""),
            "cate1_name": record.get("cate1_name", ""),
            "cate_name": record.get("cate_name", ""),
            "cpv_schema": record.get("cpv_schema", ""),
            "main_entity": record.get("main_entity", ""),
            "image_source": record.get("image_source", ""),
            "image_path": image_rel,
            "image_count": 1,
            "benchmark_cpv_results": record.get("cpv_results", []),
            "prediction_cpv_results": result.get("cpv_results", []),
            "status": result["status"],
            "error_type": result.get("error_type"),
            "finish_reason": result.get("finish_reason"),
            "attempts": result.get("attempts", 0),
            "elapsed_ms": int((time.time() - started_at) * 1000),
        }

        async with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()

        return result["status"]


async def _run(
    records: list[dict],
    output_path: str,
    client: ModelClient,
    prompt_template: str,
    data_dir: str,
    workers: int,
    request_workers: int,
    retry: int,
) -> list[str]:
    _ensure_parent_dir(output_path)
    item_semaphore = asyncio.Semaphore(workers)
    request_semaphore = asyncio.Semaphore(request_workers)
    write_lock = asyncio.Lock()

    tasks = [
        asyncio.create_task(
            _process_one(
                client, item_semaphore, request_semaphore, write_lock,
                output_path, record, prompt_template, data_dir, retry,
            )
        )
        for record in records
    ]
    results: list[str] = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="extract"):
        results.append(await fut)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-image attribute extraction")
    parser.add_argument("--input", required=True, help="Path to single_image_level.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--data-dir", default=None,
                        help="Base directory for resolving image paths (default: parent of input file)")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum generated tokens (default: 8192)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--request-workers", type=int, default=30)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    data_dir = args.data_dir or os.path.dirname(os.path.abspath(args.input))

    _ensure_parent_dir(args.output)
    if args.no_resume and os.path.exists(args.output):
        os.remove(args.output)

    if args.retry_failed:
        stripped = _strip_failed_rows(args.output)
        print(f"Stripped failed rows: {stripped}")

    records: list[dict] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                continue

    done_ids = _load_done_ids(args.output)
    pending = [r for r in records if _safe_str(r.get("record_id")).strip() not in done_ids]

    if args.shuffle:
        import random
        random.shuffle(pending)
    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"[input]   {len(records)} records")
    print(f"[done]    {len(done_ids)} records")
    print(f"[pending] {len(pending)} records")

    if not pending:
        print("Nothing to do.")
        return

    prompt_template = _load_prompt_template()
    client = create_client(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.api_base,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
    )

    results = asyncio.run(_run(
        records=pending,
        output_path=args.output,
        client=client,
        prompt_template=prompt_template,
        data_dir=data_dir,
        workers=args.workers,
        request_workers=args.request_workers,
        retry=args.retry,
    ))

    from collections import Counter
    counts = Counter(results)
    print(f"\nDone: {len(results)} records")
    for status, cnt in sorted(counts.items()):
        print(f"  {status}: {cnt}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
