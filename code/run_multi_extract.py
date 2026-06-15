"""
Multi-image attribute extraction runner.

Sends all images of a product to an MLLM in a single request,
extracting structured property-value pairs (CPV results).

Input:  item_aggregate_public.jsonl (item-level, with images list and benchmark cpv_results)
Output: per-item prediction JSONL (with benchmark_cpv_results + prediction_cpv_results)

Usage:
    python run_multi_extract.py \
        --input ../data/item_aggregate_public.jsonl \
        --output results/multi_extract_results.jsonl \
        --provider openai --model qwen-plus \
        --api-key $API_KEY --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
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

from model_client import ModelClient, create_client
from utils import parse_json_response


PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_done_item_ids(output_path: str, only_success: bool = False) -> set[str]:
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
            iid = _safe_str(rec.get("item_id")).strip()
            if iid:
                done.add(iid)
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


def _get_sorted_image_paths(item: dict) -> list[str]:
    """Sort images: main images first (by index), then detail images (by index)."""
    images = item.get("images") or []
    main_imgs = []
    detail_imgs = []
    for img in images:
        path = img.get("image_path", "").strip()
        if not path:
            continue
        if img.get("image_source") == "main_image":
            main_imgs.append((img.get("image_index", 0), path))
        else:
            detail_imgs.append((img.get("image_index", 0), path))
    main_imgs.sort(key=lambda x: x[0])
    detail_imgs.sort(key=lambda x: x[0])
    return [p for _, p in main_imgs] + [p for _, p in detail_imgs]


def _load_prompt_template() -> str:
    path = os.path.join(PROMPT_DIR, "extraction_prompt.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _build_prompt(template: str, item: dict, image_count: int) -> str:
    return Template(template).safe_substitute(
        image_count=str(image_count),
        main_entity=_safe_str(item.get("main_entity")).strip(),
        cate1_name=_safe_str(item.get("cate1_name")),
        cate_name=_safe_str(item.get("cate_name")),
        cpv_schema=_safe_str(item.get("cpv_schema")),
    )


def _clean_cpv_results(cpv_results: Any) -> list[dict]:
    cleaned: list[dict] = []
    if not isinstance(cpv_results, list):
        return cleaned
    for item in cpv_results:
        if not isinstance(item, dict):
            continue
        pn = _safe_str(item.get("property_name")).strip()
        pv = item.get("property_value")
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
    image_paths: list[Path],
    retry: int,
) -> dict:
    attempts = 0
    for attempt in range(retry):
        attempts += 1
        try:
            async with request_semaphore:
                messages = [{"role": "user", "content": prompt}]
                response_text = await client.chat(messages, images=image_paths)

            parsed = parse_json_response(response_text)
            if parsed is not None and "cpv_results" in parsed:
                return {
                    "status": "success",
                    "cpv_results": _clean_cpv_results(parsed["cpv_results"]),
                    "error_type": None,
                    "attempts": attempts,
                }

            if attempt < retry - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return {
                "status": "failed",
                "cpv_results": [],
                "error_type": "output_parse_failed",
                "attempts": attempts,
                "raw_output": (response_text or "")[:2000],
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
                "raw_output": str(exc)[:500],
            }

    return {"status": "failed", "cpv_results": [], "error_type": "unexpected_exit", "attempts": attempts}


async def _process_one(
    client: ModelClient,
    item_semaphore: asyncio.Semaphore,
    request_semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    output_path: str,
    item: dict,
    prompt_template: str,
    data_dir: str,
    retry: int,
) -> str:
    async with item_semaphore:
        started_at = time.time()
        rel_paths = _get_sorted_image_paths(item)
        abs_paths = [Path(data_dir) / p for p in rel_paths]
        prompt = _build_prompt(prompt_template, item, len(rel_paths))
        result = await _call_model(client, request_semaphore, prompt, abs_paths, retry)

        out = {
            "item_id": _safe_str(item.get("item_id")).strip(),
            "title": item.get("title", ""),
            "cate1_name": item.get("cate1_name", ""),
            "cate_name": item.get("cate_name", ""),
            "cpv_schema": item.get("cpv_schema", ""),
            "main_entity": item.get("main_entity", ""),
            "image_count": len(rel_paths),
            "image_paths": rel_paths,
            "benchmark_cpv_results": item.get("cpv_results", []),
            "prediction_cpv_results": result.get("cpv_results", []),
            "status": result["status"],
            "error_type": result.get("error_type"),
            "attempts": result.get("attempts", 0),
            "elapsed_ms": int((time.time() - started_at) * 1000),
        }
        if result.get("raw_output"):
            out["raw_output"] = result["raw_output"]

        async with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()

        return result["status"]


async def _run(
    items: list[dict],
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
                output_path, item, prompt_template, data_dir, retry,
            )
        )
        for item in items
    ]
    results: list[str] = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="extract"):
        results.append(await fut)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-image attribute extraction runner")
    parser.add_argument("--input", required=True, help="Path to item_aggregate_public.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--data-dir", default=None,
                        help="Base directory for resolving relative image paths (default: parent of input file)")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                        help="LLM API provider (default: openai)")
    parser.add_argument("--model", required=True, help="Model name (e.g., qwen-plus, claude-sonnet-4-20250514)")
    parser.add_argument("--api-key", default=None, help="API key (or set API_KEY env var)")
    parser.add_argument("--api-base", default=None, help="API base URL (or set API_BASE_URL env var)")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking/reasoning mode")
    parser.add_argument("--workers", type=int, default=10, help="Item-level concurrency (default: 10)")
    parser.add_argument("--request-workers", type=int, default=30, help="Global HTTP concurrency (default: 30)")
    parser.add_argument("--retry", type=int, default=3, help="Retries per request (default: 3)")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N items")
    parser.add_argument("--max-images", type=int, default=60,
                        help="Skip items with more than this many images (default: 60)")
    parser.add_argument("--shuffle", action="store_true", help="Randomize item order")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite output file and start fresh")
    parser.add_argument("--retry-failed", action="store_true", help="Re-run items with status=failed")
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

    items: list[dict] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                items.append(json.loads(raw))
            except json.JSONDecodeError:
                continue

    if args.max_images:
        before = len(items)
        items = [it for it in items if len(_get_sorted_image_paths(it)) <= args.max_images]
        skipped = before - len(items)
        if skipped:
            print(f"[skip]    {skipped} items with >{args.max_images} images")

    done_ids = _load_done_item_ids(args.output)
    pending = [it for it in items if _safe_str(it.get("item_id")).strip() not in done_ids]

    if args.shuffle:
        import random
        random.shuffle(pending)
    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"[input]   {len(items)} items")
    print(f"[done]    {len(done_ids)} items")
    print(f"[pending] {len(pending)} items")

    if not pending:
        print("Nothing to do.")
        return

    prompt_template = _load_prompt_template()
    client = create_client(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.api_base,
        enable_thinking=args.enable_thinking,
    )

    results = asyncio.run(_run(
        items=pending,
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
    print(f"\nDone: {len(results)} items")
    for status, cnt in sorted(counts.items()):
        print(f"  {status}: {cnt}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
