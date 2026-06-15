"""
Attribute-level evaluation: semantic matching of predicted vs. benchmark values.

Two-stage pipeline: rule-based matching -> LLM semantic judge.

v2: rule_match_all returns list[str] (all matching bench values).
    Output uses matched_bench_values (list) instead of matched_bench_value (str).

Usage:
    python run_eval.py \
        --input results/extract.jsonl \
        --output results/eval_results.jsonl \
        --provider openai --model qwen-plus \
        --api-key $API_KEY --api-base https://your-api-endpoint \
        --workers 10 --request-workers 30
"""

import argparse
import asyncio
import json
import os
import re
import unicodedata
from string import Template
from typing import Any

from tqdm import tqdm

from model_client import ModelClient, create_client
from utils import parse_json_response


PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


# ---------------------------------------------------------------------------
# Rule-based matching (collect ALL matching bench values)
# ---------------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(r"\d+\.\d+")


def _canonicalize_numbers(value: str) -> str:
    def _strip(match: re.Match) -> str:
        stripped = match.group(0).rstrip("0").rstrip(".")
        return stripped or "0"
    return _NUMBER_PATTERN.sub(_strip, value)


def _normalize_match_value(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.upper()
    value = _canonicalize_numbers(value)
    value = re.sub(r"[^\w]", "", value, flags=re.UNICODE)
    return value


def _is_subsequence(short_value: str, long_value: str) -> bool:
    if len(short_value) > len(long_value):
        return False
    cursor = 0
    for char in long_value:
        if cursor < len(short_value) and short_value[cursor] == char:
            cursor += 1
    return cursor == len(short_value)


def rule_match_all(predicted_value: str, bench_values: list[str]) -> list[str]:
    """Return ALL matching bench values (exact first, then containment/subsequence)."""
    norm_pred = _normalize_match_value(predicted_value)
    if not norm_pred:
        return []

    matched: list[str] = []
    exact_matched_indices: set[int] = set()

    for i, bv in enumerate(bench_values):
        if _normalize_match_value(bv) == norm_pred:
            matched.append(bv)
            exact_matched_indices.add(i)

    for i, bv in enumerate(bench_values):
        if i in exact_matched_indices:
            continue
        norm_bv = _normalize_match_value(bv)
        if not norm_bv:
            continue
        if (
            norm_bv in norm_pred
            or norm_pred in norm_bv
            or _is_subsequence(norm_bv, norm_pred)
            or _is_subsequence(norm_pred, norm_bv)
        ):
            matched.append(bv)

    return matched


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def make_cache_key(property_name: str, predicted_value: str, bench_values: list[str]) -> str:
    return json.dumps([property_name, predicted_value, sorted(bench_values)], ensure_ascii=False)


def load_cache(path: str) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not os.path.exists(path):
        return cache
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = json.dumps(obj["key"], ensure_ascii=False)
                cache[key] = obj
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _load_judge_prompts() -> tuple[str, str]:
    sys_path = os.path.join(PROMPT_DIR, "judge_system_prompt.txt")
    user_path = os.path.join(PROMPT_DIR, "judge_user_prompt.txt")
    with open(sys_path, encoding="utf-8") as f:
        sys_prompt = f.read()
    with open(user_path, encoding="utf-8") as f:
        user_template = f.read()
    return sys_prompt, user_template


def _extract_matched_bench_values(result: dict) -> list[str]:
    """Extract matched_bench_values from LLM response, with backward compat."""
    mbv_list = result.get("matched_bench_values")
    if isinstance(mbv_list, list):
        return [str(v) for v in mbv_list if v is not None]
    mbv_single = result.get("matched_bench_value")
    if mbv_single is not None and mbv_single != "":
        return [str(mbv_single)]
    return []


async def call_judge(
    client: ModelClient,
    request_semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_template: str,
    property_name: str,
    predicted_value: str,
    bench_values: list[str],
    retry: int,
) -> dict:
    user_prompt = Template(user_template).safe_substitute(
        property_name=property_name,
        predicted_value=predicted_value,
        bench_values=json.dumps(bench_values, ensure_ascii=False),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(retry):
        try:
            async with request_semaphore:
                response_text = await client.chat(messages)

            result = parse_json_response(response_text)
            if result is not None and "is_correct" in result:
                matched_bench_values = _extract_matched_bench_values(result)
                return {
                    "is_correct": len(matched_bench_values) > 0,
                    "matched_bench_values": matched_bench_values,
                    "reason": result.get("reason", ""),
                    "status": "success",
                }
            if attempt < retry - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return {"is_correct": False, "matched_bench_values": [], "reason": "response_parse_failed", "status": "failed"}
        except Exception as exc:
            if attempt < retry - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return {"is_correct": False, "matched_bench_values": [], "reason": f"error: {str(exc)[:200]}", "status": "failed"}
    return {"is_correct": False, "matched_bench_values": [], "reason": "unexpected", "status": "failed"}


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def prepare_eval_tasks(input_path: str) -> list[dict]:
    tasks = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if row.get("status") != "success":
                continue
            item_id = row.get("item_id", "")
            cate1 = row.get("cate1_name", "")
            cate_name = row.get("cate_name", "")

            schema_str = row.get("cpv_schema", "")
            schema_names = {s.strip() for s in schema_str.split(",") if s.strip()} if schema_str else set()

            bench_by_prop: dict[str, list[str]] = {}
            for cpv in (row.get("benchmark_cpv_results") or []):
                pn = (cpv.get("property_name") or "").strip()
                pv = cpv.get("property_value")
                if pn and pv not in (None, "", []):
                    bench_by_prop.setdefault(pn, []).append(str(pv).strip())

            for cpv in (row.get("prediction_cpv_results") or []):
                pn = (cpv.get("property_name") or "").strip()
                pv = cpv.get("property_value")
                if not pn or pv in (None, "", []):
                    continue
                pv_str = str(pv).strip()
                bench_values = bench_by_prop.get(pn, [])
                tasks.append({
                    "item_id": item_id,
                    "cate1_name": cate1,
                    "cate_name": cate_name,
                    "property_name": pn,
                    "predicted_value": pv_str,
                    "bench_values": bench_values,
                    "in_schema": pn in schema_names,
                })
    return tasks


async def run_eval(
    tasks: list[dict],
    cache: dict[str, dict],
    output_path: str,
    cache_path: str,
    client: ModelClient,
    system_prompt: str,
    user_template: str,
    workers: int,
    request_workers: int,
    retry: int,
) -> dict[str, int]:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)

    semaphore = asyncio.Semaphore(workers)
    request_semaphore = asyncio.Semaphore(request_workers)
    write_lock = asyncio.Lock()
    cache_lock = asyncio.Lock()

    stats = {"rule": 0, "cache": 0, "llm": 0, "no_bench": 0, "no_schema": 0, "failed": 0}

    async def process_one(task: dict) -> None:
        pname = task["property_name"]
        pvalue = task["predicted_value"]
        bench_values = task["bench_values"]

        if not bench_values:
            if not task.get("in_schema", True):
                method, reason = "no_schema", "predicted property not in cpv_schema"
            else:
                method, reason = "no_bench", "no benchmark values for this property"
            result = {**task, "match_method": method, "is_correct": False, "matched_bench_values": [], "reason": reason}
            stats[method] += 1
            async with write_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            return

        async with semaphore:
            rule_results = rule_match_all(pvalue, bench_values)
            if rule_results:
                result = {**task, "match_method": "rule", "is_correct": True, "matched_bench_values": rule_results, "reason": "rule match"}
                stats["rule"] += 1
                async with write_lock:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                return

            cache_key = make_cache_key(pname, pvalue, bench_values)
            async with cache_lock:
                cached = cache.get(cache_key)
            if cached is not None:
                cached_matched = cached.get("matched_bench_values", [])
                if not isinstance(cached_matched, list):
                    cached_matched = [cached_matched] if cached_matched else []
                result = {
                    **task,
                    "match_method": "cache",
                    "is_correct": len(cached_matched) > 0,
                    "matched_bench_values": cached_matched,
                    "reason": cached.get("reason", ""),
                }
                stats["cache"] += 1
                async with write_lock:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                return

            llm_result = await call_judge(
                client, request_semaphore, system_prompt, user_template,
                pname, pvalue, bench_values, retry,
            )

            if llm_result["status"] == "success":
                cache_entry = {
                    "key": [pname, pvalue, sorted(bench_values)],
                    "is_correct": llm_result["is_correct"],
                    "matched_bench_values": llm_result["matched_bench_values"],
                    "reason": llm_result["reason"],
                }
                async with cache_lock:
                    cache[cache_key] = cache_entry
                async with write_lock:
                    with open(cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")

                result = {
                    **task,
                    "match_method": "llm",
                    "is_correct": llm_result["is_correct"],
                    "matched_bench_values": llm_result["matched_bench_values"],
                    "reason": llm_result["reason"],
                }
                stats["llm"] += 1
            else:
                result = {**task, "match_method": "failed", "is_correct": False, "matched_bench_values": [], "reason": llm_result["reason"]}
                stats["failed"] += 1

            async with write_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    aws = [asyncio.create_task(process_one(t)) for t in tasks]
    for fut in tqdm(asyncio.as_completed(aws), total=len(aws), desc="eval"):
        await fut

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute-level evaluation: semantic matching")
    parser.add_argument("--input", required=True, help="Path to extract_results.jsonl")
    parser.add_argument("--output", default=None, help="Eval results JSONL path")
    parser.add_argument("--cache-dir", default=None, help="Directory for eval cache")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--model", required=True, help="Model name for semantic judge")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum generated tokens (default: 8192 for OpenAI)")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--request-workers", type=int, default=60)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if args.output is None:
        args.output = os.path.join(results_dir, "eval_results.jsonl")

    cache_dir = args.cache_dir or os.path.dirname(os.path.abspath(args.output))
    cache_path = os.path.join(cache_dir, "eval_cache.jsonl")

    cache = load_cache(cache_path)
    print(f"[cache] {len(cache)} entries loaded")

    tasks = prepare_eval_tasks(args.input)
    print(f"[tasks] {len(tasks)} attribute evaluations")

    if not tasks:
        print("No tasks to evaluate.")
        return

    if args.no_resume and os.path.exists(args.output):
        os.remove(args.output)

    if not args.no_resume and os.path.exists(args.output):
        done_keys: set[str] = set()
        failed_keys: set[str] = set()
        kept_lines: list[str] = []
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = f"{obj['item_id']}|{obj['property_name']}|{obj['predicted_value']}"
                    if obj.get("match_method") == "failed":
                        failed_keys.add(key)
                    else:
                        done_keys.add(key)
                        kept_lines.append(line)
                except (json.JSONDecodeError, KeyError):
                    kept_lines.append(line)
                    continue

        if args.retry_failed:
            with open(args.output, "w", encoding="utf-8") as f:
                for l in kept_lines:
                    f.write(l + "\n")
            print(f"[resume] {len(done_keys)} done, {len(failed_keys)} failed to retry")
            tasks = [
                t for t in tasks
                if f"{t['item_id']}|{t['property_name']}|{t['predicted_value']}" not in done_keys
            ]
        else:
            done_keys.update(failed_keys)
            print(f"[resume] {len(done_keys)} already done, skipping")
            tasks = [
                t for t in tasks
                if f"{t['item_id']}|{t['property_name']}|{t['predicted_value']}" not in done_keys
            ]

        print(f"[pending] {len(tasks)} attribute evaluations")
        if not tasks:
            print("Nothing to do.")
            return

    system_prompt, user_template = _load_judge_prompts()
    client = create_client(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.api_base,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
    )

    stats = asyncio.run(run_eval(
        tasks=tasks,
        cache=cache,
        output_path=args.output,
        cache_path=cache_path,
        client=client,
        system_prompt=system_prompt,
        user_template=user_template,
        workers=args.workers,
        request_workers=args.request_workers,
        retry=args.retry,
    ))

    print(f"\nDone: {sum(stats.values())} attributes evaluated")
    for method, cnt in sorted(stats.items()):
        print(f"  {method}: {cnt}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
