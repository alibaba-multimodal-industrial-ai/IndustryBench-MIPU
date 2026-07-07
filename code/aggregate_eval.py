"""
Aggregate evaluation results: compute Precision / Recall / F1.

Recall is computed from matched_bench_values (set-based deduplication).

Usage:
    python aggregate_eval.py \
        --input results/eval_results.jsonl \
        --bench ../data/multi_image_level.jsonl \
        --extract results/extract.jsonl

    python aggregate_eval.py \
        --input results/eval_results.jsonl \
        --bench ../data/multi_image_level.jsonl \
        --extract results/extract.jsonl \
        --by cate1_name
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any


def _get_matched_bench_values(row: dict) -> list[str]:
    """Read matched_bench_values (list), fallback to matched_bench_value (str)."""
    mbvs = row.get("matched_bench_values")
    if isinstance(mbvs, list):
        return [v for v in mbvs if v]
    mbv = row.get("matched_bench_value")
    if mbv:
        return [mbv]
    return []


def load_eval_results(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_bench(path: str) -> dict[str, dict[str, Any]]:
    bench: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            iid = record.get("record_id", "") or record.get("item_id", "")
            cpv_results = record.get("cpv_results", {})
            props: dict[str, list[str]] = {}
            if isinstance(cpv_results, dict):
                for pname, pv_list in cpv_results.items():
                    pname = pname.strip()
                    if not pname:
                        continue
                    if isinstance(pv_list, list):
                        vals = [str(v).strip() for v in pv_list if v not in (None, "", [])]
                    elif pv_list not in (None, "", []):
                        vals = [str(pv_list).strip()]
                    else:
                        continue
                    if vals:
                        props[pname] = vals
            elif isinstance(cpv_results, list):
                for item in cpv_results:
                    if not isinstance(item, dict):
                        continue
                    if item.get("dropped", False):
                        continue
                    pname = (item.get("property_name") or "").strip()
                    pvalue = item.get("property_value")
                    if not pname or pvalue in (None, "", []):
                        continue
                    props.setdefault(pname, []).append(str(pvalue).strip())
            bench[iid] = {
                "properties": props,
                "cate1_name": record.get("cate1_name", ""),
                "cate_name": record.get("cate_name", ""),
            }
    return bench


def compute_metrics(
    eval_results: list[dict],
    bench: dict[str, dict[str, Any]],
    group_by: str | None = None,
    bench_scope_ids: set[str] | None = None,
) -> dict[str, dict]:
    groups: dict[str, dict] = defaultdict(lambda: {
        "correct": 0,
        "total_pred": 0,
        "matched_bench_values": defaultdict(set),
        "total_bench": 0,
    })

    for row in eval_results:
        iid = row.get("item_id", "")
        pname = row.get("property_name", "")

        if group_by:
            group_key = row.get(group_by, "unknown")
        else:
            group_key = "overall"

        g = groups[group_key]
        g["total_pred"] += 1
        if row.get("is_correct"):
            g["correct"] += 1
            for mbv in _get_matched_bench_values(row):
                g["matched_bench_values"][(iid, pname)].add(mbv)

    if bench_scope_ids is not None:
        scope_iids = bench_scope_ids
    else:
        scope_iids = {row["item_id"] for row in eval_results}
    for iid in scope_iids:
        bdata = bench.get(iid)
        if not bdata:
            continue
        if group_by == "cate1_name":
            group_key = bdata.get("cate1_name", "unknown")
        elif group_by == "cate_name":
            group_key = bdata.get("cate_name", "unknown")
        elif group_by == "property_name":
            for pname, values in bdata["properties"].items():
                g = groups[pname]
                g["total_bench"] = g.get("total_bench", 0) + len(values)
            continue
        else:
            group_key = "overall"

        if group_by != "property_name":
            total_attrs = sum(len(v) for v in bdata["properties"].values())
            groups[group_key]["total_bench"] = groups[group_key].get("total_bench", 0) + total_attrs

    for group_key, g in groups.items():
        total_recalled = 0
        for (iid, pname), matched_set in g["matched_bench_values"].items():
            bdata = bench.get(iid)
            if bdata and pname in bdata["properties"]:
                bench_vals = set(bdata["properties"][pname])
                total_recalled += len(matched_set & bench_vals)
            else:
                total_recalled += len(matched_set)
        g["recalled"] = total_recalled

    results: dict[str, dict] = {}
    for group_key, g in sorted(groups.items()):
        precision = g["correct"] / g["total_pred"] if g["total_pred"] > 0 else 0
        recall = g["recalled"] / g["total_bench"] if g["total_bench"] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results[group_key] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "correct": g["correct"],
            "total_pred": g["total_pred"],
            "recalled": g["recalled"],
            "total_bench": g["total_bench"],
        }

    return results


def compute_method_stats(eval_results: list[dict]) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    for row in eval_results:
        method = row.get("match_method", "unknown")
        stats[method] += 1
    return dict(stats)


def compute_error_breakdown(eval_results: list[dict]) -> dict[str, int]:
    errors: dict[str, int] = defaultdict(int)
    for row in eval_results:
        if row.get("is_correct"):
            continue
        method = row.get("match_method", "")
        if method in ("no_bench", "no_schema"):
            errors["extra_property"] += 1
        elif method == "cross_name_llm":
            errors["name_mismatch"] += 1
        else:
            errors["value_mismatch"] += 1
    return dict(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate P/R/F1 from evaluation results")
    parser.add_argument("--input", required=True, help="Path to eval_results.jsonl")
    parser.add_argument("--bench", required=True, help="Path to benchmark JSONL")
    parser.add_argument("--extract", default=None,
                        help="Path to extract_results.jsonl; uses success items as recall denominator scope")
    parser.add_argument("--by", choices=["cate1_name", "cate_name", "property_name"], default=None)
    parser.add_argument("--output", default=None, help="Output JSON report path")
    args = parser.parse_args()

    eval_results = load_eval_results(args.input)
    bench = load_bench(args.bench)

    bench_scope_ids: set[str] | None = None
    if args.extract:
        bench_scope_ids = set()
        with open(args.extract, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                status = rec.get("status")
                if status == "success" or status is None:
                    iid = rec.get("record_id", "") or rec.get("item_id", "")
                    if iid:
                        bench_scope_ids.add(iid)
        print(f"[extract] {len(bench_scope_ids)} success records (bench scope)")

    print(f"[eval]  {len(eval_results)} attribute results")
    print(f"[bench] {len(bench)} items")

    method_stats = compute_method_stats(eval_results)
    print(f"\n=== Match Method Distribution ===")
    for method, count in sorted(method_stats.items(), key=lambda x: -x[1]):
        pct = count / len(eval_results) * 100 if eval_results else 0
        print(f"  {method:12s}: {count:6d} ({pct:.1f}%)")

    errors = compute_error_breakdown(eval_results)
    print(f"\n=== Error Breakdown ===")
    for etype, count in sorted(errors.items(), key=lambda x: -x[1]):
        print(f"  {etype:20s}: {count}")

    metrics = compute_metrics(eval_results, bench, group_by=args.by, bench_scope_ids=bench_scope_ids)

    if args.by is None:
        m = metrics.get("overall", {})
        print(f"\n=== Overall Metrics ===")
        print(f"  Precision: {m.get('precision', 0):.4f} ({m.get('correct', 0)}/{m.get('total_pred', 0)})")
        print(f"  Recall:    {m.get('recall', 0):.4f} ({m.get('recalled', 0)}/{m.get('total_bench', 0)})")
        print(f"  F1:        {m.get('f1', 0):.4f}")
    else:
        print(f"\n=== Metrics by {args.by} ===")
        sorted_groups = sorted(metrics.items(), key=lambda x: x[1]["f1"])
        for group_key, m in sorted_groups:
            print(f"  {group_key:30s}  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  "
                  f"(pred={m['total_pred']}, bench={m['total_bench']})")

    if args.output:
        report = {
            "method_stats": method_stats,
            "error_breakdown": errors,
            "metrics": metrics,
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
