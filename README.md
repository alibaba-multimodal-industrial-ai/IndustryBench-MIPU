# IndustryBench-MIPU

[![arXiv](https://img.shields.io/badge/arXiv-2606.14383-b31b1b.svg)](https://arxiv.org/abs/2606.14383)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E.svg)](https://huggingface.co/datasets/alibaba-multimodal-industrial-ai/IndustryBench-MIPU)

**Multi-Image Industrial Product Understanding Benchmark** — evaluating MLLMs on structured attribute extraction from industrial product images.

<p align="center">
  <img src="figs/intro.png" width="95%">
</p>

<p align="center">
  <img src="figs/pipeline.png" width="95%">
</p>

## Highlights

- **Large-scale**: 4,559 products, 27,652 images, 103,703 annotations across 18 industrial categories
- **Multi-image**: Product-level evaluation requiring cross-image evidence integration
- **Multi-model consensus**: Benchmark built from 5 MLLMs with three-tier quality assurance
- **Key finding**: Models achieve 86–94% precision but only 49.9% best recall — completeness, not accuracy, is the bottleneck

## Dataset Overview

| Statistic | Value |
|-----------|-------|
| Products | 4,559 |
| Valid images | 27,652 |
| Top-level categories | 18 |
| Unique property names | 3,564 |
| Image-level annotations | 182,527 |
| Product-level annotations | 103,703 |

Two evaluation granularities:
- **Image-level** (`single_image_level.jsonl`) — extract attributes from a single image
- **Product-level** (`multi_image_level.jsonl`) — extract attributes from all images of a product

## Quick Start

### 1. Download Data

```bash
# From HuggingFace
git lfs install
git clone https://huggingface.co/datasets/alibaba-multimodal-industrial-ai/IndustryBench-MIPU

# Or download directly
# data/multi_image_level.jsonl  (item-level benchmark)
# data/single_image_level.jsonl        (image-level benchmark)
# data/images/                       (27,652 product images)
```

### 2. Install Dependencies

```bash
pip install -r code/requirements.txt
```

### 3. Configure API

```bash
export API_KEY="your-api-key"
export API_BASE_URL="https://your-api-endpoint"  # OpenAI-compatible endpoint
```

### 4. Run Evaluation (3 steps)

```bash
cd code

# Step 1: Extract — send product images to MLLM
python run_multi_extract.py \
    --input ../data/multi_image_level.jsonl \
    --output results/extract.jsonl \
    --provider openai --model qwen-plus \
    --api-key $API_KEY --api-base $API_BASE_URL \
    --workers 10 --request-workers 30 --retry 3 --shuffle

# Step 2: Eval — semantic matching against benchmark
python run_eval.py \
    --input results/extract.jsonl \
    --output results/eval.jsonl \
    --provider openai --model qwen-plus \
    --api-key $API_KEY --api-base $API_BASE_URL \
    --workers 20 --request-workers 60 --retry 3

# Step 3: Aggregate — compute P/R/F1
python aggregate_eval.py \
    --input results/eval.jsonl \
    --bench ../data/multi_image_level.jsonl \
    --extract results/extract.jsonl
```

## Data Format

### Multi-Image Level (`multi_image_level.jsonl`)

```json
{
  "item_id": "560324848370",
  "title": "日亚铝基板 NICHIA NSSW157AT ...",
  "cate1_name": "电子元器件",
  "cate_name": "覆铜板材料",
  "main_entity": "NICHIA铝基板",
  "cpv_schema": "颜色,品牌,型号,...",
  "image_count": 6,
  "images": [
    {"record_id": "560324848370#main", "image_source": "main_image", "image_index": 0, "image_path": "images/560324848370_main_0.jpg"},
    {"record_id": "560324848370#detail_0", "image_source": "detail_image", "image_index": 0, "image_path": "images/560324848370_detail_0.jpg"}
  ],
  "cpv_results": [
    {"property_name": "颜色", "property_value": "白色"},
    {"property_name": "品牌", "property_value": "NICHIA"}
  ]
}
```

### Single-Image Level (`single_image_level.jsonl`)

```json
{
  "record_id": "589158697373#detail_3",
  "item_id": "589158697373",
  "title": "...",
  "cate1_name": "化工",
  "cate_name": "亚硫酸盐",
  "main_entity": "...",
  "cpv_schema": "...",
  "image_source": "detail_image",
  "image_index": 3,
  "image_path": "images/589158697373_detail_3.jpg",
  "cpv_results": [
    {"property_name": "含量", "property_value": "98%"}
  ]
}
```

## Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  Extract          Eval              Aggregate            │
│  (MLLM) ──────► (Rule + LLM  ──────► (P / R / F1)      │
│                   Judge)                                 │
└─────────────────────────────────────────────────────────┘
```

**Extract**: Send all product images + metadata to an MLLM, get structured `{property_name, property_value}` pairs.

**Eval**: Match each prediction against benchmark. Cascaded strategy:
1. Rule-based: Unicode normalization, subsequence matching, numeric canonicalization
2. LLM judge: Semantic equivalence for ambiguous cases

**Aggregate**: Compute precision (correct / predicted), recall (matched / benchmark), F1.

### Supported Providers

| Provider | `--provider` | Notes |
|----------|-------------|-------|
| OpenAI-compatible | `openai` | Works with Qwen, Gemini, vLLM, etc. |
| Anthropic | `anthropic` | Claude models, supports `--enable-thinking` |

### Key Options

| Option | Description |
|--------|-------------|
| `--workers` | Concurrent items |
| `--request-workers` | Concurrent HTTP requests |
| `--retry` | Max retries per item |
| `--max-images` | Skip items with too many images (default: 60) |
| `--shuffle` | Randomize processing order |
| `--retry-failed` | Re-run previously failed items |
| `--enable-thinking` | Enable reasoning mode (Anthropic) |

## Main Results

| Model | Rank | Multi-P | Multi-R | Multi-F1 | Single-P | Single-R | Single-F1 |
|-------|------|---------|---------|----------|----------|----------|-----------|
| Gemini 3.1 Pro | 1 | **93.8** | **49.9** | **65.1** | **94.0** | 65.4 | 77.1 |
| Qwen 3.5-397B-A17B | 2 | 88.2 | 48.6 | 62.7 | 80.6 | 72.0 | 76.0 |
| GPT-5.4 | 3 | 86.3 | 46.6 | 60.5 | 82.7 | 55.3 | 66.2 |
| Qwen 3.5 Plus | 4 | 88.1 | 45.4 | 59.9 | 82.9 | **79.7** | **81.3** |
| Claude Opus 4.6 | 5 | 88.2 | 42.3 | 57.2 | 85.1 | 58.6 | 69.4 |
| Kimi-K2.5-1T-A32B | 6 | 88.6 | 41.7 | 56.7 | 79.0 | 64.3 | 70.9 |
| Qwen 3.5-27B | 7 | 88.0 | 40.8 | 55.8 | 78.8 | 65.4 | 71.5 |
| Qwen 3.5-122B-A10B | 8 | 88.8 | 34.9 | 50.1 | 77.6 | 64.6 | 70.5 |
| Qwen 3.5-35B-A3B | 9 | 86.0 | 11.7 | 20.6 | 75.1 | 63.3 | 68.7 |

Rank by multi-image F1. The dominant pattern: high precision (86–94%) but low recall — the best model recovers only half the product-level attributes.

## Citation

```bibtex
@article{industrybench-mipu,
  title={IndustryBench-MIPU: Benchmarking Multi-Image Attribute Value Extraction for Industrial Products},
  author={Multimodal and Industrial AI Team},
  journal={arXiv preprint arXiv:2606.14383},
  year={2026}
}
```
