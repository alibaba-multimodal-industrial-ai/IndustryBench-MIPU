# IndustryBench-MIPU

[![arXiv](https://img.shields.io/badge/arXiv-2606.14383-b31b1b.svg)](https://arxiv.org/abs/2606.14383)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E.svg)](https://huggingface.co/datasets/alibaba-multimodal-industrial-ai/IndustryBench-MIPU)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)

**Multi-Image Industrial Product Understanding Benchmark** — evaluating MLLMs on structured attribute extraction from real-world industrial product images.

<p align="center">
  <img src="figs/intro.png" width="100%">
</p>

Industrial product specifications are scattered across multiple heterogeneous images — specification tables, nameplates, technical drawings. **IndustryBench-MIPU** tests whether MLLMs can reliably recover them through four challenges: text recognition, visual reasoning, domain knowledge, and cross-image evidence integration.

## Key Features

- **4,481 products** across 18 top-level categories (2,301 leaf categories) with 95K+ product-level annotations
- **Two evaluation granularities**: single-image and multi-image (cross-image reasoning)
- **Plug-and-play evaluation code**: OpenAI-compatible & Anthropic APIs supported out of the box
- **Cascaded judge**: rule-based normalization + LLM semantic matching for robust evaluation
- **Multi-model consensus construction**: 5 frontier MLLMs annotate independently, union-deduplicated, triple QA

---

## Main Results

<p align="center">
  <img src="figs/main_results.png" width="90%">
</p>

The dominant pattern: **high precision (86–94%) but low recall** — the best model recovers only half the product-level attributes. Failures concentrate in dense specification tables, multi-value properties, and domain-specific terminology.

---

## Dataset Overview

<div align="center">

| Products | Images | Top-level Categories | Leaf Categories | Property Names | Product-level Annotations |
|:--------:|:------:|:--------------------:|:---------------:|:--------------:|:-------------------------:|
| 4,481 | 26,310 | 18 | 2,301 | 4,554 | 95,024 |

</div>

Two evaluation settings:

| Setting | File | Granularity | Description |
|---------|------|-------------|-------------|
| Single-image | `single_image_level.jsonl` | Per image | Extract attributes visible in one image (6,000-image stratified subset, 44,111 property-value pairs) |
| Multi-image | `multi_image_level.jsonl` | Per product | Extract all attributes from a product's full image set (4,481 products) |

---

## Task Definition

**Input**: Product images + product-specific attribute schema (list of valid property names)

**Output**: Structured property-value pairs extracted from visual evidence

<p align="center">
  <img src="figs/case_study.png" width="85%">
</p>

> **Case Study**: A microscope objective with 7 images and 69 benchmark attributes. The top model achieves 100% precision but only 45% recall — failures concentrate in dense specification tables where the model stops enumerating after 4–5 values.

---

## Quick Start

### 1. Setup

```bash
git lfs install
git clone https://huggingface.co/datasets/alibaba-multimodal-industrial-ai/IndustryBench-MIPU data

pip install -r code/requirements.txt
```

### 2. Configure API credentials

```bash
export API_KEY="your-api-key"
export API_BASE_URL="https://your-api-endpoint"
```

Or pass `--api-key` and `--api-base` directly to each script.

### 3. Run evaluation

The pipeline has three steps: **Extract → Eval → Aggregate**.

<details>
<summary><b>Multi-image evaluation</b> (product-level, recommended)</summary>

```bash
cd code

# Extract: send all images of each product to the model
python run_multi_extract.py \
    --input ../data/multi_image_level.jsonl \
    --output results/extract.jsonl \
    --provider openai --model qwen-plus \
    --workers 10 --request-workers 30 --retry 3 --shuffle

# Eval: semantic matching via LLM judge
python run_eval.py \
    --input results/extract.jsonl \
    --output results/eval.jsonl \
    --provider openai --model qwen-plus \
    --workers 20 --request-workers 60 --retry 3

# Aggregate: compute P / R / F1
python aggregate_eval.py \
    --input results/eval.jsonl \
    --bench ../data/multi_image_level.jsonl \
    --extract results/extract.jsonl
```

</details>

<details>
<summary><b>Single-image evaluation</b> (image-level)</summary>

```bash
cd code

# Extract: one image per request
python run_single_extract.py \
    --input ../data/single_image_level.jsonl \
    --output results/single_extract.jsonl \
    --provider openai --model qwen-plus \
    --workers 10 --request-workers 30 --retry 3 --shuffle

# Eval
python run_eval.py \
    --input results/single_extract.jsonl \
    --output results/single_eval.jsonl \
    --provider openai --model qwen-plus \
    --workers 20 --request-workers 60 --retry 3

# Aggregate
python aggregate_eval.py \
    --input results/single_eval.jsonl \
    --bench ../data/single_image_level.jsonl \
    --extract results/single_extract.jsonl
```

</details>

<details>
<summary><b>Category-level breakdown</b></summary>

```bash
python aggregate_eval.py \
    --input results/eval.jsonl \
    --bench ../data/multi_image_level.jsonl \
    --extract results/extract.jsonl \
    --by cate1_name
```

Options: `--by cate1_name` (top-level category), `--by cate_name` (leaf category), `--by property_name`.

</details>

---

## Data Format

### Multi-Image Level (`multi_image_level.jsonl`)

```json
{
  "item_id": "560324848370",
  "title": "日亚铝基板 NICHIA ...",
  "cate1_name": "电子元器件",
  "cate_name": "铝基板",
  "main_entity": "NICHIA铝基板",
  "cpv_schema": ["颜色", "品牌", "型号", "..."],
  "image_count": 7,
  "images": [{"record_id": "560324848370#main", "image_source": "main_image", "image_index": 0, "image_path": "images/560324848370_main.jpg"}],
  "cpv_results": {"颜色": ["白色"], "品牌": ["NICHIA"], "...": ["..."]}
}
```

### Single-Image Level (`single_image_level.jsonl`)

```json
{
  "record_id": "589158697373#detail_3",
  "image_path": "images/589158697373_detail_3.jpg",
  "cpv_schema": ["含量", "纯度", "..."],
  "cpv_results": {"含量": ["98%"]}
}
```

Both use `cpv_schema` as a list of valid property names and `cpv_results` as a dict mapping `property_name` → list of property values.

---

## Evaluation Pipeline

- **Extract**: Send product images + metadata to an MLLM, obtain a `property_name → [values]` mapping
- **Eval**: Cascaded matching — schema check (predicted property not in schema → incorrect) → same-name rule match → same-name cache → rule-based cross-name match (exact / subsequence / equivalence groups / spec-model wildcard with a non-parameter blacklist) → same-name LLM semantic judge
- **Aggregate**: Precision = correct / predicted, Recall = matched / benchmark, F1 = harmonic mean

### Supported Providers

| Provider | `--provider` | Example `--model` | Notes |
|----------|-------------|-------------------|-------|
| OpenAI-compatible | `openai` | `qwen-plus`, `gemini-2.5-pro`, `gpt-4o` | Works with DashScope, Vertex, vLLM, etc. |
| Anthropic | `anthropic` | `claude-sonnet-4-20250514` | Supports `--enable-thinking` for extended reasoning |

---

## Construction Pipeline

The benchmark is built through a semi-automated pipeline:

1. **Stratified sampling** across 18 industrial categories
2. **Multi-model annotation** — 5 frontier MLLMs independently annotate via entity recognition → image filtering → per-image extraction
3. **Union & deduplication** — semantic merging across models
4. **Three-tier QA** — frontier model audit (23.9% filtered), gold-standard cross-check, human verification (96.7% pass rate)

<p align="center">
  <img src="figs/pipeline.png" width="100%">
</p>

---

## Project Structure

```
.
├── code/
│   ├── run_multi_extract.py      # Multi-image extraction (product-level)
│   ├── run_single_extract.py     # Single-image extraction (image-level)
│   ├── run_eval.py               # LLM-based semantic evaluation
│   ├── aggregate_eval.py         # Compute P / R / F1 metrics
│   ├── model_client.py           # Unified API client (OpenAI / Anthropic)
│   ├── utils.py                  # JSON parsing utilities
│   ├── prompts/                  # Prompt templates
│   │   ├── extraction_prompt_v3_multi.txt   # Multi-image extraction
│   │   ├── extraction_prompt_v3.txt         # Single-image extraction
│   │   ├── judge_system_prompt.txt          # Value semantic judge (system)
│   │   └── judge_user_prompt.txt            # Value semantic judge (user)
│   └── requirements.txt
├── data/                         # Dataset (download from HuggingFace)
│   ├── multi_image_level.jsonl
│   ├── single_image_level.jsonl
│   └── images/
└── figs/                         # Figures for README
```

---

## Citation

```bibtex
@article{industrybench-mipu,
  title={IndustryBench-MIPU: Benchmarking Multi-Image Attribute Value Extraction for Industrial Products},
  author={Multimodal and Industrial AI Team, Alibaba},
  journal={arXiv preprint arXiv:2606.14383},
  year={2026},
  url={https://arxiv.org/abs/2606.14383}
}
```

## License

This project is licensed under the MIT License — see [LICENSE.txt](LICENSE.txt) for details.
