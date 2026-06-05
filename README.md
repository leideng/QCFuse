# QCFuse

<p align="center">
  <a href="http://arxiv.org/abs/2606.05875"><img src="https://img.shields.io/badge/Paper-arXiv%3A2606.05875-b31b1b" alt="Paper"></a>
  <img src="https://img.shields.io/badge/SGLang-0.5.4-2563eb" alt="SGLang 0.5.4">
</p>

QCFuse is a **pipeline-constrained, query-aware KV cache fusion** system for
efficient long-context RAG generation. This repository contains the QCFuse
research release described in [arXiv:2606.05875](http://arxiv.org/abs/2606.05875).

## ✨ Highlights

- **Full-prefill-level quality.** QCFuse preserves the quality of full prefill.
- **Matched-quality speedup.** QCFuse achieves an average prefill-time speedup
  of **1.7x** over full prefill and **1.5x** over ProphetKV, the strongest
  quality-preserving baseline.
- **Fastest under strict quality control.** Under a **2% relative quality drop**
  criterion, QCFuse is the fastest method among all compared methods that
  satisfies the constraint, with a **2.1x** average speedup.

## 📊 Results

<p align="center">
  <img src="md/benchmark_aggregate.png" alt="Quality and TTFT trade-off on LongBench and RULER" width="900">
</p>

<p align="center">
  <em>Quality and TTFT trade-off on LongBench and RULER. Lower TTFT and higher quality are better.</em>
</p>

## 🧪 Datasets

The release runner expects each evaluation split as a local JSONL file named
`{dataset}.jsonl` under `--data_dir`.

| Benchmark | Official source | Tasks used in this release |
| --- | --- | --- |
| LongBench | [THUDM/LongBench](https://github.com/THUDM/LongBench) | `musique`, `2wikimqa`, `hotpotqa` |
| RULER | [NVIDIA/RULER](https://github.com/NVIDIA/RULER) | `ruler_mv` (`MV`), `ruler_mq` (`MQ`), `ruler_vt` (`VT`) |

## ⚙️ Installation

Install SGLang **0.5.4**:

```bash
git clone -b v0.5.4 https://github.com/sgl-project/sglang.git
cd sglang
pip install --upgrade pip
pip install -e "python"
```

Install the evaluation dependencies used by the Blend runner:

```bash
pip install rouge-score
```

Use a CUDA/PyTorch environment compatible with your GPU and SGLang 0.5.4. The
runner expects local model files and local JSONL datasets.

## 🚀 Running QCFuse

Run the SSD-backed QCFuse method:

```bash
python blend/sglang_blend_ssd.py \
  --model qwen3-8b \
  --model_dir /path/to/models \
  --data_dir /path/to/data \
  --dataset hotpotqa \
  --baseline ours \
  --size 200 \
  --cache_dir /path/to/cache
```

`--cache_dir` stores the SSD-backed chunk and query caches. With
`--baseline ours`, the runner performs offline cache preparation before the
online evaluation pass.

Run the full-prefill baseline:

```bash
python blend/sglang_blend_ssd.py \
  --model qwen3-8b \
  --model_dir /path/to/models \
  --data_dir /path/to/data \
  --dataset hotpotqa \
  --baseline fullcomp \
  --size 200 \
  --cache_dir /path/to/cache
```

Supported `--baseline` values are `ours` and `fullcomp`. Supported `--dataset`
values are `hotpotqa`, `2wikimqa`, `musique`, `ruler_mv`, `ruler_mq`, and
`ruler_vt`.

## 📚 Citation

If you find QCFuse useful, please cite:

```bibtex
@misc{yan2026qcfusequeryawarecachefusion,
      title={QCFuse: Query-Aware Cache Fusion via Compressed View for Efficient RAG Serving},
      author={Jianxin Yan and Wangze Ni and Zhenxin Li and Jiabao Jin and Zhitao Shen and Haoyang Li and Jia Zhu and Peng Cheng and Xuemin Lin and Lei Chen and Kui Ren},
      year={2026},
      eprint={2606.05875},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.05875},
}
```
