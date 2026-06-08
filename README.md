# QCFuse

<p align="center">
  <a href="http://arxiv.org/abs/2606.05875"><img src="https://img.shields.io/badge/Paper-arXiv%3A2606.05875-b31b1b" alt="Paper"></a>
</p>

QCFuse is a **pipeline-constrained, query-aware KV cache fusion** system for
efficient long-context RAG generation. This repository contains the QCFuse
research artifact described in [arXiv:2606.05875](http://arxiv.org/abs/2606.05875).

## ✨ Highlights

<p align="center">
  <img src="md/3sys_framework_01.png" alt="QCFuse framework overview" width="900">
</p>

<p align="center">
  <em>QCFuse builds a compact query-aware view for pipelined cache fusion in RAG serving.</em>
</p>

- **Query-aware compressed view.** QCFuse shortens selector analysis time,
  reduces selection-signal noise, and better balances query awareness with
  pipeline execution efficiency.
- **Matched-quality speedup.** Under matched-quality comparisons, QCFuse achieves an average prefill-time speedup
  of **1.7x** over full prefill and **1.5x** over ProphetKV, the strongest
  quality-preserving baseline.
- **Fastest under strict quality control.** Under a **1% relative quality drop**
  criterion, QCFuse is the only compared method that satisfies the constraint,
  with a **1.9x** average TTFT speedup over full prefill.

## 📊 Results

<p align="center">
  <img src="md/benchmark_aggregate.png" alt="Quality and TTFT trade-off on LongBench and RULER" width="900">
</p>

<p align="center">
  <em>Quality and TTFT trade-off on LongBench and RULER. </em>
</p>

## 🗂️ Repository Layout

```text
QCFuse/
├── blend/                         # QCFuse evaluation runner and configs
│   ├── sglang_blend_ssd.py         
│   ├── blend_common.py             
│   ├── qcfuse_config.py            
│   └── utils.py      
├── srt/                           # SGLang runtime changes for QCFuse
│   ├── entrypoints/                
│   ├── managers/                   
│   ├── layers/attention/           
│   ├── models/                     
│   └── utils/                      
├── data/                          # Dataset preprocessing               
│   ├── build_longbench_data.py      
│   └── build_ruler_data.py          
```

## 🗄️ Datasets

The evaluation runner expects each evaluation split as a local JSONL file named
`{dataset}.jsonl` under `--data_dir`.

Use the scripts in `data/` to build the evaluation splits reproducibly. See
[data/README.md](data/README.md) for preprocessing details.

| Benchmark | Official source | Tasks used in this artifact |
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

Return to the QCFuse repository root before running the commands below.

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
  --model_dir models \
  --data_dir data/final_data \
  --dataset hotpotqa \
  --baseline ours \
  --size 200 \
  --cache_dir cache/qcfuse
```

`--cache_dir` stores the SSD-backed chunk and query caches. With
`--baseline ours`, the runner performs offline cache preparation before the
online evaluation pass.

Run the full-prefill baseline:

```bash
python blend/sglang_blend_ssd.py \
  --model qwen3-8b \
  --model_dir models \
  --data_dir data/final_data \
  --dataset hotpotqa \
  --baseline fullcomp \
  --size 200 \
  --cache_dir cache/qcfuse
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
