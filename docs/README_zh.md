# EndoCA 中文说明

EndoCA 用于评估内窥镜 VQA 模型的复杂答案是否与同图像的原子问题答案保持一致。仓库提供两个已经构建完成的 benchmark：

| 数据集 | 复杂样本 | 原子问题 | 总查询数 |
|---|---:|---:|---:|
| EndoCA-Core | 12,000 | 15,736 | 27,736 |
| EndoCA-Diagnostic | 6,000 | 9,300 | 15,300 |

EndoCA-Core 用于主要模型比较和 ASR 评测；EndoCA-Diagnostic 用于观察问题复杂度增加时的表现变化。两个数据集的样本 ID 不重叠。

## 安装与准备

```bash
conda create -n endoca python=3.11 -y
conda activate endoca
pip install -e .
python -m endoca.data.prepare
```

图像来自公开的 [Kvasir-VQA-x1](https://github.com/simula/Kvasir-VQA-x1)。下载后，推理命令中的 `--data-root` 应指向包含 `Kvasir-VQA-x1/` 的父目录。

## 使用 Benchmark

下面以 Qwen3-VL-8B 为例。不同模型的 Python、CUDA 与显存配置不同，仓库只提供调用接口和简洁配置示例。

```bash
python -m endoca.inference.open_vlm \
  --config configs/models.yaml \
  --model qwen3-vl-8b \
  --input data/manifests/endoca_core.jsonl \
  --output outputs/qwen3_core.jsonl \
  --data-root /path/to/datasets \
  --prompt-style answer_only \
  --max-new-tokens 80
```

自定义模型只需逐行读取 manifest，并在输出中保留原字段，再增加 `model_id`、`prediction` 和可选的 `error`。

```bash
python -m endoca.evaluation.score \
  --predictions outputs/qwen3_core.jsonl \
  --out-jsonl outputs/qwen3_core_scored.jsonl \
  --out-metrics outputs/qwen3_core_metrics.json \
  --out-report outputs/qwen3_core_report.md
```

评分器输出复杂答案准确率、原子答案准确率、联合准确率和复杂-原子不一致率。

## 使用 ASR

ASR 使用模型自己生成的原子答案作为上下文前提。ASR-Revise 生成协调后的复杂答案，ASR-Selective 在支持不稳定时选择 abstain。最短流程是：先完成 benchmark 推理和评分，再运行 `endoca.asr.build`、`endoca.asr.run` 和 `endoca.asr.score`。完整参数可通过对应命令的 `--help` 查看。

论文结果位于 [`results/paper/`](../results/paper/README.md)。代码采用 Apache-2.0；数据遵循 Kvasir-VQA-x1 的 CC BY-NC 4.0 条款。arXiv 编号公开后需要更新主 README、`CITATION.cff` 和 `pyproject.toml` 中的占位符。
