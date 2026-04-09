# AI 检测器脚手架

此仓库是一个面向远程优先的脚手架，用于在 AutoDL 上训练和调试中文学术论文 AI 重写检测器。

## 目录结构

- `src/ai_detector/`: 可重用的 Python 模块
- `prepare_data.py`: 配对数据准备入口点
- `train.py`: 模型训练入口点
- `infer.py`: 文档推理入口点
- `configs/`: 基础配置和冒烟测试的 YAML 配置文件
- `scripts/`: AutoDL 设置和 tmux 友好的运行脚本

## 远程工作流程

1. 使用 VS Code Remote-SSH 打开 `/root/projects/ai-detector`
2. 将数据、检查点和运行日志保存在 `$PERSIST_ROOT/ai-detector` 下
3. 在 `tmux` 内运行长时间任务
4. 在进行大型实验前提交代码变更

## 快速开始

```bash
git clone <your-private-repo> /root/projects/ai-detector
cd /root/projects/ai-detector
export PERSIST_ROOT=/root/autodl-fs
bash scripts/bootstrap_autodl.sh
```

准备配对数据：

```bash
source .venv/bin/activate
python prepare_data.py \
  --input-dir "$PERSIST_ROOT/ai-detector/data/raw" \
  --output-dir "$PERSIST_ROOT/ai-detector/data/processed"
```

运行冒烟训练任务：

```bash
source .venv/bin/activate
python train.py --config configs/train_smoke.yaml --output-root "$PERSIST_ROOT/ai-detector"
```

训练成功后自动关机：

```bash
source .venv/bin/activate
export AUTO_SHUTDOWN_ON_SUCCESS=1
export SHUTDOWN_DELAY_MINUTES=1
bash scripts/run_train.sh
```

生成训练过程可视化报告：

```bash
source .venv/bin/activate
python scripts/generate_training_report.py \
  --run-dir "$PERSIST_ROOT/ai-detector/runs/<timestamp>_base"
```

运行推理：

```bash
source .venv/bin/activate
python infer.py \
  --checkpoint "$PERSIST_ROOT/ai-detector/checkpoints/<exp>" \
  --input-file "$PERSIST_ROOT/ai-detector/data/processed/documents.jsonl" \
  --output-file "$PERSIST_ROOT/ai-detector/runs/infer/predictions.jsonl"
```

## 数据假设

- 原始根目录下包含 `json/` 下的原始 JSON 文件
- 重写的 JSON 文件位于 `rewrite/outputs/per_paper/` 下
- 通过文档主干进行配对
- 仅使用 `fulltext[*].paragraphs` 作为模型输入

## 调试规则

- 在 `prepare_data.py` 中修复数据问题，而不是在训练过程中
- 将每次运行保存在 `runs/<timestamp>_<exp_name>/` 下
- 将检查点保存在 `checkpoints/<timestamp>_<exp_name>/` 下
- 在 `stderr.log` 中保留回溯信息

## 实用脚本

- `scripts/bootstrap_autodl.sh`
- `scripts/run_prepare.sh`
- `scripts/run_smoke.sh`
- `scripts/run_train.sh`
- `scripts/generate_training_report.py`
- `scripts/run_infer.sh`
- `scripts/tmux_train.sh`
