# test-agent

职责：
- 下载预训练 checkpoints 到 `pretrained_checkpoints/`。
- 在 `main` 分支下运行快速 evaluation（少量 batch）以验证 checkpoint 加载与推理流程。
- 收集日志与可视化输出供分析。

步骤：
1. 切回 `main`：`git checkout main`。
2. 下载 checkpoint（使用 `huggingface-cli` 或 `huggingface_hub` API）并放置在 `pretrained_checkpoints/`。
3. 运行快速 eval：
   ```bash
   MODEL_PATH=pretrained_checkpoints/small-droid/model-best.pt
   python eval.py --model_path "${MODEL_PATH}" --domains=droid --data_dirs=/path/to/droid/wds --batch_size=1 --eval_num_batches=10
   ```
4. 检查日志和输出目录，若需要开启可视化，运行 `--eval_viz_num` 并访问 `http://localhost:8080`。