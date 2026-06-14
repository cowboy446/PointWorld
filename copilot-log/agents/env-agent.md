# env-agent

职责：
- 在本地创建并配置 Conda 环境 `pointwm`（或按需改名）。
- 安装基础 Python 包：`huggingface_hub`、`timm`、`flash-attn`、`networkx`。
- 验证 `torch`/CUDA 是否安装并与系统驱动兼容（如果无法编译可切换到 CPU smoke 测试）。

步骤（自动化）：
1. 检查 `conda` 可用性：`conda --version`。
2. 创建环境：`conda env create -n pointwm -f environments/train_eval.yml`。
3. 激活环境并安装额外 pip 包：
   - `python -m pip install huggingface_hub==0.26.2`
   - `python -m pip install timm==1.0.19 --no-deps`
   - `python -m pip install flash-attn==2.7.4.post1 --no-build-isolation`
   - `python -m pip install networkx==3.4.2 --no-deps`
4. 检查 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`。

当遇到安装/兼容问题时，记录错误输出并回报给 test-agent 以决定下一步处理。