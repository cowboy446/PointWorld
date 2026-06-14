# data-agent

职责：
- 使用 `data` 分支的脚本从 Hugging Face 恢复数据包（支持只恢复少量 shards 用于 smoke 测试）。
- 运行数据完整性检查并将 H5 转换为 WDS 子集。
- 产出最终 WDS 子集目录供 `main` 分支消费。

步骤（自动化/交互）：
1. 切换到 `data` 分支：`git checkout data`。
2. 初始化子模块（若需要）：`git submodule update --init --recursive`。
3. 下载/放置少量数据包 parts 到本地目录（例如：`~/downloads/PointWorld-DROID-subset`）。
4. 运行恢复脚本（只恢复所需 parts）：`./recover_dataset_from_parts.sh <parts_dir> <restored_root>`。
5. 运行完整性检查：`python data_integrity_check.py --input <restored_root>`。
6. 运行转换为 WDS（只转换子集）：`python convert_wds.py --input <restored_root> --output <wds_output>`。
7. 确认输出路径 `/path/to/droid/wds` 下存在少量 shards。

注意：如果无法从 Hugging Face 下载全部文件，可只下载并恢复少量 parts 以节省磁盘空间（适用于 smoke tests）。