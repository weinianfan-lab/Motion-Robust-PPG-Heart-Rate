# PPG + ACC 运动场景心率估计：求职工程作品

本项目面向算法工程师求职和技术面试，展示 Python/SciPy 信号处理、Welch/NLMS、PyTorch TCN、多传感器融合、数据隔离评估及 ONNX 推理。是独立工程实现，不以严格复现原作者指标或发表论文为目标。

当前完成范围：**Complete fixed-config LOSO: 15 held-out subjects, 64,697 / 64,697 official-label windows**。原始 15 人、64,697 个官方 HR 标签保留。作者生成器/软件版本、RR 历史实现、重复峰来源与独立重建标为 Deferred，不再阻止开发；不联系作者。时间映射、通道/标签计数、文件哈希和受试者隔离仍强制验证。

## 已实现

- 8 秒窗、2 秒步长；PPG64Hz、ACC32Hz，工程时间映射独立版本化，旧严格审计不改成 PASS。
- Welch、ACC 频谱辅助、连续多参考 NLMS+Welch，所有方法使用相同窗口和原标签。
- PPG-only TCN（42,737 参数）和 PPG+ACC 双分支 TCN（89,697 参数）。
- 12训练/2验证/1测试的受试者级划分，验证集选 checkpoint；固定seed42、12epochs，无测试调参。
- MAE、RMSE、偏差、尾部误差、活动和受试者统计，完整分母及 coverage。
- 两种首折模型的 ONNX FP32 导出、动态batch1/7与CPU数值一致性；无数据可运行原合成demo。

## 实测结果

| Method | Subject-macro MAE | Window-micro MAE | Window-micro RMSE | Coverage |
|---|---:|---:|---:|---:|
| Train mean | 18.770 | 18.390 | 23.713 | 100% |
| Welch | 18.562 | 18.383 | 30.378 | 100% |
| ACC-assisted spectrum | 18.515 | 18.337 | 30.347 | 100% |
| NLMS + Welch | 24.479 | 24.421 | 40.012 | 100% |
| PPG-only TCN | 9.419 | 9.346 | 13.696 | 100% |
| PPG+ACC TCN | 8.413 | 8.316 | 13.605 | 100% |

全部已完成测试窗纳入，不按异常、活动或难度删除。融合模型容量更大，不能把改善全部归因于 ACC；NLMS 较差结果如实保留。完整指标和限制见[实验报告](docs/showcase_results.md)，英文项目入口与命令见[README](README.md)。

![实测模型误差比较](docs/assets/comparison.png)

## 使用及证据边界

项目用 uv 独立环境；`uv sync --locked --extra ml --default-index https://pypi.org/simple` 安装学习与导出依赖。工程数据准备使用 `scripts/prepare_showcase.py`，单折/LOSO 使用 `scripts/run_showcase.py`，汇总使用 `scripts/summarize_showcase.py`，ONNX 模型推理使用 `scripts/infer_onnx.py`。完整命令及输入预处理边界见英文 README。

旧严格 M1 历史 BLOCKED 保留，新工程数据通道 READY_ENGINEERING。官方标签不替换；无原始采集时间戳，不能保证排除所有同步误差。离线零相位滤波和对称卷积不是实时因果方案。公开数据已用于工程检查，结果不是临床或外部验证。

原始信号、缓存、逐窗预测、checkpoint/ONNX 权重、完整日志及个人信息留在本地，不纳入公开候选。现有审计、测试、失败和作者邮件草稿均保留；不再发送或继续作者核询。原创代码采用 MIT 许可证，数据集与第三方依赖保留各自许可，详见 [LICENSE](LICENSE) 与 [第三方许可说明](THIRD_PARTY_NOTICES.md)。公开发布仅使用审核后的白名单。
