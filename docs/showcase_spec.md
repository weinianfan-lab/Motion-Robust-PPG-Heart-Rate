# 求职展示工程合同：showcase_v1

日期：2026-10-25。按用户最新任务生效，覆盖旧合同冲突项。

项目：Motion-Robust-PPG-Heart-Rate。目标是展示 Python/SciPy 信号处理、Welch/NLMS、PyTorch 单模态与双分支 TCN、可追溯数据处理/评价和 ONNX CPU 推理。它是独立工程实现，不声称精确复现 Deep PPG、KID-PPG 或 BeliefPPG 论文指标。

## 保留与延期

全部已完成审计、原始官方标签、原始数据、异常表、验证代码及旧报告保留。作者联系停止，邮件仅历史草稿，永不自动发送。生成器/版本、RR 历史筛选实现、重复峰形成历史、标签完全独立重建列为 Deferred，不再阻止模型开发。旧严格审计的 BLOCKED 是历史规则下的结果，不改写为 PASS；新增版本化工程适配器单独验收。

## 工程数据门槛

采用官方 label，不生成替换目标。固定 t0=0、8 秒窗、2 秒步长、左闭右开；PPG [128k,128k+512)、ACC [64k,64k+256)、label_index=k。完整末窗 end<=duration；部分尾段不补齐、不截断标签。选择依据为官方同步记录说明、15 人通道时长与完整窗数量一致，以及已保存的固定候选全标签数值兼容结果；公开注明这是工程采用的映射，并非获得了原始生成器证明。不得搜索偏移或按模型测试误差选择时间轴。

导入验证可信官方来源/收据、文件 SHA-256、身份、各通道维度、采样率依据、有限性、所有窗口及标签数、已有显式时间索引。任一计数或时间元数据冲突即停止；不能 min(lengths)。无原始采样时间戳时保留无法排除全部包丢失的限制。所有异常窗口保留；重复 ECG 峰不属于预测输入，不因此排除主标签。工程验收结果不能设置旧 SOURCE_RULES 的 official_verified 标志。

推理仅输入 PPG/ACC；ECG、峰、HR 和活动标签不进入 estimator。活动只用于评价分组。缓存只含数值 .npy 与 JSON，allow_pickle=False，全部本地忽略。

## 冻结的初版实验

- 原 `splits/loso_v1.json`：15 折，各 12 训练 / 2 验证 / 1 测试；先 fold=0，S1 测试、S2/S3 验证、其余训练，之后相同配置完成其他折。受试者不交叉，不随机窗口分割。
- 连续记录零相位离线 Butterworth 四阶 0.5–4 Hz，PPG64/ACC32 原生采样。滤波后逐窗 PPG 标准化、ACC 每轴减均值后用当前训练受试者连续记录共同 RMS 缩放；不拟合验证/测试统计。ONNX 输入为同样预处理后的窗口，非原始传感器输入。
- 比较：Welch、ACC 辅助频谱（保留已有方法）、连续多参考 NLMS+Welch、PPG-only TCN、PPG+ACC 双分支 TCN；额外训练集均值预测作为 sanity baseline。参数在 `configs/showcase.yaml` 固定，无超参数搜索。
- TCN 初版每支 channels=[16,32,64]、kernel=5、dilations=[1,2,4]，残差卷积、全局平均池化；融合拼接。AdamW lr=0.001、weight_decay=0.0001、batch=256、最多12 epochs、seed=42。验证 MAE 挑选 checkpoint；目标均值/标准差仅训练集，测试只在训练和选择完成后读入评估。此预算定义紧凑初版，不承诺最优精度。
- 模型选择完成后每折保存所有测试窗口、失败/状态、MAE/RMSE、受试者与活动统计。总体报告 subject-macro 与 window-micro，不能用少数成功折冒充完整 LOSO。已分析过的公开数据用于工程演示，非独立临床或发表级外部验证。
- ONNX FP32：导出首折两种模型，检查图结构、batch=1/多样本的 PyTorch/ONNX Runtime CPU 数值一致性（atol=1e-4 BPM）。CPU 时间采用预热后重复测量，注明本机、batch 与预处理边界；不能宣称流式/MCU部署。

## 交付与发布

英文 README 包含 Overview、Dataset、Signal Processing、Deep Learning、Experimental Setup、Results、Visualizations、Quick Start、ONNX Inference、Limitations。所有真实指标仅由实际产物生成，所有图注明真实/合成与单折/完整 LOSO。性能不佳如实保留，不要求 ACC 一定改善结果。

候选公开范围仅原创代码、配置、测试、文档、合成图与审核过的汇总结果。原始数据、受试者窗口缓存、训练 checkpoint、ONNX 权重、完整日志、逐窗真实预测、个人信息及本机绝对路径默认不发布。ONNX 导出物本地保留，推理代码可公开。许可允许范围和发布审计分别记录。2026-10-25 用户追加授权：原创代码采用 MIT，并可创建公开仓库及上传审核后的白名单；上述数据排除规则不变。
