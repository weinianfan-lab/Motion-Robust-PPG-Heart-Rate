# 工程对比实验报告

版本：showcase_v1。当前范围：**Complete fixed-config LOSO: 15 held-out subjects, 64,697 / 64,697 official-label windows**。

## 固定协议

PPG-DaLiA 15 人，全部 64,697 个官方 HR 标签不修改。8秒窗、2秒步长，t0=0、完整尾窗<=记录时长；未搜索偏移。12/2/1 受试者角色见固定 splits/loso_v1.json。seed42，最多12epochs，验证window-micro MAE选择checkpoint，无测试调参或大规模超参搜索。

数据管线为离线0.5-4Hz零相位滤波，PPG每窗标准化，ACC尺度仅拟合当前训练受试者连续记录。TCN不读ECG、R峰、活动或HR标签作输入。全部方法同一测试分母，训练均值baseline也仅用训练标签。

## 实测指标（BPM）

| Method | Subject-macro MAE | Window-micro MAE | Window-micro RMSE | Coverage |
|---|---:|---:|---:|---:|
| Train mean | 18.770 | 18.390 | 23.713 | 100% |
| Welch | 18.562 | 18.383 | 30.378 | 100% |
| ACC-assisted spectrum | 18.515 | 18.337 | 30.347 | 100% |
| NLMS + Welch | 24.479 | 24.421 | 40.012 | 100% |
| PPG-only TCN | 9.419 | 9.346 | 13.696 | 100% |
| PPG+ACC TCN | 8.413 | 8.316 | 13.605 | 100% |

机器可读的[汇总指标](showcase_metrics.json)仅包含聚合结果。按活动和受试者的指标也在其中；mixed窗口保留。所有分母和参考值已逐行对照本地工程缓存，汇总还验证了配置、数据、固定split及依赖锁身份。

## 可视化

![模型比较](assets/comparison.png)
![逐受试者误差](assets/subject_mae.png)
![误差分布](assets/error_distribution.png)

图中没有原始信号或逐窗HR曲线；误差直方图上界按全部方法实际最大误差扩展，不隐藏溢出。所有图标注当前完成范围，单折不会称作完整LOSO。

## 解释与限制

Welch在运动频率与心率重叠时容易选错峰。ACC谱抑制无法保证从共享频率中分离脉搏。NLMS假设运动干扰可由ACC线性参考解释；ACC并非纯噪声参考，可能消除有用成分或产生残差倍频，实测失败保留、不为改善测试结果重调参数。

融合TCN参数量89,697，PPG-only为42,737；性能差不能证明只来自ACC。仅一个seed和紧凑训练预算，不宣称SOTA、论文精确复现、临床准确性或外部泛化。原公开数据已进行全体工程/标签审计，不能包装为完全未接触的独立外部队列。没有干净真实PPG参考，不能据HR误差宣称波形SNR改善。

## ONNX 与恢复记录

首折两checkpoint的FP32 ONNX图检查、动态batch1/7及CPU数值一致性通过，最大差异分别7.6293945e-06/1.5258789e-05 BPM，小于1e-4容差。CPU model-only中位数0.259/0.338ms；10次预热、100次重复、batch1、默认ORT线程，本机实测，不含预处理、采集和等窗延迟。

首轮训练和预测已完成，导出时因Windows GBK无法打印PyTorch状态emoji失败。已用GBK回归测试复现，关闭导出verbose后仅复用原checkpoint恢复导出；没有重训或重新选择checkpoint。失败及恢复收据均本地保留。后续聚合增强测试身份/原标签和运行身份检查，不改变已运行训练或预测。

## 证据身份

本页由 scripts/build_showcase_docs.py 从经过验证的汇总JSON和ONNX收据生成。完整本地来源、预测hash与配置在运行目录的results/protocol/recovery文件；公开版只保留必要汇总。原始标签、全部异常、旧严格审计和历史草稿保留，原作者实现溯源为Deferred，不再发信。
