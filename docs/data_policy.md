# 数据、来源与发布边界

核验日期：2026-10-25。经用户明确授权，已沿原作者页面的大学 sciebo 链接下载官方 PPG-DaLiA，并在仓库外提取全部 15 人记录。服务器长度/摘要及 ZIP CRC 均核对通过；以显式 `--trust-local-pickle` 进行只读审计。原始文件、官方 PDF 和逐窗诊断均保持本地。当前公开候选的结果与限制见[工程实验报告](showcase_results.md)；详细来源身份和历史异常审计保留在本地，未删除。

## 来源和引用

- UCI 数据页：[PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia)，数据 DOI：10.24432/C53890。
- 原作者页：[Deep PPG dataset](https://ubi29.informatik.uni-siegen.de/usi/data_ppgdalia.html)。
- 论文：Reiss et al., *Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks*, Sensors 19(14):3079, 2019，DOI：10.3390/s19143079。
- 官方 [readme.pdf](https://archive.ics.uci.edu/ml/machine-learning-databases/00495/readme.pdf)，获取成功并记录 SHA-256，逐页证据见 `source_verification.md`。

## 许可差异

本次核验的 UCI 页面显示 CC BY 4.0；原作者页仍限制为科学、非商业用途并要求署名。保留两种声明及链接，不在本项目中推定其法律优先关系，不把可下载解释为可商业再分发。代码许可不覆盖数据。

当前代码根据任务书和数学定义自行编写，未复制 BeliefPPG、KID-PPG 或其他第三方算法源码，也不称为这些论文的完整复现。第三方依赖保留各自许可证。后续若复用算法源文件，必须逐文件记录来源、版本与许可，增加 THIRD_PARTY_NOTICES。经用户明确授权，原创代码采用 [MIT](../LICENSE)；依赖与数据边界见 [第三方许可说明](../THIRD_PARTY_NOTICES.md)。

## 本地数据规则

真实数据应位于仓库外，由 `--data-root` 或 `PPG_DALIA_ROOT` 指定。工具不从未知镜像下载、不修改原文件、不把真实数据作为测试夹具。SHA-256 只证明内容身份，不能证明来源可信。

pickle 仅允许在用户确认其来自可信官方本地下载并传入 `--trust-local-pickle` 时加载。没有自制“安全 unpickler”承诺；pickle 仍可以执行任意代码。后续缓存禁止任意对象反序列化，数值读取始终使用 `allow_pickle=False`。

默认可作为未来发布候选：原创代码、必要配置、依赖锁、合成测试与小型 demo、方法说明、许可说明及经审核的聚合指标。发布状态以实际 GitHub 提交为准。

默认排除：原始信号、pickle、派生窗口缓存、逐窗真实预测、权重、完整日志、本机路径、凭据、原论文或说明 PDF、未经授权的第三方图。`.gitignore` 只是辅助；首次公开提交仍须逐文件白名单审核及检查暂存区。源码目录 `src/ppg_hr/data/` 不受根级 `/data/` 忽略规则影响。

2026-10-25 用户已授权创建公开仓库及上传白名单；每次发布仍须核验实际文件集合与哈希。
