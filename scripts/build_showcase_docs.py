"""Build aggregate-only portfolio documents from verified real result artifacts."""

# Markdown prose intentionally keeps each paragraph on a single editable line.
# ruff: noqa: E501

import argparse
import json
import shutil
from pathlib import Path


def build(summary: Path, onnx_fold: Path, synthetic_figure: Path | None = None):
    result = json.loads((summary / "results.json").read_text(encoding="utf-8"))
    exports = json.loads((onnx_fold / "onnx_results.json").read_text(encoding="utf-8"))
    complete = result["status"] == "COMPLETE_LOSO"
    scope = (
        f"Complete fixed-config LOSO: {result['n_subjects']} held-out subjects, "
        f"{result['n_windows']:,} / 64,697 official-label windows"
        if complete
        else f"Initial held-out experiment: {result['n_subjects']} test subject, "
        f"{result['n_windows']:,} windows; full LOSO is in progress"
    )
    table = "| Method | Subject-macro MAE | Window-micro MAE | Window-micro RMSE | Coverage |\n|---|---:|---:|---:|---:|\n"
    labels = {
        "train_mean": "Train mean",
        "welch": "Welch",
        "acc_spectral": "ACC-assisted spectrum",
        "nlms_welch": "NLMS + Welch",
        "ppg_tcn": "PPG-only TCN",
        "fusion_tcn": "PPG+ACC TCN",
    }
    for name, label in labels.items():
        m = result["methods"][name]
        table += (
            f"| {label} | {m['subject_macro_mae']:.3f} | {m['micro']['mae']:.3f} | "
            f"{m['micro']['rmse']:.3f} | {100 * m['micro']['coverage']:.0f}% |\n"
        )
    assets = Path("docs/assets")
    assets.mkdir(parents=True, exist_ok=True)
    for name in ["comparison.png", "subject_mae.png", "error_distribution.png"]:
        shutil.copyfile(summary / name, assets / name)
    if synthetic_figure is not None:
        shutil.copyfile(synthetic_figure, assets / "synthetic_demo.png")
    aggregate = {
        k: result[k]
        for k in ["status", "n_subjects", "n_windows", "methods", "config", "limitations"]
    }
    aggregate["label_source"] = "unchanged released PPG-DaLiA HR"
    (Path("docs/showcase_metrics.json")).write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    p = exports["ppg_tcn"]
    f = exports["fusion_tcn"]
    readme = f"""# Motion-Robust-PPG-Heart-Rate

Motion-aware heart-rate estimation from wrist PPG and three-axis accelerometry: SciPy signal processing, adaptive filtering, PyTorch TCNs, subject-isolated evaluation and ONNX CPU inference.

[中文说明](README_CN.md) · [Engineering scope](docs/showcase_spec.md) · [Experiment report](docs/showcase_results.md) · [Aggregate metrics](docs/showcase_metrics.json)

## Project Overview

An algorithm-engineering portfolio for interviews and practical demonstrations. The project independently implements traditional signal-processing methods, a PPG-only TCN and a PPG+ACC two-branch TCN. It is not an exact reproduction of Deep PPG, KID-PPG or BeliefPPG, and does not claim clinical accuracy.

Implemented: data adaptation, fixed windows, Welch/ACC-assisted spectrum/NLMS baselines, two TCNs, subject-level splits, activity/subject metrics, real-data charts, synthetic demo and local ONNX checks. **{scope}.**

## Dataset

[PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia) includes 15 subjects: wrist BVP at 64 Hz, three-axis ACC at 32 Hz and ECG-derived reference HR. Cite the dataset (DOI [10.24432/C53890](https://doi.org/10.24432/C53890)) and Reiss et al., *Deep PPG*, Sensors 2019 (DOI [10.3390/s19143079](https://doi.org/10.3390/s19143079)).

All **64,697 official HR labels** are retained. The engineering mapping uses `[2k, 2k+8)` seconds from the synchronized start: PPG `[128k,128k+512)`, ACC `[64k,64k+256)`, label index `k`. Complete terminal windows are included; partial trailing windows are not padded. File hashes, subject identity, durations, window counts, finite inputs and explicit time metadata are checked. Mismatches stop preparation; no minimum-length truncation or offset search is allowed.

The mapping is supported by synchronized-data documentation, consistent counts and a separately recorded numerical reference audit. Original generator/version and duplicate-annotation history are Deferred, not prerequisites for this engineering benchmark. Labels are never replaced by reconstructed ECG targets; all historical audits and anomalies remain preserved locally. Raw acquisition timestamps are absent, so every possible packet-loss or alignment issue cannot be ruled out.

## Signal Processing Methods

Offline preprocessing uses fourth-order Butterworth 0.5-4 Hz zero-phase filtering, separately per subject at native rates. Every method uses the same windows and original reference labels.

| Method | Implementation |
|---|---|
| Welch | PPG PSD peak in 30-240 BPM; Hann, 4 s segments, 50% overlap, 4096-point FFT at 64 Hz |
| ACC-assisted spectrum | Motion-associated peaks downweighted on the shared PPG/ACC physical frequency grid |
| NLMS + Welch | Three-axis ACC zero-order held to 64 Hz; 8 taps per axis, step 0.05; residual passed to Welch |
| Train mean | Sanity baseline fitted to training labels only |

NLMS runs continuously and resets per subject; overlapping windows never repeatedly update the same sample. ACC is not a clean noise reference. Shared motion/pulse frequencies can cause cancellation of useful pulse content. Failure results are retained without test-driven retuning.

## Deep Learning Models

Residual temporal convolutions use channels `[16,32,64]`, kernel 5, dilations `[1,2,4]`, dropout 0.1 and global average pooling. Padding is symmetric: these are **offline**, noncausal models.

| Model | Preprocessed input | Parameters |
|---|---|---:|
| PPG-only TCN | `[B,1,512]` | 42,737 |
| PPG+ACC two-branch TCN | `[B,1,512]` + `[B,3,256]` | 89,697 |

PPG is standardized per window. ACC is demeaned per window and divided by a common RMS fitted from unique continuous training records only. HR target mean/std also use training labels only; target rescaling is embedded in the exported model. ECG, R peaks, HR labels and activity labels never enter the prediction interface.

## Experimental Setup

The [fixed configuration](configs/showcase.yaml) uses seed 42, AdamW, learning rate 0.001, weight decay 0.0001, batch 256 and 12 epochs, with normalized-target MSE. Validation-window MAE selects the checkpoint; there is no hyperparameter search or test-set tuning.

[Subject splits](splits/loso_v1.json) provide 15 folds, each 12 train / 2 validation / 1 test. First: S1 test, S2/S3 validation, S4-S15 train. The remaining folds keep the identical settings. No random-window splitting, final 14-subject refit or nested search is performed. The public data had already been inspected during engineering/reference audits; this is a development benchmark, not an untouched external cohort.

Local artifacts record source/config/split/cache/dependency hashes, epoch histories, checkpoints, complete predictions and grouped metrics. Aggregation checks every test subject/index and original label against the verified cache, rejecting incompatible data/splits/locks/configs or missing/duplicate IDs. Metrics include MAE, RMSE, bias, tail error, coverage and activity/subject groups.

## Results

**{scope}.** All errors below are in BPM; all completed test windows are included.

{table}
The fusion model has more parameters, so the comparison does not isolate the causal benefit of ACC. NLMS failure is retained. These are measured engineering results, not published-paper reproduction scores.

## Visualizations

![Model comparison](docs/assets/comparison.png)
![Held-out subject MAE](docs/assets/subject_mae.png)
![Absolute-error distributions](docs/assets/error_distribution.png)

These real-data figures contain aggregate errors only. Per-window predictions, real signal traces and caches remain local. Synthetic demo plots are clearly marked `SYNTHETIC DATA` and do not establish real-population accuracy.

![Synthetic neighboring-frequency demonstration](docs/assets/synthetic_demo.png)

The synthetic figure illustrates PPG/ACC spectral reasoning on generated signals. It is separate from the real-data results above.

## Installation and Quick Start

Python 3.11 and uv are required. Dependencies stay in the project `.venv`. The optional ML group pins PyTorch 2.9.1 with official CUDA 12.8 wheels on Windows/Linux; system Python/drivers are not modified.

```powershell
uv sync --locked --default-index https://pypi.org/simple
uv run --locked --offline ppg-hr doctor
uv run --locked --offline ppg-hr demo --synthetic --config configs/demo.yaml --spectral-config configs/baselines/welch.yaml

uv sync --locked --extra ml --default-index https://pypi.org/simple
uv run --locked --offline --extra ml python scripts/register_dataset.py --archive "<official-PPG_FieldStudy.zip>" --destination "<new-external-data-root>" --trust-official-archive
uv run --locked --offline --extra ml python scripts/prepare_showcase.py --data-root <extracted-data-root> --trust-local-pickle
uv run --locked --offline --extra ml python scripts/run_showcase.py --cache <cache-directory> --folds 0 --device cuda
uv run --locked --offline --extra ml python scripts/run_showcase.py --cache <cache-directory> --folds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --device cuda
uv run --locked --offline --extra ml python scripts/summarize_showcase.py --runs <first-run> <remaining-run> --cache <cache-directory>
```

Replace angle-bracket placeholders with local paths before running. Download the original `PPG_FieldStudy.zip` yourself from the [author dataset page](https://ubi29.informatik.uni-siegen.de/usi/data_ppgdalia.html), then run `register_dataset.py` to verify the pinned 2,864,337,783-byte release, extract without overwriting and create `.acquisition-status.json`. The archive and extraction directory must both stay outside the repository; extraction requires about 24.2 GB plus reserve. The script does not download data. A different archive release requires a new audit, not a bypass. If the audited extraction already exists, skip registration and reuse that data root.

The extracted root contains the receipt and `PPG_FieldStudy/S1/S1.pkl` through S15. Only use trusted official pickle files; the flag acknowledges trust and does not make unknown pickle safe. CPU training is supported via `--device cpu`, with no CPU training-runtime claim. Training/aggregation require real data and are not part of the no-data smoke demo.

The original `audit-data` / `prepare` commands retain the historical strict audit. Use `scripts/prepare_showcase.py` for the explicitly versioned engineering path; do not toggle the old official-certification booleans.

Two frozen historical diagnostic protocols are included to keep the retained audit tools and synthetic tests self-contained. They are archival records; the current development scope is [showcase_v1](docs/showcase_spec.md), with author-side generator history Deferred.

## ONNX Inference

Both first-fold checkpoints have FP32 ONNX exports with dynamic batch size, graph validation and CPU parity checks at batches 1 and 7. Maximum absolute differences: `{p["max_absolute_error_bpm"]:.3g}` BPM (PPG), `{f["max_absolute_error_bpm"]:.3g}` BPM (fusion), tolerance `1e-4`.

```powershell
uv run --locked --offline --extra ml python scripts/infer_onnx.py --model <local-fold-directory>/fusion_tcn.onnx --synthetic-smoke
uv run --locked --offline --extra ml python scripts/infer_onnx.py --model <local-fold-directory>/fusion_tcn.onnx --inputs <normalized-windows.npz>
```

Inputs are already filtered/normalized PPG/ACC windows, not raw signals. The per-model metadata saves training normalization. Local model-only CPU median latency: {p["model_only_latency_ms_median"]:.3f} ms / {f["model_only_latency_ms_median"]:.3f} ms, batch 1, 10 warmups and 100 repeats, default ONNX Runtime threading on the development machine. This excludes preprocessing and acquisition/window collection and is not wearable end-to-end latency. Checkpoints and ONNX weights remain local.

No trained model is included in the showcase package. Train locally to obtain the exported files before running either ONNX command. `--synthetic-smoke` uses random signal-only inputs to check execution; its output is not an accuracy measurement. Without data or weights, the earlier synthetic spectral demo remains runnable.

## Tests and Project Structure

```powershell
uv run --locked --offline --extra ml pytest -q --basetemp=runs/test-local
uv run --locked --offline --extra ml ruff check src scripts tests
```

```text
src/ppg_hr/  # data contracts, preprocessing, baselines, TCNs, experiments
configs/     # fixed processing and experiment settings
splits/      # subject-level roles and hashes
scripts/     # preparation, training, aggregation, ONNX inference
tests/       # synthetic-only fixtures and regression checks
docs/        # scope, aggregate results, sources and historical evidence
cache/ runs/ # ignored local records, models and logs
```

## Limitations

- One dataset, one seed and a compact 12-epoch budget; no clinical or independent external validation.
- Fusion has more parameters; no capacity-matched or ACC-only attribution experiment is claimed.
- Offline filtering and TCN padding use future samples; causal streaming, MCU and smartwatch validation are Deferred.
- ACC and pulse spectra can overlap. Adaptive cancellation and motion suppression may worsen HR estimation.
- Author-side generator provenance remains unknown. All official labels and anomaly records are preserved.
- UCI and author pages have different licensing statements. Raw data and participant-derived records are not redistributed; see [data policy](docs/data_policy.md).
- Original code is licensed under MIT (see `LICENSE` and `THIRD_PARTY_NOTICES.md`); dataset and dependency licenses remain separate. Publication uses a separate whitelist; raw data, caches, checkpoints/weights, full logs, personal information and per-window predictions are excluded.
"""
    # This host has an inherited mirror setting; keep all documented commands
    # bound to the same official index as the locked installation.
    readme = readme.replace(
        "uv run --locked --offline ",
        "uv run --locked --offline --default-index https://pypi.org/simple ",
    )
    Path("README.md").write_text(readme, encoding="utf-8")
    cn = f"""# PPG + ACC 运动场景心率估计：求职工程作品

本项目面向算法工程师求职和技术面试，展示 Python/SciPy 信号处理、Welch/NLMS、PyTorch TCN、多传感器融合、数据隔离评估及 ONNX 推理。是独立工程实现，不以严格复现原作者指标或发表论文为目标。

当前完成范围：**{scope}**。原始 15 人、64,697 个官方 HR 标签保留。作者生成器/软件版本、RR 历史实现、重复峰来源与独立重建标为 Deferred，不再阻止开发；不联系作者。时间映射、通道/标签计数、文件哈希和受试者隔离仍强制验证。

## 已实现

- 8 秒窗、2 秒步长；PPG64Hz、ACC32Hz，工程时间映射独立版本化，旧严格审计不改成 PASS。
- Welch、ACC 频谱辅助、连续多参考 NLMS+Welch，所有方法使用相同窗口和原标签。
- PPG-only TCN（42,737 参数）和 PPG+ACC 双分支 TCN（89,697 参数）。
- 12训练/2验证/1测试的受试者级划分，验证集选 checkpoint；固定seed42、12epochs，无测试调参。
- MAE、RMSE、偏差、尾部误差、活动和受试者统计，完整分母及 coverage。
- 两种首折模型的 ONNX FP32 导出、动态batch1/7与CPU数值一致性；无数据可运行原合成demo。

## 实测结果

{table}
全部已完成测试窗纳入，不按异常、活动或难度删除。融合模型容量更大，不能把改善全部归因于 ACC；NLMS 较差结果如实保留。完整指标和限制见[实验报告](docs/showcase_results.md)，英文项目入口与命令见[README](README.md)。

![实测模型误差比较](docs/assets/comparison.png)

## 使用及证据边界

项目用 uv 独立环境；`uv sync --locked --extra ml --default-index https://pypi.org/simple` 安装学习与导出依赖。工程数据准备使用 `scripts/prepare_showcase.py`，单折/LOSO 使用 `scripts/run_showcase.py`，汇总使用 `scripts/summarize_showcase.py`，ONNX 模型推理使用 `scripts/infer_onnx.py`。完整命令及输入预处理边界见英文 README。

旧严格 M1 历史 BLOCKED 保留，新工程数据通道 READY_ENGINEERING。官方标签不替换；无原始采集时间戳，不能保证排除所有同步误差。离线零相位滤波和对称卷积不是实时因果方案。公开数据已用于工程检查，结果不是临床或外部验证。

原始信号、缓存、逐窗预测、checkpoint/ONNX 权重、完整日志及个人信息留在本地，不纳入公开候选。现有审计、测试、失败和作者邮件草稿均保留；不再发送或继续作者核询。没有创建公开 GitHub 仓库或 push。
"""
    Path("README_CN.md").write_text(cn, encoding="utf-8")
    report = f"""# 工程对比实验报告

版本：showcase_v1。当前范围：**{scope}**。

## 固定协议

PPG-DaLiA 15 人，全部 64,697 个官方 HR 标签不修改。8秒窗、2秒步长，t0=0、完整尾窗<=记录时长；未搜索偏移。12/2/1 受试者角色见固定 splits/loso_v1.json。seed42，最多12epochs，验证window-micro MAE选择checkpoint，无测试调参或大规模超参搜索。

数据管线为离线0.5-4Hz零相位滤波，PPG每窗标准化，ACC尺度仅拟合当前训练受试者连续记录。TCN不读ECG、R峰、活动或HR标签作输入。全部方法同一测试分母，训练均值baseline也仅用训练标签。

## 实测指标（BPM）

{table}
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

首折两checkpoint的FP32 ONNX图检查、动态batch1/7及CPU数值一致性通过，最大差异分别{p["max_absolute_error_bpm"]:.8g}/{f["max_absolute_error_bpm"]:.8g} BPM，小于1e-4容差。CPU model-only中位数{p["model_only_latency_ms_median"]:.3f}/{f["model_only_latency_ms_median"]:.3f}ms；10次预热、100次重复、batch1、默认ORT线程，本机实测，不含预处理、采集和等窗延迟。

首轮训练和预测已完成，导出时因Windows GBK无法打印PyTorch状态emoji失败。已用GBK回归测试复现，关闭导出verbose后仅复用原checkpoint恢复导出；没有重训或重新选择checkpoint。失败及恢复收据均本地保留。后续聚合增强测试身份/原标签和运行身份检查，不改变已运行训练或预测。

## 证据身份

本页由 scripts/build_showcase_docs.py 从经过验证的汇总JSON和ONNX收据生成。完整本地来源、预测hash与配置在运行目录的results/protocol/recovery文件；公开版只保留必要汇总。原始标签、全部异常、旧严格审计和历史草稿保留，原作者实现溯源为Deferred，不再发信。
"""
    Path("docs/showcase_results.md").write_text(report, encoding="utf-8")
    status = f"""# 实施状态：求职工程展示

当前合同：[showcase_v1](showcase_spec.md)，按用户最新任务替代严格论文复现的冲突条件。

| 交付 | 当前状态 | 证据与限制 |
|---|---|---|
| 独立环境 | 已验证 | Python3.11 / uv，Torch2.9.1+cu128，RTX4070 SUPER执行；未改系统环境 |
| 工程数据适配 | READY_ENGINEERING | 15人、64,697官方HR；精确窗口/标签对应、哈希与全部分母验证 |
| Welch/ACC谱/NLMS | 已实现并实测 | 相同窗口，连续NLMS每subject重置；失败结果保留 |
| PPG/融合TCN | 已实现并实测 | 42,737/89,697参数，训练尺度仅train，验证集选checkpoint |
| 真实评价 | {result["status"]} | {result["n_subjects"]}测试subject / {result["n_windows"]:,}窗；不冒充其他完成范围 |
| ONNX | 首折两模型通过 | FP32动态图，batch1/7与CPU parity，非传感器端到端延迟 |
| 合成demo | 可运行 | 六场景，标记SYNTHETIC DATA，无真实数据依赖 |
| 英文README/中文报告 | 已更新 | 从已核验实际汇总产物生成，不手写虚构指标 |
| 发布准备 | 仅本地候选 | 白名单排除原数据、cache、checkpoint、完整日志、逐窗预测、个人信息 |
| GitHub CI | 仅配置 | 已提供Windows/Linux基础测试workflow，未声称在GitHub执行 |
| 作者生成器/峰历史 | Deferred | 不再联系作者、不再阻止开发；旧审计和草稿均保留 |

## 历史边界

旧严格M1的BLOCKED是当时合同下的真实结论，未改写成作者确认PASS。新的工程映射独立版本化，采用官方HR；作者生成器和完全独立标签重建已按用户要求Deferred。没有删除原始标签、异常表、旧验证代码或失败尝试。先前状态原文存于[历史快照](history/2026-10-25-pre-showcase-implementation-status.md)。

当前结果见[实验报告](showcase_results.md)，执行与恢复细节见[进度记录](showcase_progress.md)。所有原始/逐窗记录、模型和完整日志仍在本地忽略目录；本轮未发送任何作者邮件、创建公开仓库、暂存/提交或push。

## 验证

本轮新增工程映射、NLMS、TCN、统计权限、ONNX及聚合回归检查。最后的完整测试、Ruff、发布候选身份和实验总分母以showcase_progress.md中的最终收据为准；不将历史测试次数冒充新结果。首折训练后导出曾因GBK日志编码失败，修复后仅恢复导出，无重训或测试调参。审阅发现的聚合身份缺口已补齐并回归验证。
"""
    Path("docs/implementation_status.md").write_text(status, encoding="utf-8")
    protocol = """# 当前工程实验协议

生效版本：showcase_v1，详见 [工程合同](showcase_spec.md)、configs/showcase.yaml、splits/loso_v1.json 和每次运行的 protocol.lock.json。

采用官方发布HR，不重建/替换标签。t0=0，[2k,2k+8)，PPG64Hz/ACC32Hz原生512/256点、2秒步长，完整末窗<=记录时长。检查文件身份、通道时长/维度、计数、有限性及可用显式时间元数据；不静默截断或按测试误差找偏移。没有取得原始作者生成器证明，明确作为工程采用映射；历史研究项Deferred。

离线每subject独立四阶0.5-4Hz零相位带通；逐窗PPG标准化，ACC去每轴窗均值后使用训练连续记录的公共RMS；训练target mean/std仅train。参考HR、ECG/R峰及activity不进入预测输入。NLMS参考ACC因果零阶保持到64Hz，在离线预处理后连续更新、每subject重置；全管线仍不是因果系统。

先第0折（S1test，S2/S3val，S4-S15train）跑通，再相同配置运行剩余14折。seed42、12epochs、AdamW0.001/weight_decay0.0001、batch256、normalized MSE。每模型用验证window-micro MAE选checkpoint，无嵌套搜索或14人最终重训。不同fold各自重建模型、随机种子与训练统计，测试不参与选择。

同一测试分母比较训练均值、Welch、ACC谱、NLMS+Welch、PPG-only与PPG+ACC TCN。所有ID和官方label/activity由缓存核验，拒绝缺失/重复/越界/角色错配；合并运行必须配置、cache、split、依赖锁一致。报告受试者macro及窗口micro MAE、RMSE、偏差、尾部和活动统计及coverage。完整15fold全部64,697窗后才标COMPLETE_LOSO。

首折checkpoint本地导出ONNX，输入是已预处理归一化窗口，目标尺度内嵌；动态batch1/7数值差<=1e-4 BPM。CPU仅model-only计时，不包含预处理和8秒等窗。单seed、容量不匹配、公有开发数据和未作外部验证等限制必须随结果公布。

此前严格/合成阶段协议完整保留在 history/2026-10-25-pre-showcase-protocol.md；旧报告不删除，也不冒称工程映射取得了作者认证。不得未经授权公开仓库或push。
"""
    Path("docs/protocol.md").write_text(protocol, encoding="utf-8")
    print(f"Built portfolio documents: {result['status']}, {result['n_windows']} windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--onnx-fold", type=Path, required=True)
    parser.add_argument("--synthetic-figure", type=Path)
    args = parser.parse_args()
    build(args.summary, args.onnx_fold, args.synthetic_figure)
