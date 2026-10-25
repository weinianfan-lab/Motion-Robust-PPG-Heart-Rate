# License scope and third-party notices

The [MIT license](LICENSE) covers the original code, configuration, tests and original documentation in this repository. It does not relicense datasets, reference labels, papers, third-party software or their bundled components. Dependencies are installed separately; their source code and binary distributions are not vendored here.

## PPG-DaLiA

Dataset: Reiss, A., Indlekofer, I., & Schmidt, P. (2019), PPG-DaLiA, DOI [10.24432/C53890](https://doi.org/10.24432/C53890).

Paper: Reiss et al., *Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks*, Sensors 19(14), 3079 (2019), DOI [10.3390/s19143079](https://doi.org/10.3390/s19143079).

Checked 2026-10-25: the [UCI dataset page](https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia) states CC BY 4.0, while the [authors' dataset page](https://ubi29.informatik.uni-siegen.de/usi/data_ppgdalia.html) states scientific, non-commercial use with attribution. Both statements are recorded; this project does not resolve their legal precedence or grant commercial data rights. Obtain the dataset separately and review the terms applicable to your use.

The repository includes attributed aggregate evaluation results and original plots, not raw signals, official label arrays, subject window caches, per-window predictions or trained weights. MIT does not remove rights or obligations associated with the underlying data. See [data policy](docs/data_policy.md).

## Software dependencies

This is a notice inventory for the local environment used for the reported experiments, not a replacement for upstream license texts. Consult the installed distribution's LICENSE/NOTICE files, including notices for bundled and transitive components, when redistributing software. Versions are recorded in `uv.lock`; platform-specific wheels may carry additional notices.

| Dependency | Observed version | Upstream license / scope |
|---|---|---|
| [NumPy](https://github.com/numpy/numpy) | 2.4.6 | BSD-3-Clause; installed distribution declares BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 for bundled material |
| [SciPy](https://github.com/scipy/scipy) | 1.17.1 | BSD-3-Clause; bundled components have additional notices |
| [Matplotlib](https://matplotlib.org/stable/project/license.html) | 3.11.2 | Matplotlib license agreement; bundled fonts and components retain separate terms |
| [PyYAML](https://github.com/yaml/pyyaml) | 6.0.3 | MIT |
| [PyTorch](https://github.com/pytorch/pytorch) | 2.9.1+cu128 | BSD-3-Clause; LICENSE/NOTICE and CUDA/NVIDIA component terms apply to the binary distribution |
| [ONNX](https://github.com/onnx/onnx) | 1.23.2 | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 1.31.0 | MIT; ThirdPartyNotices.txt also applies |
| [ONNX Script](https://github.com/microsoft/onnxscript) | 0.7.2 | MIT |
| [ONNX IR](https://github.com/onnx/ir-py) | 1.1.0 | Apache-2.0 |
| [pytest](https://github.com/pytest-dev/pytest) | 9.1.1 | MIT |
| [pytest-cov](https://github.com/pytest-dev/pytest-cov) | 7.1.0 | MIT |
| [Ruff](https://github.com/astral-sh/ruff) | 0.17.2 | MIT |

The build backend and environment tooling are [uv / uv_build](https://github.com/astral-sh/uv); their upstream license files govern their use. They are not covered by this repository's MIT grant. GitHub Actions referenced by CI likewise retain their upstream licenses.

No BeliefPPG or KID-PPG implementation is bundled or claimed as fully reproduced. Any future copied third-party source must include its provenance and required license notices before publication.
