"""Create a local publication-candidate whitelist; never stage, commit or push."""

import json
import re
from pathlib import Path

from ppg_hr.data.adapter import file_sha256
from ppg_hr.io import new_run_dir, write_json


def main():
    root = Path.cwd().resolve()
    files = [
        Path(x)
        for x in [
            "README.md",
            "README_CN.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "pyproject.toml",
            "uv.lock",
            ".gitignore",
            ".python-version",
            ".github/workflows/ci.yml",
            "splits/loso_v1.json",
            "docs/showcase_spec.md",
            "docs/showcase_results.md",
            "docs/showcase_metrics.json",
            "docs/data_policy.md",
            "docs/publication_policy.md",
            "docs/interview_walkthrough.md",
            "docs/m1_reference_audit_protocol.md",
            "docs/m1_forensics_protocol.md",
        ]
    ]
    for name in ["src", "tests", "scripts", "configs"]:
        files.extend(
            p
            for p in Path(name).rglob("*")
            if p.is_file()
            and p.suffix in {".py", ".yaml"}
            and p.name != "local.yaml"
            and not p.name.endswith(".local.yaml")
            and "__pycache__" not in p.parts
        )
    files.extend(Path("docs/assets").glob("*.png"))
    findings = []
    items = []
    allowed = {p.resolve() for p in files}
    for p in sorted(set(files)):
        if not p.resolve().is_relative_to(root) or not p.is_file():
            raise ValueError(f"invalid whitelist path: {p}")
        if p.suffix in {".py", ".yaml", ".toml", ".md", ".json", ".yml"}:
            text = p.read_text(encoding="utf-8")
            path_pattern = r"[A-Za-z]:[/\\](?:Users|Codex|codex)[/\\]"
            email_pattern = r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
            if re.search(path_pattern, text) or re.search(email_pattern, text):
                findings.append({"file": p.as_posix(), "reason": "local personal path/contact"})
            if p.suffix == ".md":
                for link in re.findall(r"\]\(([^)]+)\)", text):
                    if "://" in link or link.startswith("#"):
                        continue
                    target = (p.parent / link.split("#")[0]).resolve()
                    if target not in allowed:
                        findings.append(
                            {
                                "file": p.as_posix(),
                                "reason": "link target outside whitelist",
                                "target": link,
                            }
                        )
        items.append({"path": p.as_posix(), "sha256": file_sha256(p), "bytes": p.stat().st_size})
    out = new_run_dir(Path("runs"), "publication-audit")
    report = {
        "status": "PASS_LOCAL_CANDIDATE" if not findings else "BLOCKED",
        "files": items,
        "findings": findings,
        "scope": "Whitelist candidate only; no license adjudication or public action.",
        "excluded": [
            "raw data",
            "cache",
            "runs",
            "weights/checkpoints",
            "full logs",
            "per-window predictions",
            "historical author-inquiry and full audit artifacts",
            "unlisted documents",
        ],
        "licensing": "Original code: MIT; dataset and dependencies retain their own terms.",
        "public_repository_created": False,
        "git_staging_or_push": False,
    }
    write_json(out / "whitelist.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate_files": len(items),
                "report": str(out / "whitelist.json"),
            }
        )
    )
    if findings:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
