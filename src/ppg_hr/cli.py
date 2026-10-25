"""Only implemented stage commands are exposed; blocked audits return exit code 2."""

import argparse
import json
import os
import sys
from pathlib import Path

from ppg_hr import __version__
from ppg_hr.config import data_config, demo_config
from ppg_hr.errors import ConfigurationError, PPGError
from ppg_hr.io import atomic_text, new_run_dir, write_json


def _audit_artifacts(run: Path, report: dict) -> None:
    import csv
    import io

    from ppg_hr.data.adapter import file_sha256

    subjects = report.get("subjects", [])
    rows = [row for subject in subjects for row in subject.get("manifest", [])]
    artifact = {"path": None, "count": len(rows), "sha256": None}
    if rows:
        buffer = io.StringIO(newline="")
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
        manifest_path = run / "window_manifest.csv"
        atomic_text(manifest_path, buffer.getvalue())
        artifact.update(path=manifest_path.name, sha256=file_sha256(manifest_path))
    saved_subjects = []
    for subject in subjects:
        saved = dict(subject)
        if "manifest" in saved:
            manifest = saved.pop("manifest")
            saved["manifest_artifact"] = {
                **artifact,
                "count": len(manifest),
                "cohort_count": len(rows),
            }
        saved_subjects.append(saved)
    saved_report = {**report, "subjects": saved_subjects, "manifest_artifact": artifact}
    write_json(run / "data_audit.json", saved_report)

    def cell(value):
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    blockers = report.get("blockers", [])
    markdown = "# Data audit\n\nStatus: **" + report["status"] + "**\n\n"
    markdown += (
        "This report describes observed audit state. "
        "Synthetic tests do not clear real-data gates.\n\n"
    )
    markdown += (
        f"Subjects discovered: **{len(report.get('discovered_subjects', []))} / "
        f"{len(report.get('expected_subjects', []))}**; "
        f"missing: {len(report.get('missing_subjects', []))}; "
        f"unexpected: {len(report.get('unexpected_subjects', []))}.\n\n"
        f"Manifest rows: **{len(rows)}**.\n\n"
    )
    source = report.get("source_verification", {})
    source_status = "; ".join(
        f"{key}={cell(source[key])}"
        for key in ("status", "label_origin_verified", "tail_rule_verified")
        if key in source
    )
    markdown += f"Source verification: {source_status or 'not provided'}.\n\n"
    source_blockers = "; ".join(cell(reason) for reason in source.get("blockers", []))
    markdown += f"Source blockers: {source_blockers or 'none recorded'}.\n\n"
    markdown += "## Blocking conditions\n\n"
    markdown += "\n".join(f"- {cell(item)}" for item in blockers) or "None recorded."
    markdown += (
        "\n\n## Subject inventory\n\n"
        "| Subject | Status | BVP shape | ACC shape | Complete BVP / ACC | Labels | "
        "Manifest rows | Eligible | Blockers |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    by_subject = {subject["subject_id"]: subject for subject in saved_subjects}
    inventory = list(dict.fromkeys([*report.get("expected_subjects", []), *by_subject]))
    for name in inventory:
        subject = by_subject.get(name)
        if subject is None:
            status = "NOT_REPORTED" if name in report.get("discovered_subjects", []) else "MISSING"
            values = [name, status, "—", "—", "—", "—", 0, "—", "subject report unavailable"]
        else:
            signals = subject.get("signals", {})
            counts = subject.get("window_counts", {})
            summary = subject.get("manifest_summary", {})
            values = [
                name,
                subject.get("status", "not provided"),
                signals.get("signal.wrist.BVP", {}).get("shape", "—"),
                signals.get("signal.wrist.ACC", {}).get("shape", "—"),
                f"{counts.get('bvp', '—')} / {counts.get('acc', '—')}",
                counts.get("labels", "—"),
                subject.get("manifest_artifact", {}).get("count", 0),
                summary.get("eval_eligible", "—"),
                "; ".join(subject.get("blockers", [])) or "none recorded",
            ]
        markdown += "| " + " | ".join(cell(value) for value in values) + " |\n"
    if not inventory:
        markdown += "\nNo subject inventory was provided.\n"
    markdown += (
        "\n## Complete local details\n\n"
        "- [Detailed audit JSON](data_audit.json): complete audit fields and manifest references.\n"
    )
    if rows:
        markdown += (
            "- [Complete window manifest CSV](window_manifest.csv): "
            f"{len(rows)} rows; all original fields, with list/dictionary fields encoded as JSON.\n"
            f"- Manifest SHA-256: `{artifact['sha256']}`.\n"
        )
    else:
        markdown += "- No window manifest rows were available; no CSV was written (0 rows).\n"
    atomic_text(run / "data_audit.md", markdown)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Auditable PPG/ACC tools: M0/M1 foundation and synthetic spectral validation."
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="Inspect current local environment without changing it"
    )
    doctor.add_argument("--output-root", type=Path, default=Path("runs"))
    for name in ("audit-data", "prepare"):
        command = commands.add_parser(
            name, help="Read-only audit / gated numeric preparation; missing data yields BLOCKED"
        )
        command.add_argument("--config", type=Path)
        command.add_argument("--data-root", type=Path)
        command.add_argument("--output-root", type=Path)
        command.add_argument(
            "--trust-local-pickle",
            action="store_true",
            help="Acknowledge only your trusted official local pickle files",
        )
        if name == "prepare":
            command.add_argument("--cache-root", type=Path)
    splits = commands.add_parser(
        "make-splits", help="Write deterministic subject-level LOSO proposal and content hash"
    )
    splits.add_argument("--protocol", choices=["loso_v1"], default="loso_v1")
    splits.add_argument("--output", type=Path, default=Path("splits/loso_v1.json"))
    demo = commands.add_parser(
        "demo", help="Run explicitly labeled synthetic A1/A2 validation with figures"
    )
    demo.add_argument("--synthetic", action="store_true", required=True)
    demo.add_argument("--config", type=Path)
    demo.add_argument("--spectral-config", type=Path)
    demo.add_argument("--scenario")
    demo.add_argument("--seed", type=int)
    demo.add_argument("--duration", type=float)
    demo.add_argument("--hr", type=float)
    demo.add_argument("--output-root", type=Path, default=Path("runs"))
    return root


def _execute(args) -> int:
    if args.command == "doctor":
        from ppg_hr.doctor import environment_report

        run = new_run_dir(args.output_root, "doctor")
        report = environment_report()
        write_json(run / "doctor.json", report)
        print(json.dumps({"status": report["status"], "report": str(run / "doctor.json")}))
        return 0 if report["status"] == "READY_CPU" else 2
    if args.command in ("audit-data", "prepare"):
        from ppg_hr.data.audit import audit_dataset
        from ppg_hr.data.prepare import prepare_dataset

        config = data_config(args.config)
        data_root = args.data_root or config["data_root"] or os.environ.get("PPG_DALIA_ROOT")
        run = new_run_dir(
            args.output_root or config["output_root"],
            "audit" if args.command == "audit-data" else "prepare",
        )
        if args.command == "prepare":
            report = prepare_dataset(
                Path(data_root) if data_root else None,
                args.cache_root or Path(config["cache_root"]),
                trust_local_pickle=args.trust_local_pickle,
            )
            write_json(run / "prepare_status.json", report)
            audit = report["audit"]
        else:
            report = audit_dataset(
                Path(data_root) if data_root else None, trust_local_pickle=args.trust_local_pickle
            )
            audit = report
        _audit_artifacts(run, audit)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(run / "data_audit.json"),
                    "blockers": report["blockers"],
                }
            )
        )
        return 2 if report["status"] == "BLOCKED" else 0
    if args.command == "make-splits":
        from ppg_hr.data.splits import make_loso_splits

        report = make_loso_splits()
        if args.output.exists():
            if json.loads(args.output.read_text(encoding="utf-8")) != report:
                raise ConfigurationError(
                    "Split output already exists with different contents; use a new versioned path"
                )
        else:
            write_json(args.output, report)
        print(
            json.dumps(
                {
                    "status": "PROTOCOL_PROPOSAL",
                    "split_hash": report["split_hash"],
                    "report": str(args.output),
                    "real_data_verified": False,
                }
            )
        )
        return 0
    if args.command == "demo":
        from ppg_hr.demo import run_demo

        config = demo_config(
            args.config,
            seed=args.seed,
            duration_s=args.duration,
            hr_bpm=args.hr,
            scenario=args.scenario,
        )
        run, _ = run_demo(config, args.output_root, args.spectral_config)
        status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))["status"]
        print(
            json.dumps(
                {
                    "status": status,
                    "data_kind": "SYNTHETIC DATA",
                    "report": str(run / "results.json"),
                }
            )
        )
        return 0 if status == "COMPLETE" else 2
    raise ConfigurationError("Unknown command")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return _execute(args)
    except PPGError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"INPUT_OR_OUTPUT_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
