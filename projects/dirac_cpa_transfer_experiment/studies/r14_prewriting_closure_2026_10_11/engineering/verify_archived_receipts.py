"""Rehash all current first-receipt archive schemas; no scientific writes."""
from pathlib import Path
import hashlib
import json
import time

BASE = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root = BASE / "runtime/archive_integrity_checks"
    old_records = sorted(root.glob("*/verification.json")) if root.exists() else []
    previous = read(old_records[-1]) if old_records else None
    known = {row["path"]: row["receipt_sha256"] for row in previous["receipts"]} if previous else {}
    observation = read(BASE / "runtime/continuation_observation.json")
    for row in observation.get("source_preserved_receipts", []):
        assert known.get(row["path"], row["receipt_sha256"]) == row["receipt_sha256"]
        known[row["path"]] = row["receipt_sha256"]
    paths = set()
    for directory in ("cohort_receipts", "primary_comparison_receipts",
                      "mechanism_cohort_receipts", "core_case_receipts"):
        paths.update((BASE / "analysis_outputs" / directory).glob("*/receipt.json"))
    paths.update((BASE / "delivery_exports").glob("*/receipt.json"))
    paths.add(BASE / "analysis_outputs/T10_analytic_hard_primary_first_full_cohort.json")
    errors, rows = [], []
    mappings = ("preserved_file_sha256", "preserved_audit_files",
                "preserved_source_sha256", "preserved_files_sha256")
    for path in sorted(paths):
        name = path.relative_to(BASE).as_posix()
        digest = sha(path)
        if name in known and known[name] != digest:
            errors.append({"receipt": name, "reason": "previously sealed receipt changed"})
        data, checks = read(path), []

        def check(target, expected, role):
            target = target.resolve()
            target.relative_to(BASE)
            actual = sha(target) if target.is_file() else None
            row = {"path": target.relative_to(BASE).as_posix(), "role": role,
                   "expected_sha256": expected, "actual_sha256": actual,
                   "pass_check": actual == expected}
            checks.append(row)
            if not row["pass_check"]:
                errors.append({"receipt": name, **row})

        for key in mappings:
            for relative, expected in data.get(key, {}).items():
                check(path.parent / relative, expected, key)
        for key in ("preserved_independent_audit_snapshot", "preserved_independent_per_seed_rows"):
            if key in data:
                entry = data[key]
                check(BASE / entry["path"], entry["sha256"], key)
        for entry in data.get("exports", []):
            for target, role in ((BASE / entry["source"], "static source"),
                                 (BASE / entry["target"], "static root export"),
                                 (path.parent / entry["target"], "static preserved export")):
                check(target, entry["sha256"], role)
        for entry in data.get("source_proof_files", []):
            check(BASE / entry["path"], entry["sha256"], "static source proof")
        if not checks:
            errors.append({"receipt": name, "reason": "unrecognized or empty archive schema"})
        rows.append({"path": name, "receipt_sha256": digest,
                     "previous_receipt_SHA_checked": name in known,
                     "archive_file_checks": checks})
    missing = sorted(set(known) - {row["path"] for row in rows})
    errors.extend({"receipt": name, "reason": "previously sealed receipt missing"} for name in missing)
    folder = root / str(time.time_ns())
    folder.mkdir(parents=True, exist_ok=False)
    record = {"verified_utc": time.time(), "status": "FAIL" if errors else "PASS_ALL_CURRENT_PRESERVED_RECEIPT_CONTENTS",
              "checker_sha256": sha(Path(__file__)), "receipt_count": len(rows),
              "archive_file_checks": sum(len(row["archive_file_checks"]) for row in rows),
              "previous_verification": str(old_records[-1].relative_to(BASE)) if old_records else None,
              "receipts": rows, "errors": errors,
              "current_archive_scope_only": True, "no_existing_evidence_written": True,
              "whole_goal_complete": False}
    with (folder / "verification.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in record.items() if key not in ("receipts", "errors")}))
    if errors:
        print(json.dumps(errors[:10]))
        raise RuntimeError("Preserved receipt integrity failed; keep diagnostic, do not repair raw evidence")


if __name__ == "__main__":
    main()
