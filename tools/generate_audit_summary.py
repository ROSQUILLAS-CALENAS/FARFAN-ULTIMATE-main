#!/usr/bin/env python3
"""
Generate a complete, human-readable summary from system_audit_report.json
without omitting any issues.

Outputs:
- audit_summary_full.md (grouped by status, all checks listed)

Usage:
    python tools/generate_audit_summary.py [--input system_audit_report.json] [--output audit_summary_full.md]
"""
import json
import argparse
from pathlib import Path
from datetime import datetime


def load_report(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def truncate(val, limit=800):
    if isinstance(val, str) and len(val) > limit:
        return val[:limit] + f"\n... [truncated, total {len(val)} chars]"
    return val


def format_details(details):
    if not details:
        return ""
    try:
        # Pretty JSON for details
        return json.dumps(details, indent=2, ensure_ascii=False)
    except Exception:
        return str(details)


def build_markdown(report: dict) -> str:
    summary = report.get('summary', {})
    results = report.get('results', [])
    recommendations = report.get('recommendations', [])

    # Group results by status
    groups = {"FAIL": [], "WARN": [], "PASS": [], "SKIP": []}
    for r in results:
        groups.setdefault(r.get('status', 'PASS'), []).append(r)

    # Sort each group by check_name
    for k in groups:
        groups[k].sort(key=lambda x: x.get('check_name', ''))

    lines = []
    lines.append(f"# Comprehensive System Audit – Full Summary\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")

    # Top-level summary
    lines.append("## Summary\n")
    if summary:
        lines.append("```")
        for k in [
            'total_checks', 'passed_checks', 'failed_checks', 'warning_checks', 'skipped_checks',
            'critical_issues', 'success_rate', 'health_score']:
            if k in summary:
                lines.append(f"{k}: {summary[k]}")
        lines.append("``""\n")

    # Recommendations
    lines.append("## Recommendations\n")
    if recommendations:
        for rec in recommendations:
            lines.append(f"- {rec}")
    else:
        lines.append("(none)")
    lines.append("")

    # Full results by status
    for status in ["FAIL", "WARN", "PASS", "SKIP"]:
        items = groups.get(status, [])
        lines.append(f"## {status} ({len(items)})\n")
        if not items:
            lines.append("(none)\n")
            continue
        for r in items:
            check = r.get('check_name', '(unknown)')
            message = truncate(r.get('message', ''))
            severity = r.get('severity', '')
            remediation = truncate(r.get('remediation', ''))
            details = format_details(r.get('details'))

            lines.append(f"### {check}")
            lines.append(f"- status: {status}")
            if severity:
                lines.append(f"- severity: {severity}")
            if message:
                lines.append(f"- message: {message}")
            if remediation:
                lines.append(f"- remediation: {remediation}")
            if details:
                lines.append(f"<details><summary>details</summary>\n\n")
                lines.append("```json")
                lines.append(details)
                lines.append("```")
                lines.append("\n</details>")
            lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='system_audit_report.json')
    ap.add_argument('--output', default='audit_summary_full.md')
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input report not found: {input_path}")

    report = load_report(input_path)
    md = build_markdown(report)

    output_path = Path(args.output)
    output_path.write_text(md, encoding='utf-8')
    print(f"Wrote full audit summary to: {output_path}")


if __name__ == '__main__':
    main()
