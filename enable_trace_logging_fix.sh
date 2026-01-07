#!/usr/bin/env bash
set -euo pipefail

# Patch-only fixer for enable_trace_logging.sh regex error.
# Run from project root. Safe to re-run.

python - <<'PY'
from __future__ import annotations
import re
from pathlib import Path

def ensure_imports(txt: str) -> str:
    if "from orchestration.trace import new_run_id, log_task" in txt and "from orchestration.trace_logger import TraceLogger" in txt:
        return txt
    ins = "from orchestration.trace import new_run_id, log_task\nfrom orchestration.trace_logger import TraceLogger\n"
    m = re.search(r"^(?:from __future__.*\n)?(?:(?:import|from) .*\n)+", txt, flags=re.M)
    if m:
        pos = m.end()
        return txt[:pos] + ins + txt[pos:]
    return ins + "\n" + txt

def patch_pm(path: Path) -> None:
    if not path.exists():
        print(f"Skip missing {path}")
        return
    txt = path.read_text(encoding="utf-8")
    txt = ensure_imports(txt)

    if "def emit_tasks_from_epic" in txt and "run_id = new_run_id()" not in txt:
        txt = re.sub(
            r"(def\s+emit_tasks_from_epic\s*\(.*?\):\s*\n)",
            r"\1    run_id = new_run_id()\n    logger = TraceLogger()\n",
            txt,
            count=1,
            flags=re.S,
        )

    if "resp = client.responses.create" in txt and "meta={'phase': 'request'" not in txt:
        txt = txt.replace(
            "resp = client.responses.create",
            "log_task(logger, run_id=run_id, initializer='PM', recipient='PM', task=prompt, response='', task_id='EPIC', task_title='Emit tasks', meta={'phase': 'request', 'model': model})\n    resp = client.responses.create",
            1,
        )

    if "text = resp.output_text" in txt and "meta={'phase': 'response'" not in txt:
        txt = txt.replace(
            "text = resp.output_text",
            "text = resp.output_text\n    log_task(logger, run_id=run_id, initializer='PM', recipient='PM', task=prompt, response=text, task_id='EPIC', task_title='Emit tasks', meta={'phase': 'response', 'model': model})",
            1,
        )

    path.write_text(txt, encoding="utf-8")
    print(f"Patched {path}")

def patch_engineer(path: Path) -> None:
    if not path.exists():
        print(f"Skip missing {path}")
        return
    txt = path.read_text(encoding="utf-8")
    txt = ensure_imports(txt)

    if "def run_codex_on_task" in txt and "run_id = new_run_id()" not in txt:
        txt = re.sub(
            r"(def\s+run_codex_on_task\s*\(.*?\):\s*\n)",
            r"\1    run_id = new_run_id()\n    logger = TraceLogger()\n",
            txt,
            count=1,
            flags=re.S,
        )

    # Insert logs around the first subprocess.run(...) line (best-effort)
    if "subprocess.run" in txt and "phase': 'engineer_request'" not in txt:
        lines = txt.splitlines(True)
        new_lines = []
        inserted = False
        for line in lines:
            if (not inserted) and "subprocess.run" in line:
                indent = re.match(r"^\s*", line).group(0)
                new_lines.append(
                    indent
                    + "log_task(logger, run_id=run_id, initializer='PM', recipient='CODEX_ENGINEER', task=task, response='', task_id='', task_title='Codex task', meta={'phase': 'engineer_request'})\n"
                )
                new_lines.append(line)
                new_lines.append(
                    indent
                    + "log_task(logger, run_id=run_id, initializer='PM', recipient='CODEX_ENGINEER', task=task, response=(getattr(proc,'stdout','') + '\\n' + getattr(proc,'stderr','')) if 'proc' in locals() else '', task_id='', task_title='Codex task', meta={'phase': 'engineer_response', 'exit_code': getattr(proc,'returncode', None) if 'proc' in locals() else None})\n"
                )
                inserted = True
            else:
                new_lines.append(line)
        txt = ''.join(new_lines)

    path.write_text(txt, encoding="utf-8")
    print(f"Patched {path}")

patch_pm(Path("orchestration/pm.py"))
patch_engineer(Path("orchestration/codex_engineer.py"))
PY

echo "Patch complete."
echo "Ledger: monitoring/logs/task_ledger.jsonl"
echo "View: python scripts/show_traces.py --tail 20"
