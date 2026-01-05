from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="monitoring/logs/task_ledger.jsonl")
    p.add_argument("--tail", type=int, default=20)
    p.add_argument("--contains", default=None)
    args = p.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"Log not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    if args.tail:
        lines = lines[-args.tail :]

    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if args.contains:
            hay = (obj.get("task","") + "\n" + obj.get("response","")).lower()
            if args.contains.lower() not in hay:
                continue

        ts = obj.get("ts", "")
        init = obj.get("initializer", "")
        to = obj.get("recipient", "")
        rid = obj.get("run_id", "")
        tid = obj.get("task_id", "")
        title = obj.get("task_title", "")
        err = obj.get("error")
        print(f"- {ts} | {init} -> {to} | run={rid} | task_id={tid} | {title}")
        if err:
            print(f"  ERROR: {err}")


if __name__ == "__main__":
    main()
