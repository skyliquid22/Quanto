#!/usr/bin/env python3
"""Run Financial Datasets screener queries."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict, Iterable, List

from infra.ingestion.adapters import FinancialDatasetsAdapter, FinancialDatasetsRESTClient


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Financial Datasets financials screener queries.")
    parser.add_argument(
        "--filters",
        nargs="+",
        required=True,
        help="Filter expressions like field:op:value (e.g. market_cap:gt:10000000).",
    )
    parser.add_argument("--limit", type=int, default=10, help="Max number of results to return.")
    return parser.parse_args(argv)


def _parse_filter(token: str) -> Dict[str, Any]:
    parts = token.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid filter '{token}', expected field:op:value")
    field, operator, raw_value = [part.strip() for part in parts]
    if not field or not operator:
        raise ValueError(f"Invalid filter '{token}', expected field and operator")
    value: Any = raw_value
    if operator == "in":
        if "," in raw_value:
            value = [item.strip() for item in raw_value.split(",") if item.strip()]
    else:
        try:
            if "." in raw_value:
                value = float(raw_value)
            else:
                value = int(raw_value)
        except ValueError:
            value = raw_value
    return {"field": field, "operator": operator, "value": value}


async def _run(filters: List[Dict[str, Any]], limit: int) -> int:
    api_key = os.environ.get("FINANCIALDATASETS_API_KEY")
    if not api_key:
        raise SystemExit("FINANCIALDATASETS_API_KEY must be set")
    client = FinancialDatasetsRESTClient(api_key)
    adapter = FinancialDatasetsAdapter(rest_client=client)
    try:
        response = await adapter.search_financials_screener(filters=filters, limit=limit)
    finally:
        await adapter.aclose()
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    filters = [_parse_filter(token) for token in args.filters]
    return asyncio.run(_run(filters, args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
