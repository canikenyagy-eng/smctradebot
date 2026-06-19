from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.smc_parity.event_schema import json_safe


def write_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return output


def compact_pair_line(row: dict[str, Any]) -> str:
    internal = row.get("internal_summary", {})
    parity = row.get("parity")
    reference_status = row.get("reference_status", {})
    pair = row.get("pair", "UNKNOWN")
    timeframe = row.get("timeframe", "NA")
    base = (
        f"{pair} {timeframe} | internal_events={internal.get('event_count', 0)} "
        f"counts={internal.get('counts', {})}"
    )
    if parity:
        return (
            f"{base} | ref_events={parity.get('reference_event_count', 0)} "
            f"matched={parity.get('matched_event_count', 0)} "
            f"direction_agreement={parity.get('overall_direction_agreement', 0.0)}"
        )
    return f"{base} | reference_available={reference_status.get('available', False)}"
