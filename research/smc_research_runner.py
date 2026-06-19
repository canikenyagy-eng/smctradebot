from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings
from data.market_data import MarketDataCacheConfig, MarketDataClient
from research.smc_parity.feature_candidates import feature_candidate_report
from research.smc_parity.lookahead_audit import build_lookahead_report
from research.smc_parity.parity_engine import ParitySettings, compare_event_sets, internal_event_summary
from research.smc_parity.quality_diagnostics import (
    OrderBlockQualitySettings,
    QualityDiagnosticSettings,
    StructureQualitySettings,
    build_quality_diagnostics,
)
from research.smc_parity.reference_adapter import AdapterSettings, InternalSMCAdapter, build_reference_adapter
from research.smc_parity.reports import compact_pair_line, write_json_report


def _parse_pairs(raw: str) -> list[str]:
    return [item.strip().upper().replace("/", "") for item in raw.split(",") if item.strip()]


def _build_market_data(settings: Settings, *, history_limit: int, cache_only: bool) -> MarketDataClient:
    cache_mode = "cache_only" if cache_only else settings.market_data_cache_mode
    return MarketDataClient(
        history_limit=history_limit,
        data_source=settings.data_source,
        mt5_login=settings.mt5_login,
        mt5_password=settings.mt5_password,
        mt5_server=settings.mt5_server,
        mt5_path=settings.mt5_path,
        cache_config=MarketDataCacheConfig(
            enabled=settings.market_data_cache_enabled,
            cache_dir=settings.market_data_cache_dir,
            ttl_hours=settings.market_data_cache_ttl_hours,
            mode=cache_mode,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run research-only SMC parity and lookahead diagnostics.")
    parser.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--history-limit", type=int, default=600)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--reference-provider", default="smartmoneyconcepts")
    parser.add_argument("--require-reference", action="store_true")
    parser.add_argument("--swing-window", type=int, default=3)
    parser.add_argument("--max-structure-scan-bars", type=int, default=600)
    parser.add_argument("--max-liquidity-scan-bars", type=int, default=600)
    parser.add_argument("--fvg-lookback", type=int, default=600)
    parser.add_argument("--ob-lookback", type=int, default=600)
    parser.add_argument("--max-time-delta-bars", type=int, default=5)
    parser.add_argument("--max-level-distance-pips", type=float, default=5.0)
    parser.add_argument("--include-event-samples", action="store_true")
    parser.add_argument("--max-event-samples", type=int, default=10)
    parser.add_argument("--structure-min-break-pips", type=float, default=2.0)
    parser.add_argument("--structure-level-bucket-pips", type=float, default=2.0)
    parser.add_argument("--ob-min-strength", type=float, default=35.0)
    parser.add_argument("--ob-max-width-pips", type=float, default=20.0)
    parser.add_argument("--ob-max-age-bars", type=int, default=500)
    parser.add_argument("--output", default="reports/smc_research_report.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    pairs = _parse_pairs(args.pairs)
    timeframe = args.timeframe.upper()
    adapter_settings = AdapterSettings(
        swing_window=max(1, args.swing_window),
        max_structure_scan_bars=max(120, args.max_structure_scan_bars),
        max_liquidity_scan_bars=max(120, args.max_liquidity_scan_bars),
        fvg_lookback=max(50, args.fvg_lookback),
        ob_lookback=max(50, args.ob_lookback),
    )
    parity_settings = ParitySettings(
        max_time_delta_bars=max(0, args.max_time_delta_bars),
        max_level_distance_pips=max(0.0, args.max_level_distance_pips),
        include_event_samples=args.include_event_samples,
        max_event_samples=max(1, args.max_event_samples),
    )
    quality_settings = QualityDiagnosticSettings(
        structure=StructureQualitySettings(
            min_break_distance_pips=max(0.0, args.structure_min_break_pips),
            level_bucket_pips=max(0.1, args.structure_level_bucket_pips),
        ),
        order_block=OrderBlockQualitySettings(
            min_strength=max(0.0, args.ob_min_strength),
            max_width_pips=max(0.0, args.ob_max_width_pips),
            max_age_bars=max(1, args.ob_max_age_bars),
        ),
    )

    market_data = _build_market_data(settings, history_limit=args.history_limit, cache_only=args.cache_only)
    internal_adapter = InternalSMCAdapter(adapter_settings)
    reference_adapter = build_reference_adapter(args.reference_provider, adapter_settings)
    reference_status = reference_adapter.status()

    if args.require_reference and not reference_status.available:
        raise RuntimeError(f"Reference provider unavailable: {reference_status.reason}")

    pair_rows: list[dict[str, Any]] = []
    events_by_source: dict[str, list[Any]] = {"internal": []}
    if reference_status.available:
        events_by_source[reference_adapter.name] = []

    for pair in pairs:
        row: dict[str, Any] = {
            "pair": pair,
            "timeframe": timeframe,
            "error": None,
            "reference_status": reference_status.to_dict(),
        }
        try:
            frame = market_data.fetch_ohlcv(pair, timeframe, limit=args.history_limit)
            internal_events = internal_adapter.build_events(frame, pair=pair, timeframe=timeframe)
            events_by_source["internal"].extend(internal_events)
            row["bars"] = len(frame)
            row["internal_summary"] = internal_event_summary(pair=pair, timeframe=timeframe, events=internal_events)

            if reference_status.available:
                reference_events = reference_adapter.build_events(frame, pair=pair, timeframe=timeframe)
                events_by_source[reference_adapter.name].extend(reference_events)
                row["reference_summary"] = internal_event_summary(
                    pair=pair,
                    timeframe=timeframe,
                    events=reference_events,
                )
                row["parity"] = compare_event_sets(
                    pair=pair,
                    timeframe=timeframe,
                    internal_events=internal_events,
                    reference_events=reference_events,
                    settings=parity_settings,
                )
                row["quality_diagnostics"] = build_quality_diagnostics(
                    pair=pair,
                    frame_length=len(frame),
                    internal_events=internal_events,
                    reference_events=reference_events,
                    parity_settings=parity_settings,
                    settings=quality_settings,
                )
            else:
                row["reference_summary"] = None
                row["parity"] = None
                row["quality_diagnostics"] = build_quality_diagnostics(
                    pair=pair,
                    frame_length=len(frame),
                    internal_events=internal_events,
                    reference_events=[],
                    parity_settings=parity_settings,
                    settings=quality_settings,
                )
        except Exception as exc:
            row["error"] = str(exc)
        pair_rows.append(row)
        print(compact_pair_line(row), flush=True)

    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "pairs": pairs,
            "timeframe": timeframe,
            "history_limit": args.history_limit,
            "cache_only": args.cache_only,
            "reference_provider": args.reference_provider,
            "adapter": adapter_settings.__dict__,
            "parity": parity_settings.__dict__,
            "quality": quality_settings.to_dict(),
        },
        "reference_status": reference_status.to_dict(),
        "internal_status": internal_adapter.status().to_dict(),
        "pairs": pair_rows,
        "lookahead_audit": build_lookahead_report(events_by_source),
        "feature_candidates": feature_candidate_report(),
    }

    output_path = write_json_report(Path(args.output), report)
    print(f"Saved SMC research report: {output_path}", flush=True)


if __name__ == "__main__":
    main()
