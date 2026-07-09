# Phase 5 Live Health Monitor

The live health monitor detects whether the signal engine is scanning normally. It is monitoring-only and does not change signal generation, Telegram signal delivery, risk logic, or market data logic.

## Components

- `main.py` writes a heartbeat JSON file after engine start, scan start, scan completion, and scan failure.
- `research.live_health_check` checks the heartbeat age/status.
- `scripts/run_live_health_check.sh` runs the checker and logs output.
- `launchd/com.smc.healthcheck.plist` runs the checker every 5 minutes.
- Telegram alerting is cooldown-protected.

## Config

Add or keep these values in `.env`:

```env
ENABLE_LIVE_HEARTBEAT=1
LIVE_HEARTBEAT_PATH=logs/live_heartbeat.json
HEALTH_MAX_SCAN_AGE_MINUTES=15
ENABLE_HEALTH_ALERTS=1
HEALTH_ALERT_STATE_PATH=logs/live_health_alert_state.json
HEALTH_ALERT_COOLDOWN_MINUTES=60
```

`ENABLE_HEALTH_ALERTS=1` allows Telegram health alerts. Alerts are only sent when the bot is unhealthy or when it recovers after an unhealthy state.

## Manual Check

```bash
cd "/Users/kanannagiev/Documents/New project/project"
source .venv/bin/activate
python -m research.live_health_check --alert --output logs/live_health_status.json
```

Without Telegram alerts:

```bash
python -m research.live_health_check --no-alert
```

## Install Launchd Monitor

```bash
cd "/Users/kanannagiev/Documents/New project/project"
bash launchd/install_health_check.sh
```

The launch agent runs every 300 seconds.

Check status:

```bash
launchctl list | grep com.smc.healthcheck
```

Uninstall:

```bash
cd "/Users/kanannagiev/Documents/New project/project"
bash launchd/uninstall_health_check.sh
```

## Logs

Project log:

```bash
tail -f logs/live_health_check.out.log
```

Latest status JSON:

```bash
cat logs/live_health_status.json
```

Launchd stdout/stderr:

```bash
tail -f "$HOME/Library/Logs/SMCSignalEngine/health-check.out.log"
tail -f "$HOME/Library/Logs/SMCSignalEngine/health-check.err.log"
```

Heartbeat file:

```bash
cat logs/live_heartbeat.json
```

## Health Rules

The bot is unhealthy when:

- heartbeat file is missing
- heartbeat file is unreadable
- last scan failed
- heartbeat age is greater than `HEALTH_MAX_SCAN_AGE_MINUTES`

Default stale threshold is 15 minutes. With a 5-minute scanner, this allows a small safety buffer before alerting.

## Alert Cooldown

`HEALTH_ALERT_COOLDOWN_MINUTES=60` prevents repeated Telegram spam for the same unhealthy state. A new alert is sent when:

- first unhealthy state is detected
- unhealthy reason/status changes
- cooldown expires
- bot recovers after being unhealthy
