# VPS Deployment Runbook

This runbook prepares and runs the SMC Forex Signal Engine on Ubuntu 24.04 with `systemd`.

The bot is Telegram-only. No auto-trading is enabled by these scripts.

## Defaults

| Setting | Default |
| --- | --- |
| App user | `tradebot` |
| App dir | `/opt/smc-signal-engine` |
| Git branch | `main` |
| Main service | `smc-signal-engine.service` |
| Health timer | `smc-healthcheck.timer` every `5min` |
| Daily report timer | `smc-forward-report.timer` at `Mon..Fri 21:30:00` server local time |

All defaults can be overridden with environment variables:

```bash
export SMC_APP_USER=tradebot
export SMC_APP_DIR=/opt/smc-signal-engine
export SMC_REPO_URL=git@github.com:canikenyagy-eng/smctradebot.git
export SMC_BRANCH=main
```

## 1. Bootstrap the server

After SSH access works, run on the VPS:

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates
```

Clone the repo. For a private GitHub repo, prefer an SSH deploy key:

```bash
git clone git@github.com:canikenyagy-eng/smctradebot.git /tmp/smctradebot
cd /tmp/smctradebot
sudo SMC_REPO_URL=git@github.com:canikenyagy-eng/smctradebot.git scripts/vps_bootstrap.sh
```

If the repo is already present in `/opt/smc-signal-engine`, rerun bootstrap safely:

```bash
sudo scripts/vps_bootstrap.sh
```

### Alternative: sync from Mac without GitHub auth on VPS

If GitHub deploy keys are not ready yet, sync the local project directly from Mac after SSH works:

```bash
cd "/Users/kanannagiev/Documents/New project/project"
SMC_VPS_HOST=tradebot@45.137.153.215 \
SMC_SSH_KEY="$HOME/.ssh/smc_vps" \
scripts/vps_sync_from_mac.sh
```

Then on the VPS:

```bash
cd /opt/smc-signal-engine
sudo SMC_SKIP_GIT_PULL=1 scripts/vps_deploy.sh
```

## 2. Configure `.env`

On the VPS:

```bash
cd /opt/smc-signal-engine
sudo -u tradebot cp .env.vps.example .env
sudo chmod 600 .env
sudo chown tradebot:tradebot .env
sudo -u tradebot nano .env
```

Fill at minimum:

```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
ITICK_API_KEY=...
```

Do not commit `.env`.

## 3. Deploy and install systemd units

First deploy without starting live services:

```bash
cd /opt/smc-signal-engine
sudo scripts/vps_deploy.sh
```

If the env check passes, start services:

```bash
sudo SMC_START_SERVICES=1 scripts/vps_deploy.sh
```

This renders and installs:

```text
/etc/systemd/system/smc-signal-engine.service
/etc/systemd/system/smc-healthcheck.service
/etc/systemd/system/smc-healthcheck.timer
/etc/systemd/system/smc-forward-report.service
/etc/systemd/system/smc-forward-report.timer
```

## 4. Verify live operation

```bash
cd /opt/smc-signal-engine
scripts/vps_status.sh
sudo systemctl status smc-signal-engine --no-pager -l
sudo journalctl -u smc-signal-engine -f
```

Market data checks:

```bash
.venv/bin/python -m research.itick_websocket_shadow_report --recent-minutes 60
.venv/bin/python -m research.live_bar_builder_report --recent-minutes 60 --max-bar-age-seconds 180
.venv/bin/python -m research.live_bar_provider_check \
  --pairs EURUSD,EURJPY,CADJPY \
  --timeframes M5,M15,H1 \
  --limit 10 \
  --require-live-overlay
.venv/bin/python -m research.live_health_check --alert --output logs/live_health_status.json
```

Telegram test can be done through the existing project helpers or by observing service startup/health alerts.

## 5. Useful service commands

```bash
sudo systemctl restart smc-signal-engine
sudo systemctl stop smc-signal-engine
sudo systemctl start smc-signal-engine
sudo systemctl status smc-signal-engine --no-pager -l
sudo journalctl -u smc-signal-engine -n 200 --no-pager
sudo journalctl -u smc-signal-engine -f
sudo systemctl list-timers 'smc-*'
```

Run daily report manually:

```bash
cd /opt/smc-signal-engine
sudo -u tradebot .venv/bin/python -m research.daily_live_forward_report --telegram
```

Run healthcheck manually:

```bash
cd /opt/smc-signal-engine
sudo -u tradebot .venv/bin/python -m research.live_health_check --alert --output logs/live_health_status.json
```

## 6. Safe Mac-to-VPS cutover

1. Start VPS service.
2. Confirm iTick WebSocket quotes are fresh for 30-60 minutes.
3. Confirm live bars are fresh for M5/M15/H1.
4. Confirm healthcheck is OK.
5. Confirm Telegram startup/health messages arrive.
6. Stop the Mac launchd bot to avoid duplicate Telegram signals.

Mac stop commands:

```bash
cd "/Users/kanannagiev/Documents/New project/project"
bash launchd/uninstall.sh
bash launchd/uninstall_health_check.sh
bash launchd/uninstall_forward_reports.sh
```

## 7. Security hardening after SSH is stable

Recommended after deployment:

```bash
sudo apt-get install -y ufw fail2ban unattended-upgrades
sudo ufw allow OpenSSH
sudo ufw enable
sudo systemctl enable --now fail2ban
sudo dpkg-reconfigure -plow unattended-upgrades
```

Then disable password login after key-based SSH is confirmed:

```bash
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak
sudo sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
sudo sshd -t
sudo systemctl restart ssh
```

Keep one working SSH session open while testing a new one.

## 8. Troubleshooting

Check systemd logs:

```bash
sudo journalctl -u smc-signal-engine -n 200 --no-pager
sudo journalctl -u smc-healthcheck -n 100 --no-pager
sudo journalctl -u smc-forward-report -n 100 --no-pager
```

If `.env` is invalid:

```bash
cd /opt/smc-signal-engine
sudo -u tradebot .venv/bin/python - <<'PY'
from config import Settings
print(Settings.from_env())
PY
```

If the feed is stale, do not loosen strategy filters first. Check in this order:

1. `research.itick_websocket_shadow_report`
2. `research.live_bar_builder_report`
3. `research.live_bar_provider_check`
4. `research.market_data_redundancy_report`
5. `research.live_health_check`
