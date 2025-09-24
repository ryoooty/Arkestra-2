# Arkestra-2

## Sleep batch scheduler

Run the automated sleep consolidation loop with APScheduler:

```bash
python scripts/scheduler.py
```

The job will run `scripts.consolidate_sleep.run_sleep_batch` every day at 04:00 server time. Press `Ctrl+C` to stop the background scheduler.

### Cron fallback

If APScheduler is unavailable, register an equivalent cron entry instead:

```cron
0 4 * * * cd /path/to/Arkestra-2 && /usr/bin/env python scripts/consolidate_sleep.py
```

This cron entry will execute the sleep batch once per day at 04:00.