# Arkestra-2

## Sleep batch scheduler

To run the automated sleep consolidation every day at 04:00, install `APScheduler` and execute the background scheduler:

```bash
pip install apscheduler
python scripts/scheduler.py
```

If installing APScheduler is not an option, add an equivalent cron entry as a fallback (adjust the project path as needed):

```cron
0 4 * * * /usr/bin/env python /workspace/Arkestra-2/scripts/consolidate_sleep.py
```
