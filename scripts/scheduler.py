"""
APScheduler-based sleep scheduler.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from time import sleep

from scripts.consolidate_sleep import run_sleep_batch


def main():
    sched = BackgroundScheduler(daemon=True)
    # daily 04:00
    sched.add_job(run_sleep_batch, 'cron', hour=4, minute=0)
    sched.start()
    print("Scheduler started. Press Ctrl+C to stop.")
    try:
        while True:
            sleep(3600)
    except KeyboardInterrupt:
        sched.shutdown()


if __name__ == "__main__":
    main()
