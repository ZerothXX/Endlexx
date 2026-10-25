"""Single local worker with explicit restart recovery; also prevents duplicate GPU workers."""

import sys

from filelock import FileLock, Timeout

from endlexx.config import settings
from endlexx.db import init_db
from endlexx.queue import celery_app, recover_tasks


def main():
    cfg = settings()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(cfg.data_dir / "worker.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        print("Endlexx worker is already running", flush=True)
        return
    try:
        init_db()
        print(f"Recovered {recover_tasks()} pending tasks", flush=True)
        celery_app.worker_main(
            [
                "worker",
                "--pool=solo",
                "--concurrency=1",
                "--loglevel=INFO",
                "--hostname=endlexx-local@%h",
            ]
        )
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
