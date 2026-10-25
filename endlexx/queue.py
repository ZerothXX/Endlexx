from concurrent.futures import ThreadPoolExecutor

from celery import Celery

from .config import settings

celery_app = Celery("endlexx", broker=settings().redis_url, backend=settings().redis_url)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    broker_transport_options={"visibility_timeout": 7200},
    task_track_started=True,
)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="endlexx")


@celery_app.task(name="endlexx.run", soft_time_limit=2100, time_limit=2200)
def execute(task_id: str):
    from .agent import run_task

    run_task(task_id)


def enqueue(task_id: str):
    if settings().task_backend == "celery":
        execute.apply_async(args=[task_id], task_id=task_id)
    else:
        executor.submit(execute.run, task_id)


def recover_tasks() -> int:
    """Called only by the single owning worker after obtaining its process lock."""
    from sqlalchemy import select

    from .db import Session, Task

    with Session() as db:
        pending = db.scalars(select(Task).where(Task.status.in_(["running", "queued"]))).all()
        ids = []
        for task in pending:
            if task.cancel_requested:
                task.status, task.phase = "cancelled", "已取消"
                continue
            task.status, task.phase = "queued", "服务恢复，等待继续"
            ids.append(task.id)
        db.commit()
    for task_id in ids:
        enqueue(task_id)
    return len(ids)
