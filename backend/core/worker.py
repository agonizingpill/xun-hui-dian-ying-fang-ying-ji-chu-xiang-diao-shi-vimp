import json
from pathlib import Path
from traceback import print_exc

from core.queueing import QUEUE, broker
from domain import quality
from domain.learning import MODEL_VERSION, calculate

pulse = Path('/tmp/compute-heartbeat')


def job_version(job):
    return (job.get('model_version') or job.get('input', {}).get('model_version')
            or MODEL_VERSION)


def process_job(job):
    """Calculate with the version pinned when the order was accepted."""
    version = job_version(job)
    try:
        return quality.calculate_version(job['input'], version)
    except LookupError:
        # Keep old queue items usable if a registry row is unavailable; new
        # versions are never deleted, so this is mainly for restored snapshots.
        if version == MODEL_VERSION:
            return calculate(job['input'], model_version=version)
        raise


process_item = process_job
handle_job = process_job
run_job = process_job


def requeue_interrupted():
    for raw in broker.lrange(f'{QUEUE}:processing', 0, -1):
        with broker.pipeline() as pipe:
            pipe.lpush(QUEUE, raw)
            pipe.lrem(f'{QUEUE}:processing', 1, raw)
            pipe.execute()


recover_interrupted = requeue_interrupted
requeue_processing = requeue_interrupted


def serve():
    while True:
        broker.ping()
        pulse.touch()
        item = broker.blmove(QUEUE, f'{QUEUE}:processing', timeout=2, src='LEFT', dest='RIGHT')
        if item is None:
            continue
        job = json.loads(item)
        status = {'owner': job['owner'], 'input': job['input']}
        try:
            status.update(status='ready', result=process_job(job))
        except ValueError as exc:
            status.update(status='rejected', detail=str(exc))
        except Exception:
            print_exc()
            status.update(status='rejected', detail='计算失败，请检查输入和 worker 日志')
        with broker.pipeline() as pipe:
            pipe.set(f'compute:{job["token"]}', json.dumps(status, allow_nan=False), ex=86400)
            pipe.lrem(f'{QUEUE}:processing', 1, item)
            pipe.execute()
        pulse.touch()

if __name__ == '__main__':
    requeue_interrupted()
    serve()
