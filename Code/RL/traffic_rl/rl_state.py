from config import MAX_QUEUE_BUCKET, MAX_WAIT_BUCKET, STATE_BUCKET_SIZE, WAIT_BUCKET_SIZE


PHASE_TO_ACTION = {
    0: 0,
    2: 0,
    1: 1,
    3: 1,
}

ACTION_TO_PHASE = {
    0: 0,
    1: 1,
}

ACTION_TO_DIRECTIONS = {
    0: ("right", "left"),
    1: ("down", "up"),
}


def bucketize(value, bucket_size, max_bucket):
    bucket = int(value // bucket_size)
    return min(bucket, max_bucket)


def aggregate_direction_queues(direction_counts):
    ns_queue = int(direction_counts.get("up", 0) + direction_counts.get("down", 0))
    ew_queue = int(direction_counts.get("left", 0) + direction_counts.get("right", 0))
    return {"ns": ns_queue, "ew": ew_queue}


def build_state(direction_counts, current_phase, yellow_active, waiting_times=None):
    waiting_times = waiting_times or {}
    grouped_queues = aggregate_direction_queues(direction_counts)
    ns_wait = int(waiting_times.get("up", 0) + waiting_times.get("down", 0))
    ew_wait = int(waiting_times.get("left", 0) + waiting_times.get("right", 0))
    active_action = PHASE_TO_ACTION.get(current_phase, 0)
    return (
        bucketize(grouped_queues["ns"], STATE_BUCKET_SIZE, MAX_QUEUE_BUCKET),
        bucketize(grouped_queues["ew"], STATE_BUCKET_SIZE, MAX_QUEUE_BUCKET),
        active_action,
        1 if yellow_active else 0,
        bucketize(ns_wait, WAIT_BUCKET_SIZE, MAX_WAIT_BUCKET),
        bucketize(ew_wait, WAIT_BUCKET_SIZE, MAX_WAIT_BUCKET),
    )