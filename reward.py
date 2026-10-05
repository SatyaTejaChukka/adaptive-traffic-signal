from config import (
    CONGESTION_PENALTY_WEIGHT,
    CONGESTION_THRESHOLD,
    ILLEGAL_STATE_PENALTY,
    QUEUE_CLEAR_BONUS,
    SWITCH_PENALTY,
)


def compute_reward(queue_counts, waiting_times, phase_changed, previous_queue_counts=None, illegal_state=False):
    total_queue = float(sum(queue_counts.values()))
    total_wait = float(sum(waiting_times.values()))
    congestion_penalty = 0.0
    for queue_length in queue_counts.values():
        overflow = max(0.0, float(queue_length) - CONGESTION_THRESHOLD)
        congestion_penalty += overflow * CONGESTION_PENALTY_WEIGHT

    reward = -(total_queue + total_wait + congestion_penalty)

    if phase_changed:
        reward -= SWITCH_PENALTY

    if previous_queue_counts:
        previous_total = float(sum(previous_queue_counts.values()))
        if total_queue < previous_total:
            reward += QUEUE_CLEAR_BONUS * (previous_total - total_queue)

    if illegal_state:
        reward -= ILLEGAL_STATE_PENALTY

    return reward
