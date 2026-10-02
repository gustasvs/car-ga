import random

import numpy as np

def get_smooth_random_values(length, min_value=0, max_value=1):
    values = [random.uniform(min_value, max_value) for _ in range(length)]
    for i in range(1, length):
        values[i] = (values[i] + values[i-1]) / 2
    return values


def generate_road(slopes, n, smoothness=1.0):
    slopes = np.asarray(slopes)
    road = np.empty(n, dtype=slopes.dtype)
    road[0] = slopes[np.argmin(np.abs(slopes))]

    # higher - longer hills
    MEAN_REV = 0.6

    for i in range(1, n):
        current = road[i - 1]
        target = current * MEAN_REV
        weights = np.exp(-((slopes - target) ** 2) / (2 * smoothness**2))
        road[i] = np.random.choice(slopes, p=weights / weights.sum())

    return road