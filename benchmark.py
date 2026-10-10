import csv
import random
import time

import numpy as np

from game import (
    BENCHMARK_COUNTERS,
    MAX_COMBINATIONS,
    SEED,
    SLOPES,
    THROTTLE_AMOUNTS,
    brute_force_search,
    ga_search,
)
from helpers.randomization import generate_road


ROAD_LENGTHS = list(range(5, 51))
ROADS_PER_LENGTH = 3
GA_RUNS = 3
BENCHMARK_CSV = "benchmark_results.csv"

BENCHMARK_FIELDS = [
    "road_length",
    "segments",
    "road_seed",
    "algorithm",
    "run",
    "ga_seed",
    "combination_count",
    "elapsed_seconds",
    "fitness_evaluations",
    "segment_evaluations",
    "finished",
    "fuel",
    "fitness",
    "optimal_fuel",
    "fuel_gap_percent",
]


def reset_benchmark_counters():
    BENCHMARK_COUNTERS["fitness_evaluations"] = 0
    BENCHMARK_COUNTERS["segment_evaluations"] = 0


def make_benchmark_row(
    road_length, road_seed, algorithm, run, ga_seed, combination_count
):
    return {
        "road_length": road_length,
        "segments": road_length - 1,
        "road_seed": road_seed,
        "algorithm": algorithm,
        "run": run,
        "ga_seed": ga_seed,
        "combination_count": combination_count,
        "elapsed_seconds": None,
        "fitness_evaluations": 0,
        "segment_evaluations": 0,
        "finished": None,
        "fuel": None,
        "fitness": None,
        "optimal_fuel": None,
        "fuel_gap_percent": None,
    }


def benchmark_road(road, road_seed, ga_runs=GA_RUNS):
    road_length = len(road)
    combination_count = len(THROTTLE_AMOUNTS) ** (road_length - 1)
    rows = []
    optimal_fuel = None

    brute_row = make_benchmark_row(
        road_length, road_seed, "brute_force", 1, None, combination_count
    )
    if combination_count <= MAX_COMBINATIONS:
        reset_benchmark_counters()
        started = time.perf_counter()
        brute_result = brute_force_search(road, show_progress=False)
        brute_row["elapsed_seconds"] = time.perf_counter() - started
        brute_row.update(BENCHMARK_COUNTERS)

        if brute_result is not None:
            brute_row.update(
                {
                    "finished": brute_result["finished"],
                    "fuel": brute_result["fuel"],
                    "fitness": brute_result["fitness"],
                }
            )
            if brute_result["finished"]:
                optimal_fuel = brute_result["fuel"]
                brute_row["optimal_fuel"] = optimal_fuel
                brute_row["fuel_gap_percent"] = 0.0
    rows.append(brute_row)

    for run in range(1, ga_runs + 1):
        ga_seed = road_seed * 1000 + run
        random.seed(ga_seed)
        np.random.seed(ga_seed)
        reset_benchmark_counters()

        started = time.perf_counter()
        ga_result = ga_search(road)
        elapsed_seconds = time.perf_counter() - started

        ga_row = make_benchmark_row(
            road_length, road_seed, "ga", run, ga_seed, combination_count
        )
        ga_row.update(BENCHMARK_COUNTERS)
        ga_row.update(
            {
                "elapsed_seconds": elapsed_seconds,
                "finished": ga_result["finished"],
                "fuel": ga_result["fuel"],
                "fitness": ga_result["fitness"],
                "optimal_fuel": optimal_fuel,
            }
        )
        if ga_result["finished"] and optimal_fuel is not None and optimal_fuel != 0:
            ga_row["fuel_gap_percent"] = (
                (ga_result["fuel"] - optimal_fuel) / optimal_fuel * 100
            )
        rows.append(ga_row)

    return rows


def write_benchmark_csv(rows, output_path=BENCHMARK_CSV):
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = []
    for road_length in ROAD_LENGTHS:
        for road_run in range(ROADS_PER_LENGTH):
            road_seed = SEED + road_length * 1000 + road_run
            random.seed(road_seed)
            np.random.seed(road_seed)
            road = generate_road(SLOPES, road_length, 40)
            print(f"Benchmarking road length {road_length}, road {road_run + 1}/{ROADS_PER_LENGTH}")
            rows.extend(benchmark_road(road, road_seed, GA_RUNS))
            write_benchmark_csv(rows, BENCHMARK_CSV)
            print(
                f"Benchmarked road length {road_length}, "
                f"road {road_run + 1}/{ROADS_PER_LENGTH}"
            )
    print(f"Saved {len(rows)} benchmark rows to {BENCHMARK_CSV}")
    return rows


if __name__ == "__main__":
    main()
