import sys
import argparse
import random
import numpy as np
import math
from tqdm import tqdm

from helpers.randomization import generate_road

# --- vides parametri ---
ROAD_LENGTH = 17
MAX_COMBINATIONS = 500_000_000_000
MAX_SLOPE_DEG = 40
SLOPE_STEP = 5
SLOPES = list(range(-MAX_SLOPE_DEG, MAX_SLOPE_DEG + 1, SLOPE_STEP))
# one segment length (m)
DELTA_X = 10


# - physics -
K_DRAG = 0.02
K_ROLL = 0.005

THROTTLE_AMOUNTS = [0, 1, 5, 15]

# --- ga parametri ---
POPULATION_SIZE = 250
GA_STEPS = 1000
# top X vienmēr pāriet uz nākamo paaudzi
ELITE_COUNT = POPULATION_SIZE // 20
# šie tiek izvēlēti ar svērto nejaušību kā `vecāki`
RANDOM_SELECTION_COUNT = POPULATION_SIZE // 10

BENCHMARK_COUNTERS = {
    "fitness_evaluations": 0,
    "segment_evaluations": 0,
}

# --- nejaušības parametri ---
SEED = 6
random.seed(SEED)
np.random.seed(SEED)

# --- types ---
type Individual = list[float]
type Population = list[Individual]


class GA:
    def __init__(self, road):
        self.road = road
        self.population: Population = []
        self.last_scored = []
        self.iterations = 0

    # 1. uz nejaušību ģenerēta populācija
    def init_population(self, size):
        self.population = [
            [random.choice(THROTTLE_AMOUNTS) for _ in range(len(self.road) - 1)]
            for _ in range(size)
        ]

    def step(self):
        elite, selected, current_best = self.select()
        next_generation = elite[:]
        for _ in range(POPULATION_SIZE - len(elite)):
            parent1, parent2 = random.sample(selected, 2)
            child = self.crossover(parent1, parent2)
            mutated_child = self.mutate(child)
            next_generation.append(mutated_child)
        self.population = next_generation
        return current_best

    # 2. select/izvēles solis
    def select(self):
        scored = sorted(
            (
                self.eval_individual_details(individual)
                for individual in self.population
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        self.last_scored = scored

        elite = [result[0] for result in scored[:ELITE_COUNT]]

        selected = random.choices(
            scored,
            weights=[result[1] for result in scored],
            k=RANDOM_SELECTION_COUNT,
        )

        return elite, [result[0] for result in selected], scored[0]

    # smaller fn's

    def crossover(self, parent1, parent2) -> Individual:
        start = random.randint(0, len(self.road) - 2)
        end = random.randint(start + 1, len(self.road) - 1)
        return parent1[:start] + parent2[start:end] + parent1[end:]

    def mutate(self, individual) -> Individual:
        muations = 5
        for _ in range(random.randint(1, muations)):
            where = random.randint(0, len(self.road) - 2)
            individual[where] = random.choice(THROTTLE_AMOUNTS)
        return individual

    def eval_individual(self, individual):
        return self.eval_individual_details(individual)[1]

    def eval_individual_details(self, individual):
        BENCHMARK_COUNTERS["fitness_evaluations"] += 1
        self.iterations += 1
        fuel, distance, finished = measure_individual(self.road, individual)
        fitness = calculate_fitness(self.road, fuel, distance, finished)
        return individual, fitness, fuel, finished

    def get_best_fitness(self) -> float:
        return self.last_scored[0][1]

    def get_best_individual(self) -> Individual:
        return self.last_scored[0][0]


def calculate_fitness(road, fuel, distance, finished):
    if finished:
        finish_reward = len(road) * DELTA_X * 5
        return DELTA_X * (len(road) - 1) + finish_reward - fuel
    return distance


def measure_individual(
    road, individual, segment_count=None
) -> tuple[float, float, bool]:
    # returns total fuel consumption, how far the individual got, and if they finished the road
    total_fuel_consumption = 0

    # in loop this means speed entering segment i
    velocity = 0

    if segment_count is None:
        segment_count = len(individual)

    # road i-th segment describes terrain between i and i + 1
    for i in range(segment_count):
        BENCHMARK_COUNTERS["segment_evaluations"] += 1

        # for simplicity of equations, car mass is 1kg

        # road[i] holds the angle
        angle = math.radians(road[i])

        # throttle to active force (in N)
        f_engine = individual[i] ** 1.5

        # hill caused force
        f_grade = 9.81 * math.sin(angle)

        # rolling resistance
        f_roll = K_ROLL * 9.81 * math.cos(angle)

        # aerodynamic drag
        # simple
        # f_drag = K_DRAG * velocity * velocity
        # predicted
        predicted_end_velocity = math.sqrt(
            max(
                0.0,
                velocity**2
                + 2 * (f_engine - f_grade - f_roll - K_DRAG * velocity**2) * DELTA_X,
            )
        )
        f_drag = K_DRAG * ((velocity + predicted_end_velocity) / 2) ** 2

        # total force
        # IMPORTANT: since mass is 1kg can directly be used
        force = f_engine - f_grade - f_roll - f_drag

        end_speed_squared = velocity * velocity + 2 * force * DELTA_X

        # if at any point the there isnt enough velocity to continue, break
        if end_speed_squared < 0:
            stop_distance = velocity * velocity / (-2 * force)
            return total_fuel_consumption, DELTA_X * i + stop_distance, False

        end_velocity = math.sqrt(end_speed_squared)

        if velocity + end_velocity == 0:
            return total_fuel_consumption, DELTA_X * i, False

        segment_time = DELTA_X * 2 / (velocity + end_velocity)

        FUEL_K = 0.1

        fuel_rate = FUEL_K * individual[i] ** 1.2
        total_fuel_consumption += fuel_rate * segment_time

        velocity = end_velocity

    return total_fuel_consumption, DELTA_X * segment_count, True


def brute_force_search(road, show_progress=True):
    length = len(road) - 1
    choices = sorted(THROTTLE_AMOUNTS)
    total = len(choices) ** length

    if total > MAX_COMBINATIONS:
        return None

    best_individual = None
    best_fitness = -math.inf
    best_fuel = math.inf
    best_result_fuel = None
    best_finished = False
    path = []

    with tqdm(
        total=total, desc="Finding best solution", disable=not show_progress
    ) as pbar:

        def search():
            nonlocal best_individual, best_fitness, best_fuel
            nonlocal best_result_fuel, best_finished

            if path:
                fuel, distance, survived = measure_individual(
                    road,
                    path,
                    segment_count=len(path),
                )
                remaining = length - len(path)

                # A leaf has already been simulated above; do not evaluate it twice.
                if len(path) == length:
                    BENCHMARK_COUNTERS["fitness_evaluations"] += 1
                    fitness = calculate_fitness(road, fuel, distance, survived)

                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = path.copy()
                        best_result_fuel = fuel
                        best_finished = survived
                        if survived:
                            best_fuel = fuel

                    pbar.update(1)
                    return

                if not survived or fuel >= best_fuel:
                    pbar.update(len(choices) ** remaining)
                    return

            for throttle in choices:
                path.append(throttle)
                search()
                path.pop()

        search()

    if best_individual is None:
        return None

    return {
        "individual": best_individual,
        "fitness": best_fitness,
        "fuel": best_result_fuel,
        "finished": best_finished,
    }


def ga_search(road):
    ga = GA(road)
    ga.init_population(POPULATION_SIZE)
    best_result = None

    for _generation in range(GA_STEPS):
        current_best = ga.step()
        if best_result is None or current_best[1] > best_result[1]:
            best_result = current_best
            
    assert best_result is not None, "no GA result, check if GA_STEPS is positive"
    individual, fitness, fuel, finished = best_result
    return {
        "individual": individual.copy(),
        "fitness": fitness,
        "fuel": fuel,
        "finished": finished,
    }


def optimise_ga_on_road(road, visualize_results=True):

    ga_result = ga_search(road)

    if visualize_results:
        from helpers.display_run import visualize

        visualize(
            road=road,
            throttles=ga_result["individual"],
            delta_x=DELTA_X,
            k_drag=K_DRAG,
            k_roll=K_ROLL,
            zoom=True,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--example",
        nargs="?",
        const=20,
        type=int,
        default=None,
        help="Fit GA on the given road length and visualise it (default length: 20)",
    )

    args = parser.parse_args()

    if args.example is None:
        from benchmark import main as run_benchmarks

        run_benchmarks()
        sys.exit()

    road = generate_road(SLOPES, args.example, 40)
    print(road)
    optimise_ga_on_road(road, True)
