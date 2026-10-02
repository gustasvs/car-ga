import random
import numpy as np
import math
from itertools import product
from tqdm import tqdm

from helpers.display_run import visualize
from helpers.randomization import generate_road, get_smooth_random_values

# --- vides parametri ---
ROAD_LENGTH = 10
MAX_SLOPE_DEG = 40; SLOPE_STEP = 5; 
SLOPES = list(range(-MAX_SLOPE_DEG, MAX_SLOPE_DEG + 1, SLOPE_STEP))
# one segment length (m)
DELTA_X = 10


# - physics -
K_DRAG = 0.02
K_ROLL = 0.005

THROTTLE_AMOUNTS = [0, 1, 5, 20]

# --- ga parametri ---
POPULATION_SIZE = 200
# top X vienmēr pāriet uz nākamo paaudzi
ELITE_COUNT = POPULATION_SIZE // 20
# šie tiek izvēlēti ar svērto nejaušību kā `vecāki`
RANDOM_SELECTION_COUNT = POPULATION_SIZE // 10
FINISH_REWARD = ROAD_LENGTH * DELTA_X * 5


# --- nejaušības parametri ---
SEED = 6
random.seed(SEED)
np.random.seed(SEED)

# --- types ---
type Population = list[list[float]]


class GA:
    def __init__(self, road):
        self.road = road
        self.population: Population = []
    
    # 1. uz nejaušību ģenerēta populācija 
    def init_population(self, size):
        self.population = [[random.choice(THROTTLE_AMOUNTS) for _ in range(ROAD_LENGTH - 1)] for _ in range(size)]
    
    def step(self):
        elite, selected = self.select()
        next_generation = elite[:]
        for _ in range(POPULATION_SIZE - len(elite)):
            parent1, parent2 = random.sample(selected, 2)
            child = self.crossover(parent1, parent2)
            mutated_child = self.mutate(child)
            next_generation.append(mutated_child)
        self.population = next_generation
    
    # 2. select/izvēles solis
    def select(self) -> tuple[Population, Population]:
        scored = sorted(
            ((individual, self.eval_individual(individual))
            for individual in self.population),
            key=lambda x: x[1],
            reverse=True,
        )

        elite = [individual for individual, _score in scored[:ELITE_COUNT]]

        selected = random.choices(
            scored,
            weights=[score for _, score in scored],
            k=RANDOM_SELECTION_COUNT,
        )

        return elite, [individual for individual, _ in selected]
    
    # smaller fn's
    
    def crossover(self, parent1, parent2) -> list[float]:
        start = random.randint(0, ROAD_LENGTH - 2)
        end = random.randint(start + 1, ROAD_LENGTH - 1)
        return parent1[:start] + parent2[start:end] + parent1[end:]
    
    def mutate(self, individual) -> list[float]:
        muations = 5
        for _ in range(random.randint(1, muations)):
            where = random.randint(0, ROAD_LENGTH - 2)
            individual[where] = random.choice(THROTTLE_AMOUNTS)
        return individual
        
    def eval_individual(self, individual) -> float:
        fuel, distance, finished = measure_individual(self.road, individual)
        if finished:
            return DELTA_X * (ROAD_LENGTH - 1) + FINISH_REWARD - fuel
        else:
            return distance
        
    def get_best_fitness(self) -> float:
        return max(self.eval_individual(individual) for individual in self.population)
    
    def get_best_individual(self) -> list[float]:
        return max(self.population, key=self.eval_individual)
    

def measure_individual(road, individual, segment_count=None) -> tuple[float, float, bool]:
    # returns total fuel consumption, how far the individual got, and if they finished the road
    total_fuel_consumption = 0
    
    # in loop this means speed entering segment i
    velocity = 0
    
    if segment_count is None:
        segment_count = ROAD_LENGTH - 1
    
    # road i-th segment describes terrain between i and i + 1
    for i in range(segment_count):
        
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
        predicted_end_velocity = math.sqrt(max(0.0, velocity**2 + 2 * (f_engine - f_grade - f_roll - K_DRAG * velocity**2) * DELTA_X))
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

def brute_force_search(road):
    MAX_COMBINATIONS = 500_000_000

    length = ROAD_LENGTH - 1
    choices = sorted(THROTTLE_AMOUNTS)
    total = len(choices) ** length

    # if total > MAX_COMBINATIONS:
    #     print(f"Too many combinations: {total:,} > {MAX_COMBINATIONS:,}")
    #     return None

    best_individual = None
    best_fitness = -math.inf
    best_fuel = math.inf
    path = []

    with tqdm(total=total, desc="Brute force") as pbar:
        def search():
            nonlocal best_individual, best_fitness, best_fuel

            if path:
                fuel, distance, survived = measure_individual(
                    road,
                    path,
                    segment_count=len(path),
                )

                remaining = length - len(path)

                if not survived:
                    pbar.update(len(choices) ** remaining)
                    return

                if fuel >= best_fuel:
                    pbar.update(len(choices) ** remaining)
                    return

            if len(path) == length:
                fuel, distance, finished = measure_individual(road, path)

                fitness = (
                    DELTA_X * length + FINISH_REWARD - fuel
                    if finished
                    else distance
                )

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = path.copy()

                    if finished:
                        best_fuel = fuel

                pbar.update(1)
                return

            for throttle in choices:
                path.append(throttle)
                search()
                path.pop()

        search()

    print(f"Best fitness = {best_fitness}")
    print(best_individual)

    return best_individual

def main():
    # road = get_smooth_random_values(ROAD_LENGTH)
    # print(SLOPES)
    road = generate_road(SLOPES, ROAD_LENGTH, 40)
    # brute_force_search(road)

    
    print(measure_individual(road, [3.0] * (ROAD_LENGTH - 1)))
    print(measure_individual(road, [1.0] * (ROAD_LENGTH - 1)))
    print(measure_individual(road, [0.3] * (ROAD_LENGTH - 1)))
    print(measure_individual(road, [0.0] * (ROAD_LENGTH - 1)))
    
    visualize(
        road=road,
        throttles=[15.0] * 5 + [0.0] * (ROAD_LENGTH - 1 - 5),
        delta_x=DELTA_X,
        k_drag=K_DRAG,
        k_roll=K_ROLL,
    )
    
    
    ga = GA(road)
    ga.init_population(POPULATION_SIZE)
    
    best_fitness = -math.inf
    for generation in range(1000):
        ga.step()
        current_best_fitness = ga.get_best_fitness()
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            print(f"Generation {generation}: Best fitness = {best_fitness} / {FINISH_REWARD + (ROAD_LENGTH - 1) * DELTA_X}")
    
    print(ga.get_best_individual())

    visualize(
        road=road,
        throttles=ga.get_best_individual(),
        delta_x=DELTA_X,
        k_drag=K_DRAG,
        k_roll=K_ROLL,
    )

if __name__ == "__main__":
    main()
    
    