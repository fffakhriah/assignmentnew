# ==========================
# Genetic Algorithm for TV Scheduling
# ==========================
import csv
import random
import pandas as pd
import os
from pprint import pprint

# ---------- SETTINGS ----------
# ✅ Correct file path for your computer (use raw string for Windows)
CSV_PATH = r"C:\Users\User\Desktop\COMPUTER EVOLUTION\program_ratings_modified.csv"

HOUR_COLUMNS = [f"Hour {h}" for h in range(6, 24)]  # Hour 6..Hour 23 inclusive
GENERATIONS = 200
POPULATION_SIZE = 100
ELITISM_SIZE = 4
TOURNAMENT_SIZE = 2
# -------------------------------


# ==========================
# READ CSV DATA
# ==========================
def read_ratings(csv_path):
    """Read the program ratings CSV and convert it to dictionary format."""
    print("\n🔍 Checking CSV file...")
    abs_path = os.path.abspath(csv_path)
    print("Looking for file at:", abs_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"❌ File not found at: {abs_path}\n"
            "➡️ Please check your file path or move the CSV into this folder."
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"⚠️ Unable to read CSV file: {e}")

    # Check if hour columns exist
    missing = [c for c in HOUR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"⚠️ CSV missing these hour columns: {missing}\n"
            f"➡️ Make sure your headers are exactly: {HOUR_COLUMNS}"
        )

    programs = list(df.iloc[:, 0])  # First column = program names
    ratings = {}
    for idx, prog in enumerate(programs):
        row = df.loc[idx, HOUR_COLUMNS].astype(float).tolist()
        ratings[str(prog)] = row

    print(f"✅ CSV loaded successfully! Found {len(programs)} programs.")
    return ratings, programs


# ==========================
# FITNESS FUNCTION
# ==========================
def fitness_function(schedule, ratings):
    """Calculate total rating for a schedule."""
    total = 0.0
    for slot_idx, program in enumerate(schedule):
        total += ratings[program][slot_idx]
    return total


# ==========================
# INITIALIZE POPULATION
# ==========================
def init_population(programs, num_slots, pop_size):
    population = []
    for _ in range(pop_size):
        schedule = [random.choice(programs) for _ in range(num_slots)]
        population.append(schedule)
    return population


# ==========================
# SELECTION, CROSSOVER, MUTATION
# ==========================
def tournament_selection(population, ratings, k=TOURNAMENT_SIZE):
    candidates = random.sample(population, k)
    candidates.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
    return candidates[0]


def crossover(parent1, parent2):
    if len(parent1) <= 1:
        return parent1.copy(), parent2.copy()
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(schedule, programs, mutation_rate):
    new = schedule.copy()
    for i in range(len(new)):
        if random.random() < mutation_rate:
            new[i] = random.choice(programs)
    return new


# ==========================
# GENETIC ALGORITHM LOOP
# ==========================
def evolve(
    ratings,
    programs,
    num_slots,
    generations=GENERATIONS,
    pop_size=POPULATION_SIZE,
    crossover_rate=0.8,
    mutation_rate=0.02,
    elitism_size=ELITISM_SIZE,
):
    """Run the genetic algorithm."""
    population = init_population(programs, num_slots, pop_size)

    for gen in range(generations):
        population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
        new_pop = population[:elitism_size]  # Keep best individuals

        while len(new_pop) < pop_size:
            parent1 = tournament_selection(population, ratings)
            parent2 = tournament_selection(population, ratings)

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = mutate(child1, programs, mutation_rate)
            child2 = mutate(child2, programs, mutation_rate)

            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)

        population = new_pop

    population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
    best = population[0]
    best_score = fitness_function(best, ratings)
    return best, best_score


# ==========================
# OUTPUT HELPERS
# ==========================
def schedule_to_table(schedule, all_hours):
    rows = []
    for i, prog in enumerate(schedule):
        hour = all_hours[i]
        rows.append({"Hour": hour, "Program": prog})
    return pd.DataFrame(rows)


def run_three_trials(csv_path, param_sets, generations=GENERATIONS, pop_size=POPULATION_SIZE):
    ratings, programs = read_ratings(csv_path)
    num_slots = len(HOUR_COLUMNS)
    all_hours = [f"{h}:00" for h in range(6, 24)]
    results = []

    for idx, (co_r, mut_r) in enumerate(param_sets, start=1):
        print(f"\n=== Trial {idx} (CO_R={co_r}, MUT_R={mut_r}) ===")
        best_sched, best_score = evolve(
            ratings,
            programs,
            num_slots,
            generations=generations,
            pop_size=pop_size,
            crossover_rate=co_r,
            mutation_rate=mut_r,
            elitism_size=ELITISM_SIZE,
        )
        df_table = schedule_to_table(best_sched, all_hours)
        print(df_table.to_string(index=False))
        print("Total Score (fitness):", round(best_score, 4))

        df_table.to_csv(f"best_schedule_trial_{idx}.csv", index=False)
        results.append({
            "trial": idx,
            "co_r": co_r,
            "mut_r": mut_r,
            "score": best_score,
            "schedule_df": df_table
        })

    return results


# ==========================
# MAIN PROGRAM
# ==========================
if __name__ == "__main__":
    param_sets = [
        (0.8, 0.02),
        (0.9, 0.04),
        (0.7, 0.01),
    ]

    print("\n🚀 Starting Genetic Algorithm TV Scheduling...\n")

    try:
        results = run_three_trials(
            CSV_PATH,
            param_sets,
            generations=GENERATIONS,
            pop_size=POPULATION_SIZE,
        )

        print("\n📊 Summary of 3 trials:")
        for r in results:
            print(f"Trial {r['trial']}: CO_R={r['co_r']}, MUT_R={r['mut_r']}, Score={round(r['score'],4)}")

        print("\n✅ All trials completed successfully!")

    except FileNotFoundError as fnf:
        print(f"\n❌ File not found error:\n{fnf}")
    except ValueError as ve:
        print(f"\n⚠️ CSV format issue:\n{ve}")
    except Exception as e:
        print(f"\n⚠️ Unexpected error occurred:\n{e}")
