import os
import sys
from typing import List
import numpy as np


# Ensure project root is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts import image_parameters  
from modules import population, selection, crossover, mutation  


def compute_population_diversity(population_array: np.ndarray, metric: str = "mae") -> float:
    """
    Compute average pairwise diversity for a population.

    Diversity is defined as the average distance between all unique pairs of
    individuals (chromosomes). Distance metric can be 'mae' (mean absolute error)
    or 'euclidean'.
    """
    num_individuals = population_array.shape[0]
    if num_individuals < 2:
        return 0.0

    total_distance = 0.0
    num_pairs = 0

    # Cast to float to avoid uint8 wrap-around during subtraction
    pop_float = population_array.astype(np.float32, copy=False)

    for i in range(num_individuals - 1):
        diffs = pop_float[i + 1 :] - pop_float[i]
        if metric == "mae":
            # Mean absolute error per pair (average per gene), then average across pairs
            per_pair = np.mean(np.abs(diffs), axis=1)
        elif metric == "euclidean":
            per_pair = np.linalg.norm(diffs, axis=1)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        total_distance += float(np.sum(per_pair))
        num_pairs += per_pair.shape[0]

    return total_distance / float(num_pairs)


def run_diversity_evaluation(
    image_path: str,
    generations: int = 300,
    sol_per_population: int = 8,
    num_parents_mating: int = 4,
    mutation_percent: float = 0.01,
    distance_metric: str = "mae",
) -> float:
    """
    Run the GA loop functionally and compute the final generation's diversity value.
    Returns the diversity score for the last generation.
    """
    # Load image parameters
    target_img, target_arr, img_shape, target_chromosome = image_parameters.Main(image_path)

    # Initialize population
    new_population = population.initial_population(img_shape, sol_per_population)

    # Evolve
    for _ in range(generations):
        # Compute fitness: sum(target) - MAE(target, indiv)
        diffs = np.abs(new_population.astype(np.float32) - target_chromosome.astype(np.float32))
        mae_per_individual = np.mean(diffs, axis=1)
        fit_quality = np.sum(target_chromosome, dtype=np.float64) - mae_per_individual
        parents = selection.selecting_mating_pool(num_parents_mating, new_population, fit_quality.copy())
        new_population = crossover.multi_pt_crossover(parents, img_shape, sol_per_population)
        new_population = mutation.enhanced_mutation(new_population, num_parents_mating, mutation_percent)

    # Compute diversity for the final generation
    final_diversity = compute_population_diversity(new_population, metric=distance_metric)
    return final_diversity


if __name__ == "__main__":
    image_path = os.path.join("data", "raw", "test.png")
    diversity_value = run_diversity_evaluation(
        image_path=image_path,
        generations=300000,              
        sol_per_population=8,
        num_parents_mating=4,
        mutation_percent=0.01,        
        distance_metric="mae",
    )
    print(f"Final generation diversity (MAE): {diversity_value:.4f}")


