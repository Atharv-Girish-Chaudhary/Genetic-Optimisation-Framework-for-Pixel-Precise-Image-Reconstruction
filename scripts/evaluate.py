"""
Evaluation script: runs the enhanced GA on a target image and reports
normalised MAE, PSNR, and SSIM against a random baseline.

Usage:
    python scripts/evaluate.py                          # default image, 300k generations
    python scripts/evaluate.py data/raw/myimage.png     # custom image
    python scripts/evaluate.py --generations 50000      # quick smoke-test
    python scripts/evaluate.py --save-visuals           # also saves best reconstruction

Outputs a results table to stdout and writes assets/eval_results.txt.
Pass --save-visuals to additionally save assets/eval_reconstruction.png.
"""

import argparse
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts import image_parameters
from modules import population, fitness, selection, crossover, mutation
from model.helpers.saving import chromosome2img


# ---------------------------------------------------------------------------
# Metrics  (MAE normalised to [0, 1] by dividing by 255)
# ---------------------------------------------------------------------------

def compute_metrics(target_arr: np.ndarray, candidate_arr: np.ndarray) -> dict:
    target_f = target_arr.astype(np.float32)
    cand_f   = candidate_arr.astype(np.float32)

    mae_norm = float(np.mean(np.abs(target_f - cand_f)) / 255.0)
    psnr_val = float(psnr(target_arr, candidate_arr, data_range=255))
    ssim_val = float(ssim(target_arr, candidate_arr, channel_axis=2, data_range=255))

    return {"MAE": mae_norm, "PSNR": psnr_val, "SSIM": ssim_val}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def random_baseline(target_arr: np.ndarray) -> dict:
    """Single random image — no evolution at all."""
    random_img = np.random.randint(0, 256, target_arr.shape, dtype=np.uint8)
    return compute_metrics(target_arr, random_img)


# ---------------------------------------------------------------------------
# GA runner
# ---------------------------------------------------------------------------

def run_ga(
    target_chromosome: np.ndarray,
    img_shape: tuple,
    generations: int,
    sol_per_population: int = 8,
    num_parents_mating: int = 4,
    mutation_percent: float = 0.01,
) -> np.ndarray:
    """Run the enhanced GA and return the best chromosome from the final generation."""
    new_population = population.initial_population(img_shape, sol_per_population)

    for _ in tqdm(range(generations), desc="Enhanced GA", unit="gen", dynamic_ncols=True):
        fit_quality    = fitness.calc_population_fitness(target_chromosome, new_population)
        parents        = selection.selecting_mating_pool(num_parents_mating, new_population, fit_quality)
        new_population = crossover.multi_pt_crossover(parents, img_shape, sol_per_population)
        new_population = mutation.enhanced_mutation(new_population, num_parents_mating, mutation_percent)

    fit_final = fitness.calc_population_fitness(target_chromosome, new_population)
    best_idx  = int(np.argmax(fit_final))
    return new_population[best_idx]


# ---------------------------------------------------------------------------
# Visual output
# ---------------------------------------------------------------------------

def save_reconstruction(target_arr: np.ndarray, reconstruction: np.ndarray, assets_dir: str) -> None:
    target_path = os.path.join(assets_dir, "eval_target.png")
    recon_path  = os.path.join(assets_dir, "eval_reconstruction.png")
    plt.imsave(target_path, target_arr)
    plt.imsave(recon_path, reconstruction)
    print(f"Target (150×150) saved to   {target_path}")
    print(f"Reconstruction saved to     {recon_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the enhanced GA.")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=os.path.join("data", "raw", "test.png"),
        help="Path to the target image (default: data/raw/test.png)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=300_000,
        help="Generations to run (default: 300 000)",
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Save a side-by-side comparison to assets/eval_reconstruction.png",
    )
    args = parser.parse_args()

    print(f"Target image : {args.image_path}")
    print(f"Generations  : {args.generations:,}\n")

    target_img, target_arr, img_shape, target_chromosome = image_parameters.Main(args.image_path)

    assets_dir = os.path.join(project_root, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    results = {}

    # 1. Random baseline
    results["Random baseline"] = random_baseline(target_arr)

    # 2. Enhanced GA
    best_chrom  = run_ga(target_chromosome, img_shape, args.generations)
    best_img    = chromosome2img(best_chrom, img_shape)
    results["This framework (enhanced)"] = compute_metrics(target_arr, best_img)

    # -------------------------------------------------------------------
    # Print table
    # -------------------------------------------------------------------
    col_w  = 36
    header = f"{'Method':<{col_w}} {'MAE ↓':>8}  {'PSNR ↑':>8}  {'SSIM ↑':>8}"
    sep    = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, m in results.items():
        print(f"{name:<{col_w}} {m['MAE']:>8.4f}  {m['PSNR']:>7.2f} dB  {m['SSIM']:>8.4f}")
    print(sep)
    print("\nMAE is normalised to [0, 1] (raw pixel error ÷ 255).")

    # -------------------------------------------------------------------
    # Save results text
    # -------------------------------------------------------------------
    out_path = os.path.join(assets_dir, "eval_results.txt")
    with open(out_path, "w") as f:
        f.write(f"Image      : {args.image_path}\n")
        f.write(f"Generations: {args.generations:,}\n\n")
        f.write(f"{sep}\n{header}\n{sep}\n")
        for name, m in results.items():
            f.write(f"{name:<{col_w}} {m['MAE']:>8.4f}  {m['PSNR']:>7.2f} dB  {m['SSIM']:>8.4f}\n")
        f.write(f"{sep}\n")
        f.write("\nMAE is normalised to [0, 1] (raw pixel error ÷ 255).\n")
    print(f"\nResults saved to {out_path}")

    # -------------------------------------------------------------------
    # Optional visuals
    # -------------------------------------------------------------------
    if args.save_visuals:
        save_reconstruction(target_arr, best_img, assets_dir)


if __name__ == "__main__":
    main()
