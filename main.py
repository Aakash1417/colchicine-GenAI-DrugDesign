from src.reinvent_toml_parameters import get_main_section, get_parameters_section,get_learning_strategy_section,get_stage1_scoring,get_qed_scoring,get_SA_scoring, get_custom_qsar_scoring
import os
import subprocess
import shutil
import sys
import random
from skopt import gp_minimize, Optimizer
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import pandas as pd
import numpy as np

def create_config_toml(params, output_dir, filename, cell_line_name):
    """Write the combined TOML content to a file."""
    
    print(params)
    
    sections = [
        get_main_section(output_dir),
        get_parameters_section(output_dir),
        get_learning_strategy_section(params['rate'], params['sigma']),
        # get_diversity_filter_section(), # TODO get this working later
        get_stage1_scoring(output_dir),
        get_qed_scoring(output_dir),
        get_SA_scoring(output_dir),
        # get_custom_qsar_scoring(output_dir, cell_line_name)
    ]
    
    toml_content = "\n".join(sections)

    with open(filename, "w") as file:
        file.write(toml_content)
    print(f"TOML configuration written to {filename}")

space = [
    # --- Learning Strategy ---
    Integer(32, 256, name="sigma"),
    Real(1e-5, 1e-3, name="rate", prior="log-uniform"),

    # --- Diversity Filter ---
    Integer(10, 100, name="bucket_size"),
    Real(0.0, 1.0, name="minscore"),
    Real(0.0, 1.0, name="minsimilarity"),
    Real(0.1, 1.0, name="penalty_multiplier"),   

    # --- Stage 1: Termination Criteria ---
    Real(0.5, 1.0, name="max_score_stage1"),    
    Integer(10, 50, name="min_steps_stage1"),   
    Integer(50, 200, name="max_steps_stage1"),   

    # --- Stage 1: MW Transform (Double Sigmoid) ---
    Real(200, 600, name="mw_low"),                
    Real(400, 800, name="mw_high"),             
    Real(100, 1000, name="mw_coef_div"),        
    Real(5, 40, name="mw_coef_si"),            
    Real(5, 40, name="mw_coef_se"),            

    # --- Stage 2: Termination Criteria ---
    Real(0.5, 1.0, name="max_score_stage2"),     
    Integer(5, 50, name="min_steps_stage2"),     
    Integer(50, 200, name="max_steps_stage2"),    

    # --- Stage 3: Termination Criteria ---
    Real(0.5, 1.0, name="max_score_stage3"),     
    Integer(5, 50, name="min_steps_stage3"),      
    Integer(50, 200, name="max_steps_stage3"),   
]

def evaluate_params_bo(params, cell_line):
    OUTPUT_DIR = "experiment"
    config_file = f"{OUTPUT_DIR}/output_config.toml"
    log_file = f"{OUTPUT_DIR}/staged_learning.log"
    results_file = f"{OUTPUT_DIR}/staged_learning_3.csv"
    
    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing '{OUTPUT_DIR}' folder...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    create_config_toml(params, OUTPUT_DIR, config_file, cell_line)
    cmd = ["reinvent", "-l", log_file, config_file]
    score = 1e6
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        df = pd.read_csv(results_file)
        max_step = df["step"].max()
        filtered = df[(df["step"] == max_step) & (df["Score"] > 0)]
        top10 = filtered.sort_values(by="Score", ascending=False).head(10)
        score = -top10["Score"].mean()
    except subprocess.CalledProcessError as e:
        return score

    print(f"Done thingy! score: {score}")

    return score


def run_bayesian_optimization(cell_line, max_successful_calls=10, n_random_starts=2):
    opt = Optimizer(dimensions=space, n_initial_points=n_random_starts, acq_func="EI", random_state=42)
    successful_evals = 0
    tried_params = []
    tried_scores = []

    while successful_evals < max_successful_calls:
        next_params = opt.ask()
        param_dict = dict(zip([dim.name for dim in space], next_params))

        try:
            score = evaluate_params_bo(param_dict, cell_line)
            opt.tell(next_params, score)
            successful_evals += 1
            tried_params.append(next_params)
            tried_scores.append(score)
            print(f"Success {successful_evals}/{max_successful_calls}: Score = {score}\n")
        except subprocess.CalledProcessError:
            print("Reinvent subprocess failed. Retrying with different params...\n")
            continue

    best_idx = np.argmin(tried_scores)
    best_params = tried_params[best_idx]
    best_score = tried_scores[best_idx]

    print("Best parameters found:", best_params)
    print("Best score:", best_score)

if __name__ == "__main__":
    # OUTPUT_DIR = "experiment"
    # CONFIG_FILE = f"{OUTPUT_DIR}/output_config.toml"
    # LOG_FILE = f"{OUTPUT_DIR}/staged_learning.log"
    CELL_LINE = str()

    # if os.path.exists(OUTPUT_DIR):
    #     print(f"Clearing '{OUTPUT_DIR}' folder...")
    #     shutil.rmtree(OUTPUT_DIR)
    # os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        CELL_LINE = sys.argv[1]
    else:
        print("MIssing cell_line")
        sys.exit(1)
        
    run_bayesian_optimization(cell_line=CELL_LINE)

    # create_config_toml(OUTPUT_DIR, CONFIG_FILE, CELL_LINE)
    
    # command = ["reinvent", "-l", LOG_FILE, CONFIG_FILE]
    
    # try:
    #     result = subprocess.run(command, capture_output=True, text=True, check=True)
    #     print(result.stdout)
    # except subprocess.CalledProcessError as e:
    #     print("An error occurred while running REINVENT:")
    #     print(e.stderr)
