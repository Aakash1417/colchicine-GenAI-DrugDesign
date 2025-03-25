from src.reinvent_toml_parameters import get_main_section, get_parameters_section,get_learning_strategy_section,get_stage1_scoring,get_rdkit_scoring, get_custom_qsar_scoring
import os
import subprocess
import shutil
import sys
import random
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

def create_config_toml(output_dir, filename, cell_line_name):
    """Write the combined TOML content to a file."""
    
    sections = [
        get_main_section(output_dir),
        get_parameters_section(output_dir),
        get_learning_strategy_section(),
        # get_diversity_filter_section(), # TODO get this working later
        get_stage1_scoring(output_dir),
        get_rdkit_scoring(output_dir),
        # get_custom_qsar_scoring(output_dir, cell_line_name)
    ]
    
    toml_content = "\n".join(sections)

    with open(filename, "w") as file:
        file.write(toml_content)
    print(f"TOML configuration written to {filename}")

space = [
    Integer(32, 256, name='sigma'),
    Real(1e-5, 1e-3, name='rate', prior='log-uniform'),
    Real(300, 700, name='mw_high')
]

def create_config_toml_for_bo(params, output_dir, config_filename, cell_line):
    create_config_toml(output_dir, config_filename, cell_line)
    with open(config_filename, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("sigma ="):
            new_lines.append(f"sigma = {params['sigma']}\n")
        elif line.strip().startswith("rate ="):
            new_lines.append(f"rate = {params['rate']}\n")
        elif line.strip().startswith("transform.high ="):
            new_lines.append(f"transform.high = {params['mw_high']}\n")
        else:
            new_lines.append(line)
    with open(config_filename, "w") as f:
        f.writelines(new_lines)

def evaluate_params_bo(params, cell_line):
    run_id = random.randint(1000, 9999)
    output_dir = f"experiment_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    config_file = os.path.join(output_dir, "output_config.toml")
    log_file = os.path.join(output_dir, "staged_learning.log")
    create_config_toml_for_bo(params, output_dir, config_file, cell_line)
    cmd = ["reinvent", "-l", log_file, config_file]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return 1e6
    return -random.random() * 10.0

@use_named_args(space)
def objective_function(**params):
    return evaluate_params_bo(params, "MCF7")

def run_bayesian_optimization(n_calls=10, n_random_starts=5):
    result = gp_minimize(
        func=objective_function,
        dimensions=space,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        acq_func='EI',
        random_state=42
    )
    best_params = result.x
    best_score = -result.fun
    print(best_params, best_score)

if __name__ == "__main__":
    OUTPUT_DIR = "experiment"
    CONFIG_FILE = f"{OUTPUT_DIR}/output_config.toml"
    LOG_FILE = f"{OUTPUT_DIR}/staged_learning.log"
    CELL_LINE = str()

    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing '{OUTPUT_DIR}' folder...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        CELL_LINE = sys.argv[1]
    else:
        print("MIssing cell_line")
        sys.exit(1)

    create_config_toml(OUTPUT_DIR, CONFIG_FILE, CELL_LINE)
    
    command = ["reinvent", "-l", LOG_FILE, CONFIG_FILE]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running REINVENT:")
        print(e.stderr)
