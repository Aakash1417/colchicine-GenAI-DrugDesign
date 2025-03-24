from src.reinvent_toml_parameters import get_main_section, get_parameters_section,get_learning_strategy_section,get_stage1_scoring,get_rdkit_scoring, get_custom_qsar_scoring
import os
import subprocess
import shutil
import sys

def combine_sections(output_dir):
    """Combine all sections into a single TOML string."""
    sections = [
        get_main_section(output_dir),
        get_parameters_section(output_dir),
        get_learning_strategy_section(),
        # get_diversity_filter_section(), # TODO get this working later
        get_stage1_scoring(output_dir),
        get_rdkit_scoring(output_dir),
        get_custom_qsar_scoring(output_dir)
    ]
    return "\n".join(sections)

def create_config_toml(output_dir, filename):
    """Write the combined TOML content to a file."""
    
    toml_content = combine_sections(output_dir)
    with open(filename, "w") as file:
        file.write(toml_content)
    print(f"TOML configuration written to {filename}")

if __name__ == "__main__":
    OUTPUT_DIR = "experiment"
    CONFIG_FILE = f"{OUTPUT_DIR}/output_config.toml"
    LOG_FILE = f"{OUTPUT_DIR}/staged_learning.log"

    if len(sys.argv) > 1 and sys.argv[1].lower() == "clear":
        if os.path.exists(OUTPUT_DIR):
            print(f"Clearing '{OUTPUT_DIR}' folder...")
            shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    create_config_toml(OUTPUT_DIR, CONFIG_FILE)
    
    command = ["reinvent", "-l", LOG_FILE, CONFIG_FILE]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running REINVENT:")
        print(e.stderr)
