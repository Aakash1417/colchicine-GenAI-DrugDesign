from src.reinvent_toml_parameters import get_main_section, get_parameters_section,get_learning_strategy_section,get_stage1_section,get_stage2_section
import os

def combine_sections():
    """Combine all sections into a single TOML string."""
    sections = [
        get_main_section(),
        get_parameters_section(),
        get_learning_strategy_section(),
        # get_diversity_filter_section(), # TODO get this working later
        get_stage1_section(),
        get_stage2_section()
    ]
    return "\n".join(sections)

def write_toml_file(filename):
    """Write the combined TOML content to a file."""
    output_dir = "experiment"
    os.makedirs(output_dir, exist_ok=True)
    
    toml_content = combine_sections()
    with open(f"{output_dir}/{filename}", "w") as file:
        file.write(toml_content)
    print(f"TOML configuration successfully written to {filename}")

if __name__ == "__main__":
    write_toml_file("output_config.toml")
