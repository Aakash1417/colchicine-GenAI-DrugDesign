def get_main_section(output_dir):
    return f'''run_type = "staged_learning"
device = "cuda:0"
tb_logdir = "{output_dir}/tb_logs"
json_out_config = "{output_dir}/_staged_learning.json"
'''

def get_parameters_section(output_dir):
    return f'''[parameters]
summary_csv_prefix = "{output_dir}/staged_learning"
use_checkpoint = false
purge_memories = false

tb_isim = false

## LibInvent
prior_file = "priors/libinvent.prior"
agent_file = "priors/libinvent.prior"
smiles_file = "scaffolds.smi"

batch_size = 64
unique_sequences = true
randomize_smiles = false
'''

def get_learning_strategy_section():
    return '''[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001
'''

def get_diversity_filter_section():
    return '''[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4
penalty_multiplier = 0.5
'''

def get_stage1_scoring(output_dir):
    #TODO: unwated smarts filtering
    # R1 or R2 substitutions that are not -O-R (i.e., no single-bonded O to a carbon)
    # "[*:1][!O]",         # R1 is not attached to O
    # "[*:1]O[!#6]",       # R1 has O, but not bonded to carbon (bad ether)
    # "[*:2][!O]",
    # "[*:2]O[!#6]",
    # "[*:3][!#9;!#17;!#35;!#53]",  # R3 not bonded to F, Cl, Br, or I
    # "[*:5][O,S,N;!$([O,S,N][#6])]"  # R5 has O/S/N but not single bonded to carbon
    return f'''### Stage 1
[[stage]]

chkpt_file = '{output_dir}/test1.chkpt'

termination = "simple"
max_score = 0.6
min_steps = 25
max_steps = 100

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]

[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = 0.79

# a list of unwanted SMARTS(!) to be scored as zero
params.smarts = [
    "[*;r8]",
    "[*;r9]",
    "[*;r10]",
    "[*;r11]",
    "[*;r12]",
    "[*;r13]",
    "[*;r14]",
    "[*;r15]",
    "[*;r16]",
    "[*;r17]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C#C",
    "C(=[O,S])[O,S]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
]

[[stage.scoring.component]]
[stage.scoring.component.MolecularWeight]

[[stage.scoring.component.MolecularWeight.endpoint]]
name = "Molecular weight"
weight = 0.342

transform.type = "double_sigmoid"
transform.high = 500.0
transform.low = 200.0
transform.coef_div = 500.0
transform.coef_si = 20.0
transform.coef_se = 20.0
'''

def get_rdkit_scoring(output_dir):
    return f'''### Stage 2
[[stage]]

chkpt_file = '{output_dir}/test2.chkpt'

termination = "simple"
max_score = 0.7
min_steps = 10
max_steps = 100

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.QED]

[[stage.scoring.component.QED.endpoint]]
weight = 0.5
name = "QED Score"

[[stage.scoring.component]]
[stage.scoring.component.SAScore]

[[stage.scoring.component.SAScore.endpoint]]
weight = 0.5
name = "SA Score"
'''

def get_custom_qsar_scoring(output_dir, cell_line_name):
    return f'''### Stage 3
[[stage]]

chkpt_file = '{output_dir}/test3.chkpt'

termination = "simple"
max_score = 0.7
min_steps = 10
max_steps = 100

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]

[[stage.scoring.component.ExternalProcess.endpoint]]
weight = 0.6
name = "Custom QSAR model"

params.executable = "../venv-reinvent4/bin/python"
params.args = "chemprop_qsar.py XGBoost_Descriptors_pIC50_{cell_line_name}.joblib"

'''
    # [component.ExternalProcess.transform]
    # type = "sigmoid"
    # low = 5.0
    # high = 9.0
