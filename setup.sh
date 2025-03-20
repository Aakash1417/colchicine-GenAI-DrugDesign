salloc --cpus-per-task=6 --gres=gpu:p100:1 --mem-per-cpu=4000M --time=00:30:00 --account=def-aminpour --job-name=colchicine-GenAI-DrugDesign
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5 python-build-bundle/2024a scipy-stack/2024b rdkit/2024.03.4
source ../venv-reinvent4/bin/activate
