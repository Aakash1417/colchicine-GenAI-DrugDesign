import sys
import json
import joblib
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors

# Read SMILES from stdin
smiles_list = [smiles.strip() for smiles in sys.stdin]
model_filename = sys.argv[1]

try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found.", file=sys.stderr)
    sys.exit(1)

def preprocess_descriptors(descriptors):
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    descriptors_imputed = imputer.fit_transform(descriptors)
    descriptors_scaled = scaler.fit_transform(descriptors_imputed)
    descriptors_scaled = np.nan_to_num(descriptors_scaled, nan=0.0)
    return descriptors_scaled

descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
descriptor_names = descriptor_calculator.GetDescriptorNames()

descriptor_rows = []

for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        descriptor_values = descriptor_calculator.CalcDescriptors(mol)
        descriptor_rows.append(descriptor_values)

# Creating dataFrame from descriptor values
descriptors_df = pd.DataFrame(descriptor_rows, columns=descriptor_names)
descriptors_df = preprocess_descriptors(descriptors_df)
pIC50_predictions = model.predict(descriptors_df)

data = {"version": 1, "payload": {"predictions": list(map(float, pIC50_predictions))}}

print(json.dumps(data))
