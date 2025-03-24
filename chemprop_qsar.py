import sys
import json
import joblib
import pandas as pd
import xgboost as xgb
import pickle

# Read SMILES from stdin
smiles_list = [smiles.strip() for smiles in sys.stdin]
model_filename = sys.argv[1]

try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found.", file=sys.stderr)
    sys.exit(1)
    
with open("smiles_list.pkl", "wb") as f:
    pickle.dump(smiles_list, f)


# def smiles_to_descriptors(smiles_list):
#     """
#     Convert SMILES to descriptors. Replace this with actual descriptor calculation.
#     For now, returning dummy features for XGBoost.
#     """
#     import numpy as np
#     num_features = model.get_booster().num_features()
#     return pd.DataFrame(np.random.rand(len(smiles_list), num_features))

# X = smiles_to_descriptors(smiles_list)

# scores = model.predict(X)

data = {"version": 1, "payload": {"predictions": list([123,123])}}

print(json.dumps(data))
