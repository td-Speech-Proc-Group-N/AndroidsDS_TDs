import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
import os

# === Conservative Preprocessing Pipeline ===

eps = 1e-8

# column lists – adapt these to match your CSV header
rms_col = ["pcm_RMSenergy_sma"]
mfcc_cols = [f"pcm_fftMag_mfcc_sma[{i}]" for i in range(1, 13)]
f0_col = ["F0final_sma"]
zcr_col = ["pcm_zcr_sma"]
voicing_col = ["voicingFinalUnclipped_sma"]

# log transforms
log_transformer = FunctionTransformer(lambda X: np.log(X + eps), validate=False)

# RMS Energy pipeline
rms_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log", log_transformer),
    ("scale", StandardScaler())
])

# MFCCs pipeline
mfcc_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

# F0 pipeline
f0_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log", log_transformer),
    ("scale", StandardScaler())
])

# Zero Crossing Rate pipeline
zcr_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

# Voicing probability pipeline
voicing_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
    ("scale", StandardScaler())
])

# Combine everything
preprocessor = ColumnTransformer([
    ("rms", rms_pipeline, rms_col),
    ("mfcc", mfcc_pipeline, mfcc_cols),
    ("f0", f0_pipeline, f0_col),
    ("zcr", zcr_pipeline, zcr_col),
    ("voicing", voicing_pipeline, voicing_col),
], remainder="passthrough")  # keep extra cols like ID/Label

def process_file(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Apply preprocessing
    transformed = preprocessor.fit_transform(df)

    # Collect new feature column names
    all_cols = rms_col + mfcc_cols + f0_col + zcr_col + voicing_col
    passthrough_cols = [c for c in df.columns if c not in all_cols]
    final_cols = all_cols + passthrough_cols

    scaled_df = pd.DataFrame(transformed, columns=final_cols, index=df.index)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    scaled_df.to_csv(output_csv, index=False)
    # print(f"✅ Processed {input_csv} -> {output_csv}")


if __name__ == "__main__":
    input_root = "original_features/spont"
    output_root = "scaled_features/spont"
    subdirs = os.listdir("original_features/spont")
    # subdirs = ["HC", "PT"]

    for subdir in subdirs:
        input_path = os.path.join(input_root, subdir)
        output_path = os.path.join(output_root, subdir)
        os.makedirs(output_path, exist_ok=True)

        for fname in os.listdir(input_path):
            if fname.endswith(".csv"):
                in_file = os.path.join(input_path, fname)
                out_file = os.path.join(output_path, fname)
                process_file(in_file, out_file)
