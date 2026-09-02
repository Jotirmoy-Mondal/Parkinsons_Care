"""How XGboost Works
Unlike Random Forest, which builds many independent decision trees at the same time and averages them, XGBoost builds trees sequentially:

It builds a first, simple decision tree to predict the target.
It evaluates where that first tree made errors (the "residuals").
It builds a second tree specifically designed to predict and correct the errors of the first tree.
It repeats this process hundreds or thousands of times, with each new tree stepping in to fix the mistakes of the combined ensemble before it."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib # save and load model in optimized way

def train_baseline(csv_path: str, model_output_path: str):
    # ================================
    # SECTION 1: LOAD CSV DATA
    # ================================
    df = pd.read_csv(csv_path)

    # quick check — how many unique patients are actually behind these 1134 samples?
    print(df["filename"].str.extract(r'(healthy_\d+|parkinsons_\d+)')[0].nunique())

    # Drop non-feature columns — filename is an identifier, not a predictor
    feature_cols = [c for c in df.columns if c not in ("filename", "status")] # here first c is a temp var, use to append value of c to list

    # status 0 means healthy
    # ignoring filename and status

    # ==========================================
    # SECTION 2: ISOLATE VARIABLES
    # ==========================================
    
    X = df[feature_cols]
    y = df["status"]

    print(f"Loaded {len(df)} samples, {y.sum()} PD, {len(y) - y.sum()} healthy")
    print(f"Features: {feature_cols}")

    # ==========================================
    # SECTION 3: STRATIFIED SPLIT
    # ==========================================

    # test_size=0.2: Reserves 20% of data for testing, 80% for training.
     # random_state=42: Locks the random seed for reproducible results.

    # stratify=y ensures train/test split keeps the same healthy/PD ratio —
    # important with imbalanced classes so test set isn't accidentally all one class
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # SECTION 4: TRAIN XGBOOST
    # ==========================================
    model = xgb.XGBClassifier(
        n_estimators=100, # 100 tree
        max_depth=4,          # shallow trees — small dataset, avoid overfitting
        learning_rate=0.05,
        eval_metric="logloss",
        #: Evaluates training performance by penalizing predictions based on how far their calculated probability is from the actual truth. l_metric="logloss",
        
        random_state=42,
    )
    model.fit(X_train, y_train) # train the model

    # ==========================================
    # SECTION 5: EVALUATE MODEL
    # ==========================================
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    # Calculates the exact probability (0.0 to 1.0) of a positive Parkinson's diagnosis.
     # It takes the testing features (X_test) and slices [:, 1] to discard the healthy 
     # probability, returning only the raw confidence metric needed for ROC-AUC evaluation.

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "PD"]))

    # Compares the model's final guesses (y_pred) against the true answers (y_test).
     # Returns a detailed text report grading the clinical Precision, Recall, and F1-score 
     # mapped cleanly to "Healthy" and "PD" labels.



    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    # Calculates the ROC-AUC score (0.5 to 1.0) by comparing the true diagnosis (y_test) 
     # against the model's probability predictions (y_proba). The result is formatted 
     # to exactly three decimal places (:.3f) to grade the model's distinction ability.

    # --- Feature importance — this is your "explainable" signal for clinicians which show “how much did this feature help the trees make better predictions?”
     #not “how strongly is it linearly related to the output?”

    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    print("\n--- Top 10 Most Important Features ---")
    print(importance.head(10))

    # ==========================================
    # SECTION 6: SAVE MODEL
    # ==========================================
    joblib.dump(model, model_output_path)
    print(f"\nModel saved to {model_output_path}")

    

    return model

if __name__ == "__main__":
    train_baseline(
        csv_path="ml_engine/data/voice_features.csv",
        model_output_path="ml_engine/voice/weights/xgboost_baseline.pkl",
    )