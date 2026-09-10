# =========================================================================
# OVERALL FILE EXPLANATION:
# This script builds, trains, and evaluates a baseline Machine Learning 
# model (XGBoost) to detect Parkinson's disease from voice features. 
# It reads a CSV of extracted voice data, splits it into training and 
# testing sets, trains a sequential decision-tree model to recognize 
# symptom patterns, grades its own accuracy, highlights which vocal 
# features matter most, and saves the final "brain" to your hard drive.
# =========================================================================

"""
How XGboost Works
Unlike Random Forest, which builds many independent decision trees at the same time and averages them, XGBoost builds trees sequentially:

1. It builds a first, simple decision tree to predict the target.
2. It evaluates where that first tree made errors (the "residuals").
3. It builds a second tree specifically designed to predict and correct the errors of the first tree.
4. It repeats this process hundreds or thousands of times, with each new tree stepping in to fix the mistakes of the combined ensemble before it.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib # save and load model in optimized way


def train_baseline(csv_path: str, model_output_path: str):
# -------------------------------------------------------------------------
# EXPLANATION FOR FUNCTION SETUP:
# Takes: `csv_path` (string) and `model_output_path` (string).
# Returns: `model` (the fully trained XGBoost classifier).
# What it does: Encapsulates the entire machine learning pipeline into a 
# single, reusable function so you can easily run it on different datasets.
# -------------------------------------------------------------------------

    # ================================
    # SECTION 1: LOAD CSV DATA
    # ================================
    df = pd.read_csv(csv_path)

    # quick check — how many unique patients are actually behind these 1134 samples?
    # 1. Store the regex pattern in a clearly named variable so it's easy to change later
    # it return a new df
    subject_pattern = r'(healthy_\d+|parkinsons_\d+)'
    
    # 2. Extract the data into a brand new, clean column. 
    # This allows you to inspect df.head() later if something goes wrong!
    df["subject_id"] = df["filename"].str.extract(subject_pattern)[0]
        unique_subjects_count = df["subject_id"].nunique()
        print(f"Total unique subjects found: {unique_subjects_count}")

    # Drop non-feature columns — filename is an identifier, not a predictor
    feature_cols = df.columns.drop(["filename", "status"]).tolist()    
    # status 0 means healthy
    # ignoring filename and status then converting to list

    # ==========================================
    # SECTION 2: ISOLATE VARIABLES
    # ==========================================
    X = df[feature_cols] #independent 
    y = df["status"]

    print(f"Loaded {len(df)} samples, {y.sum()} PD, {len(y) - y.sum()} healthy")
    print(f"Features: {feature_cols}")
# -------------------------------------------------------------------------
# EXPLANATION FOR SECTION 1 & 2 (DATA HANDLING):
# Takes: The raw CSV file path.
# Returns: `X` (a table of just the vocal features) and `y` (the target labels: 0 or 1).
# What it does: Loads the dataset into Pandas. It counts the unique patients 
# to ensure data integrity. Then, it drops the "filename" (which isn't a symptom) 
# and isolates "status" as the answer key (y), leaving only the pure acoustic 
# features (X) for the AI to study.
# -------------------------------------------------------------------------


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
# -------------------------------------------------------------------------
# EXPLANATION FOR SECTION 3 (TRAIN/TEST SPLIT):
# Takes: The full `X` (features) and `y` (answers) datasets.
# Returns: Four separate datasets (X_train, X_test, y_train, y_test).
# What it does: Acts as a teacher holding back 20% of the exam questions. 
# `stratify=y` is crucial for medical data: it ensures both the study guide 
# (train) and the final exam (test) have the exact same ratio of Healthy to 
# Parkinson's patients, preventing the model from developing unfair biases.
# -------------------------------------------------------------------------


    # ==========================================
    # SECTION 4: TRAIN XGBOOST
    # ==========================================
    model = xgb.XGBClassifier(
        n_estimators=100,     # 100 trees
        max_depth=4,          # shallow trees — small dataset, avoid overfitting
        learning_rate=0.05,
        
        # Evaluates (optimizes) training performance by penalizing predictions based on how 
        # far their calculated probability is from the actual truth.
        eval_metric="logloss", 
        random_state=42,
    )
    model.fit(X_train, y_train) # train the model
# -------------------------------------------------------------------------
# EXPLANATION FOR SECTION 4 (MODEL TRAINING):
# Takes: The training data (`X_train` and `y_train`).
# Returns: A trained XGBoost model (saved in the `model` variable).
# What it does: Sets up the XGBoost algorithm with specific rules (like 
# shallow trees to prevent it from just memorizing the data). Then, `model.fit()` 
# forces the AI to look at the features, guess the diagnosis, check its errors, 
# and build new trees to correct those errors sequentially.
# -------------------------------------------------------------------------


    # ==========================================
    # SECTION 5: EVALUATE MODEL
    # ==========================================
    y_pred = model.predict(X_test)
    
    # Calculates the exact probability (0.0 to 1.0) of a positive Parkinson's diagnosis.
    # It takes the testing features (X_test) and slices [:, 1] to discard the healthy 
    # probability, returning only the raw confidence metric needed for ROC-AUC evaluation.
    y_proba = model.predict_proba(X_test)[:, 1] #return e.g., [0.95, 0.05],-> Patient A: 95% Healthy, 5% Parkinson's

    print("\n--- Classification Report ---")
    
    # Compares the model's final guesses (y_pred) against the true answers (y_test).
    # Returns a detailed text report grading the clinical Precision, Recall, and F1-score 
    # mapped cleanly to "Healthy" and "PD" labels.
    #0.50: Useless (Random guessing)
    #0.70 - 0.80: Acceptable screening tool
    #0.80 - 0.90: Excellent diagnostic tool
    #0.90+: Outstanding (but carefully check your code to ensure you didn't accidentally cheat or overfit your data!)
    print(classification_report(y_test, y_pred, target_names=["Healthy", "PD"]))

    # Calculates the ROC-AUC score (0.5 to 1.0) by comparing the true diagnosis (y_test) 
    # against the model's probability predictions (y_proba). The result is formatted 
    # to exactly three decimal places (:.3f) to grade the model's distinction ability.
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # --- Feature importance — this is your "explainable" signal for clinicians 
    # which show "how much did this feature help the trees make better predictions?"
    # not "how strongly is it linearly related to the output?"
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    
    print("\n--- Top 10 Most Important Features ---")
    print(importance.head(10))
# -------------------------------------------------------------------------
# EXPLANATION FOR SECTION 5 (EVALUATION):
# Takes: The hidden test data (`X_test` and `y_test`).
# Returns: Printed performance reports and a feature importance list.
# What it does: Forces the trained model to take the final exam. It checks 
# hard predictions (Healthy vs PD) using a classification report, and checks 
# confidence levels using ROC-AUC. Finally, it asks the model, "Which vocal 
# changes actually helped you decide?" and prints the top 10 most critical features.
# -------------------------------------------------------------------------


    # ==========================================
    # SECTION 6: SAVE MODEL
    # ==========================================
    joblib.dump(model, model_output_path)
    print(f"\nModel saved to {model_output_path}")

    return model

# -------------------------------------------------------------------------
# EXPLANATION FOR SECTION 6 & EXECUTION:
# Takes: The trained `model` and the output path.
# Returns: Nothing directly to Python, but saves a `.pkl` file.
# What it does: `joblib.dump` freezes the trained model and saves it. The 
# `if __name__ == "__main__":` block at the bottom is the trigger that actually 
# runs this entire function when you execute the file from the terminal.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    train_baseline(
        csv_path="ml_engine/data/voice_features.csv",
        model_output_path="ml_engine/voice/weights/xgboost_baseline.pkl",
    )
