import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

train_df = pd.read_csv("ml_engine/data/raw/uci_voice_train_data.txt", header=None)
test_df = pd.read_csv("ml_engine/data/raw/uci_voice_test_data.txt", header=None)

# Combine everything into one pool — we're going to make our own honest split
combined_df = pd.concat([train_df, test_df], ignore_index=True)

subject_ids = combined_df[0]
labels = combined_df[28]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(combined_df, groups=subject_ids))

my_train_df = combined_df.iloc[train_idx]
my_test_df = combined_df.iloc[test_idx]

# Verify zero overlap this time
train_subjects = set(my_train_df[0].unique())
test_subjects = set(my_test_df[0].unique())
print(f"Overlap: {len(train_subjects.intersection(test_subjects))}")  # should be 0
print(f"Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")