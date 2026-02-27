import pandas as pd

df = pd.read_csv("dataset.csv", engine="python", on_bad_lines="skip")

# Remove duplicate header rows
df = df[df["label"] != "label"]

# Keep only valid labels
df = df[df["label"].isin(["0", "1", "2"])]

# Drop duplicates
df = df.drop_duplicates()

# Strip whitespace
df["text"] = df["text"].str.strip()
df["label"] = df["label"].astype(int)

print("Final shape:", df.shape)
print(df["label"].value_counts())

df.to_csv("dataset_clean.csv", index=False)