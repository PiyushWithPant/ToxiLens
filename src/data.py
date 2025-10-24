import pandas as pd



# Load dataset (adjust path if needed)
df = pd.read_csv("data\\jigsaw dataset kaggle\\train.csv")

# Look at first few rows
print(df.head())

# See how many samples and columns
print("\nShape:", df.shape)
