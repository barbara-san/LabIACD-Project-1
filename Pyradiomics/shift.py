import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('1.csv')

# Shift the missing values in columns B and onward to the right
df.iloc[:, 143:] = df.iloc[:, 143:].apply(lambda x: x.shift(273) if x.isna().any() else x, axis=1)

# Save the modified DataFrame back to a CSV file
df.to_csv('pyradiomics_extraction.csv', index=False)