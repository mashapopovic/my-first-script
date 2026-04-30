import pandas as pd

# Load the original CSV file
# Replace 'data.csv' with the path to your actual file
df = pd.read_csv('/home/masas/ansys_results_ourmodel.csv')

# Remove the first column ('Name')
# .iloc[:, 1:] selects all rows and all columns starting from index 1
df_clean = df.iloc[:, 1:]

# Save the cleaned data to 'clean.csv'
# index=False ensures the row numbers are not saved as a column
df_clean.to_csv('clean.csv', index=False)

print("Data cleaning complete. Saved to 'clean.csv'.")