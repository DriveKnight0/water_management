import pandas as pd

# Read the CSV file
df = pd.read_csv('aryan.csv')

# Convert to Excel
df.to_excel('aryan.xlsx', index=False)
print("File converted successfully to aryan.xlsx")
