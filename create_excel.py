import pandas as pd

# Create empty DataFrame with column names
columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 
           'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']
df = pd.DataFrame(columns=columns)

# Save to Excel
df.to_excel('aryan.xlsx', index=False)
print("Empty Excel file 'aryan.xlsx' created successfully with water quality columns")
