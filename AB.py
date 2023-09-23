import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation
import numpy as np
import seaborn as sns

# Load the CSV file into a pandas DataFrame
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_path = (BASE_DIR/"AB_data.csv")
# c/Users/HP/Desktop/Prof_Kretchy
df = pd.read_csv(file_path)

# print(help(tree))
# Define mappings for all columns
age_mapping = {1: "18-29 years", 2: "30-39 years", 3: "40-49", 4: "50-59", 5: "60-69", 6: "70+"}
sex_mapping = {1: "Male", 2: "Female"}
marital_status_mapping = {1: "Single", 2: "Married", 3: "Divorced", 4: "Widowed"}
education_mapping = {1: "None", 2: "Primary", 3: "JHS", 4: "SHS", 5: "Tertiary"}
occupation_mapping = {1: "Unemployed", 2: "Employed", 3: "Self-employed", 4: "Other"}
religion_mapping = {1: "Christian", 2: "Muslim", 3: "Traditionalist", 4: "Other"}
b7_other_mapping = {1: "11.1-12.0 mmol/L", 2: "12.1-13.0 mmol/L", 3: "13.1-14.0 mmol/L", 4: ">14.0 mmol/L"}
b8_mapping = {1: "Type 1", 2: "Type 2"}
b9_mapping = {1: "Less than 5 years", 2: "More than 5 years"}
b10_mapping = {1: "Less than 2", 2: "More than 2", 3: "Exactly 2"}
b11_mapping = {1: "Yes", 2: "No"}
b12_mapping = {1: "Yes", 2: "No"}
b13_other_mapping = {1: "Frequency of follow-up"}  # Note: Only one value for this column
c1_to_c8_mapping = {1: "Always", 2: "Often", 3: "Sometimes", 4: "Rarely", 5: "Never"}
d1_to_d8_mapping = {i: f"Scale of 0-10 where 0 is not at all and 10 is extremely affected" for i in range(1, 9)}
e1_to_e5_mapping = {1: "Always", 2: "Often", 3: "Sometimes", 4: "Rarely", 5: "Never"}
f1_to_f5_mapping = {1: "All of the time", 2: "Most of the time", 3: "More than half of the time",
                    4: "Less than half the time", 5: "Some of the time", 6: "At no time"}

# Apply mappings to the DataFrame
df['age'] = df['age'].map(age_mapping)
df['sex'] = df['sex'].map(sex_mapping)
df['marital_status'] = df['marital_status'].map(marital_status_mapping)
df['level_of_education'] = df['level_of_education'].map(education_mapping)
df['occupation'] = df['occupation'].map(occupation_mapping)
df['religion'] = df['religion'].map(religion_mapping)
df['B7_other'] = df['B7_other'].map(b7_other_mapping)
df['B8'] = df['B8'].map(b8_mapping)
df['B9'] = df['B9'].map(b9_mapping)
df['B10'] = df['B10'].map(b10_mapping)
df['B11_No'] = df['B11_No'].map(b11_mapping)
df['B11_hyperlipedemia'] = df['B11_hyperlipedemia'].map(b11_mapping)
df['B11_stroke'] = df['B11_stroke'].map(b11_mapping)
df['B11_hypertension'] = df['B11_hypertension'].map(b11_mapping)
df['B11_heart_disease'] = df['B11_heart_disease'].map(b11_mapping)
df['B11_kidney_disease'] = df['B11_kidney_disease'].map(b11_mapping)
df['B11_obesity'] = df['B11_obesity'].map(b11_mapping)
df['B12'] = df['B12'].map(b12_mapping)
df['B13_other'] = df['B13_other'].map(b13_other_mapping)

# Map C1 to C8 columns
c_columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
for col in c_columns:
    df[col] = df[col].map(c1_to_c8_mapping)

# Map D1 to D8 columns
d_columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8']
for col in d_columns:
    df[col] = df[col].map(d1_to_d8_mapping)

# Map E1 to E5 columns
e_columns = ['E1', 'E2', 'E3', 'E4', 'E5']
for col in e_columns:
    df[col] = df[col].map(e1_to_e5_mapping)

# Map F1 to F5 columns
f_columns = ['F1', 'F2', 'F3', 'F4', 'F5']
for col in f_columns:
    df[col] = df[col].map(f1_to_f5_mapping)


# Descriptive Statistics
summary_stats = df.describe()

# Data Visualization
# Histogram for age distribution
plt.hist(df['age'], bins=6, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

# Bar chart for sex distribution
plt.figure(figsize=(8, 6))
df['sex'].value_counts().plot(kind='bar')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Sex Distribution')
plt.show()

# Pie chart for Religion Distribution
plt.figure(figsize=(8, 8))
df['religion'].value_counts().plot(kind=' pie', autopct='%1.1f%%')
plt.title('Religion Distribution')
plt.ylabel('')
plt.show()

# Box plot for age distribution by marital status
plt.figure(figsize=(10, 6))
sns.boxplot(x='marital_status', y='age', data=df)
plt.xlabel('Marital Status')
plt.ylabel('Age')
plt.title('Age Distribution by Marital Status')
plt.show()

# Select relevant columns for correlation heatmap
correlation_data = df[['acceptance', 'perception', 'marscore', 'Qscore']]

# Calculate correlations
corr_matrix = correlation_data.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# Stacked bar chart for comorbidity distribution by age group
comorbidity_columns = ['B11_hyperlipedemia', 'B11_stroke', 'B11_hypertension', 'B11_heart_disease', 'B11_kidney_disease', 'B11_obesity']
comorbidity_df = df.groupby('age')[comorbidity_columns].sum()
comorbidity_df.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Comorbidity Distribution by Age Group')
plt.legend(title='Comorbidity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# Define the mediation model using causalml
mediation_model = Mediation(data=df, treatment='acceptance', mediator='marscore', outcome='perception')

# Run the mediation analysis
mediation_results = mediation_model.estimate_effect()
mediation_summary = mediation_model.summary()


# Display mediation results
print(mediation_summary)

# For the association analysis
X = df[['perception', 'marscore']]
y = df['Qscore']

# Add a constant to the independent variables
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Display the regression results
print(model.summary())