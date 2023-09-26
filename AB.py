from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import os
from tabulate import tabulate
# build sub directory using BASE_DIRE / sub_dir
BASE_DIR = Path(__file__).resolve().parent

# Load the CSV file into a pandas DataFrame
file_path = (BASE_DIR/"AB_data.csv")
df = pd.read_csv(file_path)

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
# plt.hist(str(df['age']), bins=6, edgecolor='black')
# plt.xlabel('Age')
# # plt.ylabel('Frequency')
# plt.title('Age Distribution')
# plt.show()

# Bar chart for sex distribution
plt.figure(figsize=(8, 6))
df['sex'].value_counts().plot(kind='bar')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Sex Distribution')
plt.show()

# Pie chart for Religion Distribution
plt.figure(figsize=(8, 8))
df['religion'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Religion Distribution')
plt.ylabel('')
plt.show()

# Count plot for marital status within each age group
plt.figure(figsize=(10, 6))
sns.countplot(x='age', hue='marital_status', data=df)
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Marital Status Distribution within Each Age Group')
plt.legend(title='Marital Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# Select relevant columns for correlation heatmap
correlation_data = df[['acceptancescore', 'perceptionscore', 'marscore', 'Qscore']]

# Calculate correlations
corr_matrix = correlation_data.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# Stacked bar chart for comorbidity distribution by age group
comorbidity_columns = ['B11_hyperlipedemia', 'B11_stroke', 'B11_hypertension', 'B11_heart_disease', 'B11_kidney_disease', 'B11_obesity']

# Convert the values in the comorbidity columns to numeric (1 for 'Yes', 0 for 'No')
for col in comorbidity_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

comorbidity_df = df.groupby('age')[comorbidity_columns].sum()

# Plotting the stacked bar chart
comorbidity_df.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Comorbidity Distribution by Age Group')
plt.legend(title='Comorbidity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Mediation analysis
# print(help(Mediation))
X = df[['acceptancescore']]
M = df[['marscore']]
Y = df[['perceptionscore']]
# mediation_model = Mediation(data=df, treatment='marscore', outcome='acceptancescore', mediator='perceptionscore')
# mediation_results = mediation_model.calculate(alpha=0.05, bootstrap=1000)
# print(mediation_results)
# Create a mediation model
# Extract the relevant columns
X = df['acceptancescore']
M = df['marscore']
Y = df['perceptionscore']

# Step 1: Regress the mediator (M) on the independent variable (X)
X = sm.add_constant(X)  # Add a constant term to the independent variable
model_mediator = sm.OLS(M, X).fit()

# Step 2: Regress the dependent variable (Y) on both the independent variable (X) and the mediator (M)
X = df[['acceptancescore', 'marscore']]
X = sm.add_constant(X)  # Add a constant term to the independent variables
model_total = sm.OLS(Y, X).fit()

# Step 3: Calculate the direct effect (c') and the indirect effect (a*b)
c_prime = model_total.params['acceptancescore']
a_b = model_mediator.params['acceptancescore'] * model_total.params['marscore']

# Step 4: Calculate the total effect (c)
c = c_prime + a_b

# Step 5: Calculate the mediated effect (indirect effect) as the difference between the total effect (c) and the direct effect (c')
mediated_effect = c - c_prime

# Display the results
print(f"Direct Effect (c'): {c_prime}")
print(f"Indirect Effect (a*b): {a_b}")
print(f"Total Effect (c): {c}")
print(f"Mediated Effect (Indirect Effect): {mediated_effect}")

# For the association analysis
X = df[['perceptionscore', 'marscore']]
y = df['Qscore']

# Add a constant to the independent variables
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Save the regression results summary to a CSV file
regression_results_summary = model.summary()
with open(BASE_DIR / 'regression_results_summary.txt', 'w') as f:
    f.write(regression_results_summary.as_text())

# Save summary statistics to a new CSV file
summary_stats.to_csv(BASE_DIR / 'summary_stats.csv')

# Extract regression results to a DataFrame
regression_results_df = pd.DataFrame({
    'Coefficient': model.params,
    'Standard Error': model.bse,
    't-value': model.tvalues,
    'p-value': model.pvalues
})

# Save the regression results to a new CSV file
regression_results_df.to_csv(BASE_DIR / 'regression_results.csv')


# Table 1: Demographic Statistics
# Include sub breakdowns for certain variables
demographics_data = {
    'Variable': ['sex (Male)', 'sex (Female)', 'marital_status (Single)', 'marital_status (Married)',
                 'marital_status (Divorced)', 'marital_status (Widowed)', 'level_of_education (None)',
                 'level_of_education (Primary)', 'level_of_education (JHS)', 'level_of_education (SHS)',
                 'level_of_education (Tertiary)', 'occupation (Unemployed)', 'occupation (Employed)',
                 'occupation (Self-employed)', 'occupation (Other)', 'religion (Christian)', 'religion (Muslim)',
                 'religion (Traditionalist)', 'religion (Other)'],
    'N (Number)': [
        df[df['sex'] == 'Male']['sex'].count(),
        df[df['sex'] == 'Female']['sex'].count(),
        df[df['marital_status'] == 'Single']['marital_status'].count(),
        df[df['marital_status'] == 'Married']['marital_status'].count(),
        df[df['marital_status'] == 'Divorced']['marital_status'].count(),
        df[df['marital_status'] == 'Widowed']['marital_status'].count(),
        df[df['level_of_education'] == 'None']['level_of_education'].count(),
        df[df['level_of_education'] == 'Primary']['level_of_education'].count(),
        df[df['level_of_education'] == 'JHS']['level_of_education'].count(),
        df[df['level_of_education'] == 'SHS']['level_of_education'].count(),
        df[df['level_of_education'] == 'Tertiary']['level_of_education'].count(),
        df[df['occupation'] == 'Unemployed']['occupation'].count(),
        df[df['occupation'] == 'Employed']['occupation'].count(),
        df[df['occupation'] == 'Self-employed']['occupation'].count(),
        df[df['occupation'] == 'Other']['occupation'].count(),
        df[df['religion'] == 'Christian']['religion'].count(),
        df[df['religion'] == 'Muslim']['religion'].count(),
        df[df['religion'] == 'Traditionalist']['religion'].count(),
        df[df['religion'] == 'Other']['religion'].count()
    ]
}

# Calculate the total count for each variable
variable_total_counts = [sum(demographics_data['N (Number)'][i:i+2]) for i in range(0, len(demographics_data['Variable']), 2)]

# Calculate the percentages for each subcategory within each variable
variable_percentages = []
for i in range(0, len(demographics_data['Variable']), 2):
    total_count = variable_total_counts[i // 2]
    percentages = [(count / total_count) * 100 for count in demographics_data['N (Number)'][i:i+2]]
    variable_percentages.extend(['{:.2f}%'.format(pct) for pct in percentages])

# Update the demographics data with the correct percentage column
demographics_data['%'] = variable_percentages

## Update the percentages for each variable to ensure they add up to 100%
for i in range(0, len(demographics_data['Variable']), 2):
    total_count = sum(demographics_data['N (Number)'][i:i+2])
    percentages = [(count / total_count) * 100 for count in demographics_data['N (Number)'][i:i+2]]
    
    # Update percentages to ensure they add up to 100%
    total_percentage = sum(percentages)
    adjusted_percentages = ['{:.2f}%'.format(pct / total_percentage * 100) for pct in percentages]
    
    # Assign adjusted percentages to respective subcategories
    demographics_data['%'][i:i+2] = adjusted_percentages
    
# Display the demographic statistics table
print("Table 1: Demographic Statistics")
print(tabulate(demographics_data, headers='keys', tablefmt='pretty', showindex=False))

# Save demographic statistics to a CSV file
demographics_df = pd.DataFrame(demographics_data)
demographics_df.to_csv(BASE_DIR / 'demographics_statistics.csv', index=False)

# Table 2: Means, Standard Deviations, and Ranges for Qscore, Perception Scores, Acceptance Scores, and Marscore
means_std_ranges_table = df[['Qscore', 'perceptionscore', 'acceptancescore', 'marscore']].describe()
means_std_ranges_table.to_csv(BASE_DIR / 'means_std_ranges_table.csv')

# Table 3: Correlations between Acceptance Score, Qscore, Marscore, and Perception Score
correlation_matrix = df[['acceptancescore', 'Qscore', 'marscore', 'perceptionscore']].corr()
correlation_matrix.to_csv(BASE_DIR / 'correlation_matrix.csv')


# Table 5: Mediation Model Results
mediation_results = pd.DataFrame({
    'Measure': ['Direct Effect (c\')', 'Indirect Effect (a*b)', 'Total Effect (c)', 'Mediated Effect (Indirect Effect)'],
    'Value': [c_prime, a_b, c, mediated_effect]
})
mediation_results.to_csv(BASE_DIR / 'mediation_results.csv')

# Save demographic statistics to a CSV file
demographics_df = pd.DataFrame(demographics_data)
demographics_df.to_csv(BASE_DIR / 'demographics_statistics.csv', index=False)

# Save Table 2: Means, Standard Deviations, and Ranges for Qscore, Perception Scores, Acceptance Scores, and Marscore
means_std_ranges_table.to_csv(BASE_DIR / 'means_std_ranges_table.csv')

# Save Table 3: Correlations between Acceptance Score, Qscore, Marscore, and Perception Score
correlation_matrix.to_csv(BASE_DIR / 'correlation_matrix.csv')

# Save Table 5: Mediation Model Results
mediation_results.to_csv(BASE_DIR / 'mediation_results.csv')