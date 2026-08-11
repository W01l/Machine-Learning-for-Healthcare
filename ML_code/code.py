# Setup and import

# Show shape
patients_df.shape

# Show data types
patients_df.dtypes

# Show statistical summary
patients_df.describe(include='all')

# Show value counts for all categorical columns
categorical_cols = patients_df.select_dtypes(include=['object', 'category']).columns
{col: patients_df[col].value_counts() for col in categorical_cols}

## 3. Data Cleaning

"""One of the things necessary is to determine whether you have missing values. Handling missing values is always difficult. 
If only a few observations are missing, they can be excluded. If there is a significant number of missing values, several options 
are available depending on the type of missing value.
"""

# Check for standard missing values (NaN)
patients_df.isnull().sum()

# Check for "hidden" missing values in object columns
hidden_missing_values = {}
missing_indicators = ["Unknown", "None", "N/A", "", " "]
for col in patients_df.select_dtypes(include=['object', 'category']).columns:
    counts = patients_df[col].isin(missing_indicators).sum()
    if counts > 0:
        hidden_missing_values[col] = counts
hidden_missing_values

import numpy as np

# Replace "None" and "Unknown" with np.nan in all object and category columns
patients_nan_df = patients_df.replace({"None": np.nan, "Unknown": np.nan})

# Check for missing values in all columns after replacement
patients_nan_df.isnull().sum()

"""
Now you can see~~ ~~the true picture of missing data. The City column has ~50% missing, which is very high to impute reliably.
Also, you can observe that the Primary Condition for 495 patients is missing. On a deeper analysis, you can see that when the 
primary condition is present it has a disease or health condition associated.
"""

# Fill missing values in 'Primary_Condition' with 'No Condition'
patients_nan_replaced = patients_nan_df.copy()
patients_nan_replaced['Primary_Condition'] = patients_nan_replaced['Primary_Condition'].fillna('No Condition')

# Verify the replacement
patients_nan_replaced['Primary_Condition'].isnull().sum(), patients_nan_replaced['Primary_Condition'].value_counts(dropna=False)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set seaborn style and palette
sns.set(style="whitegrid", palette="Set2")

# Identify numeric and categorical columns
numeric_cols = patients_nan_replaced.select_dtypes(include=['number', 'float']).columns.tolist()
# Remove Preventive_Care_Flag if you don't want to plot binary as numeric
# numeric_cols = [col for col in numeric_cols if col != 'Preventive_Care_Flag']

cat_cols = ['Gender', 'Insurance_Type', 'Primary_Condition']

# Number of plots
n_num = len(numeric_cols)
n_cat = len(cat_cols)
total_plots = n_num + n_cat

# Set up grid size (3 columns)
n_cols = 3
n_rows = (total_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
axes = axes.flatten()

# Plot numeric columns: histograms with KDE
for i, col in enumerate(numeric_cols):
    sns.histplot(
        data=patients_nan_replaced,
        x=col,
        kde=True,
        ax=axes[i],
        color=sns.color_palette("Set2")[i % len(sns.color_palette("Set2"))]
    )
    axes[i].set_title(f"{col} Distribution")

# Plot categorical columns: count plots
for j, col in enumerate(cat_cols):
    idx = n_num + j
    sns.countplot(
        data=patients_nan_replaced,
        x=col,
        ax=axes[idx],
        order=patients_nan_replaced[col].value_counts().index,
        palette="Set2"
    )
    axes[idx].set_title(f"{col} Count")
    axes[idx].tick_params(axis='x', rotation=45)

# Remove any unused subplots
for k in range(total_plots, len(axes)):
    fig.delaxes(axes[k])

plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Select only numeric columns
numeric_cols = patients_nan_replaced.select_dtypes(include=[np.number])

# Compute correlation matrix
corr = numeric_cols.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr,
    mask=mask,
    cmap='coolwarm',
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .8}
)

plt.title('Correlation Heatmap (Lower Triangle)')
plt.show()
