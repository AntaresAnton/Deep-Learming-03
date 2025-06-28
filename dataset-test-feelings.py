# Import necessary libraries
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set display options for better readability
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 1000)

# Define file paths
file_paths = {"train": "train.csv", "test": "test.csv", "validation": "validation.csv"}

# Check if files exist
print("Checking if files exist:")
for name, path in file_paths.items():
    if os.path.exists(path):
        print(f"✅ {path} exists")
    else:
        print(f"❌ {path} does not exist")
print("\n")

# Load the datasets
datasets = {}
for name, path in file_paths.items():
    if os.path.exists(path):
        datasets[name] = pd.read_csv(path)
        print(f"Loaded {name} dataset with shape: {datasets[name].shape}")
    else:
        print(f"Could not load {name} dataset")
print("\n")


# Function to check column consistency
def check_column_consistency(datasets):
    print("Checking column consistency:")
    if len(datasets) < 2:
        print("Not enough datasets to compare columns")
        return

    # Get column sets for each dataset
    column_sets = {name: set(df.columns) for name, df in datasets.items()}

    # Compare columns across datasets
    reference_name = list(datasets.keys())[0]
    reference_columns = column_sets[reference_name]

    for name, columns in column_sets.items():
        if name == reference_name:
            continue

        # Check for missing columns
        missing_columns = reference_columns - columns
        extra_columns = columns - reference_columns

        if missing_columns:
            print(f"❌ {name} is missing columns: {missing_columns}")
        else:
            print(f"✅ {name} has all columns from {reference_name}")

        if extra_columns:
            print(f"⚠️ {name} has extra columns: {extra_columns}")
    print("\n")


# Function to check data types
def check_data_types(datasets):
    print("Checking data types:")
    if len(datasets) < 2:
        print("Not enough datasets to compare data types")
        return

    # Get data types for each dataset
    dtypes = {name: df.dtypes for name, df in datasets.items()}

    # Compare data types across datasets
    reference_name = list(datasets.keys())[0]
    reference_dtypes = dtypes[reference_name]

    for name, dtype_series in dtypes.items():
        if name == reference_name:
            continue

        # Check for columns with different data types
        common_columns = set(dtype_series.index) & set(reference_dtypes.index)
        for col in common_columns:
            if dtype_series[col] != reference_dtypes[col]:
                print(
                    f"❌ Column '{col}' has different data types: {reference_name}={reference_dtypes[col]}, {name}={dtype_series[col]}"
                )

    # Display data types for each dataset
    for name, df in datasets.items():
        print(f"\nData types for {name} dataset:")
        print(df.dtypes)
    print("\n")


# Function to check for missing values
def check_missing_values(datasets):
    print("Checking for missing values:")
    for name, df in datasets.items():
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]

        if missing_cols.empty:
            print(f"✅ {name} dataset has no missing values")
        else:
            print(f"⚠️ {name} dataset has missing values:")
            for col, count in missing_cols.items():
                print(f"   - {col}: {count} missing values ({count/len(df)*100:.2f}%)")
    print("\n")


# Function to check data order (assuming there's a common ID or timestamp column)
def check_data_order(datasets, order_column=None):
    print("Checking data order:")

    # If no order column is specified, try to find a common ID or timestamp column
    if order_column is None:
        common_columns = set.intersection(
            *[set(df.columns) for df in datasets.values()]
        )
        potential_order_columns = [
            col
            for col in common_columns
            if any(
                keyword in col.lower()
                for keyword in ["id", "time", "date", "timestamp", "order"]
            )
        ]

        if potential_order_columns:
            order_column = potential_order_columns[0]
            print(f"Using '{order_column}' as the order column")
        else:
            print("Could not identify a suitable order column. Please specify one.")
            return

    # Check if the order column exists in all datasets
    if not all(order_column in df.columns for df in datasets.values()):
        print(f"❌ '{order_column}' is not present in all datasets")
        return

    # Check if the order column is sorted in each dataset
    for name, df in datasets.items():
        is_sorted = df[order_column].is_monotonic_increasing
        if is_sorted:
            print(f"✅ {name} dataset is sorted by '{order_column}'")
        else:
            print(f"❌ {name} dataset is NOT sorted by '{order_column}'")
    print("\n")


# Function to visualize data distributions
def visualize_distributions(datasets, max_columns=5):
    print("Visualizing data distributions:")

    if not datasets:
        print("No datasets to visualize")
        return

    # Get common numeric columns
    numeric_columns = {}
    for name, df in datasets.items():
        numeric_columns[name] = df.select_dtypes(include=[np.number]).columns.tolist()

    common_numeric_columns = set.intersection(
        *[set(cols) for cols in numeric_columns.values()]
    )

    if not common_numeric_columns:
        print("No common numeric columns to visualize")
        return

    # Limit the number of columns to visualize
    columns_to_plot = list(common_numeric_columns)[:max_columns]

    # Create histograms for each column
    for col in columns_to_plot:
        plt.figure(figsize=(12, 6))
        for name, df in datasets.items():
            sns.histplot(df[col].dropna(), label=name, kde=True, alpha=0.5)

        plt.title(f"Distribution of {col} across datasets")
        plt.legend()
        plt.show()
    print("\n")


# Function to check for duplicates
def check_duplicates(datasets):
    print("Checking for duplicates:")
    for name, df in datasets.items():
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(
                f"⚠️ {name} dataset has {duplicates} duplicate rows ({duplicates/len(df)*100:.2f}%)"
            )
        else:
            print(f"✅ {name} dataset has no duplicate rows")
    print("\n")


# Function to check value ranges
def check_value_ranges(datasets):
    print("Checking value ranges for numeric columns:")

    if not datasets:
        print("No datasets to check")
        return

    # Get common numeric columns
    numeric_columns = {}
    for name, df in datasets.items():
        numeric_columns[name] = df.select_dtypes(include=[np.number]).columns.tolist()

    common_numeric_columns = set.intersection(
        *[set(cols) for cols in numeric_columns.values()]
    )

    if not common_numeric_columns:
        print("No common numeric columns to check")
        return

    # Check min and max values for each column
    for col in common_numeric_columns:
        print(f"\nColumn: {col}")
        for name, df in datasets.items():
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"  {name}: min={min_val}, max={max_val}")
    print("\n")


# Run all checks
if datasets:
    check_column_consistency(datasets)
    check_data_types(datasets)
    check_missing_values(datasets)
    check_duplicates(datasets)
    check_value_ranges(datasets)

    # Try to identify an order column (you can specify one if needed)
    check_data_order(datasets)

    # Visualize distributions of common numeric columns
    visualize_distributions(datasets)

    print("Data verification complete!")
else:
    print("No datasets were loaded. Please check file paths.")
