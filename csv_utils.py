#!/usr/bin/env python
# coding: utf-8

"""
CSV Utilities for reading and appending date-organized data
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_csv_by_date(csv_path, date_column=None, sort_by_date=True, parse_dates=True):
    """
    Load a CSV file organized by date and return a DataFrame.

    Args:
        csv_path (str): Path to the CSV file
        date_column (str, optional): Name of the date column. If None, will auto-detect
        sort_by_date (bool): Whether to sort the data by date (default: True)
        parse_dates (bool): Whether to parse date columns (default: True)

    Returns:
        pd.DataFrame: DataFrame with data organized by date

    Example:
        >>> df = load_csv_by_date('data.csv')
        >>> df = load_csv_by_date('data.csv', date_column='Date')
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Auto-detect date column if not specified
    if date_column is None:
        date_column = _detect_date_column(df)
        if date_column is None:
            raise ValueError(f"Could not auto-detect date column. Available columns: {df.columns.tolist()}")
        print(f"Auto-detected date column: '{date_column}'")

    # Verify the date column exists
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found. Available columns: {df.columns.tolist()}")

    # Parse dates if requested
    if parse_dates:
        df[date_column] = pd.to_datetime(df[date_column])

    # Sort by date if requested
    if sort_by_date:
        df = df.sort_values(by=date_column).reset_index(drop=True)

    print(f"Loaded {len(df)} rows from '{csv_path}'")
    print(f"Date range: {df[date_column].min()} to {df[date_column].max()}")

    return df


def append_csv_to_dataframe(existing_df, csv_path, date_column=None, remove_duplicates=True, sort_by_date=True):
    """
    Read a CSV file and append it to an existing DataFrame with the same headers.

    Args:
        existing_df (pd.DataFrame): Existing DataFrame to append to
        csv_path (str): Path to the CSV file to append
        date_column (str, optional): Name of the date column. If None, will auto-detect
        remove_duplicates (bool): Whether to remove duplicate rows (default: True)
        sort_by_date (bool): Whether to sort the combined data by date (default: True)

    Returns:
        pd.DataFrame: Combined DataFrame with appended data

    Example:
        >>> df = pd.DataFrame({'Date': ['2024-01-01'], 'Price': [100]})
        >>> df = append_csv_to_dataframe(df, 'new_data.csv')
    """
    # Load the new data
    new_df = load_csv_by_date(csv_path, date_column=date_column, sort_by_date=False)

    # Verify headers match
    if set(existing_df.columns) != set(new_df.columns):
        raise ValueError(
            f"Column mismatch!\n"
            f"Existing DataFrame columns: {existing_df.columns.tolist()}\n"
            f"New CSV columns: {new_df.columns.tolist()}"
        )

    # Ensure column order matches
    new_df = new_df[existing_df.columns]

    # Append the data
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Remove duplicates if requested
    if remove_duplicates:
        original_len = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        duplicates_removed = original_len - len(combined_df)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")

    # Sort by date if requested
    if sort_by_date and date_column:
        if date_column not in combined_df.columns:
            date_column = _detect_date_column(combined_df)
        if date_column:
            combined_df = combined_df.sort_values(by=date_column).reset_index(drop=True)

    print(f"Combined DataFrame now has {len(combined_df)} rows")

    return combined_df


def merge_multiple_csvs(csv_paths, date_column=None, remove_duplicates=True, sort_by_date=True):
    """
    Read multiple CSV files and merge them into a single DataFrame.

    Args:
        csv_paths (list): List of paths to CSV files
        date_column (str, optional): Name of the date column. If None, will auto-detect
        remove_duplicates (bool): Whether to remove duplicate rows (default: True)
        sort_by_date (bool): Whether to sort the combined data by date (default: True)

    Returns:
        pd.DataFrame: Combined DataFrame from all CSV files

    Example:
        >>> df = merge_multiple_csvs(['data1.csv', 'data2.csv', 'data3.csv'])
    """
    if not csv_paths:
        raise ValueError("No CSV paths provided")

    # Load the first CSV
    combined_df = load_csv_by_date(csv_paths[0], date_column=date_column, sort_by_date=False)
    print(f"\nMerging {len(csv_paths)} CSV files...")

    # Append remaining CSVs
    for csv_path in csv_paths[1:]:
        combined_df = append_csv_to_dataframe(
            combined_df,
            csv_path,
            date_column=date_column,
            remove_duplicates=False,  # Will dedupe at the end
            sort_by_date=False
        )

    # Final cleanup
    if remove_duplicates:
        original_len = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        duplicates_removed = original_len - len(combined_df)
        if duplicates_removed > 0:
            print(f"\nRemoved {duplicates_removed} total duplicate rows")

    # Sort by date
    if sort_by_date:
        if date_column is None:
            date_column = _detect_date_column(combined_df)
        if date_column:
            combined_df = combined_df.sort_values(by=date_column).reset_index(drop=True)

    print(f"\nFinal merged DataFrame has {len(combined_df)} rows")

    return combined_df


def _detect_date_column(df):
    """
    Auto-detect the date column in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to search

    Returns:
        str or None: Name of the date column, or None if not found
    """
    # Common date column names (case insensitive)
    date_keywords = ['date', 'dates', 'datetime', 'time', 'timestamp', 'day']

    for col in df.columns:
        if col.lower() in date_keywords:
            return col

    # Try to detect by checking if column contains date-like values
    for col in df.columns:
        try:
            pd.to_datetime(df[col].iloc[:5])  # Test first 5 rows
            return col
        except:
            continue

    return None


def save_dataframe_to_csv(df, output_path, date_column=None, sort_by_date=True):
    """
    Save a DataFrame to CSV, optionally sorting by date first.

    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path where CSV should be saved
        date_column (str, optional): Name of the date column. If None, will auto-detect
        sort_by_date (bool): Whether to sort by date before saving (default: True)

    Example:
        >>> save_dataframe_to_csv(df, 'output.csv')
    """
    # Sort by date if requested
    if sort_by_date:
        if date_column is None:
            date_column = _detect_date_column(df)
        if date_column:
            df = df.sort_values(by=date_column).reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to '{output_path}'")


# Example usage and demos
if __name__ == "__main__":
    print("CSV Utilities for Date-Organized Data")
    print("=" * 70)
    print("\nExample 1: Load a single CSV file")
    print("  df = load_csv_by_date('data.csv')")
    print("\nExample 2: Append CSV to existing DataFrame")
    print("  df = append_csv_to_dataframe(df, 'new_data.csv')")
    print("\nExample 3: Merge multiple CSV files")
    print("  df = merge_multiple_csvs(['data1.csv', 'data2.csv', 'data3.csv'])")
    print("\nExample 4: Save DataFrame to CSV")
    print("  save_dataframe_to_csv(df, 'output.csv')")
    print("\n" + "=" * 70)

    # Demo with sample data
    print("\nDemo: Creating sample data...")

    # Create sample data
    sample_data1 = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=5, freq='D'),
        'Price': [100, 102, 101, 103, 105],
        'Volume': [1000, 1100, 950, 1200, 1150]
    })

    sample_data2 = pd.DataFrame({
        'Date': pd.date_range('2024-01-06', periods=5, freq='D'),
        'Price': [106, 108, 107, 109, 110],
        'Volume': [1300, 1250, 1400, 1350, 1500]
    })

    # Save sample CSVs
    sample_data1.to_csv('sample_data1.csv', index=False)
    sample_data2.to_csv('sample_data2.csv', index=False)
    print("Created sample_data1.csv and sample_data2.csv")

    # Demo loading
    print("\n--- Demo: Load single CSV ---")
    df1 = load_csv_by_date('sample_data1.csv')
    print(df1.head())

    # Demo appending
    print("\n--- Demo: Append CSV to DataFrame ---")
    df_combined = append_csv_to_dataframe(df1, 'sample_data2.csv')
    print(df_combined)

    # Demo merging multiple
    print("\n--- Demo: Merge multiple CSVs ---")
    df_merged = merge_multiple_csvs(['sample_data1.csv', 'sample_data2.csv'])
    print(df_merged)

    # Demo saving
    print("\n--- Demo: Save to CSV ---")
    save_dataframe_to_csv(df_merged, 'merged_output.csv')

    print("\nDemo complete!")
