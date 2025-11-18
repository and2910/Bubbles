#!/usr/bin/env python
# coding: utf-8

"""
Example usage of CSV utilities for date-organized data
"""

import pandas as pd
from csv_utils import (
    load_csv_by_date,
    append_csv_to_dataframe,
    merge_multiple_csvs,
    save_dataframe_to_csv
)


def example_1_load_single_csv():
    """Example 1: Load a single CSV file organized by date"""
    print("="*70)
    print("EXAMPLE 1: Load a Single CSV File")
    print("="*70)

    # Load CSV - date column will be auto-detected
    df = load_csv_by_date('data.csv')

    # Or specify the date column explicitly
    # df = load_csv_by_date('data.csv', date_column='Date')

    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())

    return df


def example_2_append_to_existing_dataframe():
    """Example 2: Append CSV data to an existing DataFrame"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Append CSV to Existing DataFrame")
    print("="*70)

    # Start with an existing DataFrame
    existing_df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=3, freq='D'),
        'Price': [100, 102, 101],
        'Volume': [1000, 1100, 950]
    })

    print(f"\nOriginal DataFrame ({len(existing_df)} rows):")
    print(existing_df)

    # Append new data from CSV
    combined_df = append_csv_to_dataframe(existing_df, 'new_data.csv')

    print(f"\nCombined DataFrame ({len(combined_df)} rows):")
    print(combined_df.head(10))

    return combined_df


def example_3_merge_multiple_csvs():
    """Example 3: Merge multiple CSV files into one DataFrame"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Merge Multiple CSV Files")
    print("="*70)

    # List of CSV files to merge
    csv_files = [
        'data_2023.csv',
        'data_2024.csv',
        'data_2025.csv'
    ]

    # Merge all files
    merged_df = merge_multiple_csvs(csv_files)

    print(f"\nMerged DataFrame:")
    print(f"  Total rows: {len(merged_df)}")
    print(f"  Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")

    return merged_df


def example_4_iterative_append():
    """Example 4: Iteratively append multiple CSV files"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Iteratively Append CSV Files")
    print("="*70)

    # Start with the first file
    df = load_csv_by_date('initial_data.csv')

    # List of additional files to append
    additional_files = ['update_1.csv', 'update_2.csv', 'update_3.csv']

    # Append each file one by one
    for csv_file in additional_files:
        df = append_csv_to_dataframe(df, csv_file)

    print(f"\nFinal DataFrame has {len(df)} rows")

    return df


def example_5_with_custom_date_column():
    """Example 5: Work with custom date column names"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Date Column Names")
    print("="*70)

    # Load CSV with explicit date column specification
    df = load_csv_by_date('data.csv', date_column='Timestamp')

    print(f"\nLoaded data with 'Timestamp' column:")
    print(df.head())

    return df


def example_6_save_results():
    """Example 6: Save processed DataFrame to CSV"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Save DataFrame to CSV")
    print("="*70)

    # Create or load a DataFrame
    df = load_csv_by_date('data.csv')

    # Do some processing...
    # df = df[df['Price'] > 100]  # Example filtering

    # Save to new CSV file
    save_dataframe_to_csv(df, 'processed_data.csv')

    print("\nData saved successfully!")


def practical_example_time_series_analysis():
    """Practical Example: Building a time series dataset from multiple sources"""
    print("\n" + "="*70)
    print("PRACTICAL EXAMPLE: Time Series Dataset from Multiple Sources")
    print("="*70)

    # Scenario: You have historical data in multiple CSV files and want to
    # combine them into a single, date-sorted DataFrame for analysis

    # Option 1: Merge all at once
    print("\nOption 1: Merge all files at once")
    df_merged = merge_multiple_csvs([
        'historical_2020.csv',
        'historical_2021.csv',
        'historical_2022.csv',
        'historical_2023.csv',
        'historical_2024.csv'
    ])

    # Option 2: Start with base dataset and append updates
    print("\nOption 2: Incremental updates")
    df = load_csv_by_date('base_dataset.csv')

    # Append recent updates
    df = append_csv_to_dataframe(df, 'update_jan_2025.csv')
    df = append_csv_to_dataframe(df, 'update_feb_2025.csv')

    # Save the complete dataset
    save_dataframe_to_csv(df, 'complete_timeseries.csv')

    print(f"\nFinal dataset ready with {len(df)} records")

    return df


def main():
    """Main function to run all examples"""
    print("\n" + "="*70)
    print("CSV UTILITIES - COMPREHENSIVE EXAMPLES")
    print("="*70)

    print("\nThese examples demonstrate how to:")
    print("  1. Load CSV files organized by date")
    print("  2. Append CSV data to existing DataFrames")
    print("  3. Merge multiple CSV files")
    print("  4. Save processed data back to CSV")

    print("\n" + "="*70)
    print("Note: Update the file paths to match your actual data files")
    print("="*70)

    # Uncomment the examples you want to run:
    # example_1_load_single_csv()
    # example_2_append_to_existing_dataframe()
    # example_3_merge_multiple_csvs()
    # example_4_iterative_append()
    # example_5_with_custom_date_column()
    # example_6_save_results()
    # practical_example_time_series_analysis()


if __name__ == "__main__":
    main()
