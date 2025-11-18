#!/usr/bin/env python
# coding: utf-8

"""
Example: Compare Ticker Dataframes by Date

This script demonstrates how to use the compare_ticker_dataframes function
to identify differences in ticker coverage between two datasets.
"""

import pandas as pd
import numpy as np
from edhec_risk_kit import compare_ticker_dataframes


def example_basic_comparison():
    """
    Example 1: Basic ticker comparison with simple data
    """
    print("="*80)
    print("EXAMPLE 1: Basic Ticker Comparison")
    print("="*80)

    # Create sample data
    df1 = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-02'],
        'Ticker': ['AAPL', 'GOOGL', 'AAPL', 'GOOGL', 'MSFT'],
        'Price': [150, 120, 152, 122, 380]
    })

    df2 = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
        'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        'Price': [150, 380, 122, 250]
    })

    print("\nDataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)

    # Compare the dataframes
    result = compare_ticker_dataframes(df1, df2)

    print("\nComparison Result:")
    print(result)

    # Display details for each date
    print("\n" + "-"*80)
    print("Detailed Breakdown:")
    print("-"*80)
    for _, row in result.iterrows():
        print(f"\nDate: {row['Date'].strftime('%Y-%m-%d')}")
        print(f"  Tickers only in DataFrame 1: {row['Tickers_in_df1_not_df2']}")
        print(f"  Tickers only in DataFrame 2: {row['Tickers_in_df2_not_df1']}")
        print(f"  Count in both: {row['Count_in_both']}")


def example_with_multiple_dates():
    """
    Example 2: Comparison across multiple dates
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Multi-Date Ticker Comparison")
    print("="*80)

    # Generate data with multiple dates
    dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')

    # DataFrame 1: Tech stocks with varying coverage
    df1_data = []
    for date in dates:
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        if date >= pd.Timestamp('2024-01-03'):
            tickers.append('NVDA')  # NVDA added later
        for ticker in tickers:
            df1_data.append({
                'Date': date,
                'Ticker': ticker,
                'Volume': np.random.randint(1000, 10000)
            })
    df1 = pd.DataFrame(df1_data)

    # DataFrame 2: Different set of tech stocks
    df2_data = []
    for date in dates:
        tickers = ['AAPL', 'MSFT', 'TSLA']
        if date <= pd.Timestamp('2024-01-03'):
            tickers.append('META')  # META removed after 2024-01-03
        for ticker in tickers:
            df2_data.append({
                'Date': date,
                'Ticker': ticker,
                'Volume': np.random.randint(1000, 10000)
            })
    df2 = pd.DataFrame(df2_data)

    print(f"\nDataFrame 1 unique tickers: {sorted(df1['Ticker'].unique())}")
    print(f"DataFrame 2 unique tickers: {sorted(df2['Ticker'].unique())}")

    # Compare
    result = compare_ticker_dataframes(df1, df2)

    print("\nComparison Summary:")
    print(result[['Date', 'Count_only_in_df1', 'Count_only_in_df2', 'Count_in_both']])

    # Identify dates with differences
    differences = result[
        (result['Count_only_in_df1'] > 0) | (result['Count_only_in_df2'] > 0)
    ]

    if len(differences) > 0:
        print("\n" + "-"*80)
        print("Dates with Differences:")
        print("-"*80)
        for _, row in differences.iterrows():
            print(f"\nDate: {row['Date'].strftime('%Y-%m-%d')}")
            if row['Tickers_in_df1_not_df2']:
                print(f"  Only in DF1: {row['Tickers_in_df1_not_df2']}")
            if row['Tickers_in_df2_not_df1']:
                print(f"  Only in DF2: {row['Tickers_in_df2_not_df1']}")


def example_export_to_csv():
    """
    Example 3: Export comparison results to CSV
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Export Results to CSV")
    print("="*80)

    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')

    df1 = pd.DataFrame({
        'Date': np.repeat(dates, 5),
        'Ticker': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
                                  size=len(dates)*5),
        'Price': np.random.uniform(100, 500, size=len(dates)*5)
    })

    df2 = pd.DataFrame({
        'Date': np.repeat(dates, 5),
        'Ticker': np.random.choice(['AAPL', 'TSLA', 'NVDA', 'AMZN', 'NFLX'],
                                  size=len(dates)*5),
        'Price': np.random.uniform(100, 500, size=len(dates)*5)
    })

    # Compare
    result = compare_ticker_dataframes(df1, df2)

    # Export to CSV
    output_file = 'ticker_comparison_results.csv'
    result.to_csv(output_file, index=False)
    print(f"\nResults exported to: {output_file}")

    # Also export a summary
    summary = result[['Date', 'Count_only_in_df1', 'Count_only_in_df2', 'Count_in_both']]
    summary_file = 'ticker_comparison_summary.csv'
    summary.to_csv(summary_file, index=False)
    print(f"Summary exported to: {summary_file}")

    print("\nFirst few rows of results:")
    print(result.head())


def example_custom_column_names():
    """
    Example 4: Using custom column names
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Column Names")
    print("="*80)

    # Create data with different column names
    df1 = pd.DataFrame({
        'TradeDate': ['2024-01-01', '2024-01-01', '2024-01-02'],
        'Symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'Value': [100, 200, 300]
    })

    df2 = pd.DataFrame({
        'TradeDate': ['2024-01-01', '2024-01-02', '2024-01-02'],
        'Symbol': ['AAPL', 'GOOGL', 'TSLA'],
        'Value': [100, 200, 400]
    })

    print("\nDataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)

    # Compare using custom column names
    result = compare_ticker_dataframes(
        df1, df2,
        date_col='TradeDate',
        ticker_col='Symbol'
    )

    print("\nComparison Result:")
    print(result)


def main():
    """Run all examples"""
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  Ticker Dataframe Comparison Examples                             ║
    ║  Compare ticker coverage across datasets by date                   ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    # Run all examples
    example_basic_comparison()
    example_with_multiple_dates()
    example_export_to_csv()
    example_custom_column_names()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
