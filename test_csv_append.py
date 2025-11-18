#!/usr/bin/env python
"""
Test script for the append_csv_files_by_date function
"""

import pandas as pd
import numpy as np
import edhec_risk_kit as erk

def test_append_csv_files_by_date():
    """
    Test the append_csv_files_by_date function
    """
    print("Testing append_csv_files_by_date function...")

    # Create sample data to simulate CSV files
    # File 1: Later dates (should appear second when sorted)
    data1 = {
        'Date': [20240201, 20240202, 20240203],
        'Ticker': ['AAPL', 'GOOGL', 'MSFT'],
        'Price': [150.0, 2800.0, 350.0],
        'Volume': [1000000, 500000, 750000]
    }

    # File 2: Earlier dates (should appear first when sorted)
    data2 = {
        'Date': [20240101, 20240102, 20240103],
        'Ticker': ['AAPL', 'GOOGL', 'MSFT'],
        'Price': [145.0, 2750.0, 340.0],
        'Volume': [900000, 450000, 700000]
    }

    # File 3: Middle dates
    data3 = {
        'Date': [20240115, 20240116, 20240117],
        'Ticker': ['AAPL', 'GOOGL', 'MSFT'],
        'Price': [148.0, 2775.0, 345.0],
        'Volume': [950000, 475000, 725000]
    }

    # Save to temporary CSV files
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)

    df1.to_csv('/tmp/test_file1.csv', index=False)
    df2.to_csv('/tmp/test_file2.csv', index=False)
    df3.to_csv('/tmp/test_file3.csv', index=False)

    # Test the function
    file_list = ['test_file1', 'test_file2', 'test_file3']
    output_columns = ['Date', 'Ticker', 'Price', 'Volume']

    result = erk.append_csv_files_by_date(
        file_list=file_list,
        file_path_template='/tmp/{}.csv',
        output_columns=output_columns,
        date_column='Date',
        date_format='%Y%m%d'
    )

    print("\nResult DataFrame:")
    print(result)
    print(f"\nShape: {result.shape}")
    print(f"Date range: {result['Date'].min()} to {result['Date'].max()}")

    # Verify that dates are sorted
    dates_sorted = result['Date'].is_monotonic_increasing
    print(f"\nDates are sorted: {dates_sorted}")

    # Verify that all columns are present
    columns_present = all(col in result.columns for col in output_columns)
    print(f"All columns present: {columns_present}")

    # Verify total number of rows
    expected_rows = len(data1['Date']) + len(data2['Date']) + len(data3['Date'])
    actual_rows = len(result)
    print(f"Expected rows: {expected_rows}, Actual rows: {actual_rows}")

    if dates_sorted and columns_present and expected_rows == actual_rows:
        print("\n✓ Test PASSED!")
        return True
    else:
        print("\n✗ Test FAILED!")
        return False


if __name__ == "__main__":
    test_append_csv_files_by_date()
