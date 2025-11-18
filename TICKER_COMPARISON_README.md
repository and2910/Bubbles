# Ticker Dataframe Comparison

This document explains how to use the `compare_ticker_dataframes()` function to compare ticker coverage between two dataframes by date.

## Overview

The `compare_ticker_dataframes()` function identifies which tickers appear in one dataframe but not the other for each date. This is useful for:
- Detecting missing data in one dataset
- Comparing ticker coverage across different data sources
- Identifying when tickers were added or removed from datasets
- Data quality validation

## Function Signature

```python
def compare_ticker_dataframes(df1, df2, date_col='Date', ticker_col='Ticker')
```

### Parameters

- **df1**: First DataFrame with date and ticker columns
- **df2**: Second DataFrame with date and ticker columns
- **date_col** (optional): Name of the date column (default: 'Date')
- **ticker_col** (optional): Name of the ticker column (default: 'Ticker')

### Returns

A DataFrame with the following columns:
- **Date**: The date
- **Tickers_in_df1_not_df2**: List of tickers present in df1 but not in df2
- **Tickers_in_df2_not_df1**: List of tickers present in df2 but not in df1
- **Count_only_in_df1**: Number of tickers only in df1
- **Count_only_in_df2**: Number of tickers only in df2
- **Count_in_both**: Number of tickers present in both dataframes

## Usage Examples

### Basic Example

```python
import pandas as pd
from edhec_risk_kit import compare_ticker_dataframes

# Create two dataframes with ticker data
df1 = pd.DataFrame({
    'Date': ['2024-01-01', '2024-01-01', '2024-01-02'],
    'Ticker': ['AAPL', 'GOOGL', 'MSFT'],
    'Price': [150, 120, 380]
})

df2 = pd.DataFrame({
    'Date': ['2024-01-01', '2024-01-02', '2024-01-02'],
    'Ticker': ['AAPL', 'GOOGL', 'TSLA'],
    'Price': [150, 122, 250]
})

# Compare the dataframes
result = compare_ticker_dataframes(df1, df2)
print(result)
```

### Custom Column Names

If your dataframes use different column names:

```python
df1 = pd.DataFrame({
    'TradeDate': ['2024-01-01', '2024-01-02'],
    'Symbol': ['AAPL', 'GOOGL'],
    'Value': [100, 200]
})

df2 = pd.DataFrame({
    'TradeDate': ['2024-01-01', '2024-01-02'],
    'Symbol': ['AAPL', 'TSLA'],
    'Value': [100, 400]
})

result = compare_ticker_dataframes(
    df1, df2,
    date_col='TradeDate',
    ticker_col='Symbol'
)
```

### Export Results to CSV

```python
# Compare dataframes
result = compare_ticker_dataframes(df1, df2)

# Export full results
result.to_csv('ticker_comparison_results.csv', index=False)

# Export just the summary counts
summary = result[['Date', 'Count_only_in_df1', 'Count_only_in_df2', 'Count_in_both']]
summary.to_csv('ticker_comparison_summary.csv', index=False)
```

### Identify Dates with Differences

```python
result = compare_ticker_dataframes(df1, df2)

# Find dates where there are differences
differences = result[
    (result['Count_only_in_df1'] > 0) | (result['Count_only_in_df2'] > 0)
]

for _, row in differences.iterrows():
    print(f"\nDate: {row['Date']}")
    if row['Tickers_in_df1_not_df2']:
        print(f"  Only in DF1: {row['Tickers_in_df1_not_df2']}")
    if row['Tickers_in_df2_not_df1']:
        print(f"  Only in DF2: {row['Tickers_in_df2_not_df1']}")
```

## Example Script

A complete example script is provided in `example_ticker_comparison.py`. Run it with:

```bash
python example_ticker_comparison.py
```

This script demonstrates:
1. Basic ticker comparison
2. Multi-date ticker comparison
3. Exporting results to CSV
4. Using custom column names

## Use Cases

### Data Quality Validation

```python
# Check if production data has all tickers from reference data
result = compare_ticker_dataframes(reference_df, production_df)

# Identify missing tickers
missing_tickers = result[result['Count_only_in_df1'] > 0]
if len(missing_tickers) > 0:
    print("WARNING: Missing tickers in production data:")
    print(missing_tickers[['Date', 'Tickers_in_df1_not_df2']])
```

### Tracking Coverage Changes

```python
# Compare historical data to current data
result = compare_ticker_dataframes(historical_df, current_df)

# Identify when tickers were added
new_tickers = result[result['Count_only_in_df2'] > 0]
print("Dates when new tickers appeared:")
print(new_tickers[['Date', 'Tickers_in_df2_not_df1']])
```

### Dataset Reconciliation

```python
# Compare two data sources
result = compare_ticker_dataframes(source1_df, source2_df)

# Summary statistics
print(f"Total dates analyzed: {len(result)}")
print(f"Dates with perfect match: {len(result[result['Count_only_in_df1'] == 0])}")
print(f"Average tickers in both: {result['Count_in_both'].mean():.1f}")
```

## Function Location

The function is located in `edhec_risk_kit.py` at line 765.

## Requirements

- pandas
- numpy

## See Also

- `example_ticker_comparison.py` - Complete working examples
- `edhec_risk_kit.py` - Source code
