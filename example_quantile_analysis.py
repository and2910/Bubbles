#!/usr/bin/env python
# coding: utf-8

"""
Example: Quantile Crash Date Analysis for Gold & Platinum
Based on Geraskin et al. methodology

This script demonstrates the new quantile range analysis functionality
for LPPLS bubble detection.
"""

from bubble_diagnostic import (
    BubbleDiagnostic,
    demo_gold_platinum_quantile_analysis,
    create_comparative_quantile_plot
)
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def example_with_real_data(data_path='GoldPlatData.csv'):
    """
    Example 1: Full analysis with real Gold & Platinum data

    This will:
    - Load the data
    - Run LPPLS analysis on both assets
    - Generate quantile crash date plots
    - Create comparative visualization
    """
    print("="*80)
    print("EXAMPLE 1: Full Gold & Platinum Analysis")
    print("="*80)

    # Run the comprehensive analysis
    results = demo_gold_platinum_quantile_analysis(
        data_path=data_path,
        window_days=504,      # 2 year windows
        step_days=21,         # Monthly steps
        recent_years=10       # Analyze last 10 years
    )

    return results


def example_single_asset(data_path='GoldPlatData.csv'):
    """
    Example 2: Detailed single asset analysis

    Shows how to use individual methods for custom analysis
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Detailed Single Asset Analysis (Gold)")
    print("="*80)

    # Load data
    df = pd.read_csv(data_path)
    dates = pd.to_datetime(df['Dates'])
    gold_prices = df['Gold'].values

    # Filter to recent 5 years
    cutoff = dates.max() - pd.Timedelta(days=5*365)
    mask = dates >= cutoff
    dates = dates[mask]
    gold_prices = gold_prices[mask]

    print(f"\nAnalyzing Gold from {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")

    # Create detector
    detector = BubbleDiagnostic(dates=dates, prices=gold_prices)

    # Run analysis
    print("\nRunning LPPLS analysis...")
    results = detector.analyze(window_days=252, step_days=14)

    if len(results) > 0:
        # Print summary
        detector.print_summary()

        # Generate standard diagnostic plot
        print("\nGenerating standard diagnostic plot...")
        detector.plot_bubble_diagnostics(save_path='example_gold_diagnostic.png')

        # Generate quantile crash date plot
        print("\nGenerating quantile crash date analysis...")
        detector.plot_quantile_crash_dates(
            save_path='example_gold_quantile.png',
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]  # Custom quantiles
        )

        # Export results
        detector.export_results('example_gold_results.csv')

        print("\n" + "="*70)
        print("Analysis complete! Generated files:")
        print("  - example_gold_diagnostic.png")
        print("  - example_gold_quantile.png")
        print("  - example_gold_results.csv")
    else:
        print("\nNo bubble signals detected in this period.")

    return detector


def example_synthetic_data():
    """
    Example 3: Test with synthetic bubble data

    Creates synthetic price data with known bubble and tests detection
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Synthetic Bubble Test")
    print("="*80)

    # Generate synthetic bubble
    print("\nGenerating synthetic data with embedded bubble...")

    n_days = 1000
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

    # LPPLS bubble parameters
    tc_true = 800  # Crash at day 800
    A = 5.0
    B = -0.5
    C = 0.4
    m = 0.7
    omega = 10
    phi = 1.0

    # Generate LPPLS prices
    t = np.arange(n_days)
    dt = np.maximum(tc_true - t, 0.1)
    log_prices = A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))

    # Add noise
    log_prices += np.random.normal(0, 0.02, n_days)
    prices = np.exp(log_prices)

    print(f"Generated {n_days} days of data")
    print(f"True crash date: {dates[tc_true].strftime('%Y-%m-%d')}")

    # Analyze
    detector = BubbleDiagnostic(dates=dates, prices=prices)
    results = detector.analyze(window_days=252, step_days=21)

    if len(results) > 0:
        print(f"\nDetected {len(results)} bubble signals")

        # Plot diagnostics
        detector.plot_bubble_diagnostics(save_path='example_synthetic_diagnostic.png')

        # Plot quantile analysis
        detector.plot_quantile_crash_dates(save_path='example_synthetic_quantile.png')

        # Check if detected tc is close to true tc
        latest = results.iloc[-1]
        detected_tc_day = int((latest['tc_date'] - dates[0]).days)
        error_days = abs(detected_tc_day - tc_true)

        print(f"\nDetection accuracy:")
        print(f"  True tc: Day {tc_true} ({dates[tc_true].strftime('%Y-%m-%d')})")
        print(f"  Detected tc: Day {detected_tc_day} ({latest['tc_date'].strftime('%Y-%m-%d')})")
        print(f"  Error: {error_days} days")

        detector.print_summary()

    return detector


def example_custom_comparison():
    """
    Example 4: Custom comparative analysis

    Shows how to create custom comparative plots
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Comparative Analysis")
    print("="*80)

    # This assumes you've already run analysis on both assets
    # In practice, you'd load or run the analysis first

    print("\nTo create a custom comparison:")
    print("""
    # After running analysis on two detectors:
    detector1 = BubbleDiagnostic(dates=dates1, prices=prices1)
    detector1.analyze()

    detector2 = BubbleDiagnostic(dates=dates2, prices=prices2)
    detector2.analyze()

    # Create comparison plot with custom quantiles
    create_comparative_quantile_plot(
        detector1, detector2,
        label1='Asset A',
        label2='Asset B',
        quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
        save_path='custom_comparison.png'
    )
    """)


def main():
    """Run all examples"""
    import sys

    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  Quantile Crash Date Analysis - Examples                          ║
    ║  Based on Geraskin et al. methodology                             ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    print("\nAvailable examples:")
    print("  1. Full Gold & Platinum analysis (requires GoldPlatData.csv)")
    print("  2. Detailed single asset analysis (requires GoldPlatData.csv)")
    print("  3. Synthetic bubble test (no data required)")
    print("  4. Custom comparative analysis (info only)")
    print("  5. Run all examples")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter example number (1-5): ").strip()

    try:
        if choice == '1':
            example_with_real_data()
        elif choice == '2':
            example_single_asset()
        elif choice == '3':
            example_synthetic_data()
        elif choice == '4':
            example_custom_comparison()
        elif choice == '5':
            # Run all that don't require data file
            example_synthetic_data()
            example_custom_comparison()

            # Try to run with real data if available
            try:
                example_with_real_data()
            except FileNotFoundError:
                print("\nSkipping real data examples (GoldPlatData.csv not found)")
        else:
            print("Invalid choice. Please run with argument 1-5")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure GoldPlatData.csv is in the current directory")
        print("or provide the correct path.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
