#!/usr/bin/env python
# coding: utf-8

"""
Quick unit test for quantile crash date analysis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from bubble_diagnostic import BubbleDiagnostic

def test_quantile_method_exists():
    """Test that the new method exists"""
    print("Test 1: Checking if plot_quantile_crash_dates method exists...")

    # Create simple test data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = np.exp(np.linspace(4, 5, 100) + np.random.normal(0, 0.02, 100))

    detector = BubbleDiagnostic(dates=dates, prices=prices)

    # Check method exists
    assert hasattr(detector, 'plot_quantile_crash_dates'), "Method plot_quantile_crash_dates not found"
    print("✓ Method exists")

    return True

def test_basic_functionality():
    """Test basic functionality with minimal data"""
    print("\nTest 2: Testing basic functionality...")

    # Create simple LPPLS-like data
    n = 300
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')

    # Simple exponential trend
    t = np.arange(n)
    log_prices = 4.5 + 0.002 * t + np.random.normal(0, 0.05, n)
    prices = np.exp(log_prices)

    detector = BubbleDiagnostic(dates=dates, prices=prices)

    print("  Running analysis (small window)...")
    results = detector.analyze(window_days=100, step_days=50)

    print(f"  Found {len(results)} results")

    if len(results) > 0:
        print("  Attempting to plot quantile analysis...")
        try:
            fig = detector.plot_quantile_crash_dates(
                quantiles=[0.25, 0.5, 0.75],
                save_path='test_quantile_output.png'
            )
            if fig is not None:
                print("✓ Quantile plot created successfully")
                import os
                if os.path.exists('test_quantile_output.png'):
                    print("✓ Output file created")
                    os.remove('test_quantile_output.png')
                return True
            else:
                print("  (Not enough data for quantile plot - this is expected)")
                return True
        except Exception as e:
            print(f"✗ Error creating plot: {e}")
            return False
    else:
        print("  (No bubble signals detected - this is okay for test data)")
        return True

def test_demo_function_exists():
    """Test that demo functions exist"""
    print("\nTest 3: Checking demo functions...")

    from bubble_diagnostic import (
        demo_gold_platinum_quantile_analysis,
        create_comparative_quantile_plot
    )

    print("✓ demo_gold_platinum_quantile_analysis exists")
    print("✓ create_comparative_quantile_plot exists")

    return True

def main():
    print("="*70)
    print("UNIT TESTS FOR QUANTILE CRASH DATE ANALYSIS")
    print("="*70)

    tests = [
        test_quantile_method_exists,
        test_basic_functionality,
        test_demo_function_exists
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
