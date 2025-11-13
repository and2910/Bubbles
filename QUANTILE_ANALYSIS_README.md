# Quantile Crash Date Analysis for LPPLS Bubble Detection

## Overview

This implementation adds **quantile range analysis for predicted crash dates** to the LPPLS bubble diagnostic system, based on the methodology described in **Geraskin et al.** - "Everything You Always Wanted to Know about Log Period Power Laws for Bubble Modelling but Were Afraid to Ask" (page 29).

## New Features

### 1. `plot_quantile_crash_dates()` Method

Visualizes the distribution of predicted crash dates (tc) over time using quantile bands.

**Key Insights:**
- Shows how crash date predictions evolve as new data arrives
- Quantile bands represent uncertainty in predictions
- Converging bands suggest increasing confidence in crash timing
- Diverging bands suggest market uncertainty or transition periods

**Usage:**
```python
from bubble_diagnostic import BubbleDiagnostic

# Load your data
detector = BubbleDiagnostic('data.csv')
detector.analyze()

# Generate quantile crash date plot
detector.plot_quantile_crash_dates(
    save_path='quantile_analysis.png',
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]  # Customizable quantiles
)
```

**Interpretation:**
- **Median line (50th percentile)**: Central prediction of crash date
- **Quantile bands**: Show distribution of predictions
  - Narrow bands = high confidence/consensus
  - Wide bands = high uncertainty
- **Diagonal reference line**: Shows "present" (where tc = observation date)
- Predictions above the diagonal are future crash dates
- Predictions converging to a specific date suggest imminent crash

### 2. `demo_gold_platinum_quantile_analysis()` Function

Comprehensive analysis function for Gold & Platinum data with automatic quantile visualization.

**Features:**
- Analyzes both assets automatically
- Generates all diagnostic plots including quantile analysis
- Creates comparative visualization
- Exports results to CSV

**Usage:**
```python
from bubble_diagnostic import demo_gold_platinum_quantile_analysis

# Full analysis
results = demo_gold_platinum_quantile_analysis(
    data_path='GoldPlatData.csv',
    window_days=504,      # 2-year analysis windows
    step_days=21,         # Monthly steps
    recent_years=10       # Analyze last 10 years (optional)
)

# Access individual detectors
gold_detector = results['gold']
platinum_detector = results['platinum']
```

**Generated Files:**
- `gold_bubble_diagnostic.png` - Standard LPPLS diagnostic plot
- `gold_quantile_crash_dates.png` - Quantile analysis for gold
- `gold_bubble_results.csv` - Detailed results
- `platinum_bubble_diagnostic.png` - Standard LPPLS diagnostic plot
- `platinum_quantile_crash_dates.png` - Quantile analysis for platinum
- `platinum_bubble_results.csv` - Detailed results
- `gold_platinum_quantile_comparison.png` - Side-by-side comparison

### 3. `create_comparative_quantile_plot()` Function

Creates side-by-side quantile plots for two assets.

**Usage:**
```python
from bubble_diagnostic import create_comparative_quantile_plot

create_comparative_quantile_plot(
    detector1=gold_detector,
    detector2=platinum_detector,
    label1='Gold',
    label2='Platinum',
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    save_path='comparison.png'
)
```

## Methodology

### What are Quantile Ranges of Crash Dates?

In LPPLS analysis, each rolling window produces an estimate of tc (the critical time or expected crash date). These estimates have uncertainty. The quantile range analysis:

1. **Collects all tc predictions** made at each observation point
2. **Calculates quantiles** (e.g., 10%, 25%, 50%, 75%, 90%) of the distribution
3. **Plots these quantiles over time** to show:
   - Central tendency (median)
   - Spread/uncertainty (quantile ranges)
   - Evolution of predictions

### Based on Geraskin et al. (Page 29)

The paper describes this technique for assessing:
- **Stability of predictions**: Do predictions converge or diverge over time?
- **Confidence intervals**: How much uncertainty exists in crash timing?
- **Early warning signals**: Narrowing quantile bands can indicate approaching crash

### Interpretation Guide

**Converging Quantiles:**
```
      |  [====]    narrowing bands
  t1  |   [==]
  t2  |    []      converging to specific date
      +------------> time
```
→ Increasing confidence in crash timing

**Diverging Quantiles:**
```
      |  [=]
  t1  | [====]     widening bands
  t2  |[========]
      +------------> time
```
→ Increasing uncertainty or market transition

**Stable Quantiles:**
```
      | [====]
  t1  | [====]     consistent bands
  t2  | [====]
      +------------> time
```
→ Stable bubble with consistent predicted timing

## Examples

See `example_quantile_analysis.py` for complete examples:

```bash
# Run all examples
python example_quantile_analysis.py 5

# Run specific examples
python example_quantile_analysis.py 1  # Full Gold & Platinum analysis
python example_quantile_analysis.py 2  # Detailed single asset
python example_quantile_analysis.py 3  # Synthetic bubble test
```

## Quick Test

Verify installation with unit tests:

```bash
python test_quantile_analysis.py
```

Should output:
```
======================================================================
UNIT TESTS FOR QUANTILE CRASH DATE ANALYSIS
======================================================================
Test 1: Checking if plot_quantile_crash_dates method exists...
✓ Method exists
...
RESULTS: 3 passed, 0 failed
======================================================================
```

## Integration with Existing Code

The new functionality integrates seamlessly with existing `BubbleDiagnostic` code:

```python
# Existing workflow
detector = BubbleDiagnostic('data.csv')
detector.analyze()
detector.plot_bubble_diagnostics()  # Standard plot
detector.print_summary()

# NEW: Add quantile analysis
detector.plot_quantile_crash_dates()  # Quantile plot
```

## Technical Details

### Quantile Calculation

For each observation date t:
1. Collect all tc predictions where tc > t (future predictions)
2. Calculate specified quantiles of these predictions
3. Plot quantile values vs observation date

### Default Quantiles

Default: `[0.1, 0.25, 0.5, 0.75, 0.9]`
- 10% and 90%: Outer uncertainty bounds
- 25% and 75%: Inner confidence range
- 50%: Median (central prediction)

Can be customized to any set of quantiles in range (0, 1).

### Minimum Data Requirements

- Need at least 3 future tc predictions per observation point
- Typically requires ~10+ successful LPPLS fits for meaningful quantiles
- More data = more robust quantile estimates

## References

**Geraskin, P., & Fantazzini, D.** (2013). "Everything You Always Wanted to Know about Log Period Power Laws for Bubble Modelling but Were Afraid to Ask." *The European Journal of Finance*, 19(5), 366-391.

Key contributions:
- Methodology for quantile-based crash date analysis
- Interpretation guidelines for quantile evolution
- Application to real financial bubbles

## File Structure

```
bubble_diagnostic.py              # Main module with new functions
example_quantile_analysis.py      # Usage examples and demos
test_quantile_analysis.py         # Unit tests
QUANTILE_ANALYSIS_README.md       # This file
```

## Requirements

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.3.0
scipy>=1.7.0
```

Install with:
```bash
pip install numpy pandas matplotlib scipy
```

## Support

For issues or questions:
1. Check examples in `example_quantile_analysis.py`
2. Run unit tests: `python test_quantile_analysis.py`
3. Review original paper (Geraskin et al., 2013, page 29)

## License

Same as parent project.
