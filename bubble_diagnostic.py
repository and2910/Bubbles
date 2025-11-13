#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Generalized LPPLS Bubble Diagnostic System
Simplified version focused on bubble detection and visualization

Based on Sornette et al. (2013): "Financial Bubbles: mechanisms and diagnostics"

Usage:
    detector = BubbleDiagnostic('data.csv')  # CSV with Date, Price columns
    detector.analyze()
    detector.plot_bubble_diagnostics()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize, differential_evolution
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class LPPLSFitter:
    """
    Core LPPLS model fitting
    
    LPPLS model: ln(p(t)) = A + B(tc - t)^m [1 + C*cos(ω*ln(tc - t) + φ)]
    
    Parameters:
        tc: Critical time (expected regime change)
        A: Log price level at tc
        B: Amplitude (B<0 for bubble)
        C: Oscillation amplitude
        m: Power law exponent (0.1 - 0.9)
        ω: Angular frequency (6 - 13)
        φ: Phase
    """
    
    def __init__(self):
        self.fitted_params = None
        
    def lppls(self, t, tc, A, B, C, m, omega, phi):
        """LPPLS function"""
        dt = np.maximum(tc - t, 1e-10)
        return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))
    
    def fit(self, time_array, log_price_array):
        """
        Fit LPPLS model to data
        
        Args:
            time_array: Days since first observation
            log_price_array: Log of prices
            
        Returns:
            dict with parameters and fit quality, or None if failed
        """
        # Convert to numpy arrays to support negative indexing
        t = np.asarray(time_array)
        p = np.asarray(log_price_array)

        # Parameter bounds
        tc_min = t[-1] + 1
        tc_max = t[-1] + 365
        
        bounds = [
            (tc_min, tc_max),           # tc
            (p.min() - 0.5, p.max() + 0.5),  # A
            (-2, 2),                    # B
            (-2, 2),                    # C
            (0.01, 0.99),               # m
            (6, 13),                    # omega
            (0, 2*np.pi)                # phi
        ]
        
        def cost(params):
            pred = self.lppls(t, *params)
            return np.sum((p - pred) ** 2)
        
        # Try multiple starting points
        best_result = None
        best_cost = np.inf
        
        for _ in range(5):
            x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
            
            try:
                result = minimize(cost, x0, method='L-BFGS-B', bounds=bounds)
                
                if result.success and result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result
            except:
                continue
        
        # Try differential evolution if needed
        if best_result is None or best_cost > 1.0:
            try:
                result = differential_evolution(cost, bounds, seed=42, maxiter=1000)
                best_result = result
            except:
                return None
        
        if best_result is None:
            return None
        
        # Extract parameters
        tc, A, B, C, m, omega, phi = best_result.x
        
        # Calculate fit quality
        pred = self.lppls(t, tc, A, B, C, m, omega, phi)
        residuals = p - pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((p - np.mean(p)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'tc': tc,
            'A': A,
            'B': B,
            'C': C,
            'm': m,
            'omega': omega,
            'phi': phi,
            'r_squared': r_squared,
            'residuals': residuals
        }


class BubbleDiagnostic:
    """
    Comprehensive bubble diagnostic system
    
    Analyzes price time series to detect:
    1. Early stage bubble formation
    2. Mature bubble presence
    3. Expected regime change window
    """
    
    def __init__(self, data_path=None, dates=None, prices=None):
        """
        Initialize with either CSV path or data arrays
        
        Args:
            data_path: Path to CSV with 'Date' and 'Price' columns
            dates: Array of dates (if not using CSV)
            prices: Array of prices (if not using CSV)
        """
        if data_path:
            self.load_from_csv(data_path)
        elif dates is not None and prices is not None:
            self.dates = pd.to_datetime(dates)
            self.prices = np.array(prices)
        else:
            raise ValueError("Must provide either data_path or (dates, prices)")
        
        self.fitter = LPPLSFitter()
        self.results = []
        
    def load_from_csv(self, path):
        """Load data from CSV with Date and Price columns"""
        df = pd.read_csv(path)
        
        # Find date column (case insensitive)
        date_col = None
        for col in df.columns:
            if col.lower() in ['date', 'dates', 'datetime', 'time']:
                date_col = col
                break
        
        # Find price column
        price_col = None
        for col in df.columns:
            if col.lower() in ['price', 'close', 'value']:
                price_col = col
                break
        
        if date_col is None or price_col is None:
            raise ValueError(f"Could not find Date and Price columns. Found: {df.columns.tolist()}")
        
        self.dates = pd.to_datetime(df[date_col])
        self.prices = df[price_col].values
        
        # Sort by date
        sort_idx = np.argsort(self.dates)
        self.dates = self.dates[sort_idx]
        self.prices = self.prices[sort_idx]
        
        print(f"Loaded {len(self.dates)} observations")
        print(f"Date range: {self.dates.min()} to {self.dates.max()}")
    
    def analyze(self, window_days=504, step_days=21, min_data_points=200):
        """
        Run rolling window LPPLS analysis
        
        Args:
            window_days: Analysis window size in days (default: 504 = 2 years)
            step_days: Step size in days (default: 21 = 1 month)
            min_data_points: Minimum data points required
            
        Returns:
            DataFrame with analysis results
        """
        print(f"\nRunning LPPLS Analysis")
        print(f"  Window: {window_days} days")
        print(f"  Step: {step_days} days")
        
        # Convert to numerical time
        time_numerical = (self.dates - self.dates[0]).dt.days.astype(float)
        log_prices = np.log(self.prices)
        
        results = []
        total_windows = 0
        successful_fits = 0
        
        # Rolling window analysis
        for i in range(0, len(self.dates) - window_days, step_days):
            window_end_idx = i + window_days
            
            if window_end_idx > len(self.dates):
                break
            
            total_windows += 1
            
            # Extract window
            t_window = time_numerical[i:window_end_idx].values
            p_window = log_prices[i:window_end_idx].values
            date_window_end = self.dates[window_end_idx - 1]

            if len(t_window) < min_data_points:
                continue
            
            # Fit LPPLS
            fit = self.fitter.fit(t_window, p_window)
            
            if fit is not None and fit['r_squared'] > 0.3:
                successful_fits += 1
                
                # Calculate bubble metrics
                metrics = self._calculate_bubble_metrics(fit, t_window, p_window)
                
                # Bootstrap confidence intervals for tc
                tc_ci = self._bootstrap_tc_confidence(t_window, p_window, fit['tc'])
                
                # Convert tc to dates
                tc_date = self.dates[0] + pd.Timedelta(days=float(fit['tc']))
                tc_lower = self.dates[0] + pd.Timedelta(days=float(tc_ci['lower']))
                tc_upper = self.dates[0] + pd.Timedelta(days=float(tc_ci['upper']))
                
                results.append({
                    'window_end': date_window_end,
                    'tc_date': tc_date,
                    'tc_lower': tc_lower,
                    'tc_upper': tc_upper,
                    'days_to_tc': int(fit['tc'] - t_window[-1]),
                    'B': fit['B'],
                    'm': fit['m'],
                    'omega': fit['omega'],
                    'r_squared': fit['r_squared'],
                    'bubble_confidence': metrics['bubble_confidence'],
                    'stage': metrics['stage']
                })
                
                print(f"  {date_window_end.strftime('%Y-%m-%d')}: "
                      f"Stage={metrics['stage']}, "
                      f"Conf={metrics['bubble_confidence']:.2f}, "
                      f"tc={tc_date.strftime('%Y-%m-%d')}")
        
        self.results = pd.DataFrame(results)
        
        print(f"\nAnalysis Complete:")
        print(f"  Windows analyzed: {total_windows}")
        print(f"  Successful fits: {successful_fits} ({successful_fits/total_windows*100:.1f}%)")
        print(f"  Bubble signals: {len(self.results)}")
        
        return self.results
    
    def _calculate_bubble_metrics(self, fit, t_window, p_window):
        """
        Calculate bubble stage and confidence
        
        Returns:
            dict with bubble_confidence and stage
        """
        B = fit['B']
        m = fit['m']
        omega = fit['omega']
        r_squared = fit['r_squared']
        
        # Criteria from Sornette methodology
        criteria_score = 0
        
        # 1. Power law exponent in valid range
        if 0.1 <= m <= 0.9:
            criteria_score += 0.25
        
        # 2. Frequency in valid range
        if 6 <= omega <= 13:
            criteria_score += 0.25
        
        # 3. Good fit
        if r_squared > 0.5:
            criteria_score += 0.25
        elif r_squared > 0.3:
            criteria_score += 0.15
        
        # 4. B sign and magnitude (negative B = bubble)
        if B < -0.1:
            criteria_score += 0.25
        elif B < 0:
            criteria_score += 0.10
        
        # Determine bubble stage
        days_to_tc = fit['tc'] - t_window[-1]
        
        if B >= 0:
            stage = 'No Bubble'
        elif days_to_tc > 180:
            stage = 'Early Formation'
        elif days_to_tc > 60:
            stage = 'Developing'
        elif days_to_tc > 0:
            stage = 'Mature (High Risk)'
        else:
            stage = 'Post-Critical'
        
        return {
            'bubble_confidence': criteria_score,
            'stage': stage
        }
    
    def _bootstrap_tc_confidence(self, t, p, tc_estimate, n_bootstrap=50, confidence=0.80):
        """
        Calculate confidence interval for tc using bootstrap
        
        Returns:
            dict with lower and upper bounds
        """
        tc_estimates = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(t), size=len(t), replace=True)
            t_boot = t[indices]
            p_boot = p[indices]
            
            # Sort
            sort_idx = np.argsort(t_boot)
            t_boot = t_boot[sort_idx]
            p_boot = p_boot[sort_idx]
            
            # Fit
            fit = self.fitter.fit(t_boot, p_boot)
            
            if fit is not None:
                tc_estimates.append(fit['tc'])
        
        if len(tc_estimates) < 10:
            # Not enough estimates, use wider bounds
            return {
                'lower': tc_estimate - 90,
                'upper': tc_estimate + 90
            }
        
        # Calculate percentiles
        alpha = 1 - confidence
        lower = np.percentile(tc_estimates, alpha/2 * 100)
        upper = np.percentile(tc_estimates, (1 - alpha/2) * 100)
        
        return {
            'lower': lower,
            'upper': upper
        }
    
    def get_current_bubble_status(self):
        """
        Get current bubble status based on most recent analysis
        
        Returns:
            dict with current bubble assessment
        """
        if len(self.results) == 0:
            return {
                'status': 'No Analysis',
                'message': 'Run analyze() first'
            }
        
        # Get most recent detection
        recent = self.results.iloc[-1]
        
        status = {
            'date': recent['window_end'],
            'stage': recent['stage'],
            'confidence': recent['bubble_confidence'],
            'tc_date': recent['tc_date'],
            'tc_range': (recent['tc_lower'], recent['tc_upper']),
            'days_to_tc': recent['days_to_tc']
        }
        
        # Interpretation
        if status['confidence'] > 0.75 and status['stage'] == 'Mature (High Risk)':
            status['message'] = "HIGH RISK: Strong bubble signal with imminent reversal expected"
        elif status['confidence'] > 0.60 and 'Developing' in status['stage']:
            status['message'] = "CAUTION: Bubble appears to be developing"
        elif status['confidence'] > 0.50 and status['stage'] == 'Early Formation':
            status['message'] = "WATCH: Early signs of potential bubble formation"
        else:
            status['message'] = "NORMAL: No strong bubble signals detected"
        
        return status
    
    def plot_bubble_diagnostics(self, save_path=None, recent_only=False):
        """
        Create comprehensive bubble diagnostic plot
        Similar to Sornette et al. (2013) Figure on page 20
        
        Args:
            save_path: Path to save figure
            recent_only: If True, only plot last 2 years
        """
        if len(self.results) == 0:
            print("No results to plot. Run analyze() first.")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Filter to recent data if requested
        if recent_only:
            cutoff_date = self.dates.max() - pd.Timedelta(days=730)  # 2 years
            plot_mask = self.dates >= cutoff_date
            plot_dates = self.dates[plot_mask]
            plot_prices = self.prices[plot_mask]
        else:
            plot_dates = self.dates
            plot_prices = self.prices
        
        # Panel 1: Price with bubble periods and tc confidence intervals
        ax1 = axes[0]
        ax1.plot(plot_dates, plot_prices, 'k-', linewidth=1.5, label='Price')
        
        # Shade bubble periods with confidence intervals for tc
        for _, row in self.results.iterrows():
            if row['window_end'] < plot_dates.min():
                continue
            
            # Color by bubble stage
            if row['stage'] == 'Mature (High Risk)':
                color = 'red'
                alpha = 0.3
            elif row['stage'] == 'Developing':
                color = 'orange'
                alpha = 0.2
            elif row['stage'] == 'Early Formation':
                color = 'yellow'
                alpha = 0.15
            else:
                continue  # Don't plot non-bubble periods
            
            # Shade from window end to tc upper bound (80% CI)
            if row['tc_upper'] > plot_dates.min() and row['window_end'] < plot_dates.max():
                ax1.axvspan(
                    row['window_end'],
                    min(row['tc_upper'], plot_dates.max()),
                    alpha=alpha,
                    color=color
                )
                
                # Mark tc point estimate
                if row['tc_date'] >= plot_dates.min() and row['tc_date'] <= plot_dates.max():
                    ax1.axvline(row['tc_date'], color=color, linestyle='--', 
                               linewidth=1, alpha=0.7)
        
        ax1.set_ylabel('Price', fontsize=11)
        ax1.set_title('Price History with Bubble Periods and Expected Regime Change (tc)', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Create legend for bubble stages
        stage_patches = [
            mpatches.Patch(color='red', alpha=0.3, label='Mature (High Risk)'),
            mpatches.Patch(color='orange', alpha=0.2, label='Developing'),
            mpatches.Patch(color='yellow', alpha=0.15, label='Early Formation')
        ]
        ax1.legend(handles=stage_patches, loc='upper left', fontsize=9)
        
        # Panel 2: Bubble confidence over time
        ax2 = axes[1]
        
        # Plot confidence as line
        ax2.plot(self.results['window_end'], self.results['bubble_confidence'], 
                'b-', linewidth=2, label='Bubble Confidence')
        
        # Shade by stage
        for _, row in self.results.iterrows():
            if row['window_end'] < plot_dates.min():
                continue
                
            if row['stage'] == 'Mature (High Risk)':
                color = 'red'
            elif row['stage'] == 'Developing':
                color = 'orange'
            elif row['stage'] == 'Early Formation':
                color = 'yellow'
            else:
                color = 'gray'
            
            ax2.scatter(row['window_end'], row['bubble_confidence'], 
                       c=color, s=50, zorder=3, edgecolors='black', linewidth=0.5)
        
        # Threshold lines
        ax2.axhline(y=0.75, color='red', linestyle='--', linewidth=1, 
                   label='High Confidence (0.75)', alpha=0.7)
        ax2.axhline(y=0.60, color='orange', linestyle='--', linewidth=1, 
                   label='Medium Confidence (0.60)', alpha=0.7)
        
        ax2.set_ylabel('Bubble Confidence', fontsize=11)
        ax2.set_title('Bubble Detection Confidence Over Time', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Days to critical time (tc)
        ax3 = axes[2]
        
        # Plot days to tc
        colors = []
        for stage in self.results['stage']:
            if stage == 'Mature (High Risk)':
                colors.append('red')
            elif stage == 'Developing':
                colors.append('orange')
            elif stage == 'Early Formation':
                colors.append('yellow')
            else:
                colors.append('gray')
        
        scatter = ax3.scatter(self.results['window_end'], self.results['days_to_tc'],
                            c=colors, s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Zero line (tc has arrived)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5, label='tc Arrived')
        
        # Shade the "danger zone" (< 30 days to tc)
        ax3.axhspan(-30, 30, alpha=0.1, color='red', label='High Risk Zone (±30 days)')
        
        ax3.set_ylabel('Days to Critical Time (tc)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_title('Time Horizon to Expected Regime Change', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Set x-axis to show only plot date range
        for ax in axes:
            ax.set_xlim([plot_dates.min(), plot_dates.max()])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_quantile_crash_dates(self, save_path=None, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
        """
        Plot quantile ranges of predicted crash dates over time
        Based on Geraskin et al. - shows distribution of tc predictions

        This visualization shows how the predicted crash date (tc) evolves over time
        with uncertainty bands representing different quantiles of the predictions.

        Args:
            save_path: Path to save figure
            quantiles: List of quantiles to plot (default: [0.1, 0.25, 0.5, 0.75, 0.9])
        """
        if len(self.results) == 0:
            print("No results to plot. Run analyze() first.")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        # For each window_end date, collect all tc predictions made up to that point
        # This creates a forward-looking distribution of predicted crash dates
        window_ends = sorted(self.results['window_end'].unique())

        quantile_data = {q: [] for q in quantiles}
        plot_dates = []

        for window_date in window_ends:
            # Get all predictions made up to this window_end
            predictions = self.results[self.results['window_end'] <= window_date]

            # Get tc_date predictions that are in the future relative to window_date
            future_tc = predictions[predictions['tc_date'] > window_date]['tc_date']

            if len(future_tc) >= 3:  # Need minimum predictions for quantiles
                # Convert to days from window_date for quantile calculation
                days_to_tc = [(tc - window_date).days for tc in future_tc]

                # Calculate quantiles
                quantile_values = np.percentile(days_to_tc, [q*100 for q in quantiles])

                plot_dates.append(window_date)
                for i, q in enumerate(quantiles):
                    # Convert back to absolute dates
                    tc_quantile = window_date + pd.Timedelta(days=float(quantile_values[i]))
                    quantile_data[q].append(tc_quantile)

        if len(plot_dates) == 0:
            print("Not enough data to plot quantiles")
            return

        plot_dates = pd.to_datetime(plot_dates)

        # Plot quantile bands
        colors_fill = ['#ff9999', '#ffb366', '#ffff99', '#ffb366', '#ff9999']
        colors_line = ['#ff0000', '#ff6600', '#ffcc00', '#ff6600', '#ff0000']

        # Plot the median (50th percentile) as the central line
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles)//2
        median_q = quantiles[median_idx]
        ax.plot(plot_dates, quantile_data[median_q],
               color='darkred', linewidth=2.5, label='Median tc prediction', zorder=10)

        # Plot quantile bands symmetrically
        sorted_quantiles = sorted(quantiles)
        mid_idx = len(sorted_quantiles) // 2

        for i in range(mid_idx):
            lower_q = sorted_quantiles[i]
            upper_q = sorted_quantiles[-(i+1)]

            if lower_q == upper_q:
                continue

            ax.fill_between(plot_dates,
                           quantile_data[lower_q],
                           quantile_data[upper_q],
                           alpha=0.3,
                           color=colors_fill[i],
                           label=f'{int(lower_q*100)}%-{int(upper_q*100)}% quantile range')

        # Plot individual quantile lines
        for i, q in enumerate(sorted_quantiles):
            if q != median_q:  # Don't redraw median
                ax.plot(plot_dates, quantile_data[q],
                       color=colors_line[min(i, len(colors_line)-1)],
                       linewidth=1, alpha=0.7, linestyle='--')

        # Add diagonal reference line (tc = observation date)
        min_date = min(plot_dates.min(), min([min(v) for v in quantile_data.values()]))
        max_date = max(plot_dates.max(), max([max(v) for v in quantile_data.values()]))
        ax.plot([min_date, max_date], [min_date, max_date],
               'k--', linewidth=1, alpha=0.5, label='Present (tc = observation date)')

        # Formatting
        ax.set_xlabel('Observation Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Crash Date (tc)', fontsize=12, fontweight='bold')
        ax.set_title('Quantile Ranges of Predicted Crash Dates\n' +
                    'Shows distribution of tc predictions over time',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()

        return fig

    def export_results(self, output_path='bubble_diagnostic_results.csv'):
        """Export results to CSV"""
        if len(self.results) > 0:
            self.results.to_csv(output_path, index=False)
            print(f"Results exported to: {output_path}")
        else:
            print("No results to export")
    
    def print_summary(self):
        """Print summary of bubble diagnostic"""
        print("\n" + "="*70)
        print("BUBBLE DIAGNOSTIC SUMMARY")
        print("="*70)
        
        print(f"\nData Period: {self.dates.min().strftime('%Y-%m-%d')} to {self.dates.max().strftime('%Y-%m-%d')}")
        print(f"Total Observations: {len(self.dates)}")
        
        if len(self.results) > 0:
            print(f"\nBubble Detections: {len(self.results)}")
            
            # Count by stage
            stage_counts = self.results['stage'].value_counts()
            print("\nBy Stage:")
            for stage, count in stage_counts.items():
                print(f"  {stage}: {count}")
            
            # High confidence signals
            high_conf = self.results[self.results['bubble_confidence'] > 0.75]
            print(f"\nHigh Confidence Signals: {len(high_conf)}")
            
            # Current status
            status = self.get_current_bubble_status()
            print(f"\nCurrent Status ({status['date'].strftime('%Y-%m-%d')}):")
            print(f"  Stage: {status['stage']}")
            print(f"  Confidence: {status['confidence']:.2f}")
            print(f"  Expected tc: {status['tc_date'].strftime('%Y-%m-%d')}")
            print(f"  Days to tc: {status['days_to_tc']}")
            print(f"  Assessment: {status['message']}")
        else:
            print("\nNo bubble signals detected")
        
        print("\n" + "="*70)


def demo_gold_platinum_quantile_analysis(data_path, window_days=504, step_days=21,
                                         recent_years=None):
    """
    Comprehensive demonstration with gold and platinum data including quantile analysis

    Args:
        data_path: Path to CSV with 'Dates', 'Gold', 'Platinum' columns
        window_days: Analysis window size in days (default: 504 = ~2 years)
        step_days: Step size in days (default: 21 = ~1 month)
        recent_years: If specified, only analyze last N years of data

    Returns:
        dict with detectors for both gold and platinum
    """
    print("="*80)
    print("GOLD & PLATINUM BUBBLE ANALYSIS WITH QUANTILE CRASH DATE RANGES")
    print("Based on Geraskin et al. methodology")
    print("="*80)

    # Load the combined data
    df = pd.read_csv(data_path)

    # Find date column (case insensitive)
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'dates', 'datetime', 'time']:
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"Could not find date column. Found: {df.columns.tolist()}")

    dates = pd.to_datetime(df[date_col])

    # Filter to recent years if specified
    if recent_years is not None:
        cutoff = dates.max() - pd.Timedelta(days=recent_years*365)
        mask = dates >= cutoff
        dates = dates[mask].reset_index(drop=True)
        df = df[mask].reset_index(drop=True)
        print(f"\nFiltered to last {recent_years} years")

    print(f"\nData range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
    print(f"Total observations: {len(dates)}")

    results = {}

    # Analyze Gold
    print("\n" + "="*80)
    print("ANALYZING GOLD")
    print("="*80)

    if 'Gold' in df.columns:
        gold_detector = BubbleDiagnostic(dates=dates, prices=df['Gold'].values)
        gold_results = gold_detector.analyze(window_days=window_days, step_days=step_days)

        if len(gold_results) > 0:
            print("\nGenerating Gold diagnostic plots...")
            gold_detector.plot_bubble_diagnostics(save_path='gold_bubble_diagnostic.png')

            print("\nGenerating Gold quantile crash date analysis...")
            gold_detector.plot_quantile_crash_dates(save_path='gold_quantile_crash_dates.png')

            gold_detector.export_results('gold_bubble_results.csv')
            gold_detector.print_summary()

        results['gold'] = gold_detector

    # Analyze Platinum
    print("\n" + "="*80)
    print("ANALYZING PLATINUM")
    print("="*80)

    if 'Platinum' in df.columns:
        platinum_detector = BubbleDiagnostic(dates=dates, prices=df['Platinum'].values)
        platinum_results = platinum_detector.analyze(window_days=window_days, step_days=step_days)

        if len(platinum_results) > 0:
            print("\nGenerating Platinum diagnostic plots...")
            platinum_detector.plot_bubble_diagnostics(save_path='platinum_bubble_diagnostic.png')

            print("\nGenerating Platinum quantile crash date analysis...")
            platinum_detector.plot_quantile_crash_dates(save_path='platinum_quantile_crash_dates.png')

            platinum_detector.export_results('platinum_bubble_results.csv')
            platinum_detector.print_summary()

        results['platinum'] = platinum_detector

    # Create comparison plot
    print("\n" + "="*80)
    print("GENERATING COMPARATIVE QUANTILE ANALYSIS")
    print("="*80)

    if 'gold' in results and 'platinum' in results:
        create_comparative_quantile_plot(results['gold'], results['platinum'],
                                        label1='Gold', label2='Platinum',
                                        save_path='gold_platinum_quantile_comparison.png')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - gold_bubble_diagnostic.png")
    print("  - gold_quantile_crash_dates.png")
    print("  - gold_bubble_results.csv")
    print("  - platinum_bubble_diagnostic.png")
    print("  - platinum_quantile_crash_dates.png")
    print("  - platinum_bubble_results.csv")
    print("  - gold_platinum_quantile_comparison.png")

    return results


def create_comparative_quantile_plot(detector1, detector2, save_path=None,
                                    label1='Asset 1', label2='Asset 2',
                                    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """
    Create comparative quantile crash date plot for two assets

    Args:
        detector1: First BubbleDiagnostic instance (e.g., gold)
        detector2: Second BubbleDiagnostic instance (e.g., platinum)
        save_path: Path to save figure
        label1: Label for first asset
        label2: Label for second asset
        quantiles: List of quantiles to plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    for idx, (detector, label, ax) in enumerate([(detector1, label1, axes[0]),
                                                   (detector2, label2, axes[1])]):
        if len(detector.results) == 0:
            ax.text(0.5, 0.5, f'No results for {label}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Calculate quantile data
        window_ends = sorted(detector.results['window_end'].unique())
        quantile_data = {q: [] for q in quantiles}
        plot_dates = []

        for window_date in window_ends:
            predictions = detector.results[detector.results['window_end'] <= window_date]
            future_tc = predictions[predictions['tc_date'] > window_date]['tc_date']

            if len(future_tc) >= 3:
                days_to_tc = [(tc - window_date).days for tc in future_tc]
                quantile_values = np.percentile(days_to_tc, [q*100 for q in quantiles])

                plot_dates.append(window_date)
                for i, q in enumerate(quantiles):
                    tc_quantile = window_date + pd.Timedelta(days=float(quantile_values[i]))
                    quantile_data[q].append(tc_quantile)

        if len(plot_dates) == 0:
            ax.text(0.5, 0.5, f'Not enough data for {label}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        plot_dates = pd.to_datetime(plot_dates)

        # Plot quantile bands
        colors_fill = ['#ff9999', '#ffb366', '#ffff99', '#ffb366', '#ff9999']
        colors_line = ['#ff0000', '#ff6600', '#ffcc00', '#ff6600', '#ff0000']

        # Plot median
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles)//2
        median_q = quantiles[median_idx]
        ax.plot(plot_dates, quantile_data[median_q],
               color='darkred', linewidth=2.5, label='Median tc', zorder=10)

        # Plot quantile bands
        sorted_quantiles = sorted(quantiles)
        mid_idx = len(sorted_quantiles) // 2

        for i in range(mid_idx):
            lower_q = sorted_quantiles[i]
            upper_q = sorted_quantiles[-(i+1)]

            if lower_q != upper_q:
                ax.fill_between(plot_dates,
                               quantile_data[lower_q],
                               quantile_data[upper_q],
                               alpha=0.3,
                               color=colors_fill[i],
                               label=f'{int(lower_q*100)}%-{int(upper_q*100)}%')

        # Plot individual quantile lines
        for i, q in enumerate(sorted_quantiles):
            if q != median_q:
                ax.plot(plot_dates, quantile_data[q],
                       color=colors_line[min(i, len(colors_line)-1)],
                       linewidth=1, alpha=0.7, linestyle='--')

        # Diagonal reference
        min_date = min(plot_dates.min(), min([min(v) for v in quantile_data.values()]))
        max_date = max(plot_dates.max(), max([max(v) for v in quantile_data.values()]))
        ax.plot([min_date, max_date], [min_date, max_date],
               'k--', linewidth=1, alpha=0.5, label='Present')

        # Formatting
        ax.set_ylabel('Predicted Crash Date (tc)', fontsize=11, fontweight='bold')
        ax.set_title(f'{label} - Quantile Ranges of Predicted Crash Dates',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Observation Date', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

    return fig


def demo_gold_recent():
    """Demonstrate with recent gold data (legacy function for backward compatibility)"""
    print("="*70)
    print("DEMO: Gold Market - Last 2 Years")
    print("="*70)

    # This would load your full gold data and filter to last 2 years
    # For demo purposes, showing the structure
    detector = BubbleDiagnostic('gold_data.csv')  # CSV with Date, Price

    # Filter to last 2 years
    cutoff = detector.dates.max() - pd.Timedelta(days=730)
    mask = detector.dates >= cutoff
    detector.dates = detector.dates[mask].reset_index(drop=True)
    detector.prices = detector.prices[mask]

    # Analyze
    results = detector.analyze(window_days=504, step_days=21)

    # Print summary
    detector.print_summary()

    # Create diagnostic plots
    detector.plot_bubble_diagnostics(save_path='gold_bubble_diagnostic.png')

    # Export results
    detector.export_results('gold_bubble_results.csv')

    return detector


if __name__ == "__main__":
    # Example usage
    print("LPPLS Bubble Diagnostic System")
    print("="*70)
    print("\nBasic Usage:")
    print("  detector = BubbleDiagnostic('your_data.csv')")
    print("  detector.analyze()")
    print("  detector.plot_bubble_diagnostics()")
    print("  detector.plot_quantile_crash_dates()  # NEW: Quantile analysis")
    print("  detector.print_summary()")
    print("\nGold & Platinum Demo:")
    print("  # Run comprehensive analysis with quantile ranges")
    print("  results = demo_gold_platinum_quantile_analysis('GoldPlatData.csv')")
    print("  ")
    print("  # Or analyze specific recent period:")
    print("  results = demo_gold_platinum_quantile_analysis('GoldPlatData.csv', recent_years=5)")
    print("\nComparative Analysis:")
    print("  # Compare two assets side by side")
    print("  create_comparative_quantile_plot(gold_detector, platinum_detector,")
    print("                                   label1='Gold', label2='Platinum',")
    print("                                   save_path='comparison.png')")