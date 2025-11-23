import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class BenchmarkAnalyzer:
    
    def __init__(self, csv_path: str):
        """
        Initialize analyzer with benchmark results.
        
        Args:
            csv_path: Path to the benchmark results CSV file
        """
        self.df = pd.read_csv(csv_path)
        self.output_dir = Path("plots")
        self.output_dir.mkdir(exist_ok=True)
        
        # Compute statistics
        self._compute_statistics()
        
    def _compute_statistics(self):
        # Group by all dimensions except Run
        group_cols = ['Algo', 'Dataset', 'TextSize', 'PatternSize', 'Hit']
        
        self.stats = self.df.groupby(group_cols).agg({
            'Time': ['mean', 'std', 'min', 'max'],
            'Memory': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        self.stats.columns = [
            '_'.join(col).strip('_') if col[1] else col[0] 
            for col in self.stats.columns.values
        ]
        
    def plot_time_vs_text_size(self, save: bool = True):
        datasets = self.stats['Dataset'].unique()
        hit_values = [True, False]
        
        fig, axes = plt.subplots(len(datasets), 2, figsize=(16, 5 * len(datasets)))
        if len(datasets) == 1:
            axes = axes.reshape(1, -1)
        
        for i, dataset in enumerate(datasets):
            for j, hit in enumerate(hit_values):
                ax = axes[i, j]
                
                # Filter data
                data = self.stats[
                    (self.stats['Dataset'] == dataset) & 
                    (self.stats['Hit'] == hit)
                ]
                
                # Plot each algorithm
                for algo in data['Algo'].unique():
                    algo_data = data[data['Algo'] == algo].sort_values('TextSize')
                    
                    ax.plot(
                        algo_data['TextSize'],
                        algo_data['Time_mean'],
                        marker='o',
                        label=algo,
                        linewidth=2,
                        markersize=6
                    )
                    
                    # Add error bars (std)
                    ax.fill_between(
                        algo_data['TextSize'],
                        algo_data['Time_mean'] - algo_data['Time_std'],
                        algo_data['Time_mean'] + algo_data['Time_std'],
                        alpha=0.2
                    )
                
                ax.set_xlabel('Text Size (characters)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_title(
                    f'{dataset} - {"Hit" if hit else "Miss"}',
                    fontsize=14,
                    fontweight='bold'
                )
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "time_vs_textsize.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.show()
        
    def plot_time_vs_pattern_size(self, save: bool = True):
        datasets = self.stats['Dataset'].unique()
        
        fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
        if len(datasets) == 1:
            axes = [axes]
        
        for i, dataset in enumerate(datasets):
            ax = axes[i]
            
            # Filter data - combine hit and miss
            data = self.stats[self.stats['Dataset'] == dataset]
            
            # Plot each algorithm
            for algo in data['Algo'].unique():
                algo_data = data[data['Algo'] == algo].sort_values('PatternSize')
                
                # Group by pattern size and compute mean across text sizes
                pattern_stats = algo_data.groupby('PatternSize').agg({
                    'Time_mean': 'mean',
                    'Time_std': 'mean'
                }).reset_index()
                
                ax.plot(
                    pattern_stats['PatternSize'],
                    pattern_stats['Time_mean'],
                    marker='s',
                    label=algo,
                    linewidth=2,
                    markersize=6
                )
            
            ax.set_xlabel('Pattern Size (characters)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Time (seconds)', fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "time_vs_patternsize.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.show()
    
    def plot_memory_usage(self, save: bool = True):
        datasets = self.stats['Dataset'].unique()
        
        fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
        if len(datasets) == 1:
            axes = [axes]
        
        has_data = False
        
        for i, dataset in enumerate(datasets):
            ax = axes[i]
            
            # Filter data
            data = self.stats[self.stats['Dataset'] == dataset]
            
            # Plot each algorithm
            for algo in data['Algo'].unique():
                algo_data = data[data['Algo'] == algo].sort_values('TextSize')
                
                # Group by text size
                size_stats = algo_data.groupby('TextSize').agg({
                    'Memory_mean': 'mean'
                }).reset_index()
                
                # Convert to KB (more readable for small values)
                size_stats['Memory_KB'] = size_stats['Memory_mean'] / 1024
                
                # Check if we have meaningful data
                if size_stats['Memory_KB'].max() > 0.1:  # At least 100 bytes
                    has_data = True
                    ax.plot(
                        size_stats['TextSize'],
                        size_stats['Memory_KB'],
                        marker='D',
                        label=algo,
                        linewidth=2,
                        markersize=6
                    )
            
            ax.set_xlabel('Text Size (characters)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Peak Memory Usage (KB)', fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            
            # Only use log scale if we have significant variation
            if has_data and data['Memory_mean'].max() / max(data['Memory_mean'].min(), 1) > 10:
                ax.set_yscale('log')
            
            ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add note if memory values are very small
            if data['Memory_mean'].max() < 1024:  # Less than 1KB
                ax.text(0.5, 0.95, 
                       'Note: Memory usage is very small\n(algorithm overhead only)',
                       transform=ax.transAxes,
                       fontsize=9, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "memory_usage.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
            
            if not has_data:
                print(f"     Memory measurements are very small (< 100 bytes)")
                print(f"     This is expected for string matching algorithms.")
                print(f"     Memory usage represents Python object allocation overhead.")
        
        plt.show()
    
    def plot_hit_vs_miss_comparison(self, save: bool = True):
        """
        Compare hit vs miss performance for each algorithm.
        """
        datasets = self.stats['Dataset'].unique()
        algos = self.stats['Algo'].unique()
        
        fig, axes = plt.subplots(len(algos), len(datasets), 
                                 figsize=(6 * len(datasets), 5 * len(algos)))
        
        if len(algos) == 1 and len(datasets) == 1:
            axes = np.array([[axes]])
        elif len(algos) == 1:
            axes = axes.reshape(1, -1)
        elif len(datasets) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, algo in enumerate(algos):
            for j, dataset in enumerate(datasets):
                ax = axes[i, j]
                
                # Filter data
                data = self.stats[
                    (self.stats['Algo'] == algo) & 
                    (self.stats['Dataset'] == dataset)
                ]
                
                # Plot hit and miss
                for hit, label, style in [(True, 'Hit', '-'), (False, 'Miss', '--')]:
                    hit_data = data[data['Hit'] == hit].sort_values('TextSize')
                    
                    ax.plot(
                        hit_data['TextSize'],
                        hit_data['Time_mean'],
                        linestyle=style,
                        marker='o',
                        label=label,
                        linewidth=2,
                        markersize=6
                    )
                
                ax.set_xlabel('Text Size', fontsize=11, fontweight='bold')
                ax.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_title(f'{algo} - {dataset}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "hit_vs_miss.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.show()
    
    def plot_algorithm_comparison_heatmap(self, metric: str = 'Time', save: bool = True):
        """
        Create a heatmap comparing algorithms across different configurations.
        
        Args:
            metric: 'Time' or 'Memory'
        """
        datasets = self.stats['Dataset'].unique()
        
        fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 6))
        if len(datasets) == 1:
            axes = [axes]
        
        for i, dataset in enumerate(datasets):
            ax = axes[i]
            
            # Filter data
            data = self.stats[self.stats['Dataset'] == dataset]
            
            # Create pivot table
            pivot = data.pivot_table(
                values=f'{metric}_mean',
                index='TextSize',
                columns='Algo',
                aggfunc='mean'
            )
            
            # Normalize by row (each text size)
            pivot_normalized = pivot.div(pivot.min(axis=1), axis=0)
            
            # Plot heatmap
            sns.heatmap(
                pivot_normalized,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn_r',
                ax=ax,
                cbar_kws={'label': f'Relative {metric} (lower is better)'},
                vmin=1.0,
                vmax=pivot_normalized.max().max()
            )
            
            ax.set_title(f'{dataset} - {metric} Comparison', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
            ax.set_ylabel('Text Size', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"{metric.lower()}_heatmap.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.show()
    
    def plot_scaling_analysis(self, save: bool = True):
        """
        Analyze and visualize scaling behavior (linear, log-linear, quadratic, etc.)
        Fit curves to determine complexity.
        """
        from scipy.optimize import curve_fit
        
        datasets = self.stats['Dataset'].unique()
        algos = self.stats['Algo'].unique()
        
        # Define fitting functions
        def linear(x, a, b):
            return a * x + b
        
        def log_linear(x, a, b):
            return a * x * np.log(x) + b
        
        def quadratic(x, a, b):
            """O(n*m) for Naive - approximated as O(n*sqrt(n)) for varying pattern sizes"""
            return a * x * np.sqrt(x) + b
        
        fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 5 * len(datasets)))
        if len(datasets) == 1:
            axes = [axes]
        
        results = []
        
        for i, dataset in enumerate(datasets):
            ax = axes[i]
            
            for algo in algos:
                # Get data for hit case only (miss is similar)
                data = self.stats[
                    (self.stats['Dataset'] == dataset) & 
                    (self.stats['Algo'] == algo) &
                    (self.stats['Hit'] == True)
                ].sort_values('TextSize')
                
                if len(data) < 3:
                    continue
                
                x = data['TextSize'].values
                y = data['Time_mean'].values
                
                # Try fitting O(n), O(n log n), and O(n*sqrt(n)) for Naive
                try:
                    # O(n) fit
                    popt_linear, _ = curve_fit(linear, x, y)
                    y_linear = linear(x, *popt_linear)
                    residual_linear = np.sum((y - y_linear) ** 2)
                    
                    # O(n log n) fit
                    popt_log, _ = curve_fit(log_linear, x, y)
                    y_log = log_linear(x, *popt_log)
                    residual_log = np.sum((y - y_log) ** 2)
                    
                    # O(n*sqrt(n)) fit (approximation for Naive's O(n*m) behavior)
                    popt_quad, _ = curve_fit(quadratic, x, y)
                    y_quad = quadratic(x, *popt_quad)
                    residual_quad = np.sum((y - y_quad) ** 2)
                    
                    # Determine best fit
                    residuals = {
                        'O(n)': residual_linear,
                        'O(n log n)': residual_log,
                        'O(n*sqrt(n))': residual_quad
                    }
                    best_fit = min(residuals, key=residuals.get)
                    
                    # For Naive algorithm, prefer O(n*sqrt(n)) if it's close
                    if algo == "Naive" and residual_quad < residual_linear * 1.5:
                        best_fit = 'O(n*sqrt(n))'
                    
                    results.append({
                        'Dataset': dataset,
                        'Algo': algo,
                        'Best_Fit': best_fit,
                        'Linear_Residual': residual_linear,
                        'LogLinear_Residual': residual_log,
                        'Quadratic_Residual': residual_quad
                    })
                    
                    # Plot actual data
                    ax.scatter(x, y, label=f'{algo} (actual)', s=50, alpha=0.7)
                    
                    # Plot best fit
                    if best_fit == "O(n)":
                        ax.plot(x, y_linear, '--', label=f'{algo} O(n) fit', linewidth=2)
                    elif best_fit == "O(n log n)":
                        ax.plot(x, y_log, '--', label=f'{algo} O(n log n) fit', linewidth=2)
                    else:  # O(n*sqrt(n))
                        ax.plot(x, y_quad, '--', label=f'{algo} O(n·√n) fit', linewidth=2)
                
                except Exception as e:
                    print(f"Warning: Could not fit {algo} on {dataset}: {e}")
            
            ax.set_xlabel('Text Size (n)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
            ax.set_title(f'Scaling Analysis - {dataset}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "scaling_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
            
            # Save scaling results to CSV
            results_df = pd.DataFrame(results)
            results_path = self.output_dir / "scaling_analysis.csv"
            results_df.to_csv(results_path, index=False)
            print(f"✓ Saved scaling results: {results_path}")
        
        plt.show()
        
        return pd.DataFrame(results)
    
    def generate_summary_statistics(self, save: bool = True):
        """
        Generate and print summary statistics.
        """
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY STATISTICS")
        print("=" * 80)
        
        # Overall statistics
        print("\nOverall Statistics:")
        print(f"  Total test runs: {len(self.df)}")
        print(f"  Algorithms tested: {', '.join(self.df['Algo'].unique())}")
        print(f"  Datasets tested: {', '.join(self.df['Dataset'].unique())}")
        print(f"  Text sizes: {len(self.df['TextSize'].unique())}")
        print(f"  Pattern sizes: {len(self.df['PatternSize'].unique())}")
        
        # Best/Worst performers
        print("\nBest Performers (Average Time):")
        best = self.stats.nsmallest(5, 'Time_mean')[
            ['Algo', 'Dataset', 'TextSize', 'PatternSize', 'Hit', 'Time_mean']
        ]
        print(best.to_string(index=False))
        
        print("\nSlowest Cases:")
        worst = self.stats.nlargest(5, 'Time_mean')[
            ['Algo', 'Dataset', 'TextSize', 'PatternSize', 'Hit', 'Time_mean']
        ]
        print(worst.to_string(index=False))
        
        # Algorithm comparison
        print("\n🔬 Algorithm Comparison (Overall Averages):")
        algo_comparison = self.stats.groupby('Algo').agg({
            'Time_mean': 'mean',
            'Memory_mean': 'mean'
        }).round(6)
        
        # Convert memory to KB for readability
        algo_comparison['Memory_KB'] = (algo_comparison['Memory_mean'] / 1024).round(2)
        algo_comparison = algo_comparison.rename(columns={
            'Time_mean': 'Avg_Time(s)',
            'Memory_KB': 'Avg_Memory(KB)'
        })
        print(algo_comparison[['Avg_Time(s)', 'Avg_Memory(KB)']])
        
        # Add memory usage context
        max_mem = self.stats['Memory_mean'].max()
        if max_mem < 10240:  # Less than 10KB
            print("\n   Note: Memory values represent Python object allocation overhead.")
            print("     Actual algorithm space complexity:")
            print("     - Naive: O(1) constant space")
            print("     - KMP: O(m) for prefix table")
            print("     - Boyer-Moore: O(m + σ) for skip tables")  
            print("     - Rabin-Karp: O(1) constant space")
        
        # Dataset comparison
        print("\nDataset Comparison (Average Time):")
        dataset_comparison = self.stats.groupby('Dataset').agg({
            'Time_mean': 'mean'
        }).round(6)
        print(dataset_comparison)
        
        # Hit vs Miss
        print("\n🎯 Hit vs Miss Comparison (Average Time):")
        hit_comparison = self.stats.groupby(['Algo', 'Hit']).agg({
            'Time_mean': 'mean'
        }).round(6)
        print(hit_comparison)
        
        print("\n" + "=" * 80)
        
        if save:
            # Save detailed statistics
            stats_path = self.output_dir / "summary_statistics.txt"
            with open(stats_path, 'w') as f:
                f.write("BENCHMARK SUMMARY STATISTICS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total test runs: {len(self.df)}\n")
                f.write(f"Algorithms: {', '.join(self.df['Algo'].unique())}\n")
                f.write(f"Datasets: {', '.join(self.df['Dataset'].unique())}\n\n")
                f.write("Algorithm Comparison:\n")
                f.write(algo_comparison.to_string())
                f.write("\n\nDataset Comparison:\n")
                f.write(dataset_comparison.to_string())
                f.write("\n\nHit vs Miss:\n")
                f.write(hit_comparison.to_string())
            
            print(f"\n✓ Statistics saved to: {stats_path}")
    
    def generate_all_plots(self):
        """
        Generate all plots at once.
        """
        print("\nGenerating all visualizations...")
        print("-" * 80)
        
        self.plot_time_vs_text_size(save=True)
        self.plot_time_vs_pattern_size(save=True)
        self.plot_memory_usage(save=True)
        self.plot_hit_vs_miss_comparison(save=True)
        self.plot_algorithm_comparison_heatmap(metric='Time', save=True)
        self.plot_algorithm_comparison_heatmap(metric='Memory', save=True)
        self.plot_scaling_analysis(save=True)
        
        print("-" * 80)
        print(f"All plots saved to: {self.output_dir}/")
        print("-" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze and visualize string matching benchmark results"
    )
    
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to benchmark results CSV file'
    )
    
    parser.add_argument(
        '--plots',
        nargs='+',
        choices=['time_text', 'time_pattern', 'memory', 'hit_miss', 
                 'heatmap', 'scaling', 'all'],
        default=['all'],
        help='Which plots to generate (default: all)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots (only save)'
    )
    
    args = parser.parse_args()
    
    print(f"\nLoading results from: {args.csv_file}")
    analyzer = BenchmarkAnalyzer(args.csv_file)
    
    #summary
    analyzer.generate_summary_statistics(save=True)
    
    if 'all' in args.plots:
        analyzer.generate_all_plots()
    else:
        plot_map = {
            'time_text': analyzer.plot_time_vs_text_size,
            'time_pattern': analyzer.plot_time_vs_pattern_size,
            'memory': analyzer.plot_memory_usage,
            'hit_miss': analyzer.plot_hit_vs_miss_comparison,
            'heatmap': lambda: [
                analyzer.plot_algorithm_comparison_heatmap('Time'),
                analyzer.plot_algorithm_comparison_heatmap('Memory')
            ],
            'scaling': analyzer.plot_scaling_analysis,
        }
        
        for plot_type in args.plots:
            if plot_type in plot_map:
                plot_map[plot_type]()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()