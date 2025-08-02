import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from time_simulator import simulate_boarding_time, BoardingMethod


def main():
    # Run simulations
    n_simulations = 100
    results = run_boarding_simulations(n_simulations)

    # Create visualizations
    create_visualizations(results)

    # Perform statistical tests
    perform_statistical_tests(results)


def run_boarding_simulations(n_simulations: int) -> pd.DataFrame:
    """Run boarding simulations for all methods."""
    print(f"Running {n_simulations} simulations for each boarding method...")

    methods = list(BoardingMethod)
    all_results = []

    for method in methods:
        print(f"Simulating {method.value}...")
        method_times = []

        for i in range(n_simulations):
            # Run simulation
            boarding_time = simulate_boarding_time(method)
            method_times.append(boarding_time)

            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{n_simulations} simulations")

        # Add results to list
        for time in method_times:
            all_results.append(
                {
                    "method": method.value,
                    "boarding_time_seconds": time,
                    "boarding_time_minutes": time / 60,
                }
            )

    df = pd.DataFrame(all_results)
    print("\nSimulation Summary:")
    print(
        df.groupby("method")["boarding_time_minutes"]
        .agg(["mean", "std", "min", "max"])
        .round(2)
    )

    return df


def create_visualizations(df: pd.DataFrame):
    """Create visualizations for the boarding simulation results."""
    print("\nCreating visualizations...")

    # Set up the plotting style with custom parameters
    plt.style.use("default")  # Start with default style
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )

    # Custom color palette
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]
    sns.set_palette(colors)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Airplane Boarding Time Analysis", fontsize=18, fontweight="bold", y=0.95
    )

    # 1. Enhanced Box plot
    sns.boxplot(
        data=df,
        x="method",
        y="boarding_time_minutes",
        ax=axes[0, 0],
        hue="method",
        palette=colors,
        linewidth=1.2,
        legend=False,
    )
    axes[0, 0].set_title(
        "Boarding Time Distribution by Method", fontweight="bold", pad=15
    )
    axes[0, 0].set_xlabel("Boarding Method", fontweight="bold")
    axes[0, 0].set_ylabel("Boarding Time (minutes)", fontweight="bold")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Add median value labels on box plot
    medians = df.groupby("method")["boarding_time_minutes"].median()
    for i, method in enumerate(df["method"].unique()):
        axes[0, 0].text(
            i,
            medians[method] + 0.5,
            f"{medians[method]:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    # 2. Enhanced Violin plot
    sns.violinplot(
        data=df,
        x="method",
        y="boarding_time_minutes",
        ax=axes[0, 1],
        hue="method",
        palette=colors,
        inner="quart",
        linewidth=1.2,
        legend=False,
    )
    axes[0, 1].set_title(
        "Boarding Time Density Distribution", fontweight="bold", pad=15
    )
    axes[0, 1].set_xlabel("Boarding Method", fontweight="bold")
    axes[0, 1].set_ylabel("Boarding Time (minutes)", fontweight="bold")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # 3. Enhanced Bar plot with error bars and styling
    summary_stats = (
        df.groupby("method")["boarding_time_minutes"].agg(["mean", "std"]).reset_index()
    )
    bars = axes[1, 0].bar(
        summary_stats["method"],
        summary_stats["mean"],
        yerr=summary_stats["std"],
        capsize=8,
        alpha=0.8,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        error_kw={"linewidth": 2, "ecolor": "black"},
    )
    axes[1, 0].set_title(
        "Average Boarding Time with Standard Deviation", fontweight="bold", pad=15
    )
    axes[1, 0].set_xlabel("Boarding Method", fontweight="bold")
    axes[1, 0].set_ylabel("Average Boarding Time (minutes)", fontweight="bold")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars with better formatting
    for bar, mean_val, std_val in zip(
        bars, summary_stats["mean"], summary_stats["std"]
    ):
        height = bar.get_height()
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std_val + 0.2,
            f"{mean_val:.1f}±{std_val:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    # 4. Enhanced Histogram overlay with better styling
    methods = df["method"].unique()

    for i, method in enumerate(methods):
        method_data = df[df["method"] == method]["boarding_time_minutes"]
        axes[1, 1].hist(
            method_data,
            alpha=0.7,
            label=method,
            color=colors[i],
            bins=15,
            edgecolor="black",
            linewidth=0.8,
        )

    axes[1, 1].set_title(
        "Boarding Time Distribution Histograms", fontweight="bold", pad=15
    )
    axes[1, 1].set_xlabel("Boarding Time (minutes)", fontweight="bold")
    axes[1, 1].set_ylabel("Frequency", fontweight="bold")
    axes[1, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for the main title
    plt.show()

    # Create an enhanced ranking plot
    plt.figure(figsize=(14, 8))
    ranking_data = df.groupby("method")["boarding_time_minutes"].mean().sort_values()

    # Create gradient colors for ranking
    gradient_colors = plt.colormaps.get_cmap("RdYlGn_r")(
        np.linspace(0.2, 0.8, len(ranking_data))
    )

    bars = plt.bar(
        range(len(ranking_data)),
        np.asarray(ranking_data.values, dtype=float),
        color=gradient_colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )

    plt.xticks(
        range(len(ranking_data)),
        list(map(str, ranking_data.index)),
        rotation=45,
        fontweight="bold",
    )
    plt.ylabel("Average Boarding Time (minutes)", fontweight="bold", fontsize=12)
    plt.title(
        "Boarding Methods Efficiency Ranking\n(Lower Time = Better Performance)",
        fontweight="bold",
        fontsize=16,
        pad=20,
    )
    plt.grid(True, alpha=0.3, axis="y")

    # Add enhanced value labels on bars
    for i, (bar, value) in enumerate(zip(bars, ranking_data.values)):
        # Time label
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.3,
            f"{value:.1f} min",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )
        # Rank label
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() / 2,
            f"#{i + 1}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

    plt.tight_layout()
    plt.show()


def perform_statistical_tests(df: pd.DataFrame):
    """Perform statistical tests on the boarding simulation results."""
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    # Prepare data for tests
    methods = df["method"].unique()
    method_data = [
        df[df["method"] == method]["boarding_time_minutes"].values for method in methods
    ]

    # 1. Descriptive Statistics
    print("\n1. DESCRIPTIVE STATISTICS")
    print("-" * 30)
    desc_stats = df.groupby("method")["boarding_time_minutes"].describe().round(2)
    print(desc_stats)

    # 2. ANOVA Test
    print("\n2. ONE-WAY ANOVA TEST")
    print("-" * 30)
    f_stat, p_value_anova = stats.f_oneway(*method_data)
    print(f"F-statistic: {f_stat:.4f}")

    # Display p-value in scientific notation if very small
    if p_value_anova < 1e-10:
        print(f"P-value: {p_value_anova:.2e} (extremely small)")
    else:
        print(f"P-value: {p_value_anova:.6f}")

    alpha = 0.05
    if p_value_anova < alpha:
        print(f"Result: SIGNIFICANT (p < {alpha})")
        print("Conclusion: There are significant differences between boarding methods.")
    else:
        print(f"Result: NOT SIGNIFICANT (p >= {alpha})")
        print(
            "Conclusion: No significant differences detected between boarding methods."
        )

    # 3. Kruskal-Wallis Test (non-parametric alternative)
    print("\n3. KRUSKAL-WALLIS TEST")
    print("-" * 30)
    h_stat, p_value_kw = stats.kruskal(*method_data)
    print(f"H-statistic: {h_stat:.4f}")

    # Display p-value in scientific notation if very small
    if p_value_kw < 1e-10:
        print(f"P-value: {p_value_kw:.2e} (extremely small)")
    else:
        print(f"P-value: {p_value_kw:.6f}")

    if p_value_kw < alpha:
        print(f"Result: SIGNIFICANT (p < {alpha})")
        print(
            "Conclusion: There are significant differences between boarding methods (non-parametric)."
        )
    else:
        print(f"Result: NOT SIGNIFICANT (p >= {alpha})")
        print(
            "Conclusion: No significant differences detected between boarding methods (non-parametric)."
        )

    # 4. Post-hoc analysis if significant
    if p_value_anova < alpha or p_value_kw < alpha:
        print("\n4. POST-HOC ANALYSIS (Pairwise Comparisons)")
        print("-" * 30)

        # Pairwise t-tests with Bonferroni correction
        from itertools import combinations

        method_pairs = list(combinations(methods, 2))
        n_comparisons = len(method_pairs)
        bonferroni_alpha = alpha / n_comparisons

        print(f"Bonferroni corrected alpha: {bonferroni_alpha:.6f}")
        print("\nPairwise t-test results:")

        significant_pairs = []
        for method1, method2 in method_pairs:
            data1 = df[df["method"] == method1]["boarding_time_minutes"]
            data2 = df[df["method"] == method2]["boarding_time_minutes"]

            ttest_result = stats.ttest_ind(data1, data2)
            t_stat = ttest_result.statistic
            p_val = ttest_result.pvalue

            is_significant = p_val < bonferroni_alpha
            if is_significant:
                significant_pairs.append((method1, method2, p_val))

            # Format p-value display
            if p_val < 1e-10:
                p_val_str = f"{p_val:.2e}"
            else:
                p_val_str = f"{p_val:.6f}"

            print(
                f"{method1} vs {method2}: t={t_stat:.3f}, p={p_val_str} {'*' if is_significant else ''}"
            )

        if significant_pairs:
            print("\nSignificant pairs (after Bonferroni correction):")
            for method1, method2, p_val in significant_pairs:
                mean1 = df[df["method"] == method1]["boarding_time_minutes"].mean()
                mean2 = df[df["method"] == method2]["boarding_time_minutes"].mean()

                # Format p-value for display
                if p_val < 1e-10:
                    p_val_str = f"{p_val:.2e}"
                else:
                    p_val_str = f"{p_val:.6f}"

                print(
                    f"  {method1} ({mean1:.1f} min) vs {method2} ({mean2:.1f} min): p={p_val_str}"
                )
        else:
            print("\nNo significant pairwise differences after Bonferroni correction.")

    # 5. Effect size (eta-squared for ANOVA)
    print("\n5. EFFECT SIZE")
    print("-" * 30)

    # Calculate eta-squared
    ss_between = sum(
        len(group) * (np.mean(group) - np.mean(df["boarding_time_minutes"])) ** 2
        for group in method_data
    )
    ss_total = sum(
        (
            df["boarding_time_minutes"].to_numpy()
            - np.mean(df["boarding_time_minutes"].to_numpy())
        )
        ** 2
    )
    eta_squared = ss_between / ss_total

    print(f"Eta-squared (η²): {eta_squared:.4f}")

    if eta_squared < 0.01:
        effect_size = "negligible"
    elif eta_squared < 0.06:
        effect_size = "small"
    elif eta_squared < 0.14:
        effect_size = "medium"
    else:
        effect_size = "large"

    print(f"Effect size interpretation: {effect_size}")

    # 6. Ranking and Recommendations
    print("\n6. FINAL RANKINGS AND RECOMMENDATIONS")
    print("-" * 30)

    rankings = (
        df.groupby("method")["boarding_time_minutes"].agg(["mean", "std"]).round(2)
    )
    rankings = rankings.sort_values("mean")

    print("Boarding methods ranked by efficiency (fastest to slowest):")
    for i, (method, row) in enumerate(rankings.iterrows(), 1):
        print(f"{i}. {method}: {row['mean']:.1f} ± {row['std']:.1f} minutes")

    best_method = rankings.index[0]
    worst_method = rankings.index[-1]
    time_saved = rankings.loc[worst_method, "mean"] - rankings.loc[best_method, "mean"]

    print(f"\nRecommendation: Use '{best_method}' method")
    print(
        f"Potential time savings: {time_saved:.1f} minutes compared to '{worst_method}'"
    )


if __name__ == "__main__":
    main()
