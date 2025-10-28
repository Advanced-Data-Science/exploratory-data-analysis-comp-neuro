#!/usr/bin/env python3
"""
Concatonated version of all EDA script for presentation

Following functions are included:
- create_univariate_analysis
- create_summary_statistics_boxplots
- create_bivariate_analysis
- create_multivariate_analysis
- create_pattern_recognition
- create_time_series_analysis
- create_segmentation_analysis
- create_advanced_visualizations
- create_summary_dashboard

This file also uses UV, run with uv run scripts/eda_visualizations.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# styling for latex doc compatibility
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


def load_data():
    """Load all required data"""

    with open("Data/Raw/daily_data.json", "r") as f:
        git_data = json.load(f)

    # vector data (unused)
    git_vectors = np.load("Data/Raw/ctm_git_timeseries_vectors/git_vectors.npy")

    # frontmatter
    with open("Data/Raw/frontmatter.json", "r") as f:
        frontmatter = json.load(f)

    # wikilink
    with open("Data/Raw/wikilink_graph.json", "r") as f:
        wikilink = json.load(f)

    # dates for vectors
    with open("Data/Raw/ctm_git_timeseries_vectors/dates.json", "r") as f:
        dates = json.load(f)

    # unused, but a temporal stability analysis
    with open("Data/Analysis/temporal_stability_results.json", "r") as f:
        stability = json.load(f)

    return git_data, git_vectors, frontmatter, wikilink, dates, stability


def create_univariate_analysis(git_data, git_vectors):
    """For 1.1 - Univariate Analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # total edits per day
    daily_totals = [day["total_changes"] for day in git_data]
    axes[0, 0].hist(
        daily_totals, bins=30, color="steelblue", edgecolor="black", alpha=0.7
    )
    axes[0, 0].set_xlabel("Total Edits per Day")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Daily Total Edits")
    axes[0, 0].axvline(
        np.mean(daily_totals),
        color="red",
        linestyle="--",
        label=f"Mean = {np.mean(daily_totals):.1f}",
    )
    axes[0, 0].legend()

    # individual files edited per day
    files_per_day = [day["total_files"] for day in git_data]
    axes[0, 1].hist(
        files_per_day, bins=25, color="forestgreen", edgecolor="black", alpha=0.7
    )
    axes[0, 1].set_xlabel("Files Edited per Day")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Files Edited per Day")
    axes[0, 1].axvline(
        np.mean(files_per_day),
        color="red",
        linestyle="--",
        label=f"Mean = {np.mean(files_per_day):.1f}",
    )
    axes[0, 1].legend()

    # edits per file (only active days)
    edits_per_file = []
    for day in git_data:
        if day["total_files"] > 0:
            edits_per_file.append(day["total_changes"] / day["total_files"])

    axes[1, 0].hist(
        edits_per_file, bins=30, color="coral", edgecolor="black", alpha=0.7
    )
    axes[1, 0].set_xlabel("Edits per File (Active Days)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Distribution of Edits per File")
    axes[1, 0].axvline(
        np.mean(edits_per_file),
        color="red",
        linestyle="--",
        label=f"Mean = {np.mean(edits_per_file):.1f}",
    )
    axes[1, 0].legend()

    # file activty
    file_totals = np.sum(git_vectors, axis=0)
    file_totals_nonzero = file_totals[file_totals > 0]

    axes[1, 1].hist(
        np.log10(file_totals_nonzero + 1),
        bins=30,
        color="mediumpurple",
        edgecolor="black",
        alpha=0.7,
    )
    axes[1, 1].set_xlabel("Log10(Total Edits per File)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution of File Activity (Log Scale)")

    plt.tight_layout()
    plt.savefig("eda_figures/univariate_analysis.png", bbox_inches="tight")
    plt.close()
    print("Created univariate_analysis.png")


def create_summary_statistics_boxplots(git_data):
    """1.2 - Summary Statistics"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # data preparation
    daily_totals = [day["total_changes"] for day in git_data]
    files_per_day = [day["total_files"] for day in git_data]

    # distribution of daily total edits
    bp1 = axes[0].boxplot(
        [daily_totals], vert=True, patch_artist=True, labels=["Daily Total Edits"]
    )
    bp1["boxes"][0].set_facecolor("steelblue")
    axes[0].set_ylabel("Edits")
    axes[0].set_title("Daily Total Edits Distribution")
    axes[0].grid(axis="y", alpha=0.3)

    # distribution of files edited per day
    bp2 = axes[1].boxplot(
        [files_per_day], vert=True, patch_artist=True, labels=["Files per Day"]
    )
    bp2["boxes"][0].set_facecolor("forestgreen")
    axes[1].set_ylabel("Number of Files")
    axes[1].set_title("Files Edited per Day Distribution")
    axes[1].grid(axis="y", alpha=0.3)

    # activiy organized by day of week
    daily_data_with_dow = []
    for i, day in enumerate(git_data):
        dow = i % 7  # rough approximation of day of week
        daily_data_with_dow.append((dow, day["total_changes"]))

    dow_groups = [[] for _ in range(7)]
    for dow, changes in daily_data_with_dow:
        dow_groups[dow].append(changes)

    bp3 = axes[2].boxplot(
        dow_groups,
        vert=True,
        patch_artist=True,
        labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )
    for patch in bp3["boxes"]:
        patch.set_facecolor("coral")
    axes[2].set_xlabel("Day of Week")
    axes[2].set_ylabel("Edits")
    axes[2].set_title("Activity by Day of Week")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/summary_statistics.png", bbox_inches="tight")
    plt.close()
    print("summary_statistics.png created")


def create_bivariate_analysis(git_data, git_vectors):
    """1.3 - Bivariate Analysis"""
    fig = plt.figure(figsize=(14, 10))

    # df preparation
    daily_totals = np.array([day["total_changes"] for day in git_data])
    files_per_day = np.array([day["total_files"] for day in git_data])
    edits_per_file = daily_totals / (files_per_day + 1e-8)

    # rolling volatility (std dev over 7-day window)
    window = 7
    rolling_mean = np.convolve(daily_totals, np.ones(window) / window, mode="valid")
    rolling_std = np.array(
        [
            np.std(daily_totals[max(0, i - window) : i + 1])
            for i in range(len(daily_totals))
        ]
    )

    # corr matrix
    ax1 = plt.subplot(2, 3, 1)
    corr_data = np.column_stack(
        [daily_totals, files_per_day, edits_per_file, rolling_std]
    )
    corr_matrix = np.corrcoef(corr_data.T)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        xticklabels=["Daily Total", "Files/Day", "Edits/File", "Volatility"],
        yticklabels=["Daily Total", "Files/Day", "Edits/File", "Volatility"],
        vmin=-1,
        vmax=1,
        ax=ax1,
        cbar_kws={"label": "Correlation"},
    )
    ax1.set_title("Correlation Matrix")

    # scatter offiles v total Edits
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(files_per_day, daily_totals, alpha=0.5, c="steelblue", s=30)
    z = np.polyfit(files_per_day, daily_totals, 1)
    p = np.poly1d(z)
    ax2.plot(files_per_day, p(files_per_day), "r--", alpha=0.8, linewidth=2)
    ax2.set_xlabel("Files Edited per Day")
    ax2.set_ylabel("Total Edits per Day")
    ax2.set_title(
        f"Files vs Total Edits\n(r = {np.corrcoef(files_per_day, daily_totals)[0,1]:.3f})"
    )
    ax2.grid(alpha=0.3)

    # scatter breadth v depth
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(files_per_day, edits_per_file, alpha=0.5, c="forestgreen", s=30)
    ax3.set_xlabel("Files Edited per Day")
    ax3.set_ylabel("Edits per File")
    ax3.set_title(
        f"Breadth vs Depth\n(r = {np.corrcoef(files_per_day, edits_per_file)[0,1]:.3f})"
    )
    ax3.grid(alpha=0.3)

    # scatter activity v volatility
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(daily_totals, rolling_std, alpha=0.5, c="coral", s=30)
    ax4.set_xlabel("Daily Total Edits")
    ax4.set_ylabel("7-day Volatility (SD)")
    ax4.set_title("Activity vs Volatility")
    ax4.grid(alpha=0.3)

    # corr file activity
    ax5 = plt.subplot(2, 3, 5)
    file_totals = np.sum(git_vectors, axis=0)
    top_files = np.argsort(file_totals)[-100:]
    file_sample = git_vectors[:, top_files[:50]]

    file_corr = np.corrcoef(file_sample.T)
    sns.heatmap(
        file_corr,
        cmap="viridis",
        ax=ax5,
        cbar_kws={"label": "Correlation"},
        xticklabels=False,
        yticklabels=False,
    )
    ax5.set_title("File Activity Correlations\n(Top 50 Files)")

    # Lag correlation
    ax6 = plt.subplot(2, 3, 6)
    lags = range(1, 31)
    autocorrs = []
    for lag in lags:
        if lag < len(daily_totals):
            corr = np.corrcoef(daily_totals[:-lag], daily_totals[lag:])[0, 1]
            autocorrs.append(corr if not np.isnan(corr) else 0)
        else:
            autocorrs.append(0)

    ax6.bar(lags, autocorrs, color="mediumpurple", alpha=0.7)
    ax6.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax6.axhline(
        y=1.96 / np.sqrt(len(daily_totals)),
        color="red",
        linestyle="--",
        label="95% CI",
        linewidth=1,
    )
    ax6.axhline(
        y=-1.96 / np.sqrt(len(daily_totals)), color="red", linestyle="--", linewidth=1
    )
    ax6.set_xlabel("Lag (days)")
    ax6.set_ylabel("Autocorrelation")
    ax6.set_title("Temporal Autocorrelation")
    ax6.legend()
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/bivariate_analysis.png", bbox_inches="tight")
    plt.close()
    print(" Created bivariate_analysis.png")


def create_multivariate_analysis(git_data, git_vectors):
    """1.4 Multivariate Analysis - Pair plot and 3D scatter"""
    # Prepare data
    daily_totals = np.array([day["total_changes"] for day in git_data])
    files_per_day = np.array([day["total_files"] for day in git_data])
    edits_per_file = daily_totals / (files_per_day + 1e-8)

    window = 7
    rolling_std = np.array(
        [
            np.std(daily_totals[max(0, i - window) : i + 1])
            for i in range(len(daily_totals))
        ]
    )

    # Create DataFrame
    df = pd.DataFrame(
        {
            "Total_Edits": daily_totals,
            "Files_Edited": files_per_day,
            "Edits_per_File": edits_per_file,
            "Volatility": rolling_std,
        }
    )

    # Pair plot
    fig = plt.figure(figsize=(12, 10))

    # 3D scatter plot
    ax = fig.add_subplot(221, projection="3d")
    scatter = ax.scatter(
        files_per_day,
        edits_per_file,
        daily_totals,
        c=rolling_std,
        cmap="viridis",
        s=30,
        alpha=0.6,
    )
    ax.set_xlabel("Files Edited")
    ax.set_ylabel("Edits per File")
    ax.set_zlabel("Total Edits")
    ax.set_title("3D Multivariate View")
    plt.colorbar(scatter, ax=ax, label="Volatility", pad=0.1)

    # PCA visualization
    ax2 = fig.add_subplot(222)
    data_scaled = (df - df.mean()) / df.std()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)

    scatter2 = ax2.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=daily_totals,
        cmap="coolwarm",
        s=30,
        alpha=0.6,
    )
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax2.set_title("PCA: First Two Components")
    plt.colorbar(scatter2, ax=ax2, label="Total Edits")
    ax2.grid(alpha=0.3)

    # Interaction effect: Files × Volatility → Total Edits
    ax3 = fig.add_subplot(223)
    # Bin files and volatility
    files_bins = np.digitize(files_per_day, bins=np.percentile(files_per_day, [33, 67]))
    vol_bins = np.digitize(rolling_std, bins=np.percentile(rolling_std, [33, 67]))

    interaction_means = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            mask = (files_bins == i) & (vol_bins == j)
            if np.sum(mask) > 0:
                interaction_means[i, j] = np.mean(daily_totals[mask])

    sns.heatmap(
        interaction_means,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax3,
        xticklabels=["Low Vol", "Med Vol", "High Vol"],
        yticklabels=["Few Files", "Med Files", "Many Files"],
    )
    ax3.set_title("Interaction: Files × Volatility → Total Edits")
    ax3.set_xlabel("Volatility Level")
    ax3.set_ylabel("Files Edited Level")

    # Conditional distributions
    ax4 = fig.add_subplot(224)
    low_files = daily_totals[files_per_day < np.percentile(files_per_day, 33)]
    high_files = daily_totals[files_per_day > np.percentile(files_per_day, 67)]

    ax4.hist(
        low_files,
        bins=20,
        alpha=0.5,
        label="Few Files (<33%)",
        color="blue",
        density=True,
    )
    ax4.hist(
        high_files,
        bins=20,
        alpha=0.5,
        label="Many Files (>67%)",
        color="red",
        density=True,
    )
    ax4.set_xlabel("Total Edits")
    ax4.set_ylabel("Density")
    ax4.set_title("Conditional Distribution of Total Edits")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/multivariate_analysis.png", bbox_inches="tight")
    plt.close()
    print("Created multivariate_analysis.png")


def create_pattern_recognition(git_data):
    """2.1 Pattern Recognition - Outliers and trends"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    daily_totals = np.array([day["total_changes"] for day in git_data])

    # Outlier detection using Z-score
    z_scores = zscore(daily_totals)
    outliers = np.abs(z_scores) > 2

    ax1 = axes[0, 0]
    ax1.scatter(
        range(len(daily_totals)),
        daily_totals,
        alpha=0.5,
        c="steelblue",
        s=20,
        label="Normal",
    )
    ax1.scatter(
        np.where(outliers)[0],
        daily_totals[outliers],
        c="red",
        s=60,
        marker="*",
        label="Outliers (|Z|>2)",
        zorder=5,
    )
    ax1.set_xlabel("Day Index")
    ax1.set_ylabel("Total Edits")
    ax1.set_title(
        f"Outlier Detection (Z-score method)\n{np.sum(outliers)} outliers detected"
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Box plot with outliers
    ax2 = axes[0, 1]
    bp = ax2.boxplot(
        [daily_totals],
        vert=True,
        patch_artist=True,
        showfliers=True,
        labels=["Daily Edits"],
    )
    bp["boxes"][0].set_facecolor("lightblue")
    ax2.set_ylabel("Total Edits")
    ax2.set_title("Box Plot Outlier Detection")
    ax2.grid(axis="y", alpha=0.3)

    # Add statistics
    q1, q3 = np.percentile(daily_totals, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    n_outliers = np.sum((daily_totals < lower_bound) | (daily_totals > upper_bound))
    ax2.text(1.15, upper_bound, f"{n_outliers} outliers", fontsize=9)

    # Trend analysis with moving average
    ax3 = axes[1, 0]
    window = 7
    ma = np.convolve(daily_totals, np.ones(window) / window, mode="valid")

    ax3.plot(daily_totals, alpha=0.3, color="gray", label="Raw Data")
    ax3.plot(
        range(window - 1, len(daily_totals)),
        ma,
        color="blue",
        linewidth=2,
        label=f"{window}-day MA",
    )

    # Fit linear trend
    x = np.arange(len(ma))
    z = np.polyfit(x, ma, 1)
    p = np.poly1d(z)
    ax3.plot(
        range(window - 1, len(daily_totals)),
        p(x),
        "r--",
        linewidth=2,
        label=f"Trend (slope={z[0]:.2f})",
    )

    ax3.set_xlabel("Day Index")
    ax3.set_ylabel("Total Edits")
    ax3.set_title("Trend Analysis with Moving Average")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Distribution comparison: Normal vs Actual
    ax4 = axes[1, 1]
    ax4.hist(
        daily_totals,
        bins=30,
        density=True,
        alpha=0.6,
        color="steelblue",
        label="Actual Distribution",
    )

    # Overlay normal distribution
    mu, sigma = np.mean(daily_totals), np.std(daily_totals)
    x = np.linspace(daily_totals.min(), daily_totals.max(), 100)
    ax4.plot(
        x,
        stats.norm.pdf(x, mu, sigma),
        "r-",
        linewidth=2,
        label=f"Normal(μ={mu:.0f}, σ={sigma:.0f})",
    )

    ax4.set_xlabel("Total Edits")
    ax4.set_ylabel("Density")
    ax4.set_title("Distribution vs Normal\n(Assessing Pattern Normality)")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Add skewness and kurtosis
    skew = stats.skew(daily_totals)
    kurt = stats.kurtosis(daily_totals)
    ax4.text(
        0.6,
        0.9,
        f"Skewness: {skew:.2f}\nKurtosis: {kurt:.2f}",
        transform=ax4.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig("figures/pattern_recognition.png", bbox_inches="tight")
    plt.close()
    print("Created pattern_recognition.png")


def create_time_series_analysis(git_data, dates):
    """2.2 Time Series Analysis"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    daily_totals = np.array([day["total_changes"] for day in git_data])

    # Time series plot with trend
    ax1 = axes[0]
    ax1.plot(daily_totals, alpha=0.6, color="steelblue", linewidth=1)

    # Add moving averages
    window_short = 7
    window_long = 30
    ma_short = np.convolve(
        daily_totals, np.ones(window_short) / window_short, mode="valid"
    )
    ma_long = np.convolve(
        daily_totals, np.ones(window_long) / window_long, mode="valid"
    )

    ax1.plot(
        range(window_short - 1, len(daily_totals)),
        ma_short,
        color="orange",
        linewidth=2,
        label="7-day MA",
    )
    ax1.plot(
        range(window_long - 1, len(daily_totals)),
        ma_long,
        color="red",
        linewidth=2,
        label="30-day MA",
    )

    ax1.set_xlabel("Days")
    ax1.set_ylabel("Total Edits")
    ax1.set_title("Time Series of Daily Activity with Moving Averages")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Seasonal decomposition approximation
    ax2 = axes[1]
    # Weekly pattern
    weekly_avg = np.zeros(7)
    for i in range(7):
        weekly_avg[i] = np.mean(
            [daily_totals[j] for j in range(i, len(daily_totals), 7)]
        )

    weekly_pattern = np.tile(weekly_avg, len(daily_totals) // 7 + 1)[
        : len(daily_totals)
    ]

    ax2.plot(weekly_pattern, color="green", linewidth=2, label="Weekly Pattern")
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Average Edits")
    ax2.set_title("Detected Weekly Seasonality")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Residuals (detrended, deseasonalized)
    ax3 = axes[2]
    trend = np.convolve(daily_totals, np.ones(window_long) / window_long, mode="same")
    residuals = daily_totals - trend - (weekly_pattern - np.mean(weekly_pattern))

    ax3.plot(residuals, color="purple", alpha=0.6, linewidth=1)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.axhline(
        y=2 * np.std(residuals), color="red", linestyle="--", linewidth=1, label="±2σ"
    )
    ax3.axhline(y=-2 * np.std(residuals), color="red", linestyle="--", linewidth=1)
    ax3.fill_between(
        range(len(residuals)),
        -2 * np.std(residuals),
        2 * np.std(residuals),
        alpha=0.1,
        color="red",
    )
    ax3.set_xlabel("Days")
    ax3.set_ylabel("Residuals")
    ax3.set_title("Residuals (After Trend & Seasonality Removal)")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/time_series_analysis.png", bbox_inches="tight")
    plt.close()
    print("Created time_series_analysis.png")


def create_segmentation_analysis(git_data, git_vectors):
    """2.3 Segmentation Analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    daily_totals = np.array([day["total_changes"] for day in git_data])
    files_per_day = np.array([day["total_files"] for day in git_data])

    # Segment by activity level
    low_activity = daily_totals < np.percentile(daily_totals, 33)
    med_activity = (daily_totals >= np.percentile(daily_totals, 33)) & (
        daily_totals < np.percentile(daily_totals, 67)
    )
    high_activity = daily_totals >= np.percentile(daily_totals, 67)

    # Segment statistics
    ax1 = axes[0, 0]
    segments = [
        "Low\nActivity\n(<33%)",
        "Medium\nActivity\n(33-67%)",
        "High\nActivity\n(>67%)",
    ]
    segment_counts = [np.sum(low_activity), np.sum(med_activity), np.sum(high_activity)]
    segment_means = [
        np.mean(daily_totals[low_activity]),
        np.mean(daily_totals[med_activity]),
        np.mean(daily_totals[high_activity]),
    ]

    x = np.arange(len(segments))
    width = 0.35

    ax1_twin = ax1.twinx()
    bars1 = ax1.bar(
        x - width / 2,
        segment_counts,
        width,
        label="Count",
        color="steelblue",
        alpha=0.7,
    )
    bars2 = ax1_twin.bar(
        x + width / 2,
        segment_means,
        width,
        label="Mean Edits",
        color="coral",
        alpha=0.7,
    )

    ax1.set_xlabel("Activity Segment")
    ax1.set_ylabel("Number of Days", color="steelblue")
    ax1_twin.set_ylabel("Mean Edits per Day", color="coral")
    ax1.set_title("Segmentation by Activity Level")
    ax1.set_xticks(x)
    ax1.set_xticklabels(segments)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1_twin.tick_params(axis="y", labelcolor="coral")
    ax1.grid(alpha=0.3)

    # filed edited comparison
    ax2 = axes[0, 1]
    data_to_plot = [
        files_per_day[low_activity],
        files_per_day[med_activity],
        files_per_day[high_activity],
    ]

    bp = ax2.boxplot(data_to_plot, labels=["Low", "Medium", "High"], patch_artist=True)
    colors = ["lightblue", "lightgreen", "lightcoral"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax2.set_xlabel("Activity Segment")
    ax2.set_ylabel("Files Edited per Day")
    ax2.set_title("Files Edited Across Segments")
    ax2.grid(axis="y", alpha=0.3)

    # 2D segmentation: Activity × Files
    ax3 = axes[1, 0]
    scatter = ax3.scatter(
        files_per_day, daily_totals, c=daily_totals, cmap="viridis", s=50, alpha=0.6
    )

    # quadrant setup
    ax3.axhline(y=np.median(daily_totals), color="red", linestyle="--", alpha=0.5)
    ax3.axvline(x=np.median(files_per_day), color="red", linestyle="--", alpha=0.5)

    ax3.text(
        0.75 * np.max(files_per_day),
        0.75 * np.max(daily_totals),
        "High Activity\nMany Files",
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
    )
    ax3.text(
        0.25 * np.max(files_per_day),
        0.75 * np.max(daily_totals),
        "High Activity\nFew Files",
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="orange", alpha=0.5),
    )

    ax3.set_xlabel("Files Edited per Day")
    ax3.set_ylabel("Total Edits per Day")
    ax3.set_title("2D Segmentation: Activity vs Breadth")
    plt.colorbar(scatter, ax=ax3, label="Total Edits")
    ax3.grid(alpha=0.3)

    # Temporal segmentation over time
    ax4 = axes[1, 1]
    colors_timeline = [
        "blue" if low_activity[i] else "green" if med_activity[i] else "red"
        for i in range(len(daily_totals))
    ]
    ax4.scatter(
        range(len(daily_totals)), daily_totals, c=colors_timeline, s=20, alpha=0.6
    )
    ax4.set_xlabel("Days")
    ax4.set_ylabel("Total Edits")
    ax4.set_title("Temporal View of Activity Segments")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="blue", label="Low Activity"),
        Patch(facecolor="green", label="Medium Activity"),
        Patch(facecolor="red", label="High Activity"),
    ]
    ax4.legend(handles=legend_elements, loc="upper left")
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/segmentation_analysis.png", bbox_inches="tight")
    plt.close()
    print("Created segmentation_analysis.png")


def create_advanced_visualizations(git_data, git_vectors):
    """3.2 Advanced Visualizations - KDE, 3D, dashboard"""
    fig = plt.figure(figsize=(16, 12))

    daily_totals = np.array([day["total_changes"] for day in git_data])
    files_per_day = np.array([day["total_files"] for day in git_data])

    # 2D KDE plot
    ax1 = plt.subplot(2, 3, 1)
    from scipy.stats import gaussian_kde

    # Remove zeros for better KDE
    mask = (files_per_day > 0) & (daily_totals > 0)
    x_kde = files_per_day[mask]
    y_kde = daily_totals[mask]

    if len(x_kde) > 10:
        xy = np.vstack([x_kde, y_kde])
        z = gaussian_kde(xy)(xy)

        scatter = ax1.scatter(x_kde, y_kde, c=z, s=30, cmap="viridis", alpha=0.6)
        plt.colorbar(scatter, ax=ax1, label="Density")

    ax1.set_xlabel("Files Edited per Day")
    ax1.set_ylabel("Total Edits per Day")
    ax1.set_title("2D KDE Plot: Activity Density")
    ax1.grid(alpha=0.3)

    # 3D surface plot (activity over time and rolling window)
    ax2 = plt.subplot(2, 3, 2, projection="3d")

    window_sizes = [3, 7, 14, 30]
    time_points = np.arange(50, len(daily_totals) - 30, 10)

    X, Y = np.meshgrid(window_sizes, time_points)
    Z = np.zeros_like(X, dtype=float)

    for i, t in enumerate(time_points):
        for j, w in enumerate(window_sizes):
            if t >= w:
                Z[i, j] = np.mean(daily_totals[t - w : t])

    surf = ax2.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.8)
    ax2.set_xlabel("Window Size (days)")
    ax2.set_ylabel("Time Point")
    ax2.set_zlabel("Rolling Mean")
    ax2.set_title("3D: Rolling Statistics Surface")

    # hexbin plot
    ax3 = plt.subplot(2, 3, 3)
    hexbin = ax3.hexbin(
        files_per_day, daily_totals, gridsize=20, cmap="YlOrRd", mincnt=1
    )
    ax3.set_xlabel("Files Edited per Day")
    ax3.set_ylabel("Total Edits per Day")
    ax3.set_title("Hexbin Density Plot")
    plt.colorbar(hexbin, ax=ax3, label="Count")

    # ridgeline plot approx
    ax4 = plt.subplot(2, 3, 4)

    # segemtns by time periods
    n_periods = 5
    period_size = len(daily_totals) // n_periods

    for i in range(n_periods):
        start = i * period_size
        end = (i + 1) * period_size if i < n_periods - 1 else len(daily_totals)
        period_data = daily_totals[start:end]

        if len(period_data) > 10:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(period_data)
            x_range = np.linspace(0, np.max(daily_totals), 200)
            density = kde(x_range)

            offset = i * 0.0002
            ax4.fill_between(
                x_range, offset, density + offset, alpha=0.6, label=f"Period {i+1}"
            )

    ax4.set_xlabel("Total Edits")
    ax4.set_ylabel("Period (Density)")
    ax4.set_title("Ridgeline Plot: Temporal Evolution")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # circular/polar plot for weekly pattern
    ax5 = plt.subplot(2, 3, 5, projection="polar")

    weekly_avg = np.zeros(7)
    for i in range(7):
        weekly_avg[i] = np.mean(
            [daily_totals[j] for j in range(i, len(daily_totals), 7)]
        )

    theta = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    radii = weekly_avg
    width = 2 * np.pi / 7

    bars = ax5.bar(theta, radii, width=width, bottom=0.0, alpha=0.7, color="steelblue")
    ax5.set_theta_zero_location("N")
    ax5.set_theta_direction(-1)
    ax5.set_xticks(theta)
    ax5.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax5.set_title("Weekly Activity Pattern\n(Polar Plot)", pad=20)

    # heatmap calendar-style
    ax6 = plt.subplot(2, 3, 6)

    # reshape into weeks
    n_weeks = len(daily_totals) // 7
    calendar_data = daily_totals[: n_weeks * 7].reshape(n_weeks, 7)

    sns.heatmap(
        calendar_data,
        cmap="YlOrRd",
        ax=ax6,
        cbar_kws={"label": "Edits"},
        xticklabels=["M", "T", "W", "T", "F", "S", "S"],
        yticklabels=False,
    )
    ax6.set_xlabel("Day of Week")
    ax6.set_ylabel("Week Number")
    ax6.set_title("Calendar Heatmap: Activity by Week")

    plt.tight_layout()
    plt.savefig("figures/advanced_visualizations.png", bbox_inches="tight")
    plt.close()
    print("Created advanced_visualizations.png")


def create_summary_dashboard(git_data):
    """Additional dashboard visualization"""
    fig = plt.figure(figsize=(16, 10))

    daily_totals = np.array([day["total_changes"] for day in git_data])
    files_per_day = np.array([day["total_files"] for day in git_data])

    # primary time series
    ax1 = plt.subplot(3, 3, (1, 3))
    ax1.fill_between(
        range(len(daily_totals)), daily_totals, alpha=0.3, color="steelblue"
    )
    ax1.plot(daily_totals, color="steelblue", linewidth=1)

    window = 7
    ma = np.convolve(daily_totals, np.ones(window) / window, mode="valid")
    ax1.plot(
        range(window - 1, len(daily_totals)),
        ma,
        color="red",
        linewidth=2,
        label="7-day MA",
    )

    ax1.set_xlabel("Days")
    ax1.set_ylabel("Total Edits")
    ax1.set_title("Daily Activity Overview", fontweight="bold", fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # distribution
    ax2 = plt.subplot(3, 3, 4)
    ax2.hist(daily_totals, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.axvline(
        np.mean(daily_totals), color="red", linestyle="--", linewidth=2, label="Mean"
    )
    ax2.axvline(
        np.median(daily_totals),
        color="green",
        linestyle="--",
        linewidth=2,
        label="Median",
    )
    ax2.set_xlabel("Total Edits")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # boxplot
    ax3 = plt.subplot(3, 3, 5)
    bp = ax3.boxplot([daily_totals], vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    ax3.set_ylabel("Total Edits")
    ax3.set_title("Box Plot")
    ax3.grid(axis="y", alpha=0.3)

    # stattable
    ax4 = plt.subplot(3, 3, 6)
    ax4.axis("off")

    stats_text = f"""
    Summary Statistics
    Mean:       {np.mean(daily_totals):.1f}
    Median:     {np.median(daily_totals):.1f}
    Std Dev:    {np.std(daily_totals):.1f}
    Min:        {np.min(daily_totals):.1f}
    Max:        {np.max(daily_totals):.1f}

    Skewness:   {stats.skew(daily_totals):.2f}
    Kurtosis:   {stats.kurtosis(daily_totals):.2f}
    CV:         {np.std(daily_totals)/np.mean(daily_totals):.2f}
    """
    ax4.text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
    )

    # scatter for files vs edits
    ax5 = plt.subplot(3, 3, 7)
    ax5.scatter(files_per_day, daily_totals, alpha=0.5, s=30, c="steelblue")
    z = np.polyfit(files_per_day, daily_totals, 1)
    p = np.poly1d(z)
    ax5.plot(files_per_day, p(files_per_day), "r--", linewidth=2)
    ax5.set_xlabel("Files Edited")
    ax5.set_ylabel("Total Edits")
    ax5.set_title(
        f"Files vs Edits (r={np.corrcoef(files_per_day, daily_totals)[0,1]:.3f})"
    )
    ax5.grid(alpha=0.3)

    # autocorrelation plot
    ax6 = plt.subplot(3, 3, 8)
    lags = range(1, 31)
    autocorrs = []
    for lag in lags:
        if lag < len(daily_totals):
            corr = np.corrcoef(daily_totals[:-lag], daily_totals[lag:])[0, 1]
            autocorrs.append(corr if not np.isnan(corr) else 0)

    ax6.bar(lags, autocorrs, color="mediumpurple", alpha=0.7)
    ax6.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax6.set_xlabel("Lag (days)")
    ax6.set_ylabel("Autocorrelation")
    ax6.set_title("Autocorrelation Function")
    ax6.grid(alpha=0.3)

    # activity segments pie chart
    ax7 = plt.subplot(3, 3, 9)
    low = np.sum(daily_totals < np.percentile(daily_totals, 33))
    med = np.sum(
        (daily_totals >= np.percentile(daily_totals, 33))
        & (daily_totals < np.percentile(daily_totals, 67))
    )
    high = np.sum(daily_totals >= np.percentile(daily_totals, 67))

    ax7.pie(
        [low, med, high],
        labels=["Low", "Medium", "High"],
        autopct="%1.1f%%",
        colors=["lightblue", "lightgreen", "lightcoral"],
        startangle=90,
    )
    ax7.set_title("Activity Segments")

    plt.suptitle("figures Dashboard", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig("eda_figures/summary_dashboard.png", bbox_inches="tight")
    plt.close()
    print("Created summary_dashboard.png")


if __name__ == "__main__":
    import os

    git_data, git_vectors, frontmatter, wikilink, dates, stability = load_data()

    print("\nGenerating visualizations...")
    print("\n Variable Analysis:")
    create_univariate_analysis(git_data, git_vectors)
    create_summary_statistics_boxplots(git_data)
    create_bivariate_analysis(git_data, git_vectors)
    create_multivariate_analysis(git_data, git_vectors)

    print("\n Pattern Analysis:")
    create_pattern_recognition(git_data)
    create_time_series_analysis(git_data, dates)
    create_segmentation_analysis(git_data, git_vectors)

    print("\n Visualizations:")
    create_advanced_visualizations(git_data, git_vectors)
    create_summary_dashboard(git_data)
