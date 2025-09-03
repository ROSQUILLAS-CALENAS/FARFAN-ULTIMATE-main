#!/usr/bin/env python3
"""
Conformal Risk Control System Demonstration

This example demonstrates the usage of the conformal risk control system
based on Angelopoulos et al. (2024) "Conformal Risk Control" for generating
distribution-free prediction guarantees.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# # # from sklearn.calibration import CalibratedClassifierCV  # Module not found  # Module not found  # Module not found
# # # from sklearn.datasets import make_classification, make_regression  # Module not found  # Module not found  # Module not found
# # # from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Module not found  # Module not found  # Module not found
# # # from sklearn.metrics import accuracy_score, classification_report  # Module not found  # Module not found  # Module not found
# # # from sklearn.model_selection import train_test_split  # Module not found  # Module not found  # Module not found

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import conformal risk control components
# # # from egw_query_expansion.core.conformal_risk_control import (  # Module not found  # Module not found  # Module not found
    RiskControlConfig,
    create_conformal_system,
)


def demo_classification():
    """Demonstrate conformal risk control for classification"""
    print("\n" + "=" * 60)
    print("CONFORMAL RISK CONTROL - CLASSIFICATION DEMO")
    print("=" * 60)

    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_classes=4,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=42,
    )

    print(
        f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes"
    )

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42, stratify=y
    )

    # Train base classifier
    print("Training base classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Get calibrated predictions
    calibrated_clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    calibrated_clf.fit(X_train, y_train)

    # Evaluate base classifier
    y_pred = calibrated_clf.predict(X_test)
    base_accuracy = accuracy_score(y_test, y_pred)
    print(f"Base classifier accuracy: {base_accuracy:.3f}")

    # Get prediction probabilities for conformal prediction
    predictions = calibrated_clf.predict_proba(X_test)

    # Create conformal risk control system
    print("\nCreating conformal risk control system...")

    # Test different risk levels
    alphas = [0.05, 0.1, 0.15, 0.2]
    results = {}

    for alpha in alphas:
        print(f"\nTesting with α = {alpha} (target coverage = {1-alpha:.2f})")

        controller, score_fn = create_conformal_system(
            task_type="classification", alpha=alpha, seed=42
        )

        # Generate certificate
        result = controller.certify_predictions(
            X_test, y_test, predictions, score_fn, task_type="classification"
        )

        # Validate certificate
        validation = controller.validate_certificate(result)

        results[alpha] = {"result": result, "validation": validation}

        print(f"  Empirical coverage: {result.empirical_coverage:.3f}")
        print(f"  Risk bound: {result.risk_bound:.3f}")
        print(f"  Lambda threshold: {result.lambda_hat:.3f}")
        print(f"  Miscoverage risk: {result.miscoverage_risk:.3f}")
        print(f"  Certificate hash: {result.certificate_hash}")

        # Prediction set statistics
        set_sizes = [len(ps) for ps in result.prediction_sets]
        print(f"  Avg prediction set size: {np.mean(set_sizes):.2f}")
        print(f"  Set size range: [{min(set_sizes)}, {max(set_sizes)}]")

        # Validation results
        print(f"  Coverage guarantee satisfied: {validation['coverage_guarantee']}")
        print(f"  Risk bound valid: {validation['risk_bound_valid']}")

    # Visualization
    plot_classification_results(results, y_test)

    return results


def demo_regression():
    """Demonstrate conformal risk control for regression"""
    print("\n" + "=" * 60)
    print("CONFORMAL RISK CONTROL - REGRESSION DEMO")
    print("=" * 60)

    # Generate synthetic regression dataset
    X, y = make_regression(n_samples=1500, n_features=15, noise=0.1, random_state=42)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42
    )

    # Train base regressor
    print("Training base regressor...")
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)

    # Get predictions
    predictions = reg.predict(X_test)

    # Evaluate base regressor
    mse = np.mean((predictions - y_test) ** 2)
    print(f"Base regressor MSE: {mse:.3f}")

    # Create conformal risk control system
    print("\nCreating conformal risk control system...")

    # Test different risk levels
    alphas = [0.05, 0.1, 0.2]
    results = {}

    for alpha in alphas:
        print(f"\nTesting with α = {alpha} (target coverage = {1-alpha:.2f})")

        controller, score_fn = create_conformal_system(
            task_type="regression", alpha=alpha, seed=42
        )

        # Generate certificate
        result = controller.certify_predictions(
            X_test, y_test, predictions, score_fn, task_type="regression"
        )

        # Validate certificate
        validation = controller.validate_certificate(result)

        results[alpha] = {"result": result, "validation": validation}

        print(f"  Empirical coverage: {result.empirical_coverage:.3f}")
        print(f"  Risk bound: {result.risk_bound:.3f}")
        print(f"  Lambda threshold: {result.lambda_hat:.3f}")
        print(f"  Certificate hash: {result.certificate_hash}")

        # Interval statistics
        intervals = result.confidence_intervals
        interval_widths = intervals[:, 1] - intervals[:, 0]
        print(f"  Avg interval width: {np.mean(interval_widths):.3f}")
        print(
            f"  Width range: [{interval_widths.min():.3f}, {interval_widths.max():.3f}]"
        )

        # Validation results
        print(f"  Coverage guarantee satisfied: {validation['coverage_guarantee']}")
        print(f"  Risk bound valid: {validation['risk_bound_valid']}")

    # Visualization
    plot_regression_results(results, y_test, predictions)

    return results


def plot_classification_results(results, y_test):
    """Visualize classification conformal prediction results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Conformal Risk Control - Classification Results", fontsize=16)

    alphas = list(results.keys())
    coverages = [results[a]["result"].empirical_coverage for a in alphas]
    targets = [1 - a for a in alphas]
    risk_bounds = [results[a]["result"].risk_bound for a in alphas]

    # Coverage vs Target
    axes[0, 0].plot(targets, coverages, "bo-", label="Empirical Coverage")
    axes[0, 0].plot(
        [min(targets), max(targets)],
        [min(targets), max(targets)],
        "r--",
        label="Perfect Coverage",
    )
    axes[0, 0].set_xlabel("Target Coverage")
    axes[0, 0].set_ylabel("Empirical Coverage")
    axes[0, 0].set_title("Coverage Calibration")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Risk bounds
    axes[0, 1].bar(range(len(alphas)), risk_bounds)
    axes[0, 1].set_xlabel("Risk Level (α)")
    axes[0, 1].set_ylabel("Risk Bound")
    axes[0, 1].set_title("Risk Bounds")
    axes[0, 1].set_xticks(range(len(alphas)))
    axes[0, 1].set_xticklabels([f"{a:.2f}" for a in alphas])

    # Prediction set sizes
    for i, alpha in enumerate(alphas):
        set_sizes = [len(ps) for ps in results[alpha]["result"].prediction_sets]
        axes[1, 0].hist(set_sizes, alpha=0.7, label=f"α={alpha:.2f}", bins=10)

    axes[1, 0].set_xlabel("Prediction Set Size")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Distribution of Prediction Set Sizes")
    axes[1, 0].legend()

    # Coverage efficiency
    efficiencies = []
    for alpha in alphas:
        set_sizes = [len(ps) for ps in results[alpha]["result"].prediction_sets]
        avg_size = np.mean(set_sizes)
        coverage = results[alpha]["result"].empirical_coverage
        efficiency = coverage / avg_size if avg_size > 0 else 0
        efficiencies.append(efficiency)

    axes[1, 1].bar(range(len(alphas)), efficiencies)
    axes[1, 1].set_xlabel("Risk Level (α)")
    axes[1, 1].set_ylabel("Coverage/Average Set Size")
    axes[1, 1].set_title("Prediction Efficiency")
    axes[1, 1].set_xticks(range(len(alphas)))
    axes[1, 1].set_xticklabels([f"{a:.2f}" for a in alphas])

    plt.tight_layout()
    plt.savefig("conformal_classification_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_regression_results(results, y_test, predictions):
    """Visualize regression conformal prediction results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Conformal Risk Control - Regression Results", fontsize=16)

    alphas = list(results.keys())

    # Coverage comparison
    coverages = [results[a]["result"].empirical_coverage for a in alphas]
    targets = [1 - a for a in alphas]

    axes[0, 0].plot(targets, coverages, "bo-", label="Empirical Coverage")
    axes[0, 0].plot(
        [min(targets), max(targets)],
        [min(targets), max(targets)],
        "r--",
        label="Perfect Coverage",
    )
    axes[0, 0].set_xlabel("Target Coverage")
    axes[0, 0].set_ylabel("Empirical Coverage")
    axes[0, 0].set_title("Coverage Calibration")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Prediction intervals visualization (for smallest alpha)
    alpha_min = min(alphas)
    intervals = results[alpha_min]["result"].confidence_intervals

    # Sort by prediction for cleaner visualization
    sort_idx = np.argsort(predictions)
    y_test_sorted = y_test[sort_idx]
    pred_sorted = predictions[sort_idx]
    intervals_sorted = intervals[sort_idx]

    # Sample for visualization (avoid overcrowding)
    n_plot = min(200, len(y_test))
    step = len(y_test) // n_plot

    x_plot = np.arange(n_plot)
    axes[0, 1].scatter(x_plot, y_test_sorted[::step], alpha=0.6, s=20, label="True")
    axes[0, 1].scatter(x_plot, pred_sorted[::step], alpha=0.6, s=20, label="Predicted")
    axes[0, 1].fill_between(
        x_plot,
        intervals_sorted[::step, 0],
        intervals_sorted[::step, 1],
        alpha=0.3,
        label=f"Conformal Interval (α={alpha_min})",
    )
    axes[0, 1].set_xlabel("Sample Index")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].set_title(f"Conformal Prediction Intervals")
    axes[0, 1].legend()

    # Interval widths
    for i, alpha in enumerate(alphas):
        intervals = results[alpha]["result"].confidence_intervals
        widths = intervals[:, 1] - intervals[:, 0]
        axes[1, 0].hist(widths, alpha=0.7, label=f"α={alpha:.2f}", bins=20)

    axes[1, 0].set_xlabel("Interval Width")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Distribution of Interval Widths")
    axes[1, 0].legend()

    # Coverage vs interval width trade-off
    avg_widths = []
    for alpha in alphas:
        intervals = results[alpha]["result"].confidence_intervals
        widths = intervals[:, 1] - intervals[:, 0]
        avg_widths.append(np.mean(widths))

    axes[1, 1].scatter(avg_widths, coverages, s=100)
    for i, alpha in enumerate(alphas):
        axes[1, 1].annotate(
            f"α={alpha:.2f}",
            (avg_widths[i], coverages[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    axes[1, 1].set_xlabel("Average Interval Width")
    axes[1, 1].set_ylabel("Empirical Coverage")
    axes[1, 1].set_title("Coverage vs Interval Width Trade-off")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("conformal_regression_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def demo_certificate_reproducibility():
    """Demonstrate certificate reproducibility and caching"""
    print("\n" + "=" * 60)
    print("CERTIFICATE REPRODUCIBILITY DEMO")
    print("=" * 60)

    # Generate small test dataset
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=3, random_state=42
    )

    # Mock predictions (deterministic)
    np.random.seed(42)
    predictions = np.random.rand(len(X), 3)
    predictions = predictions / predictions.sum(axis=1, keepdims=True)

    # Create system
    controller, score_fn = create_conformal_system(
        task_type="classification", alpha=0.1, seed=42
    )

    # Run certification multiple times
    print("Running certification 3 times with identical inputs...")

    certificates = []
    for i in range(3):
        result = controller.certify_predictions(
            X, y, predictions, score_fn, task_type="classification"
        )
        certificates.append(result)
        print(
            f"Run {i+1}: Hash = {result.certificate_hash}, "
            f"Coverage = {result.empirical_coverage:.3f}"
        )

    # Verify reproducibility
    all_same = all(
        cert.certificate_hash == certificates[0].certificate_hash
        for cert in certificates
    )

    print(f"\nAll certificates identical: {all_same}")

    if all_same:
        print("✓ Certificate reproducibility verified!")
    else:
        print("✗ Certificate reproducibility failed!")

    return all_same


def main():
    """Run complete conformal risk control demonstration"""
    print("CONFORMAL RISK CONTROL SYSTEM DEMONSTRATION")
    print("Based on Angelopoulos et al. (2024) 'Conformal Risk Control'")
    print("Distribution-free prediction guarantees with finite-sample validity")

    try:
        # Run demonstrations
        classification_results = demo_classification()
        regression_results = demo_regression()
        reproducible = demo_certificate_reproducibility()

        # Summary
        print("\n" + "=" * 60)
        print("DEMONSTRATION SUMMARY")
        print("=" * 60)
        print("✓ Classification conformal prediction completed")
        print("✓ Regression conformal prediction completed")
        print(f"✓ Certificate reproducibility: {'PASS' if reproducible else 'FAIL'}")

        print("\nKey Features Demonstrated:")
        print("- Distribution-free coverage guarantees")
        print("- Finite-sample validity bounds")
        print("- Adaptive threshold selection")
        print("- Deterministic certification with caching")
        print("- Multiple risk levels and validation")

        return True

    except Exception as e:
        logging.error(f"Demonstration failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
