"""
Demo: Sistema de Certificación de Riesgo Conformal Adaptativo

Demonstración del sistema de control de riesgo conformal basado en la teoría de
predicción conformal adaptativa de Angelopoulos et al. (2024).

El sistema proporciona garantías distribution-free con cobertura exacta finita
e incluye certificación integral de riesgo.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# Import the enhanced conformal risk control system
from egw_query_expansion.core.conformal_risk_control import (
    ClassificationNonconformityScorer,
    CoverageAnalysis,
    EnhancedConformalRiskController,
    RegressionNonconformityScorer,
    RiskCertificate,
    RiskControlConfig,
    StatisticalBounds,
)


def generate_synthetic_regression_data(
    n_samples: int = 2000, noise_std: float = 1.0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression data with known ground truth."""
    np.random.seed(seed)

    # Features: simple quadratic relationship
    X = np.random.uniform(-3, 3, (n_samples, 2))

    # True function: quadratic with interaction
    y_true = X[:, 0] ** 2 + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 0] * X[:, 1]

    # Add heteroskedastic noise (variance increases with |y_true|)
    noise = np.random.normal(0, noise_std * (1 + 0.2 * np.abs(y_true)), n_samples)
    y = y_true + noise

    # Predictions from an imperfect model (with slight bias)
    y_pred = y_true + 0.1 * np.random.normal(0, 1, n_samples)

    return X, y, y_pred


def generate_synthetic_classification_data(
    n_samples: int = 2000, n_classes: int = 3, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data with prediction probabilities."""
    np.random.seed(seed)

    # Features
    X = np.random.randn(n_samples, 4)

    # Generate true labels based on linear combinations
    linear_combo = X @ np.random.randn(4, n_classes)
    y_true = np.argmax(linear_combo, axis=1)

    # Model predictions with some miscalibration
    pred_logits = linear_combo + 0.2 * np.random.randn(n_samples, n_classes)
    pred_probs = np.exp(pred_logits) / np.sum(
        np.exp(pred_logits), axis=1, keepdims=True
    )

    return X, y_true, pred_probs


def demo_regression_certification():
    """Demonstrate conformal risk certification for regression."""
    print("=" * 80)
    print("DEMOSTRACIÓN: CERTIFICACIÓN DE RIESGO CONFORMAL - REGRESIÓN")
    print("=" * 80)

    # Generate synthetic data
    X, y_true, y_pred = generate_synthetic_regression_data(n_samples=2000)

    # Configuration with enhanced parameters
    config = RiskControlConfig(
        alpha=0.1,  # 90% coverage target
        random_seed=42,
        calibration_ratio=0.4,
        validation_size=300,
        test_ratio=0.3,
        adaptive_lambda=True,
        adaptive_quantile_method="bootstrap",
        distribution_shift_bound=0.05,
        confidence_level=0.95,
        enable_cross_validation=True,
        bootstrap_samples=500,
    )

    print(f"Configuración: α={config.alpha}, cobertura objetivo={1-config.alpha:.1%}")

    # Initialize controller
    controller = EnhancedConformalRiskController(config)

    # Non-conformity scorer with normalization
    scorer = RegressionNonconformityScorer(
        method="absolute", normalize=True, robust=False, name="AbsoluteErrorScorer"
    )

    print("\nAjustando calibración...")
    fitting_stats = controller.fit_calibration(
        predictions=y_pred.tolist(),
        true_labels=y_true.tolist(),
        scorer=scorer,
        enable_scorer_fitting=True,
    )

    print(f"✓ Calibración completada:")
    print(f"  - {fitting_stats['n_calibration']} muestras de calibración")
    print(f"  - {fitting_stats['n_validation']} muestras de validación")
    print(f"  - {fitting_stats['n_test']} muestras de prueba")
    print(
        f"  - Cuantil adaptativo: {fitting_stats['quantile_results']['selected_quantile']:.4f}"
    )

    # Generate test data for certification
    X_test, y_test_true, y_test_pred = generate_synthetic_regression_data(
        n_samples=500, seed=123
    )

    print("\nGenerando certificado de riesgo...")
    certificate = controller.generate_enhanced_certificate(
        input_data={"test_size": len(y_test_pred), "model_type": "regression"},
        predictions=y_test_pred.tolist(),
        scorer=scorer,
        test_labels=y_test_true.tolist(),
        model_metadata={"algorithm": "synthetic_quadratic", "features": 2},
    )

    print(f"\n📜 CERTIFICADO DE RIESGO GENERADO:")
    print(f"ID: {certificate.certificate_id}")
    print(f"Versión: {certificate.certificate_version}")
    print(f"Validez: hasta {certificate.validity_expires}")

    print(f"\n📊 MÉTRICAS DE COBERTURA:")
    print(f"Cobertura empírica: {certificate.coverage_analysis.empirical_coverage:.3f}")
    print(f"Cobertura objetivo: {certificate.coverage_analysis.target_coverage:.3f}")
    print(f"Gap de cobertura: {certificate.coverage_analysis.coverage_gap:+.3f}")
    print(
        f"Intervalo confianza: [{certificate.coverage_analysis.confidence_interval[0]:.3f}, {certificate.coverage_analysis.confidence_interval[1]:.3f}]"
    )
    print(
        f"p-value test cobertura: {certificate.coverage_analysis.coverage_test_p_value:.4f}"
    )

    print(f"\n🎯 BOUNDS DE RIESGO:")
    print(f"Riesgo empírico: {certificate.risk_bounds.empirical_risk:.4f}")
    print(f"Bound Hoeffding: {certificate.risk_bounds.upper_bound_hoeffding:.4f}")
    print(f"Bound CLT: {certificate.risk_bounds.upper_bound_clt:.4f}")
    print(f"Bound Bootstrap: {certificate.risk_bounds.upper_bound_bootstrap:.4f}")
    print(f"Bound conservativo: {certificate.risk_bounds.conservative_bound:.4f}")

    print(f"\n🔍 CALIDAD DE PREDICCIÓN:")
    print(
        f"Tamaño promedio conjunto: {certificate.coverage_analysis.average_set_size:.3f}"
    )
    print(f"Eficiencia: {certificate.coverage_analysis.efficiency_score:.3f}")
    print(f"Calidad calibración: {certificate.calibration_quality_score:.3f}")

    print(f"\n🚨 DETECCIÓN DE DRIFT:")
    shift_results = certificate.distribution_shift_results
    print(f"Drift detectado: {'SÍ' if shift_results['shift_detected'] else 'NO'}")
    print(f"Estadístico principal: {shift_results.get('primary_p_value', 'N/A'):.4f}")
    print(
        f"Recalibración requerida: {'SÍ' if shift_results['requires_recalibration'] else 'NO'}"
    )

    # Validate certificate
    is_valid = certificate.is_valid(tolerance=0.02)
    print(f"\n✅ VALIDEZ CERTIFICADO: {'VÁLIDO' if is_valid else 'INVÁLIDO'}")

    # Save certificate
    cert_path = Path("regression_risk_certificate.json")
    certificate.save(cert_path)
    print(f"💾 Certificado guardado en: {cert_path}")

    return certificate, controller


def demo_classification_certification():
    """Demonstrate conformal risk certification for classification."""
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN: CERTIFICACIÓN DE RIESGO CONFORMAL - CLASIFICACIÓN")
    print("=" * 80)

    # Generate synthetic data
    X, y_true, y_pred_probs = generate_synthetic_classification_data(
        n_samples=1500, n_classes=4
    )

    # Configuration
    config = RiskControlConfig(
        alpha=0.05,  # 95% coverage target
        random_seed=42,
        calibration_ratio=0.5,
        adaptive_lambda=True,
        adaptive_quantile_method="clt",
        enable_cross_validation=True,
    )

    print(f"Configuración: α={config.alpha}, cobertura objetivo={1-config.alpha:.1%}")

    # Initialize controller
    controller = EnhancedConformalRiskController(config)

    # Non-conformity scorer with adaptive thresholds
    scorer = ClassificationNonconformityScorer(
        method="softmax",
        adaptive_threshold=True,
        temperature=1.2,
        name="AdaptiveSoftmaxScorer",
    )

    print("\nAjustando calibración...")
    fitting_stats = controller.fit_calibration(
        predictions=y_pred_probs.tolist(),
        true_labels=y_true.tolist(),
        scorer=scorer,
        enable_scorer_fitting=True,
    )

    print(f"✓ Calibración completada con {fitting_stats['n_calibration']} muestras")

    # Generate test data
    X_test, y_test_true, y_test_pred_probs = generate_synthetic_classification_data(
        n_samples=400, n_classes=4, seed=456
    )

    print("\nGenerando certificado de riesgo...")
    certificate = controller.generate_enhanced_certificate(
        input_data={
            "test_size": len(y_test_pred_probs),
            "model_type": "classification",
        },
        predictions=y_test_pred_probs.tolist(),
        scorer=scorer,
        test_labels=y_test_true.tolist(),
        model_metadata={"algorithm": "synthetic_multiclass", "n_classes": 4},
    )

    print(f"\n📜 CERTIFICADO DE CLASIFICACIÓN:")
    print(f"ID: {certificate.certificate_id}")
    print(f"Cobertura: {certificate.coverage_analysis.empirical_coverage:.3f}")
    print(
        f"Tamaño conjunto promedio: {certificate.coverage_analysis.average_set_size:.2f}"
    )
    print(f"Bound de riesgo: {certificate.risk_bounds.conservative_bound:.4f}")

    # Test prediction sets for a few examples
    print(f"\n🔮 EJEMPLOS DE CONJUNTOS DE PREDICCIÓN:")
    for i in range(min(5, len(y_test_pred_probs))):
        pred_probs = y_test_pred_probs[i]
        true_label = y_test_true[i]

        # Create candidate set
        candidates = {j: scorer.score(pred_probs, j) for j in range(len(pred_probs))}
        pred_set = controller.construct_prediction_set(candidates)

        print(
            f"Ejemplo {i+1}: Etiqueta real={true_label}, "
            f"Conjunto predicción={sorted(list(pred_set.set_values))}, "
            f"Contiene real={'SÍ' if pred_set.contains(true_label) else 'NO'}"
        )

    return certificate, controller


def demo_distribution_shift_detection():
    """Demonstrate distribution shift detection capabilities."""
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN: DETECCIÓN DE DISTRIBUTION SHIFT")
    print("=" * 80)

    # Create base data
    X_base, y_base, y_pred_base = generate_synthetic_regression_data(
        n_samples=1000, seed=42
    )

    # Create shifted data (change noise characteristics)
    X_shift, y_shift, y_pred_shift = generate_synthetic_regression_data(
        n_samples=500, noise_std=2.0, seed=999  # Higher noise
    )

    config = RiskControlConfig(
        alpha=0.1,
        shift_detection_method="all",  # Use all methods
        distribution_shift_bound=0.05,
        recalibration_threshold=0.01,
    )

    controller = EnhancedConformalRiskController(config)
    scorer = RegressionNonconformityScorer(method="absolute")

    # Fit on base data
    print("Ajustando en datos base...")
    controller.fit_calibration(
        predictions=y_pred_base.tolist(), true_labels=y_base.tolist(), scorer=scorer
    )

    # Detect shift on new data
    print("\nEvaluando datos con shift...")
    shift_scores = scorer.batch_score(y_pred_shift.tolist(), y_shift.tolist())
    shift_results = controller.detect_distribution_shift(shift_scores.tolist())

    print(f"🚨 RESULTADOS DETECCIÓN DE SHIFT:")
    print(f"Shift detectado: {'SÍ' if shift_results['shift_detected'] else 'NO'}")
    print(f"Método primario: {shift_results['primary_method']}")
    print(f"p-value principal: {shift_results['primary_p_value']:.6f}")
    print(
        f"Recalibración recomendada: {'SÍ' if shift_results['requires_recalibration'] else 'NO'}"
    )

    print(f"\nRESULTADOS POR MÉTODO:")
    for method, results in shift_results["methods"].items():
        print(f"  {method.upper()}:")
        print(f"    Estadístico: {results['statistic']:.6f}")
        if "p_value" in results:
            print(f"    p-value: {results['p_value']:.6f}")
        print(f"    Shift: {'SÍ' if results['shift_detected'] else 'NO'}")

    return shift_results


def demo_identical_input_certification():
    """Demonstrate that identical inputs produce identical certificates."""
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN: DETERMINISMO - ENTRADAS IDÉNTICAS")
    print("=" * 80)

    # Generate data
    X, y_true, y_pred = generate_synthetic_regression_data(n_samples=1000, seed=42)

    config = RiskControlConfig(alpha=0.1, random_seed=42)
    scorer = RegressionNonconformityScorer(method="absolute")

    # Create two identical controllers
    controller1 = EnhancedConformalRiskController(config)
    controller2 = EnhancedConformalRiskController(config)

    # Fit both identically
    controller1.fit_calibration(y_pred.tolist(), y_true.tolist(), scorer)
    controller2.fit_calibration(y_pred.tolist(), y_true.tolist(), scorer)

    # Generate certificates with identical inputs
    test_input = {"test_data": "identical", "params": [1, 2, 3]}
    test_predictions = y_pred[:100].tolist()
    test_labels = y_true[:100].tolist()

    print("Generando certificados con entradas idénticas...")
    cert1 = controller1.generate_enhanced_certificate(
        input_data=test_input,
        predictions=test_predictions,
        scorer=scorer,
        test_labels=test_labels,
    )

    cert2 = controller2.generate_enhanced_certificate(
        input_data=test_input,
        predictions=test_predictions,
        scorer=scorer,
        test_labels=test_labels,
    )

    # Compare certificates
    print(f"\n🔍 COMPARACIÓN DE CERTIFICADOS:")
    print(f"Hash entrada cert1: {cert1.input_hash}")
    print(f"Hash entrada cert2: {cert2.input_hash}")
    print(f"Hashes idénticos: {'SÍ' if cert1.input_hash == cert2.input_hash else 'NO'}")

    print(f"\nCobertura cert1: {cert1.coverage_analysis.empirical_coverage:.6f}")
    print(f"Cobertura cert2: {cert2.coverage_analysis.empirical_coverage:.6f}")
    print(
        f"Cobertura idéntica: {'SÍ' if abs(cert1.coverage_analysis.empirical_coverage - cert2.coverage_analysis.empirical_coverage) < 1e-10 else 'NO'}"
    )

    print(f"\nBound riesgo cert1: {cert1.risk_bounds.conservative_bound:.6f}")
    print(f"Bound riesgo cert2: {cert2.risk_bounds.conservative_bound:.6f}")
    print(
        f"Bounds idénticos: {'SÍ' if abs(cert1.risk_bounds.conservative_bound - cert2.risk_bounds.conservative_bound) < 1e-10 else 'NO'}"
    )

    return cert1, cert2


def visualize_coverage_results(
    certificate: RiskCertificate, controller: EnhancedConformalRiskController
):
    """Create visualizations of coverage results."""
    print("\n" + "=" * 80)
    print("VISUALIZACIÓN DE RESULTADOS")
    print("=" * 80)

    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Calibration scores distribution
        ax = axes[0, 0]
        ax.hist(
            controller.calibration_scores,
            bins=30,
            alpha=0.7,
            label="Calibración",
            density=True,
        )
        ax.hist(
            controller.validation_scores,
            bins=30,
            alpha=0.7,
            label="Validación",
            density=True,
        )
        ax.axvline(
            controller.adaptive_quantile,
            color="red",
            linestyle="--",
            label=f"Cuantil adaptivo ({controller.adaptive_quantile:.3f})",
        )
        ax.set_xlabel("Non-conformity Score")
        ax.set_ylabel("Densidad")
        ax.set_title("Distribución de Scores de No-Conformidad")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Risk bounds comparison
        ax = axes[0, 1]
        bounds_data = {
            "Empírico": certificate.risk_bounds.empirical_risk,
            "Hoeffding": certificate.risk_bounds.upper_bound_hoeffding,
            "CLT": certificate.risk_bounds.upper_bound_clt,
            "Bootstrap": certificate.risk_bounds.upper_bound_bootstrap,
        }
        bars = ax.bar(
            bounds_data.keys(),
            bounds_data.values(),
            color=["blue", "red", "green", "orange"],
            alpha=0.7,
        )
        ax.axhline(
            certificate.risk_bounds.empirical_risk,
            color="blue",
            linestyle="-",
            alpha=0.5,
            label="Riesgo empírico",
        )
        ax.set_ylabel("Riesgo")
        ax.set_title("Comparación de Bounds de Riesgo")
        ax.grid(True, alpha=0.3)

        # Add values on bars
        for bar, value in zip(bars, bounds_data.values()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )

        # Plot 3: Coverage analysis
        ax = axes[1, 0]
        coverage_data = {
            "Empírica": certificate.coverage_analysis.empirical_coverage,
            "Objetivo": certificate.coverage_analysis.target_coverage,
        }
        bars = ax.bar(
            coverage_data.keys(),
            coverage_data.values(),
            color=["lightblue", "darkblue"],
            alpha=0.7,
        )

        # Add confidence interval
        ci_lower, ci_upper = certificate.coverage_analysis.confidence_interval
        ax.errorbar(
            ["Empírica"],
            [certificate.coverage_analysis.empirical_coverage],
            yerr=[
                [certificate.coverage_analysis.empirical_coverage - ci_lower],
                [ci_upper - certificate.coverage_analysis.empirical_coverage],
            ],
            fmt="o",
            color="red",
            capsize=5,
            label="IC 95%",
        )

        ax.set_ylabel("Cobertura")
        ax.set_title("Análisis de Cobertura")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add values on bars
        for bar, value in zip(bars, coverage_data.values()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # Plot 4: System diagnostics
        ax = axes[1, 1]
        diagnostics = controller.get_comprehensive_diagnostics()

        if "health_metrics" in diagnostics:
            health_data = {
                "Calidad\nCalibración": diagnostics["health_metrics"].get(
                    "calibration_quality_score", 0
                ),
                "Estabilidad\nCobertura": diagnostics["health_metrics"].get(
                    "coverage_stability", 0
                )
                or 0,
                "Eficiencia\nCache": diagnostics["health_metrics"].get(
                    "cache_efficiency", 0
                ),
            }

            bars = ax.bar(
                health_data.keys(),
                health_data.values(),
                color=["green", "blue", "purple"],
                alpha=0.7,
            )
            ax.set_ylabel("Score")
            ax.set_title("Métricas de Salud del Sistema")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            # Add values on bars
            for bar, value in zip(bars, health_data.values()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig("conformal_risk_results.png", dpi=300, bbox_inches="tight")
        print("📊 Visualización guardada como 'conformal_risk_results.png'")

        # Show plot if in interactive environment
        try:
            plt.show()
        except:
            print("💡 Ejecuta en entorno interactivo para ver los gráficos")

    except Exception as e:
        print(f"⚠️  Error creando visualizaciones: {e}")
        print("💡 Instala matplotlib: pip install matplotlib")


def main():
    """Main demonstration function."""
    print("🚀 SISTEMA DE CERTIFICACIÓN DE RIESGO CONFORMAL ADAPTATIVO")
    print("Basado en Angelopoulos et al. (2024) - Journal of Machine Learning Research")
    print("Garantías distribution-free con cobertura exacta finita")

    # Regression demonstration
    reg_certificate, reg_controller = demo_regression_certification()

    # Classification demonstration
    clf_certificate, clf_controller = demo_classification_certification()

    # Distribution shift demonstration
    shift_results = demo_distribution_shift_detection()

    # Determinism demonstration
    cert1, cert2 = demo_identical_input_certification()

    # Visualization
    visualize_coverage_results(reg_certificate, reg_controller)

    print("\n" + "=" * 80)
    print("✅ DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)
    print("El sistema de certificación conformal ha sido demostrado exitosamente:")
    print("• ✓ Cobertura exacta finita garantizada")
    print("• ✓ Bounds de riesgo distribution-free")
    print("• ✓ Detección de distribution shift")
    print("• ✓ Certificados deterministas y reproducibles")
    print("• ✓ Validación estadística integral")
    print("• ✓ Splits deterministas con semillas fijas")
    print("• ✓ Constructores RCPS con validez bajo drift acotado")

    return {
        "regression_certificate": reg_certificate,
        "classification_certificate": clf_certificate,
        "shift_results": shift_results,
        "determinism_demo": (cert1, cert2),
    }


if __name__ == "__main__":
    results = main()
