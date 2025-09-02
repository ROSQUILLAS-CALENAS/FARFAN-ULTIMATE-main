# Guía de Integración del Sistema de Puntuación PDT

## Resumen Ejecutivo

El sistema de análisis PDT ha sido actualizado con un **Controlador Autoritativo de Puntuación** (`PDTScoringController`) que centraliza todos los cálculos numéricos según los procedimientos formalizados. Esta integración garantiza consistencia, trazabilidad y precisión en todas las evaluaciones.

## Arquitectura de Integración

### Componentes Principales

1. **PDTScoringController** - Autoridad central de cálculo
2. **AdaptiveScoringEngine** - Motor predictivo integrado
3. **IntelligentRecommendationEngine** - Sistema de recomendaciones contextual
4. **ContinuousAnalyticalCalibration** - Calibración continua
5. **PDTEvaluationSystem** - Orquestador principal

### Flujo de Datos

```
PDT Document → PDTEvaluationSystem → PDTScoringController 
                                          ↓
              AdaptiveScoringEngine ←── Authoritative Scores
                                          ↓
              IntelligentRecommendationEngine ←── Context Data
                                          ↓
              ContinuousAnalyticalCalibration ←── Calibration Feedback
```

## Procedimientos de Cálculo Implementados

### DE1: Lógica de Intervención y Coherencia Interna
```
DE1 = P_{B,c} + P_R + P_G

Donde:
- P_G = ((Q_1 + Q_2) / 2) × 10    [10%]
- P_R = ((Q_3 + Q_4) / 2) × 30    [30%]  
- P_B = ((Q_5 + Q_6) / 2) × 60    [60%]
- P_{B,c} = P_B × FC (Factor Causal)
```

### DE2: Inclusión Temática
```
DE2 = (∑(EMI_sub × w_sub)) × 4

Subdimensiones:
- EMI_{2.1}: Diagnóstico (25%)
- EMI_{2.2}: Alineación Estratégica (30%)
- EMI_{2.3}: Territorialización (25%)
- EMI_{2.4}: Seguimiento (20%)
```

### DE3: Planificación y Adecuación Presupuestal
```
DE3 = ∑φ(respuesta_i)

Donde φ: {Sí=10, Parcial=5, No/NI=0}
Variables: G_1, G_2, A_1, A_2, R_1, R_2, S_1, S_2
```

### DE4: Cadena de Valor
```
DE4 = μ(NE, SC)

Donde:
- NE = Número de eslabones cumplidos (0-8)
- SC = Suficiencia Causal (heredada de DE1)
- μ: {Alto=90, Medio=65, Bajo=35}
```

### Puntaje Agregado (PA)
```
PA = ∑(DE_i × w_i)

Pesos: w = (0.35, 0.25, 0.20, 0.20)
```

## Integración con Engines

### AdaptiveScoringEngine
- **Integración**: Automática en `__init__`
- **Funcionalidad**: Usa cálculos autoritativos cuando hay datos PDT
- **Fallback**: Predicciones adaptativas para análisis exploratorio
- **Método principal**: `_get_authoritative_scores()`

### IntelligentRecommendationEngine  
- **Integración**: Automática en `__init__`
- **Funcionalidad**: Contexto enriquecido con datos autoritativos
- **Análisis**: Detalles de cálculo para recomendaciones precisas
- **Método principal**: `_enrich_context_with_authoritative_data()`

### ContinuousAnalyticalCalibration
- **Integración**: Condicional en `__init__`
- **Funcionalidad**: Calibración basada en procedimientos formales
- **Autoridad**: Validación de parámetros contra reglas oficiales

## API del Controlador Autoritativo

### Cálculo Principal
```python
scoring_controller = PDTScoringController()

result = scoring_controller.calculate_all_scores(
    responses={
        'DE1': {'Q_1': 'Sí', 'Q_2': 'Parcial', ...},
        'DE2': {'D_1': 'Sí', 'D_2': 'No', ...},
        'DE3': {'G_1': 'Sí', 'A_1': 'Parcial', ...},
        'DE4': {'E_1': 'Sí', 'E_2': 'Sí', ...}
    },
    matriz_causal_data={
        'coherencia_diagnostico': 0.8,
        'articulacion_logica': 0.7,
        'consistencia_temporal': 0.6
    }
)
```

### Resultado Estructurado
```python
result = ScoringCalculationResult(
    dimension_scores={'DE1': 78.5, 'DE2': 65.0, 'DE3': 70.0, 'DE4': 90.0},
    aggregated_score=75.175,
    calculation_details={
        'DE1': {
            'final_score': 78.5,
            'components': {'p_g': 7.5, 'p_r': 22.5, 'p_b_c': 48.5},
            'causal_factor': 0.81
        },
        # ... detalles completos
    },
    validation_status={'all_validations_passed': True},
    timestamp=datetime.now()
)
```

### Validación Automática
- **Condición 1**: `|∑w_i - 1| ≤ 0.01`
- **Condición 2**: `DE_i ∈ [0, 100] ∀i`
- **Condición 3**: `PA ∈ [0, 100]`

## Flujo de Integración en el Sistema

### 1. Evaluación de Documento
```python
system = PDTEvaluationSystem()
results = system.evaluate_document(pdf_path, questions)

# El sistema automáticamente:
# - Extrae respuestas PDT de los resultados
# - Ejecuta cálculo autoritativo vía PDTScoringController
# - Integra scores con AdaptiveScoringEngine
# - Enriquece contexto para recomendaciones
# - Calibra parámetros continuamente
```

### 2. Conversión de Formatos
```python
# Para AdaptiveScoringEngine (escala 0-1)
adaptive_scores = scoring_controller.convert_to_adaptive_scoring_format(result)

# Para IntelligentRecommendationEngine 
context = scoring_controller.get_recommendation_context(result)
```

### 3. Extracción de Respuestas
```python
# Desde resultados de evaluación automática
pdt_responses = system._extract_pdt_responses_from_results(results)
matriz_causal = system._extract_matriz_causal_data_from_results(results)
```

## Beneficios de la Integración

### Consistencia
- Todos los cálculos siguen procedimientos formalizados
- Eliminación de discrepancias entre componentes
- Trazabilidad completa de cada puntuación

### Precisión
- Implementación exacta de fórmulas matemáticas
- Validación automática de resultados
- Factor causal integrado en DE1 y heredado en DE4

### Escalabilidad  
- Arquitectura modular y extensible
- Integración transparente con sistema existente
- Compatibilidad con procesamiento secuencial

### Calibración Inteligente
- Feedback basado en cálculos autoritativos
- Recomendaciones contextualizadas con detalles de procedimiento
- Calibración continua respetando reglas formales

## Configuración y Uso

### Instalación
El controlador se integra automáticamente al inicializar `PDTEvaluationSystem`. No requiere configuración adicional.

### Verificación de Estado
```python
status = system.get_system_status()
assert status['components_status']['scoring_controller'] == 'active'
```

### Logs de Integración
```
INFO - PDT Evaluation System initialized with authoritative scoring controller
INFO - AdaptiveScoringEngine integrated with PDTScoringController
INFO - IntelligentRecommendationEngine integrated with PDTScoringController
INFO - ContinuousAnalyticalCalibration integrated with PDTScoringController
```

## Mantenimiento y Evolución

### Actualización de Procedimientos
Para modificar procedimientos de cálculo, actualizar únicamente `PDTScoringController`. Los engines se adaptan automáticamente.

### Validación de Integridad
El sistema incluye validación automática en cada cálculo. Cualquier desviación de los procedimientos formales genera alertas.

### Historial de Calibración
Todas las calibraciones se almacenan con referencia al controlador autoritativo, manteniendo trazabilidad completa.

---

Este sistema garantiza que todas las puntuaciones PDT fluyan a través de la autoridad de cálculo centralizada, manteniendo coherencia matemática y procedural en todo el pipeline analítico.
