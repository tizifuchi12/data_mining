# Predicción de Consumo de Combustible usando Machine Learning

## 1. Introducción
- Contexto: Munic.io, 600,000 vehículos
- Problema: Predecir consumo usando solo velocidad
- Objetivo: Reducir uso de ancho de banda

## 2. Metodología

### 2.1 Preprocesamiento
- Datos: 148 vehículos, probado con vehículo ID 863609060548678
- Resampling: 30 segundos (óptimo según análisis)
- Features: Velocidad GPS/OBD + Aceleración calculada
- Limpieza: Interpolación de gaps <5s, eliminación de outliers

### 2.2 Modelos evaluados
1. Regresión Lineal (baseline)
2. Random Forest (100 árboles)
3. Gradient Boosting
4. Neural Network (capas: 50-25)

## 3. Resultados

### 3.1 Comparación de modelos
[INSERTAR: model_comparison_final.png]

**Tabla de resultados:**
| Modelo | R² (Speed) | R² (Speed+Accel) | Mejora |
|--------|------------|------------------|--------|
| Linear Regression | 0.684 | 0.707 | +3.4% |
| Random Forest | 0.713 | 0.764 | +7.1% |
| Gradient Boosting | 0.719 | 0.764 | +6.3% |
| **Neural Network** | **0.725** | **0.765** | **+5.5%** |

### 3.2 Importancia de features
[INSERTAR: feature_importance.png]

- Velocidad: 58% de importancia
- Aceleración: 42% de importancia
- **Conclusión**: Ambas variables son relevantes

### 3.3 Performance del mejor modelo
[INSERTAR: final_model_performance.png]

- R² = 0.765 (explica 76.5% de la varianza)
- RMSE = 25.45 ml/30s (0.85 ml/s)
- Error medio: prácticamente 0 (no sesgado)

### 3.4 Predicción temporal
[INSERTAR: timeline_prediction.png]

El modelo captura bien las variaciones de consumo en el tiempo.

## 4. Análisis de Frecuencias

[INSERTAR: frequency_sensitivity.png]

| Frecuencia | R² | RMSE | Reducción datos |
|------------|-----|------|-----------------|
| 2s | 0.648 | 0.96 ml/s | 0% (baseline) |
| 30s | **0.764** | **0.85 ml/s** | **44%** |
| 60s | 0.776 | 0.84 ml/s | 68% |

**Conclusión clave**: 
- Frecuencia de 30s es óptima
- Reduce 44% de transmisión de datos
- Mantiene excelente precisión (R²=0.764)

## 5. Conclusiones

✅ **Modelo óptimo**: Neural Network con velocidad + aceleración
✅ **Frecuencia óptima**: 30 segundos
✅ **Impacto**: Reducción de 44% en uso de ancho de banda
✅ **Precisión**: R² = 0.765 (aceptable para aplicación real)

### Recomendaciones
1. Implementar modelo con muestreo cada 30s
2. Considerar agregar: temperatura exterior, carga del vehículo, pendiente
3. Validar con múltiples vehículos (próximo paso)

## 6. Trabajo Futuro
- Validar modelo con los 148 vehículos
- Analizar diferencias por tipo de vehículo
- Probar con datos de RPM si están disponibles


RESULTADOS VALIDACIÓN MULTI-VEHÍCULO:

✓ Single-vehicle model: R² = -1.56 (no generaliza)
✓ Multi-vehicle model: R² = 0.34 (generaliza mejor)

CONCLUSIÓN PRINCIPAL:
El consumo de combustible depende fuertemente de características
del vehículo no disponibles en los datos (tipo de motor, peso, etc.).

RECOMENDACIÓN PRÁCTICA:
Para aplicación real en Munic.io:
1. Modelo general entrenado con múltiples vehículos (R² = 0.34)
2. Fine-tuning personalizado usando primeras 500-1000 lecturas del vehículo
3. Resultado esperado: R² > 0.6 por vehículo después de calibración