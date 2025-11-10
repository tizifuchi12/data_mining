import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("="*60)
print("FASE 3: ENTRENAMIENTO DE MODELOS")
print("="*60)

# Cargar datos limpios
data = pd.read_pickle('vehicle_data_clean_no_outliers.pkl')

print(f"\nDatos cargados: {len(data)} registros")

# =============================================================================
# OBJETIVO 1: MODELOS CON VELOCIDAD SOLA (10 MODELOS)
# =============================================================================
print("\n" + "="*60)
print("OBJETIVO 1: PREDICCIÓN CON VELOCIDAD SOLA (10 MODELOS)")
print("="*60)

# Preparar datos: X = velocidad, y = consumo incremental
X_speed_only = data[['speed']].values
y = data['fuel_increment'].values

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_speed_only, y, test_size=0.2, random_state=42
)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ 10 Modelos a probar
models_speed_only = {
    'Linear Regression': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
    'K-Neighbors (k=5)': KNeighborsRegressor(n_neighbors=5),
    'SVM (RBF)': SVR(kernel='rbf', C=1.0, gamma='scale'),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
}

results_speed_only = {}

for name, model in models_speed_only.items():
    print(f"\n🔄 Entrenando {name}...")
    
    # Entrenar (modelos lineales sin scaling, el resto con scaling)
    if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    # Métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results_speed_only[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'model': model
    }
    
    print(f"   RMSE: {rmse:.2f} ml")
    print(f"   MAE:  {mae:.2f} ml")
    print(f"   R²:   {r2:.4f}")

# =============================================================================
# OBJETIVO 1: MODELOS CON VELOCIDAD + ACELERACIÓN (10 MODELOS)
# =============================================================================
print("\n" + "="*60)
print("OBJETIVO 1: PREDICCIÓN CON VELOCIDAD + ACELERACIÓN (10 MODELOS)")
print("="*60)

# Preparar datos: X = [velocidad, aceleración], y = consumo
X_with_accel = data[['speed', 'acceleration']].values
y = data['fuel_increment'].values

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_with_accel, y, test_size=0.2, random_state=42
)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ 10 Modelos con velocidad + aceleración
models_with_accel = {
    'Linear Regression': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
    'K-Neighbors (k=5)': KNeighborsRegressor(n_neighbors=5),
    'SVM (RBF)': SVR(kernel='rbf', C=1.0, gamma='scale'),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
}

results_with_accel = {}

for name, model in models_with_accel.items():
    print(f"\n🔄 Entrenando {name}...")
    
    # Entrenar
    if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    # Métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results_with_accel[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'model': model
    }
    
    print(f"   RMSE: {rmse:.2f} ml")
    print(f"   MAE:  {mae:.2f} ml")
    print(f"   R²:   {r2:.4f}")

# =============================================================================
# OBJETIVO 2: COMPARACIÓN DE RESULTADOS
# =============================================================================
print("\n" + "="*60)
print("OBJETIVO 2: COMPARACIÓN DE 10 MODELOS")
print("="*60)

# Crear tabla comparativa
comparison = pd.DataFrame({
    'Model': list(results_speed_only.keys()),
    'RMSE (Speed Only)': [r['RMSE'] for r in results_speed_only.values()],
    'R² (Speed Only)': [r['R2'] for r in results_speed_only.values()],
    'RMSE (Speed+Accel)': [r['RMSE'] for r in results_with_accel.values()],
    'R² (Speed+Accel)': [r['R2'] for r in results_with_accel.values()]
})

# Ordenar por R² (Speed+Accel) descendente
comparison = comparison.sort_values('R² (Speed+Accel)', ascending=False)

print("\n" + comparison.to_string(index=False))

# =============================================================================
# VISUALIZACIÓN MEJORADA
# =============================================================================

# Figura 1: Comparación R² de ambos enfoques
fig1, ax = plt.subplots(figsize=(14, 7))
x_pos = np.arange(len(comparison))
width = 0.35

bars1 = ax.bar(x_pos - width/2, comparison['R² (Speed Only)'], width, 
               label='Speed Only', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x_pos + width/2, comparison['R² (Speed+Accel)'], width, 
               label='Speed + Accel', alpha=0.8, edgecolor='black')

# Colorear barras según performance
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    r2_speed = comparison.iloc[i]['R² (Speed Only)']
    r2_accel = comparison.iloc[i]['R² (Speed+Accel)']
    
    # Speed Only
    if r2_speed > 0.7:
        bar1.set_color('green')
    elif r2_speed > 0.5:
        bar1.set_color('orange')
    else:
        bar1.set_color('red')
    
    # Speed + Accel
    if r2_accel > 0.7:
        bar2.set_color('darkgreen')
    elif r2_accel > 0.5:
        bar2.set_color('darkorange')
    else:
        bar2.set_color('darkred')

ax.set_xlabel('Modelo', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Comparación de R² - 10 Modelos (Ordenados por Performance)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison['Model'], rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)

# Agregar valores encima de las barras
for i, (idx, row) in enumerate(comparison.iterrows()):
    ax.text(i - width/2, row['R² (Speed Only)'] + 0.01, f"{row['R² (Speed Only)']:.3f}", 
            ha='center', fontsize=8, fontweight='bold')
    ax.text(i + width/2, row['R² (Speed+Accel)'] + 0.01, f"{row['R² (Speed+Accel)']:.3f}", 
            ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('10_models_r2_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Gráfico 1 guardado: '10_models_r2_comparison.png'")
plt.show()

# Figura 2: RMSE Comparison
fig2, ax2 = plt.subplots(figsize=(14, 7))
x_pos = np.arange(len(comparison))

ax2.bar(x_pos - width/2, comparison['RMSE (Speed Only)'], width, 
        label='Speed Only', alpha=0.8, color='steelblue', edgecolor='black')
ax2.bar(x_pos + width/2, comparison['RMSE (Speed+Accel)'], width, 
        label='Speed + Accel', alpha=0.8, color='darkblue', edgecolor='black')

ax2.set_xlabel('Modelo', fontsize=12)
ax2.set_ylabel('RMSE (ml)', fontsize=12)
ax2.set_title('Comparación de RMSE - 10 Modelos', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(comparison['Model'], rotation=45, ha='right', fontsize=10)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('10_models_rmse_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico 2 guardado: '10_models_rmse_comparison.png'")
plt.show()

# Figura 3: Mejora relativa (Speed+Accel vs Speed Only)
fig3, ax3 = plt.subplots(figsize=(14, 7))
improvement = ((comparison['R² (Speed+Accel)'] - comparison['R² (Speed Only)']) / 
               comparison['R² (Speed Only)'].abs() * 100)

bars = ax3.bar(x_pos, improvement, alpha=0.8, edgecolor='black')

# Colorear: verde si mejora, rojo si empeora
for i, bar in enumerate(bars):
    if improvement.iloc[i] > 0:
        bar.set_color('green')
    else:
        bar.set_color('red')

ax3.set_xlabel('Modelo', fontsize=12)
ax3.set_ylabel('Mejora (%)', fontsize=12)
ax3.set_title('Mejora al Agregar Aceleración (% cambio en R²)', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(comparison['Model'], rotation=45, ha='right', fontsize=10)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.grid(True, alpha=0.3, axis='y')

# Agregar valores
for i, val in enumerate(improvement):
    ax3.text(i, val + 1, f"{val:.1f}%", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('10_models_improvement.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico 3 guardado: '10_models_improvement.png'")
plt.show()

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)

print("\n🏆 TOP 3 MODELOS (Speed Only):")
top3_speed = comparison.nlargest(3, 'R² (Speed Only)')[['Model', 'R² (Speed Only)', 'RMSE (Speed Only)']]
for idx, row in top3_speed.iterrows():
    print(f"   {row['Model']:20s}: R² = {row['R² (Speed Only)']:.4f}, RMSE = {row['RMSE (Speed Only)']:.2f} ml")

print("\n🏆 TOP 3 MODELOS (Speed + Accel):")
top3_accel = comparison.nlargest(3, 'R² (Speed+Accel)')[['Model', 'R² (Speed+Accel)', 'RMSE (Speed+Accel)']]
for idx, row in top3_accel.iterrows():
    print(f"   {row['Model']:20s}: R² = {row['R² (Speed+Accel)']:.4f}, RMSE = {row['RMSE (Speed+Accel)']:.2f} ml")

best_model = comparison.iloc[0]
print(f"\n🎯 MEJOR MODELO OVERALL: {best_model['Model']}")
print(f"   R² (Speed Only):  {best_model['R² (Speed Only)']:.4f}")
print(f"   R² (Speed+Accel): {best_model['R² (Speed+Accel)']:.4f}")
print(f"   Mejora: +{((best_model['R² (Speed+Accel)'] - best_model['R² (Speed Only)']) / best_model['R² (Speed Only)'] * 100):.1f}%")

# Guardar tabla de resultados
comparison.to_csv('models_comparison_results.csv', index=False)
print("\n✅ Tabla de resultados guardada: 'models_comparison_results.csv'")