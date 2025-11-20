"""
MODELO FINAL OPTIMIZADO
=======================
Configuración óptima encontrada:
- Algoritmo: Gradient Boosting
- Features: speed + acceleration (2 features)
- Frecuencia: 120 segundos (2 minutos)
- Performance: R²=0.522, RMSE=0.420 ml/s
- Reducción bandwidth: 99.2%
"""

import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ✅ CONFIGURACIÓN ÓPTIMA (BASADA EN ANÁLISIS DE FRECUENCIA)
# =============================================================================
OPTIMAL_FREQ = '120s'        # ✅ Mejor frecuencia encontrada
RANDOM_SEED = 42
NUM_TRAIN_VEHICLES = 120
MAX_TRAINING_SAMPLES = 1500000

print("="*80)
print("ENTRENAMIENTO DEL MODELO FINAL OPTIMIZADO")
print("="*80)
print(f"""
🎯 CONFIGURACIÓN ÓPTIMA:
   • Algoritmo: Gradient Boosting (200 árboles, depth=7)
   • Features: speed + acceleration (2 features)
   • Frecuencia: {OPTIMAL_FREQ} (2 minutos)
   • Reducción bandwidth: 99.2% (vs 1 segundo)
   
📊 PERFORMANCE ESPERADA:
   • R² medio: ~0.52
   • R² mediano: ~0.61
   • RMSE: ~0.42 ml/s
   • Vehículos con R²>0.5: ~70%
""")

# =============================================================================
# FUNCIÓN DE PROCESAMIENTO CON FRECUENCIA ÓPTIMA
# =============================================================================
def process_vehicle_optimal(file_path):
    """Procesa vehículo con frecuencia óptima (120s)"""
    try:
        data = pd.read_csv(file_path)
        data['time'] = pd.to_datetime(data['time'], errors='coerce')
        data['time'] = data['time'].dt.floor('s')
        data = data.dropna(subset=['time'])
        
        if len(data) == 0:
            return None
        
        signals = {
            'gps_speed': 'TRACKS.MUNIC.GPS_SPEED (km/h)',
            'obd_speed': 'TRACKS.MUNIC.MDI_OBD_SPEED (km/h)',
            'fuel_consumed': 'TRACKS.MUNIC.MDI_OBD_FUEL (ml)'
        }
        
        dfs = {}
        for signal_name, column_name in signals.items():
            if column_name in data.columns:
                df_signal = data[['time', column_name]].copy()
                df_signal = df_signal.dropna(subset=[column_name])
                df_signal = df_signal.rename(columns={column_name: signal_name})
                df_signal = df_signal.set_index('time')
                dfs[signal_name] = df_signal
        
        if 'fuel_consumed' not in dfs:
            return None
        if 'gps_speed' not in dfs and 'obd_speed' not in dfs:
            return None
        
        # Resamplear a 120s
        resampled = {}
        for signal_name, df in dfs.items():
            resampled[signal_name] = df.resample(OPTIMAL_FREQ).mean()
        
        data_aligned = pd.concat(resampled.values(), axis=1)
        data_aligned.columns = resampled.keys()
        
        # Combinar velocidades
        if 'gps_speed' in data_aligned.columns and 'obd_speed' in data_aligned.columns:
            data_aligned['speed'] = data_aligned['gps_speed'].fillna(data_aligned['obd_speed'])
        elif 'gps_speed' in data_aligned.columns:
            data_aligned['speed'] = data_aligned['gps_speed']
        elif 'obd_speed' in data_aligned.columns:
            data_aligned['speed'] = data_aligned['obd_speed']
        else:
            return None
        
        # Interpolación simple
        for col in ['speed', 'fuel_consumed']:
            if col in data_aligned.columns:
                data_aligned[col] = data_aligned[col].fillna(method='ffill', limit=3)
                data_aligned[col] = data_aligned[col].fillna(method='bfill', limit=3)
        
        data_freq = data_aligned.dropna(subset=['speed', 'fuel_consumed']).copy()
        
        if len(data_freq) < 50:
            return None
        
        # Features
        freq_seconds = 120
        data_freq['acceleration'] = data_freq['speed'].diff()
        data_freq['fuel_increment'] = data_freq['fuel_consumed'].diff()
        data_freq.loc[data_freq['fuel_increment'] < 0, 'fuel_increment'] = np.nan
        data_freq['fuel_rate'] = data_freq['fuel_increment'] / freq_seconds
        
        data_freq = data_freq.replace([np.inf, -np.inf], np.nan)
        data_freq = data_freq.dropna()
        
        if len(data_freq) < 50:
            return None
        
        # Eliminar outliers
        Q1 = data_freq['fuel_rate'].quantile(0.25)
        Q3 = data_freq['fuel_rate'].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return None
        
        data_freq = data_freq[(data_freq['fuel_rate'] >= Q1 - 1.5*IQR) & 
                              (data_freq['fuel_rate'] <= Q3 + 1.5*IQR)].copy()
        
        if len(data_freq) < 100:
            return None
        
        return data_freq[['speed', 'acceleration', 'fuel_rate']].copy()
        
    except Exception as e:
        return None

# =============================================================================
# CARGAR Y DIVIDIR DATOS
# =============================================================================
print("\n" + "="*80)
print("CARGANDO DATOS")
print("="*80)

all_files = glob.glob("fuel_data/*.csv")
np.random.seed(RANDOM_SEED)
np.random.shuffle(all_files)

train_files = all_files[:NUM_TRAIN_VEHICLES]
test_files = all_files[NUM_TRAIN_VEHICLES:]

print(f"\n📂 División:")
print(f"   • Archivos train: {len(train_files)}")
print(f"   • Archivos test: {len(test_files)}")

# Procesar train
print(f"\n🔄 Procesando {len(train_files)} vehículos de entrenamiento...")
train_data_list = []
for i, file in enumerate(train_files):
    if i % 20 == 0:
        print(f"   Progreso: {i}/{len(train_files)}")
    d = process_vehicle_optimal(file)
    if d is not None:
        train_data_list.append(d)

print(f"\n✅ Vehículos procesados: {len(train_data_list)}/{len(train_files)}")

combined_train = pd.concat(train_data_list, ignore_index=True)
print(f"✅ Total registros: {len(combined_train):,}")

# Submuestreo si necesario
if len(combined_train) > MAX_TRAINING_SAMPLES:
    print(f"   Submuestreando a {MAX_TRAINING_SAMPLES:,}...")
    combined_train = combined_train.sample(n=MAX_TRAINING_SAMPLES, random_state=RANDOM_SEED)

combined_train = combined_train.replace([np.inf, -np.inf], np.nan)
combined_train = combined_train.dropna()
print(f"✅ Registros finales: {len(combined_train):,}")

# Preparar datos
feature_cols = ['speed', 'acceleration']
X_train = combined_train[feature_cols].values
y_train = combined_train['fuel_rate'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# =============================================================================
# ENTRENAR MODELO ÓPTIMO
# =============================================================================
print("\n" + "="*80)
print("ENTRENANDO MODELO GRADIENT BOOSTING")
print("="*80)

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    random_state=RANDOM_SEED,
    verbose=1
)

print("\n🤖 Entrenando Gradient Boosting...")
model.fit(X_train_scaled, y_train)
print("✅ Modelo entrenado")

# =============================================================================
# EVALUAR EN TEST
# =============================================================================
print("\n" + "="*80)
print(f"EVALUANDO EN {len(test_files)} VEHÍCULOS DE TEST")
print("="*80)

results = []

for i, file in enumerate(test_files):
    if i % 5 == 0:
        print(f"\nProcesando: {i}/{len(test_files)}")
    
    vehicle_id = file.split('\\')[-1].replace('.csv', '')
    test_data = process_vehicle_optimal(file)
    
    if test_data is None:
        continue
    
    X_test = test_data[feature_cols].values
    y_test = test_data['fuel_rate'].values
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        'Vehicle_ID': vehicle_id,
        'R²': r2,
        'RMSE': rmse,
        'N_samples': len(y_test)
    })

df_results = pd.DataFrame(results)

# =============================================================================
# RESULTADOS
# =============================================================================
print("\n" + "="*80)
print("RESULTADOS FINALES")
print("="*80)

r2_mean = df_results['R²'].mean()
r2_median = df_results['R²'].median()
r2_std = df_results['R²'].std()
rmse_mean = df_results['RMSE'].mean()
r2_above_05 = (df_results['R²'] > 0.5).sum()
r2_03_05 = ((df_results['R²'] >= 0.3) & (df_results['R²'] <= 0.5)).sum()
r2_below_03 = (df_results['R²'] < 0.3).sum()

print(f"""
📊 MÉTRICAS GLOBALES:
   • R² medio: {r2_mean:.3f} ± {r2_std:.3f}
   • R² mediano: {r2_median:.3f}
   • RMSE medio: {rmse_mean:.3f} ml/s
   • Vehículos evaluados: {len(df_results)}

📈 DISTRIBUCIÓN:
   • R² > 0.5 (bueno):     {r2_above_05}/{len(df_results)} ({r2_above_05/len(df_results)*100:.1f}%)
   • R² 0.3-0.5 (regular): {r2_03_05}/{len(df_results)} ({r2_03_05/len(df_results)*100:.1f}%)
   • R² < 0.3 (malo):      {r2_below_03}/{len(df_results)} ({r2_below_03/len(df_results)*100:.1f}%)
""")

# Guardar resultados
df_results.to_csv('optimal_model_results.csv', index=False)
print("✅ Resultados guardados: 'optimal_model_results.csv'")

# =============================================================================
# GUARDAR MODELO ENTRENADO
# =============================================================================
print("\n" + "="*80)
print("GUARDANDO MODELO")
print("="*80)

# Guardar modelo
with open('optimal_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Modelo guardado: 'optimal_model.pkl'")

# Guardar scaler
with open('optimal_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler guardado: 'optimal_scaler.pkl'")

# Guardar configuración
config = {
    'algorithm': 'Gradient Boosting',
    'features': feature_cols,
    'frequency': OPTIMAL_FREQ,
    'n_estimators': 200,
    'max_depth': 7,
    'r2_mean': r2_mean,
    'r2_median': r2_median,
    'rmse_mean': rmse_mean,
    'bandwidth_reduction': 99.2
}

with open('optimal_config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("✅ Configuración guardada: 'optimal_config.pkl'")

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Distribución de R²
ax1 = axes[0, 0]
ax1.hist(df_results['R²'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(r2_mean, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {r2_mean:.3f}')
ax1.axvline(r2_median, color='green', linestyle='--', linewidth=2, label=f'Médiane: {r2_median:.3f}')
ax1.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_ylabel('Nombre de véhicules', fontsize=12, fontweight='bold')
ax1.set_title('Distribution des Scores R²', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. R² por vehículo (ordenado)
ax2 = axes[0, 1]
df_sorted = df_results.sort_values('R²', ascending=False).reset_index(drop=True)
colors = ['green' if r2 > 0.5 else 'orange' if r2 > 0.3 else 'red' for r2 in df_sorted['R²']]
ax2.bar(range(len(df_sorted)), df_sorted['R²'], color=colors, edgecolor='black', alpha=0.7)
ax2.axhline(0.5, color='green', linestyle=':', alpha=0.5, label='Seuil bon (0.5)')
ax2.axhline(0.3, color='orange', linestyle=':', alpha=0.5, label='Seuil acceptable (0.3)')
ax2.set_xlabel('Véhicule (trié)', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('Performance par Véhicule', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. RMSE vs R²
ax3 = axes[1, 0]
scatter = ax3.scatter(df_results['R²'], df_results['RMSE'], 
                     s=100, alpha=0.6, c=df_results['R²'], cmap='RdYlGn', 
                     edgecolors='black', linewidth=1)
ax3.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax3.set_ylabel('RMSE (ml/s)', fontsize=12, fontweight='bold')
ax3.set_title('Trade-off R² vs RMSE', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='R²')

# 4. Métricas clave
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
🏆 MODELO FINAL OPTIMIZADO

📊 CONFIGURACIÓN:
   • Algoritmo: Gradient Boosting
   • Features: speed + acceleration
   • Frecuencia: 120s (2 minutos)
   • Árboles: 200
   • Profundidad: 7

📈 PERFORMANCE:
   • R² medio: {r2_mean:.3f} ± {r2_std:.3f}
   • R² mediano: {r2_median:.3f}
   • RMSE: {rmse_mean:.3f} ml/s

✅ DISTRIBUCIÓN:
   • Vehículos con R²>0.5: {r2_above_05}/{len(df_results)} ({r2_above_05/len(df_results)*100:.0f}%)
   • Vehículos con R²>0.3: {r2_above_05+r2_03_05}/{len(df_results)} ({(r2_above_05+r2_03_05)/len(df_results)*100:.0f}%)

💰 AHORRO:
   • Reducción bandwidth: 99.2%
   • Datos enviados: 0.8% vs 100%
"""

ax4.text(0.1, 0.5, summary_text, fontsize=13, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('optimal_model_summary.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: 'optimal_model_summary.png'")

print("\n" + "="*80)
print("✅ ENTRENAMIENTO DEL MODELO ÓPTIMO COMPLETADO")
print("="*80)

plt.show()