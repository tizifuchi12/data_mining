# RESULTADOS
#  Análisis de tus resultados:
# 🔴 ESTRATEGIA 1: FALLA COMPLETAMENTE
# R² promedio: -1.556 (¡NEGATIVO!)
# RMSE: 0.993 ml/s

# ¿Qué significa R² negativo?

# El modelo es PEOR que predecir la media
# No generaliza a otros vehículos
# Cada auto tiene características muy diferentes (peso, motor, aerodinámica)
# ✅ ESTRATEGIA 2: MUCHO MEJOR
# R² promedio: 0.341 ± 0.315
# RMSE: 0.583 ml/s (41% mejor que Estrategia 1)

# Esto demuestra:

# ✅ Entrenar con múltiples vehículos mejora la generalización
# ✅ El modelo aprende patrones comunes, no específicos de 1 auto
# ⚠️ R² = 0.34 es bajo, pero esperado (cada auto es diferente)

import pandas as pd
import numpy as np
import glob
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor  # ✅ AGREGAR
import matplotlib.pyplot as plt

print("="*60)
print("VALIDACIÓN MULTI-VEHÍCULO")
print("="*60)

# =============================================================================
# FUNCIÓN PARA PROCESAR VEHÍCULOS
# =============================================================================
def process_vehicle(file_path, freq='30s'):
    """Procesa un archivo de vehículo y retorna datos limpios"""
    try:
        data = pd.read_csv(file_path)
        
        # Parsear fechas de forma flexible y truncar a segundos
        data['time'] = pd.to_datetime(data['time'], errors='coerce')
        data['time'] = data['time'].dt.floor('s')  # Elimina milisegundos
        data = data.dropna(subset=['time'])
        
        if len(data) == 0:
            return None
        
        # Preparar señales
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
        
        # Verificar datos mínimos
        if 'fuel_consumed' not in dfs:
            return None
        if 'gps_speed' not in dfs and 'obd_speed' not in dfs:
            return None
        
        # Resamplear
        resampled = {}
        for signal_name, df in dfs.items():
            resampled[signal_name] = df.resample(freq).mean()
        
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
        
        # Interpolar
        for col in data_aligned.columns:
            data_aligned[col] = data_aligned[col].interpolate(method='linear', limit=5)
        
        data_freq = data_aligned.dropna(subset=['speed', 'fuel_consumed']).copy()
        
        if len(data_freq) < 50:
            return None
        
        # Feature engineering
        freq_seconds = int(freq[:-1])
        data_freq['acceleration'] = data_freq['speed'].diff()
        data_freq['fuel_increment'] = data_freq['fuel_consumed'].diff()
        data_freq.loc[data_freq['fuel_increment'] < 0, 'fuel_increment'] = np.nan
        data_freq['fuel_rate'] = data_freq['fuel_increment'] / freq_seconds
        data_freq = data_freq.dropna(subset=['fuel_rate']).copy()
        
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
# ESTRATEGIA 1: Entrenar con 1 vehículo, validar con otros
# =============================================================================
print("\n" + "="*60)
print("ESTRATEGIA 1: TRAIN EN 1 VEHÍCULO, TEST EN OTROS")
print("="*60)

train_vehicle = "fuel_data/863609060548678.csv"
train_data = process_vehicle(train_vehicle)
results_strategy1 = []  # ← asegúrate de tenerla definida

if train_data is not None:
    X_train = train_data[['speed', 'acceleration']].values
    y_train = train_data['fuel_rate'].values

    scaler_s1 = StandardScaler()
    X_train_scaled = scaler_s1.fit_transform(X_train)

    # CAMBIO: MLP → Extra Trees
    model_s1 = ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42)
    model_s1.fit(X_train_scaled, y_train)

    # usar lista propia para S1
    files_all = glob.glob("fuel_data/*.csv")
    test_files_s1 = [f for f in files_all if f != train_vehicle.replace('/', '\\')]

    for i, file in enumerate(test_files_s1):
        vehicle_id = file.split('\\')[-1].replace('.csv', '')
        test_data = process_vehicle(file)
        if test_data is None:
            continue

        X_test = test_data[['speed', 'acceleration']].values
        y_test = test_data['fuel_rate'].values
        y_pred = model_s1.predict(scaler_s1.transform(X_test))

        results_strategy1.append({
            'Vehicle_ID': vehicle_id,
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'Registros': len(test_data)
        })
# =============================================================================
# ESTRATEGIA 2: Entrenar con 100 vehículos, test con otros
# =============================================================================
print("\n" + "="*60)
print("ESTRATEGIA 2: TRAIN CON ~120 VEHÍCULOS, TEST CON OTROS")
print("="*60)

all_files = glob.glob("fuel_data/*.csv")
np.random.seed(42); np.random.shuffle(all_files)
num_train = min(120, int(len(all_files) * 0.8))
train_files = all_files[:num_train]
test_files_s2 = all_files[num_train:]  # ← lista propia para S2

train_data_list = []
for file in train_files:
    d = process_vehicle(file)
    if d is not None:
        train_data_list.append(d)

results_strategy2 = []

if len(train_data_list) > 0:
    combined_train = pd.concat(train_data_list, ignore_index=True)
    X_train = combined_train[['speed', 'acceleration']].values
    y_train = combined_train['fuel_rate'].values

    scaler_s2 = StandardScaler()
    X_train_scaled = scaler_s2.fit_transform(X_train)

    # CAMBIO: MLP → Extra Trees
    model_s2 = ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42)
    model_s2.fit(X_train_scaled, y_train)

    for file in test_files_s2:
        vehicle_id = file.split('\\')[-1].replace('.csv', '')
        test_data = process_vehicle(file)
        if test_data is None:
            continue

        X_test = test_data[['speed', 'acceleration']].values
        y_test = test_data['fuel_rate'].values
        y_pred = model_s2.predict(scaler_s2.transform(X_test))

        results_strategy2.append({
            'Vehicle_ID': vehicle_id,
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'Registros': len(test_data)
        })
# =============================================================================
# COMPARACIÓN DE ESTRATEGIAS
# =============================================================================
print("\n" + "="*60)
print("COMPARACIÓN DE ESTRATEGIAS")
print("="*60)

if len(results_strategy1) > 0:
    df_s1 = pd.DataFrame(results_strategy1)
    print("\n📊 ESTRATEGIA 1 (Train: 1 vehículo, Test: otros)")
    print(f"   R² promedio: {df_s1['R²'].mean():.3f} ± {df_s1['R²'].std():.3f}")
    print(f"   R² mediano: {df_s1['R²'].median():.3f}")
    print(f"   RMSE promedio: {df_s1['RMSE'].mean():.3f} ± {df_s1['RMSE'].std():.3f} ml/s")
    print(f"   Vehículos exitosos: {len(df_s1)}/{len(test_files_s1)}")  # ← usa test_files_s1

if len(results_strategy2) > 0:
    df_s2 = pd.DataFrame(results_strategy2)
    print(f"\n📊 ESTRATEGIA 2 (Train: {len(train_data_list)} vehículos, Test: {len(test_files_s2)} vehículos)")
    print(f"   R² promedio: {df_s2['R²'].mean():.3f} ± {df_s2['R²'].std():.3f}")
    print(f"   R² mediano: {df_s2['R²'].median():.3f}")
    print(f"   RMSE promedio: {df_s2['RMSE'].mean():.3f} ± {df_s2['RMSE'].std():.3f} ml/s")
    print(f"   Vehículos exitosos: {len(df_s2)}/{len(test_files_s2)}")  # ← usa test_files_s2
    
    # Estadísticas adicionales
    print(f"\n📈 DISTRIBUCIÓN DE PERFORMANCE:")
    print(f"   R² > 0.5 (bueno):     {(df_s2['R²'] > 0.5).sum()} vehículos ({(df_s2['R²'] > 0.5).sum()/len(df_s2)*120:.1f}%)")
    print(f"   R² 0.3-0.5 (regular): {((df_s2['R²'] >= 0.3) & (df_s2['R²'] <= 0.5)).sum()} vehículos ({((df_s2['R²'] >= 0.3) & (df_s2['R²'] <= 0.5)).sum()/len(df_s2)*120:.1f}%)")
    print(f"   R² < 0.3 (malo):      {(df_s2['R²'] < 0.3).sum()} vehículos ({(df_s2['R²'] < 0.3).sum()/len(df_s2)*120:.1f}%)")
    
    df_s2.to_csv('strategy2_results.csv', index=False)
    print("\n✅ Resultados guardados en 'strategy2_results.csv'")

# =============================================================================
# VISUALIZACIÓN MEJORADA
# =============================================================================
if len(results_strategy1) > 0 and len(results_strategy2) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Comparación R² (Boxplot)
    ax1 = axes[0, 0]
    bp1 = ax1.boxplot([df_s1['R²'], df_s2['R²']], 
                       tick_labels=['1 vehículo\n(train)', f'{len(train_data_list)} vehículos\n(train)'],
                       patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightcoral')
    bp1['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Comparación de R² - Generalización', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='R²=0.5')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='R²=0')
    ax1.legend()
    
    # 2. Comparación RMSE (Boxplot)
    ax2 = axes[0, 1]
    bp2 = ax2.boxplot([df_s1['RMSE'], df_s2['RMSE']], 
                       tick_labels=['1 vehículo\n(train)', f'{len(train_data_list)} vehículos\n(train)'],
                       patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightcoral')
    bp2['boxes'][1].set_facecolor('lightgreen')
    ax2.set_ylabel('RMSE (ml/s)', fontsize=12)
    ax2.set_title('Comparación de RMSE - Error de Predicción', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Histograma R² Estrategia 2
    ax3 = axes[1, 0]
    ax3.hist(df_s2['R²'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(df_s2['R²'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Media: {df_s2["R²"].mean():.3f}')
    ax3.axvline(df_s2['R²'].median(), color='green', linestyle='--', linewidth=2,
                label=f'Mediana: {df_s2["R²"].median():.3f}')
    ax3.set_xlabel('R² Score', fontsize=12)
    ax3.set_ylabel('Frecuencia', fontsize=12)
    ax3.set_title(f'Distribución de R² - Estrategia 2 ({len(train_data_list)} vehículos train)', 
                  fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Top 10 y Bottom 10 vehículos
    ax4 = axes[1, 1]
    df_s2_sorted = df_s2.sort_values('R²', ascending=False)
    top5 = df_s2_sorted.head(5)
    bottom5 = df_s2_sorted.tail(5)
    
    y_pos = range(10)
    colors = ['green']*5 + ['red']*5
    r2_values = list(top5['R²']) + list(bottom5['R²'])
    labels = [f"Top {i+1}" for i in range(5)] + [f"Bottom {i+1}" for i in range(5)]
    
    bars = ax4.barh(y_pos, r2_values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels)
    ax4.set_xlabel('R² Score', fontsize=12)
    ax4.set_title('Top 5 y Bottom 5 Vehículos (Estrategia 2)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Agregar valores
    for i, (bar, val) in enumerate(zip(bars, r2_values)):
        ax4.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('multi_vehicle_comparison_full.png', dpi=300, bbox_inches='tight')
    print("\n✅ Gráfico comparativo guardado: 'multi_vehicle_comparison_full.png'")
    plt.show()

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "="*60)
print("RESUMEN EJECUTIVO")
print("="*60)

if len(results_strategy2) > 0:
    print(f"""
🎯 CONFIGURACIÓN FINAL:
   • Vehículos de entrenamiento: {len(train_data_list)}
   • Vehículos de test: {len(test_files)}
   • Frecuencia de muestreo: 30 segundos
   • Total registros entrenamiento: {len(combined_train_data):,}

📊 PERFORMANCE EN TEST:
   • R² promedio: {df_s2['R²'].mean():.3f} ± {df_s2['R²'].std():.3f}
   • R² mediano: {df_s2['R²'].median():.3f}
   • RMSE promedio: {df_s2['RMSE'].mean():.3f} ml/s
   
✅ CONCLUSIÓN:
   Entrenar con ~100 vehículos permite generalizar a otros vehículos
   con R² ≈ 0.3-0.4, lo cual es razonable considerando la heterogeneidad
   del parque vehicular (diferentes motores, pesos, estilos de conducción).
   
💡 RECOMENDACIÓN:
   Para producción, usar modelo entrenado con múltiples vehículos + 
   fine-tuning personalizado con primeras 500-1000 lecturas del vehículo.
""")

print("\n" + "="*60)
print("✅ VALIDACIÓN MULTI-VEHÍCULO COMPLETADA")
print("="*60)