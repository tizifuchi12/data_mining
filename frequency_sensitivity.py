import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

RANDOM_SEED = 42
NUM_TRAIN_VEHICLES = 120
FREQUENCIES = ['1s', '5s', '10s', '30s', '60s', '120s']

print("="*80)
print("ANÁLISIS DE SENSIBILIDAD A FRECUENCIA DE MUESTREO")
print("="*80)
print("\n🎯 Usando configuración óptima: Gradient Boosting + 2 features")

def process_vehicle_freq(file_path, freq):
    """Procesa vehículo con frecuencia específica"""
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
        
        resampled = {}
        for signal_name, df in dfs.items():
            resampled[signal_name] = df.resample(freq).mean()
        
        data_aligned = pd.concat(resampled.values(), axis=1)
        data_aligned.columns = resampled.keys()
        
        if 'gps_speed' in data_aligned.columns and 'obd_speed' in data_aligned.columns:
            data_aligned['speed'] = data_aligned['gps_speed'].fillna(data_aligned['obd_speed'])
        elif 'gps_speed' in data_aligned.columns:
            data_aligned['speed'] = data_aligned['gps_speed']
        elif 'obd_speed' in data_aligned.columns:
            data_aligned['speed'] = data_aligned['obd_speed']
        else:
            return None
        
        for col in data_aligned.columns:
            data_aligned[col] = data_aligned[col].interpolate(method='linear', limit=5)
        
        data_freq = data_aligned.dropna(subset=['speed', 'fuel_consumed']).copy()
        
        if len(data_freq) < 50:
            return None
        
        freq_seconds = int(freq[:-1])
        data_freq['acceleration'] = data_freq['speed'].diff()
        data_freq['fuel_increment'] = data_freq['fuel_consumed'].diff()
        data_freq.loc[data_freq['fuel_increment'] < 0, 'fuel_increment'] = np.nan
        data_freq['fuel_rate'] = data_freq['fuel_increment'] / freq_seconds
        
        data_freq = data_freq.replace([np.inf, -np.inf], np.nan)
        data_freq = data_freq.dropna()
        
        if len(data_freq) < 50:
            return None
        
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

# Cargar y dividir archivos
all_files = glob.glob("fuel_data/*.csv")
np.random.seed(RANDOM_SEED)
np.random.shuffle(all_files)

train_files = all_files[:NUM_TRAIN_VEHICLES]
test_files = all_files[NUM_TRAIN_VEHICLES:]

print(f"\n📂 División de datos:")
print(f"   • Archivos train: {len(train_files)}")
print(f"   • Archivos test: {len(test_files)}")

results = []

for freq in FREQUENCIES:
    print(f"\n{'='*80}")
    print(f"🔄 PROBANDO FRECUENCIA: {freq}")
    print(f"{'='*80}")
    
    # Procesar train
    print(f"\n📥 Procesando {len(train_files)} vehículos de entrenamiento...")
    train_data_list = []
    for i, file in enumerate(train_files):
        if i % 20 == 0:
            print(f"   Progreso: {i}/{len(train_files)}")
        d = process_vehicle_freq(file, freq)
        if d is not None:
            train_data_list.append(d)
    
    print(f"✅ Vehículos procesados: {len(train_data_list)}/{len(train_files)}")
    
    if len(train_data_list) == 0:
        print(f"❌ No hay datos para frecuencia {freq}")
        continue
    
    combined_train = pd.concat(train_data_list, ignore_index=True)
    print(f"✅ Total registros: {len(combined_train):,}")
    
    # Submuestreo
    if len(combined_train) > 100000:
        print(f"   ⚠️ Submuestreando a 100,000 registros...")
        combined_train = combined_train.sample(n=100000, random_state=RANDOM_SEED)
    
    combined_train = combined_train.replace([np.inf, -np.inf], np.nan)
    combined_train = combined_train.dropna()
    print(f"✅ Registros finales: {len(combined_train):,}")
    
    feature_cols = ['speed', 'acceleration']
    X_train = combined_train[feature_cols].values
    y_train = combined_train['fuel_rate'].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Entrenar Gradient Boosting (mejor modelo)
    print(f"\n🤖 Entrenando Gradient Boosting...")
    model = GradientBoostingRegressor(n_estimators=200, max_depth=7, random_state=RANDOM_SEED)
    model.fit(X_train_scaled, y_train)
    print(f"   ✅ Modelo entrenado")
    
    # Evaluar en test
    print(f"\n📊 Evaluando en {len(test_files)} vehículos de test...")
    r2_scores = []
    rmse_scores = []
    
    for i, file in enumerate(test_files):
        if i % 5 == 0:
            print(f"   Progreso: {i}/{len(test_files)}")
        
        test_data = process_vehicle_freq(file, freq)
        if test_data is None:
            continue
        
        X_test = test_data[feature_cols].values
        y_test = test_data['fuel_rate'].values
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        r2_scores.append(r2)
        rmse_scores.append(rmse)
    
    r2_mean = np.mean(r2_scores)
    r2_median = np.median(r2_scores)
    r2_std = np.std(r2_scores)
    rmse_mean = np.mean(rmse_scores)
    
    bandwidth_reduction = (1 / int(freq[:-1])) * 100
    
    results.append({
        'Frecuencia': freq,
        'Segundos': int(freq[:-1]),
        'R² medio': r2_mean,
        'R² mediano': r2_median,
        'R² std': r2_std,
        'RMSE medio': rmse_mean,
        'Vehículos evaluados': len(r2_scores),
        'Uso Bandwidth (%)': bandwidth_reduction,
        'R² > 0.5': sum(1 for r2 in r2_scores if r2 > 0.5),
        'R² 0.3-0.5': sum(1 for r2 in r2_scores if 0.3 <= r2 <= 0.5),
        'R² < 0.3': sum(1 for r2 in r2_scores if r2 < 0.3)
    })
    
    print(f"\n✅ RESULTADOS para {freq}:")
    print(f"   R² medio: {r2_mean:.3f} ± {r2_std:.3f}")
    print(f"   R² mediano: {r2_median:.3f}")
    print(f"   RMSE: {rmse_mean:.3f} ml/s")
    print(f"   Uso bandwidth: {bandwidth_reduction:.1f}% (vs 1s)")
    print(f"   Vehículos con R²>0.5: {sum(1 for r2 in r2_scores if r2 > 0.5)}/{len(r2_scores)}")

# Guardar resultados
df_results = pd.DataFrame(results)
df_results.to_csv('frequency_sensitivity_results.csv', index=False)

print("\n" + "="*80)
print("📊 RESUMEN - SENSIBILIDAD A FRECUENCIA")
print("="*80)
print("\n" + df_results.to_string(index=False))

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gráfico 1: R² vs Frecuencia
ax1 = axes[0, 0]
ax1.plot(df_results['Segundos'], df_results['R² medio'], 
         marker='o', linewidth=3, markersize=12, label='R² medio', color='steelblue')
ax1.fill_between(df_results['Segundos'], 
                  df_results['R² medio'] - df_results['R² std'],
                  df_results['R² medio'] + df_results['R² std'],
                  alpha=0.2, color='steelblue')
ax1.plot(df_results['Segundos'], df_results['R² mediano'], 
         marker='s', linewidth=2, markersize=10, label='R² mediano', 
         color='green', linestyle='--')
ax1.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Seuil R²=0.5')
ax1.axhline(y=0.4, color='orange', linestyle=':', alpha=0.5, label='Seuil R²=0.4')
ax1.set_xlabel('Période d\'échantillonnage (secondes)', fontsize=13, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=13, fontweight='bold')
ax1.set_title('Impact de la Fréquence sur la Performance', fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_xscale('log')

# Gráfico 2: RMSE vs Frecuencia
ax2 = axes[0, 1]
ax2.plot(df_results['Segundos'], df_results['RMSE medio'],
         marker='o', linewidth=3, markersize=12, color='coral')
ax2.set_xlabel('Période d\'échantillonnage (secondes)', fontsize=13, fontweight='bold')
ax2.set_ylabel('RMSE (ml/s)', fontsize=13, fontweight='bold')
ax2.set_title('Erreur de Prédiction vs Fréquence', fontsize=15, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# Anotar valores
for i, row in df_results.iterrows():
    ax2.annotate(f"{row['RMSE medio']:.3f}", 
                (row['Segundos'], row['RMSE medio']),
                textcoords="offset points", xytext=(0,10), 
                ha='center', fontsize=10, fontweight='bold')

# Gráfico 3: Trade-off Performance vs Bandwidth
ax3 = axes[1, 0]
colors = ['darkgreen' if r2 > 0.45 else 'green' if r2 > 0.4 else 'orange' if r2 > 0.35 else 'red' 
          for r2 in df_results['R² medio']]
scatter = ax3.scatter(df_results['Uso Bandwidth (%)'], df_results['R² medio'], 
                     s=300, alpha=0.7, c=colors, edgecolors='black', linewidth=2)

for i, row in df_results.iterrows():
    ax3.annotate(row['Frecuencia'], 
                (row['Uso Bandwidth (%)'], row['R² medio']),
                textcoords="offset points", xytext=(0,12), 
                ha='center', fontsize=11, fontweight='bold')

ax3.set_xlabel('Utilisation de la Bande Passante (%)', fontsize=13, fontweight='bold')
ax3.set_ylabel('R² moyen', fontsize=13, fontweight='bold')
ax3.set_title('Trade-off: Performance vs Bandwidth', fontsize=15, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')

# Leyenda colores
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='darkgreen', label='Excellent (R²>0.45)'),
    Patch(facecolor='green', label='Bon (R²>0.40)'),
    Patch(facecolor='orange', label='Acceptable (R²>0.35)'),
    Patch(facecolor='red', label='Faible (R²<0.35)')
]
ax3.legend(handles=legend_elements, loc='best', fontsize=10)

# Gráfico 4: Distribución de vehículos
ax4 = axes[1, 1]
x = range(len(df_results))
width = 0.25

bars1 = ax4.bar([i - width for i in x], df_results['R² > 0.5'], width, 
                label='R² > 0.5 (bon)', color='green', alpha=0.8, edgecolor='black')
bars2 = ax4.bar(x, df_results['R² 0.3-0.5'], width, 
                label='R² 0.3-0.5 (moyen)', color='orange', alpha=0.8, edgecolor='black')
bars3 = ax4.bar([i + width for i in x], df_results['R² < 0.3'], width, 
                label='R² < 0.3 (faible)', color='red', alpha=0.8, edgecolor='black')

ax4.set_xticks(x)
ax4.set_xticklabels(df_results['Frecuencia'])
ax4.set_xlabel('Fréquence', fontsize=13, fontweight='bold')
ax4.set_ylabel('Nombre de véhicules', fontsize=13, fontweight='bold')
ax4.set_title('Distribution de la Performance', fontsize=15, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('frequency_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Gráfico guardado: 'frequency_sensitivity_analysis.png'")
plt.show()

# Análisis de recomendación
print("\n" + "="*80)
print("💡 RECOMMANDATION OPTIMALE")
print("="*80)

best_idx = df_results['R² medio'].idxmax()
best_freq = df_results.loc[best_idx]

print(f"""
🏆 MEILLEURE FRÉQUENCE: {best_freq['Frecuencia']}
   • R² moyen: {best_freq['R² medio']:.3f} ± {best_freq['R² std']:.3f}
   • R² médian: {best_freq['R² mediano']:.3f}
   • RMSE: {best_freq['RMSE medio']:.3f} ml/s
   • Véhicules avec R²>0.5: {best_freq['R² > 0.5']}/{best_freq['Vehículos evaluados']}
   • Utilisation bandwidth: {best_freq['Uso Bandwidth (%)']:.1f}%

📊 COMPROMISE RECOMMENDÉ: 30s
   • R² acceptable (≈{df_results[df_results['Frecuencia']=='30s']['R² medio'].values[0]:.3f})
   • Réduction bandwidth: {100 - df_results[df_results['Frecuencia']=='30s']['Uso Bandwidth (%)'].values[0]:.1f}%
   • Bon équilibre performance/coût
""")

print("\n" + "="*80)
print("✅ ANALYSE DE SENSIBILITÉ COMPLÉTÉE")
print("="*80)