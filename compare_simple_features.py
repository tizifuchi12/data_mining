import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

# =============================================================================
# ✅ CONFIGURACIÓN
# =============================================================================
NUM_TRAIN_VEHICLES = 120
NUM_TEST_VEHICLES = None
RANDOM_SEED = 42
SAMPLING_FREQ = '30s'
MAX_TRAINING_SAMPLES = 1500000

print("="*80)
print("COMPARACIÓN CON FEATURE SIMPLE: speed²")
print("="*80)
print(f"\n📊 CONFIGURACIÓN:")
print(f"   • Vehículos para entrenar: {NUM_TRAIN_VEHICLES}")
print(f"   • Features: speed, acceleration, speed² (3 features)")
print(f"   • Frecuencia de muestreo: {SAMPLING_FREQ}")

# =============================================================================
# FUNCIÓN PARA PROCESAR VEHÍCULOS CON FEATURE SIMPLE
# =============================================================================
def process_vehicle(file_path, freq=SAMPLING_FREQ):
    """Procesa un archivo de vehículo y retorna datos con speed²"""
    try:
        data = pd.read_csv(file_path)
        
        # Parsear fechas
        data['time'] = pd.to_datetime(data['time'], errors='coerce')
        data['time'] = data['time'].dt.floor('s')
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
        
        # ========================================================================
        # ✅ FEATURE ENGINEERING SIMPLE (solo speed²)
        # ========================================================================
        freq_seconds = int(freq[:-1])
        
        # Features básicas
        data_freq['acceleration'] = data_freq['speed'].diff()
        data_freq['fuel_increment'] = data_freq['fuel_consumed'].diff()
        data_freq.loc[data_freq['fuel_increment'] < 0, 'fuel_increment'] = np.nan
        data_freq['fuel_rate'] = data_freq['fuel_increment'] / freq_seconds
        
        # ✅ SOLO 1 FEATURE ADICIONAL: speed² (resistencia aerodinámica)
        data_freq['speed_squared'] = data_freq['speed'] ** 2
        
        # Limpiar NaN
        data_freq = data_freq.dropna(subset=['fuel_rate', 'speed', 'acceleration', 'speed_squared'])
        
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
        
        # Eliminar inf
        data_freq = data_freq.replace([np.inf, -np.inf], np.nan)
        data_freq = data_freq.dropna()
        
        if len(data_freq) < 100:
            return None
        
        return data_freq[['speed', 'acceleration', 'speed_squared', 'fuel_rate']].copy()
        
    except Exception as e:
        return None

# =============================================================================
# CARGAR Y DIVIDIR DATOS
# =============================================================================
print("\n" + "="*80)
print("CARGANDO DATOS")
print("="*80)

all_files = glob.glob("fuel_data/*.csv")
print(f"\nTotal de archivos CSV disponibles: {len(all_files)}")

np.random.seed(RANDOM_SEED)
np.random.shuffle(all_files)

if len(all_files) > NUM_TRAIN_VEHICLES:
    num_train = NUM_TRAIN_VEHICLES
else:
    num_train = max(1, int(len(all_files) * 0.8))

train_files = all_files[:num_train]

if NUM_TEST_VEHICLES is None:
    test_files = all_files[num_train:]
else:
    test_files = all_files[num_train:num_train + NUM_TEST_VEHICLES]

print(f"\n📊 División:")
print(f"   • Archivos para TRAIN: {len(train_files)}")
print(f"   • Archivos para TEST:  {len(test_files)}")

# Cargar datos de entrenamiento
print(f"\n🔄 Procesando {len(train_files)} vehículos de entrenamiento...")
train_data_list = []
for i, file in enumerate(train_files):
    if i % 20 == 0:
        print(f"   Progreso: {i}/{len(train_files)}")
    d = process_vehicle(file)
    if d is not None:
        train_data_list.append(d)

print(f"\n✅ Vehículos de entrenamiento procesados: {len(train_data_list)}/{len(train_files)}")

if len(train_data_list) == 0:
    print("❌ ERROR: No hay datos de entrenamiento")
    exit()

combined_train = pd.concat(train_data_list, ignore_index=True)
print(f"✅ Total registros de entrenamiento: {len(combined_train):,}")

# Submuestreo si es necesario
if len(combined_train) > MAX_TRAINING_SAMPLES:
    print(f"\n⚠️ Dataset muy grande ({len(combined_train):,} registros)")
    print(f"   Submuestreando a {MAX_TRAINING_SAMPLES:,} registros...")
    combined_train = combined_train.sample(n=MAX_TRAINING_SAMPLES, random_state=RANDOM_SEED)
    print(f"   ✅ Nuevo tamaño: {len(combined_train):,} registros")

# Limpiar NaN e Inf
print(f"\n🔧 Limpieza final de datos...")
combined_train = combined_train.replace([np.inf, -np.inf], np.nan)
combined_train = combined_train.dropna()
print(f"   ✅ Registros finales: {len(combined_train):,}")

# ✅ USAR 3 FEATURES
feature_cols = ['speed', 'acceleration', 'speed_squared']

print(f"\n✅ Features utilizadas: {feature_cols}")

X_train = combined_train[feature_cols].values
y_train = combined_train['fuel_rate'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# =============================================================================
# DEFINIR MODELOS
# =============================================================================
print("\n" + "="*80)
print("ENTRENANDO 9 MODELOS CON 3 FEATURES (speed, accel, speed²)")
print("="*80)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=15, random_state=RANDOM_SEED),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, max_depth=15, n_jobs=-1),
    'Extra Trees': ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_SEED, max_depth=15, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_SEED, max_depth=7),
    'K-Neighbors (k=5)': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=1000, random_state=RANDOM_SEED)
}

# Entrenar todos los modelos
trained_models = {}

for name, model in models.items():
    print(f"\n🔄 Entrenando: {name}")
    
    if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train_scaled, y_train)
    
    trained_models[name] = model
    print(f"   ✅ Entrenado")

# =============================================================================
# EVALUAR EN VEHÍCULOS DE TEST
# =============================================================================
print("\n" + "="*80)
print(f"EVALUANDO EN {len(test_files)} VEHÍCULOS DE TEST")
print("="*80)

results_by_model = {name: [] for name in trained_models.keys()}

for i, file in enumerate(test_files):
    if i % 5 == 0:
        print(f"\nProcesando vehículos de test: {i}/{len(test_files)}")
    
    vehicle_id = file.split('\\')[-1].replace('.csv', '')
    test_data = process_vehicle(file)
    
    if test_data is None:
        continue
    
    X_test = test_data[feature_cols].values
    y_test = test_data['fuel_rate'].values
    
    # Evaluar con cada modelo
    for name, model in trained_models.items():
        if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
            y_pred = model.predict(X_test)
        else:
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results_by_model[name].append({
            'Vehicle_ID': vehicle_id,
            'R²': r2,
            'RMSE': rmse
        })

# =============================================================================
# ANÁLISIS DE RESULTADOS
# =============================================================================
print("\n" + "="*80)
print("RESULTADOS CON 3 FEATURES (speed, accel, speed²)")
print("="*80)

summary = []

for name, results in results_by_model.items():
    if len(results) == 0:
        continue
    
    df = pd.DataFrame(results)
    
    summary.append({
        'Modelo': name,
        'R² medio': df['R²'].mean(),
        'R² mediano': df['R²'].median(),
        'R² std': df['R²'].std(),
        'RMSE medio': df['RMSE'].mean(),
        'Vehículos evaluados': len(df),
        'R² > 0.5': (df['R²'] > 0.5).sum(),
        'R² 0.3-0.5': ((df['R²'] >= 0.3) & (df['R²'] <= 0.5)).sum(),
        'R² < 0.3': (df['R²'] < 0.3).sum()
    })
    
    df.to_csv(f'results_simple_{name.replace(" ", "_")}.csv', index=False)

df_summary = pd.DataFrame(summary)
df_summary = df_summary.sort_values('R² medio', ascending=False)

print("\n" + df_summary.to_string(index=False))

df_summary.to_csv('all_models_summary_simple_features.csv', index=False)
print("\n✅ Resumen guardado en 'all_models_summary_simple_features.csv'")

# =============================================================================
# COMPARACIÓN CON VERSIÓN BÁSICA (2 features)
# =============================================================================
print("\n" + "="*80)
print("📊 COMPARACIÓN: 2 features vs 3 features")
print("="*80)

# Intentar cargar resultados anteriores
try:
    df_baseline = pd.read_csv('all_models_summary.csv')
    
    print("\n| Modelo            | R² (2 feat) | R² (3 feat) | Mejora |")
    print("|-------------------|-------------|-------------|--------|")
    
    for idx, row in df_summary.iterrows():
        modelo = row['Modelo']
        r2_new = row['R² medio']
        
        baseline_row = df_baseline[df_baseline['Modelo'] == modelo]
        if len(baseline_row) > 0:
            r2_old = baseline_row.iloc[0]['R² medio']
            mejora = ((r2_new - r2_old) / abs(r2_old)) * 100 if r2_old != 0 else 0
            print(f"| {modelo:17s} | {r2_old:11.3f} | {r2_new:11.3f} | {mejora:+5.1f}% |")
        else:
            print(f"| {modelo:17s} | {'N/A':11s} | {r2_new:11.3f} | {'N/A':6s} |")
            
except FileNotFoundError:
    print("\n⚠️ No se encontró 'all_models_summary.csv' (versión con 2 features)")
    print("   Ejecuta primero 'compare_all_models.py' para comparar")

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
print("\n" + "="*80)
print("GENERANDO GRÁFICOS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. R² medio por modelo
ax1 = axes[0, 0]
bars = ax1.barh(range(len(df_summary)), df_summary['R² medio'], 
                color='steelblue', edgecolor='black', alpha=0.8)

for i, (idx, row) in enumerate(df_summary.iterrows()):
    if row['R² medio'] > 0.5:
        bars[i].set_color('darkgreen')
    elif row['R² medio'] > 0.4:
        bars[i].set_color('green')
    elif row['R² medio'] > 0.2:
        bars[i].set_color('orange')
    else:
        bars[i].set_color('red')

ax1.set_yticks(range(len(df_summary)))
ax1.set_yticklabels(df_summary['Modelo'])
ax1.set_xlabel('R² medio', fontsize=12)
ax1.set_title(f'Ranking con 3 Features (speed, accel, speed²)', 
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.axvline(x=0.4, color='green', linestyle='--', alpha=0.5, label='R²=0.4')
ax1.axvline(x=0.5, color='darkgreen', linestyle='--', alpha=0.5, label='R²=0.5')
ax1.legend()

for i, (idx, row) in enumerate(df_summary.iterrows()):
    ax1.text(row['R² medio'] + 0.01, i, f"{row['R² medio']:.3f}", 
             va='center', fontsize=10, fontweight='bold')

# 2. RMSE
ax2 = axes[0, 1]
ax2.barh(range(len(df_summary)), df_summary['RMSE medio'], 
         color='coral', edgecolor='black', alpha=0.8)
ax2.set_yticks(range(len(df_summary)))
ax2.set_yticklabels(df_summary['Modelo'])
ax2.set_xlabel('RMSE medio (ml/s)', fontsize=12)
ax2.set_title('Error de Predicción', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# 3. Distribución
ax3 = axes[1, 0]
x_models = range(len(df_summary))
width = 0.25

ax3.bar([x - width for x in x_models], df_summary['R² > 0.5'], width, 
        label='R² > 0.5 (bueno)', color='green', alpha=0.8, edgecolor='black')
ax3.bar(x_models, df_summary['R² 0.3-0.5'], width, 
        label='R² 0.3-0.5 (regular)', color='orange', alpha=0.8, edgecolor='black')
ax3.bar([x + width for x in x_models], df_summary['R² < 0.3'], width, 
        label='R² < 0.3 (malo)', color='red', alpha=0.8, edgecolor='black')

ax3.set_xticks(x_models)
ax3.set_xticklabels(df_summary['Modelo'], rotation=45, ha='right')
ax3.set_ylabel('Número de vehículos', fontsize=12)
ax3.set_title('Distribución de Performance', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Boxplot Top 5
ax4 = axes[1, 1]
top5_models = df_summary.head(5)['Modelo'].tolist()
boxplot_data = []

for name in top5_models:
    df_model = pd.DataFrame(results_by_model[name])
    boxplot_data.append(df_model['R²'])

bp = ax4.boxplot(boxplot_data, tick_labels=top5_models, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_edgecolor('black')

ax4.set_ylabel('R² Score', fontsize=12)
ax4.set_title('Distribución R² - Top 5', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax4.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
ax4.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('all_models_comparison_simple_features.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: 'all_models_comparison_simple_features.png'")
plt.show()

# =============================================================================
# RESUMEN EJECUTIVO
# =============================================================================
print("\n" + "="*80)
print("RESUMEN EJECUTIVO - VERSIÓN CON FEATURE SIMPLE (speed²)")
print("="*80)

best_model = df_summary.iloc[0]

print(f"""
🎯 CONFIGURACIÓN:
   • Vehículos entrenamiento: {len(train_data_list)}
   • Vehículos test: {len(test_files)}
   • Registros entrenamiento: {len(combined_train):,}
   • Features utilizadas: 3 (speed, acceleration, speed²)

🏆 MEJOR MODELO: {best_model['Modelo']}
   • R² medio: {best_model['R² medio']:.3f} ± {best_model['R² std']:.3f}
   • R² mediano: {best_model['R² mediano']:.3f}
   • RMSE medio: {best_model['RMSE medio']:.3f} ml/s
   • Vehículos con R² > 0.5: {best_model['R² > 0.5']}/{best_model['Vehículos evaluados']} ({best_model['R² > 0.5']/best_model['Vehículos evaluados']*100:.1f}%)

📊 TOP 3 MODELOS:
""")

for i, (idx, row) in enumerate(df_summary.head(3).iterrows(), 1):
    print(f"   {i}. {row['Modelo']:20s}: R² = {row['R² medio']:.3f} (mediano: {row['R² mediano']:.3f})")

print("\n" + "="*80)
print("✅ ANÁLISIS CON FEATURE SIMPLE COMPLETADO")
print("="*80)