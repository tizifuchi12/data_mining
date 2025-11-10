import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar un solo archivo para empezar
file_path = "fuel_data/863609060548678.csv"
data = pd.read_csv(file_path)

print("=== INFORMACIÓN DEL VEHÍCULO ===")
print(f"Archivo: {file_path}")
print(f"Total de registros: {len(data)}")

# Convertir time a datetime
data['time'] = pd.to_datetime(data['time'])

print(f"\nRango temporal: {data['time'].min()} a {data['time'].max()}")
print(f"Duración: {data['time'].max() - data['time'].min()}")

# Definir las señales que queremos extraer (con los nombres correctos)
signals = {
    'gps_speed': 'TRACKS.MUNIC.GPS_SPEED (km/h)',
    'obd_speed': 'TRACKS.MUNIC.MDI_OBD_SPEED (km/h)',
    'fuel_consumed': 'TRACKS.MUNIC.MDI_OBD_FUEL (ml)'
}

# Crear DataFrames individuales para cada señal
dfs = {}
for signal_name, column_name in signals.items():
    if column_name in data.columns:
        df_signal = data[['time', column_name]].copy()
        df_signal = df_signal.dropna(subset=[column_name])
        df_signal = df_signal.rename(columns={column_name: signal_name})
        df_signal = df_signal.set_index('time')
        dfs[signal_name] = df_signal
        print(f"\n{signal_name}: {len(df_signal)} registros válidos")

# Elegir frecuencia de resampling (1 segundo)
freq = '30s'

print(f"\n=== RESAMPLING A {freq} ===")

# Resamplear cada señal
resampled = {}
for signal_name, df in dfs.items():
    # Usar mean para agregación
    resampled[signal_name] = df.resample(freq).mean()
    print(f"{signal_name}: {len(resampled[signal_name])} puntos después de resample")

# Combinar todas las señales en un solo DataFrame
data_aligned = pd.concat(resampled.values(), axis=1)
data_aligned.columns = resampled.keys()

print(f"\n=== DATOS ALINEADOS ===")
print(f"Total de timestamps: {len(data_aligned)}")
print(f"\nPrimeras filas:")
print(data_aligned.head(10))

print(f"\nValores nulos después de alineación:")
print(data_aligned.isnull().sum())

# Crear columna de velocidad combinada (GPS prioritario, OBD como backup)
data_aligned['speed'] = data_aligned['gps_speed'].fillna(data_aligned['obd_speed'])

# Interpolar valores faltantes (solo gaps pequeños)
print(f"\n=== INTERPOLACIÓN ===")
for col in data_aligned.columns:
    nulls_before = data_aligned[col].isnull().sum()
    if nulls_before > 0:
        # Interpolar solo si el gap es pequeño (max 5 segundos)
        data_aligned[col] = data_aligned[col].interpolate(method='linear', limit=5)
        nulls_after = data_aligned[col].isnull().sum()
        print(f"{col}: {nulls_before} → {nulls_after} nulos")

# Eliminar filas que aún tienen nulos en variables críticas
critical_vars = ['speed', 'fuel_consumed']
data_clean = data_aligned.dropna(subset=critical_vars)

print(f"\n=== DATOS FINALES ===")
print(f"Registros: {len(data_clean)}")
print(f"Porcentaje conservado: {(len(data_clean)/len(data))*100:.1f}%")
print(f"\nEstadísticas:")
print(data_clean.describe())

# Guardar datos limpios
data_clean.to_pickle('vehicle_data_clean.pkl')
print("\nDatos limpios guardados en 'vehicle_data_clean.pkl'")

# FEATURE ENGINEERING
print("\n=== FEATURE ENGINEERING ===")

# 1. Calcular aceleración (cambio de velocidad)
data_clean['acceleration'] = data_clean['speed'].diff() / 9  # m/s²  (9 segundos)

# 2. Calcular consumo incremental
data_clean['fuel_increment'] = data_clean['fuel_consumed'].diff()

# 3. Detectar cuando el consumo se reinicia (vuelve a 0)
data_clean.loc[data_clean['fuel_increment'] < 0, 'fuel_increment'] = np.nan

# 4. Calcular consumo por distancia (L/100km aproximado)
# Distancia recorrida en 9s a velocidad promedio
data_clean['distance_km'] = (data_clean['speed'] * 9) / 3600  # km
data_clean['consumption_per_100km'] = (data_clean['fuel_increment'] / 1000) / data_clean['distance_km'] * 100

# Limpiar infinitos y valores extremos
data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
data_clean = data_clean.dropna(subset=['fuel_increment'])

print(f"\nRegistros finales después de feature engineering: {len(data_clean)}")
print("\nNuevas features:")
print(data_clean[['speed', 'acceleration', 'fuel_increment', 'consumption_per_100km']].describe())

# Guardar
data_clean.to_pickle('vehicle_data_features.pkl')
print("\nDatos con features guardados")



# FASE 2: ANÁLISIS EXPLORATORIO
print("\n" + "="*60)
print("FASE 2: ANÁLISIS EXPLORATORIO")
print("="*60)

# Cargar datos procesados
data_clean = pd.read_pickle('vehicle_data_features.pkl')

# Crear figura con múltiples subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Análisis Exploratorio - Consumo de Combustible', fontsize=16, fontweight='bold')

# 1. Velocidad vs Consumo Incremental
ax1 = axes[0, 0]
scatter1 = ax1.scatter(data_clean['speed'], data_clean['fuel_increment'], 
                       alpha=0.3, s=10, c=data_clean['acceleration'], cmap='RdYlGn_r')
ax1.set_xlabel('Velocidad (km/h)')
ax1.set_ylabel('Consumo Incremental (ml/9s)')
ax1.set_title('Velocidad vs Consumo')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Aceleración')

# 2. Aceleración vs Consumo Incremental
ax2 = axes[0, 1]
ax2.scatter(data_clean['acceleration'], data_clean['fuel_increment'], 
            alpha=0.3, s=10, color='orange')
ax2.set_xlabel('Aceleración (km/h/s)')
ax2.set_ylabel('Consumo Incremental (ml/9s)')
ax2.set_title('Aceleración vs Consumo')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)

# 3. Distribución de Velocidad
ax3 = axes[0, 2]
ax3.hist(data_clean['speed'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Velocidad (km/h)')
ax3.set_ylabel('Frecuencia')
ax3.set_title('Distribución de Velocidad')
ax3.axvline(data_clean['speed'].mean(), color='red', linestyle='--', 
            label=f'Media: {data_clean["speed"].mean():.1f} km/h')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Distribución de Consumo Incremental
ax4 = axes[1, 0]
ax4.hist(data_clean['fuel_increment'], bins=50, color='green', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Consumo Incremental (ml/9s)')
ax4.set_ylabel('Frecuencia')
ax4.set_title('Distribución de Consumo')
ax4.axvline(data_clean['fuel_increment'].mean(), color='red', linestyle='--',
            label=f'Media: {data_clean["fuel_increment"].mean():.1f} ml')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Serie temporal: Velocidad y Consumo
ax5 = axes[1, 1]
ax5_twin = ax5.twinx()

# Tomar solo una muestra para visualizar (primeros 500 puntos)
sample = data_clean.head(500)
time_index = range(len(sample))

line1 = ax5.plot(time_index, sample['speed'], color='blue', alpha=0.7, label='Velocidad')
line2 = ax5_twin.plot(time_index, sample['fuel_increment'], color='red', alpha=0.7, label='Consumo')

ax5.set_xlabel('Tiempo (muestras cada 9s)')
ax5.set_ylabel('Velocidad (km/h)', color='blue')
ax5_twin.set_ylabel('Consumo (ml/9s)', color='red')
ax5.set_title('Serie Temporal (primeros 500 puntos)')
ax5.tick_params(axis='y', labelcolor='blue')
ax5_twin.tick_params(axis='y', labelcolor='red')
ax5.grid(True, alpha=0.3)

# Combinar leyendas
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax5.legend(lines, labels, loc='upper left')

# 6. Boxplot de Consumo por rangos de velocidad
ax6 = axes[1, 2]
# Crear bins de velocidad
data_clean['speed_bin'] = pd.cut(data_clean['speed'], 
                                  bins=[0, 30, 60, 90, 120, 150],
                                  labels=['0-30', '30-60', '60-90', '90-120', '120+'])
data_clean.boxplot(column='fuel_increment', by='speed_bin', ax=ax6)
ax6.set_xlabel('Rango de Velocidad (km/h)')
ax6.set_ylabel('Consumo Incremental (ml/9s)')
ax6.set_title('Consumo por Rango de Velocidad')
plt.sca(ax6)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('exploratory_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Gráficos guardados en 'exploratory_analysis.png'")
plt.show()

# Análisis de correlaciones
print("\n=== MATRIZ DE CORRELACIONES ===")
correlations = data_clean[['speed', 'acceleration', 'fuel_increment', 'consumption_per_100km']].corr()
print(correlations)

# Insights automáticos
print("\n=== INSIGHTS ===")
print(f"📊 Correlación velocidad-consumo: {correlations.loc['speed', 'fuel_increment']:.3f}")
print(f"📊 Correlación aceleración-consumo: {correlations.loc['acceleration', 'fuel_increment']:.3f}")
print(f"🚗 Velocidad promedio: {data_clean['speed'].mean():.1f} km/h")
print(f"⛽ Consumo promedio: {data_clean['fuel_increment'].mean():.1f} ml/9s")
print(f"📈 Consumo promedio equivalente: {data_clean['consumption_per_100km'].median():.1f} L/100km")

# Identificar outliers
print("\n=== DETECCIÓN DE OUTLIERS ===")
Q1 = data_clean['fuel_increment'].quantile(0.25)
Q3 = data_clean['fuel_increment'].quantile(0.75)
IQR = Q3 - Q1
outliers = data_clean[(data_clean['fuel_increment'] < Q1 - 1.5*IQR) | 
                      (data_clean['fuel_increment'] > Q3 + 1.5*IQR)]
print(f"Outliers detectados: {len(outliers)} ({len(outliers)/len(data_clean)*100:.1f}%)")

# Guardar datos sin outliers para modelado
data_no_outliers = data_clean[(data_clean['fuel_increment'] >= Q1 - 1.5*IQR) & 
                               (data_clean['fuel_increment'] <= Q3 + 1.5*IQR)]
data_no_outliers.to_pickle('vehicle_data_clean_no_outliers.pkl')
print(f"✅ Datos sin outliers guardados: {len(data_no_outliers)} registros")





#VEO QUE ME FREQENCIA ME CONVIENE (30s O 60S, 30 xq tiene 2880 muestra x dia)

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# print("="*60)
# print("OBJETIVO 3: SENSIBILIDAD A FRECUENCIAS")
# print("="*60)

# # Cargar datos originales (antes del resampling)
# data = pd.read_csv("fuel_data/863609060548678.csv")
# data['time'] = pd.to_datetime(data['time'])

# frequencies = ['2s', '5s', '10s', '30s', '60s']
# results = []

# for freq in frequencies:
#     print(f"\n{'='*60}")
#     print(f"Procesando frecuencia: {freq}")
#     print(f"{'='*60}")
    
#     # Preparar señales
#     signals = {
#         'gps_speed': 'TRACKS.MUNIC.GPS_SPEED (km/h)',
#         'obd_speed': 'TRACKS.MUNIC.MDI_OBD_SPEED (km/h)',
#         'fuel_consumed': 'TRACKS.MUNIC.MDI_OBD_FUEL (ml)'
#     }
    
#     dfs = {}
#     for signal_name, column_name in signals.items():
#         if column_name in data.columns:
#             df_signal = data[['time', column_name]].copy()
#             df_signal = df_signal.dropna(subset=[column_name])
#             df_signal = df_signal.rename(columns={column_name: signal_name})
#             df_signal = df_signal.set_index('time')
#             dfs[signal_name] = df_signal
    
#     # Resamplear
#     resampled = {}
#     for signal_name, df in dfs.items():
#         resampled[signal_name] = df.resample(freq).mean()
    
#     data_aligned = pd.concat(resampled.values(), axis=1)
#     data_aligned.columns = resampled.keys()
#     data_aligned['speed'] = data_aligned['gps_speed'].fillna(data_aligned['obd_speed'])
    
#     # Interpolar
#     for col in data_aligned.columns:
#         data_aligned[col] = data_aligned[col].interpolate(method='linear', limit=5)
    
#     data_freq = data_aligned.dropna(subset=['speed', 'fuel_consumed'])
    
# # Feature engineering
#     data_freq['acceleration'] = data_freq['speed'].diff()
#     data_freq['fuel_increment'] = data_freq['fuel_consumed'].diff()
#     data_freq.loc[data_freq['fuel_increment'] < 0, 'fuel_increment'] = np.nan
    
#     # ✅ NORMALIZAR POR TIEMPO
#     freq_seconds = int(freq[:-1])
#     data_freq['fuel_rate'] = data_freq['fuel_increment'] / freq_seconds  # ml/segundo
    
#     data_freq = data_freq.dropna(subset=['fuel_rate'])
    
#     # Eliminar outliers
#     Q1 = data_freq['fuel_rate'].quantile(0.25)
#     Q3 = data_freq['fuel_rate'].quantile(0.75)
#     IQR = Q3 - Q1
#     data_freq = data_freq[(data_freq['fuel_rate'] >= Q1 - 1.5*IQR) & 
#                           (data_freq['fuel_rate'] <= Q3 + 1.5*IQR)]
    
#     print(f"Registros disponibles: {len(data_freq)}")
    
#     if len(data_freq) < 100:
#         print("⚠️ Muy pocos datos, saltando...")
#         continue
    
#     # Preparar datos para modelo
#     X = data_freq[['speed', 'acceleration']].values
#     y = data_freq['fuel_rate'].values  # ✅ Usar fuel_rate en lugar de fuel_increment
    
    
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
    
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Entrenar Random Forest (mejor modelo anterior)
#     model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
    
#     # Métricas
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     r2 = r2_score(y_test, y_pred)
    
#     # Calcular "datos por día" equivalente
#     freq_seconds = int(freq[:-1])
#     samples_per_day = (24 * 60 * 60) / freq_seconds
    
#     results.append({
#         'Frecuencia': freq,
#         'Segundos': freq_seconds,
#         'Registros': len(data_freq),
#         'Muestras/día': samples_per_day,
#         'RMSE': rmse,
#         'R²': r2
#     })
    
#     print(f"RMSE: {rmse:.3f} ml/s")  # ✅ Ahora es ml/segundo
#     print(f"R²: {r2:.4f}")

# # Crear DataFrame de resultados
# df_results = pd.DataFrame(results)
# print("\n" + "="*60)
# print("RESUMEN DE RESULTADOS")
# print("="*60)
# print(df_results.to_string(index=False))

# # Visualizar resultados
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # R² vs Frecuencia
# ax1 = axes[0]
# ax1.plot(df_results['Segundos'], df_results['R²'], marker='o', linewidth=2, markersize=8)
# ax1.set_xlabel('Frecuencia de muestreo (segundos)')
# ax1.set_ylabel('R² Score')
# ax1.set_title('R² vs Frecuencia de Muestreo')
# ax1.grid(True, alpha=0.3)
# ax1.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='R²=0.6 (umbral)')
# ax1.legend()

# # RMSE vs Frecuencia
# ax2 = axes[1]
# ax2.plot(df_results['Segundos'], df_results['RMSE'], marker='o', linewidth=2, markersize=8, color='orange')
# ax2.set_xlabel('Frecuencia de muestreo (segundos)')
# ax2.set_ylabel('RMSE (ml)')
# ax2.set_title('RMSE vs Frecuencia de Muestreo')
# ax2.grid(True, alpha=0.3)

# # Trade-off: Precisión vs Datos
# ax3 = axes[2]
# ax3_twin = ax3.twinx()
# line1 = ax3.plot(df_results['Segundos'], df_results['R²'], marker='o', 
#                  linewidth=2, markersize=8, color='blue', label='R²')
# line2 = ax3_twin.plot(df_results['Segundos'], df_results['Registros'], marker='s', 
#                       linewidth=2, markersize=8, color='green', label='Registros')
# ax3.set_xlabel('Frecuencia de muestreo (segundos)')
# ax3.set_ylabel('R² Score', color='blue')
# ax3_twin.set_ylabel('Número de Registros', color='green')
# ax3.set_title('Trade-off: Precisión vs Cantidad de Datos')
# ax3.tick_params(axis='y', labelcolor='blue')
# ax3_twin.tick_params(axis='y', labelcolor='green')
# ax3.grid(True, alpha=0.3)

# lines = line1 + line2
# labels = [l.get_label() for l in lines]
# ax3.legend(lines, labels, loc='upper left')

# plt.tight_layout()
# plt.savefig('frequency_sensitivity.png', dpi=300, bbox_inches='tight')
# print("\n✅ Gráficos guardados en 'frequency_sensitivity.png'")
# plt.show()

# # Conclusión
# print("\n" + "="*60)
# print("CONCLUSIÓN")
# print("="*60)
# best_tradeoff = df_results.loc[df_results['R²'] > 0.6].iloc[-1]  # Mayor frecuencia con R²>0.6
# print(f"✅ Mejor trade-off: {best_tradeoff['Frecuencia']}")
# print(f"   - R²: {best_tradeoff['R²']:.4f}")
# print(f"   - RMSE: {best_tradeoff['RMSE']:.3f} ml")
# print(f"   - Muestras/día: {best_tradeoff['Muestras/día']:.0f}")
# print(f"   - Reducción de datos vs 2s: {(1 - best_tradeoff['Registros']/df_results.iloc[0]['Registros'])*100:.1f}%")