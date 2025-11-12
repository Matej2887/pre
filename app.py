import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

# --- 1. Definice souborů ---
file_temp = 'mly-0-20000-0-11723-T.csv'
file_srazky = 'mly-0-20000-0-11723-SRA.csv'
file_vitr = 'mly-0-20000-0-11723-F (3).csv'

# Kontrola, jestli soubory existují
files_to_check = [file_temp, file_srazky, file_vitr]
for f in files_to_check:
    if not os.path.exists(f):
        print(f"❌ CHYBA: Soubor '{f}' nebyl nalezen. Ujisti se, že je ve stejné složce jako skript.")
        exit()

print("Všechny soubory nalezeny. Načítám a zpracovávám...")

# --- 2. Načtení a filtrace dat ---

try:
    # Načtení, prázdné hodnoty " " se převedou na NaN
    df_t_raw = pd.read_csv(file_temp, na_values=['""'])
    df_s_raw = pd.read_csv(file_srazky, na_values=['""'])
    df_f_raw = pd.read_csv(file_vitr, na_values=['""'])

    # --- TEPLOTA (T) ---
    # Chceme průměrnou měsíční teplotu = TIMEFUNCTION 'AVG', MDFUNCTION 'AVG'
    df_temp_monthly = df_t_raw[
        (df_t_raw['TIMEFUNCTION'] == 'AVG') & (df_t_raw['MDFUNCTION'] == 'AVG')
    ].copy()
    # Převedeme VALUE na číslo, chyby (NaN) zahodíme
    df_temp_monthly['VALUE'] = pd.to_numeric(df_temp_monthly['VALUE'], errors='coerce')
    # Vytvoříme datum (vždy první den v měsíci)
    df_temp_monthly['date'] = pd.to_datetime(df_temp_monthly['YEAR'].astype(str) + '-' + df_temp_monthly['MONTH'].astype(str).str.zfill(2) + '-01')
    # Vybereme jen potřebné sloupce a přejmenujeme
    df_temp_monthly = df_temp_monthly.set_index('date')[['VALUE']].rename(columns={'VALUE': 'teplota_avg_C'})

    # --- SRÁŽKY (SRA) ---
    # Chceme měsíční úhrn srážek = MDFUNCTION 'SUM'
    df_srazky_monthly = df_s_raw[
        (df_s_raw['MDFUNCTION'] == 'SUM')
    ].copy()
    df_srazky_monthly['VALUE'] = pd.to_numeric(df_srazky_monthly['VALUE'], errors='coerce')
    df_srazky_monthly['date'] = pd.to_datetime(df_srazky_monthly['YEAR'].astype(str) + '-' + df_srazky_monthly['MONTH'].astype(str).str.zfill(2) + '-01')
    df_srazky_monthly = df_srazky_monthly.set_index('date')[['VALUE']].rename(columns={'VALUE': 'srazky_sum_mm'})

    # --- VÍTR (F) ---
    # Chceme průměrnou měsíční rychlost větru = TIMEFUNCTION 'AVG', MDFUNCTION 'AVG'
    df_vitr_monthly = df_f_raw[
        (df_f_raw['TIMEFUNCTION'] == 'AVG') & (df_f_raw['MDFUNCTION'] == 'AVG')
    ].copy()
    df_vitr_monthly['VALUE'] = pd.to_numeric(df_vitr_monthly['VALUE'], errors='coerce')
    df_vitr_monthly['date'] = pd.to_datetime(df_vitr_monthly['YEAR'].astype(str) + '-' + df_vitr_monthly['MONTH'].astype(str).str.zfill(2) + '-01')
    df_vitr_monthly = df_vitr_monthly.set_index('date')[['VALUE']].rename(columns={'VALUE': 'vitr_avg_ms'})

except Exception as e:
    print(f"❌ CHYBA: Nastala chyba při zpracování CSV souborů: {e}")
    print("Zkontroluj formát CSV souborů, jestli odpovídají očekávání.")
    exit()
    
print("Měsíční data zpracována.")

# --- 3. Spojení dat a roční agregace ---
# Spojíme všechny měsíční data do jedné tabulky
df_monthly = pd.concat([df_temp_monthly, df_srazky_monthly, df_vitr_monthly], axis=1)

# Zahodíme měsíce, kde chybí jakýkoliv údaj (pro čistou analýzu)
df_monthly.dropna(inplace=True)

# Agregujeme na ROKY
df_yearly = df_monthly.resample('YE').agg({
    'teplota_avg_C': 'mean',   # Průměrná roční teplota (průměr průměrů)
    'srazky_sum_mm': 'sum',    # Celkový roční úhrn (suma měsíčních sum)
    'vitr_avg_ms': 'mean'      # Průměrná roční rychlost (průměr průměrů)
})
df_yearly['year'] = df_yearly.index.year

# --- 4. Export do Excelu ---
# Tohle je jeden z tvých výstupů
try:
    with pd.ExcelWriter('brno_weather_data_zpracovano.xlsx') as writer:
        df_monthly.to_excel(writer, sheet_name='Mesicni_data_agregovana')
        df_yearly.to_excel(writer, sheet_name='Rocni_data_agregovana')
    print(f"\n✅ ÚSPĚCH: Data exportována do 'brno_weather_data_zpracovano.xlsx'")
except PermissionError:
    print("\n❌ CHYBA: Soubor 'brno_weather_data_zpracovano.xlsx' je otevřený. Zavři ho a spusť skript znovu.")
    exit()
except Exception as e:
    print(f"\n❌ CHYBA: Nepodařilo se uložit Excel: {e}")
    exit()

# --- 5. Grafy historie ---
print("Generuji grafy historie...")

# Teplota
plt.figure(figsize=(12, 6))
plt.plot(df_yearly.index, df_yearly['teplota_avg_C'], marker='o', linestyle='-')
plt.title(f'Vývoj průměrné roční teploty v Brně ({df_yearly["year"].min()}-{df_yearly["year"].max()})')
plt.xlabel('Rok')
plt.ylabel('Průměrná teplota (°C)')
plt.grid(True)
plt.savefig('graf_teplota_historie_csv.png')

# Srážky
plt.figure(figsize=(12, 6))
plt.bar(df_yearly.index, df_yearly['srazky_sum_mm'], width=300)
plt.title(f'Vývoj ročního úhrnu srážek v Brně ({df_yearly["year"].min()}-{df_yearly["year"].max()})')
plt.xlabel('Rok')
plt.ylabel('Roční úhrn srážek (mm)')
plt.grid(axis='y')
plt.savefig('graf_srazky_historie_csv.png')

# Vítr
plt.figure(figsize=(12, 6))
plt.plot(df_yearly.index, df_yearly['vitr_avg_ms'], marker='o', linestyle='-')
plt.title(f'Vývoj průměrné roční rychlosti větru v Brně ({df_yearly["year"].min()}-{df_yearly["year"].max()})')
plt.xlabel('Rok')
plt.ylabel('Průměrná rychlost větru (m/s)')
plt.grid(True)
plt.savefig('graf_vitr_historie_csv.png')

print("Grafy historie uloženy (teplota, srazky, vitr).")

# --- 6. "Predikce" (DEMONSTRACE OMEZENÍ) ---
print("\nProvádím lineární extrapolaci (jako příklad pro diskuzi)...")

# Připravíme data pro model (X = rok)
# Použijeme jen data, kde máme všechny 3 proměnné
df_clean_yearly = df_yearly.dropna()
X = df_clean_yearly[['year']]

# Defice roků pro predikci
posledni_rok = df_clean_yearly['year'].max()
future_years = np.array([
    posledni_rok + 10, 
    posledni_rok + 100, 
    posledni_rok + 1000
]).reshape(-1, 1)

# Funkce pro trénování a predikci
def run_prediction(y_data, variable_name, unit):
    model = LinearRegression()
    model.fit(X, y_data)
    predictions = model.predict(future_years)
    
    print(f"\n--- Naivní extrapolace pro: {variable_name} ---")
    print(f"Predikce pro rok {future_years[0][0]} (+10 let): {predictions[0]:.2f} {unit}")
    print(f"Predikce pro rok {future_years[1][0]} (+100 let): {predictions[1]:.2f} {unit}")
    print(f"Predikce pro rok {future_years[2][0]} (+1000 let): {predictions[2]:.2f} {unit}")
    return model

# Spuštění pro všechny 3 proměnné
model_temp = run_prediction(df_clean_yearly['teplota_avg_C'], "Průměrná teplota", "°C")
model_srazky = run_prediction(df_clean_yearly['srazky_sum_mm'], "Roční úhrn srážek", "mm")
model_vitr = run_prediction(df_clean_yearly['vitr_avg_ms'], "Průměrná rychlost větru", "m/s")
print("-------------------------------------------------")


# --- 7. Graf predikce (ukázka jen pro teplotu) ---
plt.figure(figsize=(12, 6))
# Predikce pro delší časovou osu (do +100 let)
plot_years = np.array(range(X['year'].min(), posledni_rok + 101)).reshape(-1, 1)
trend_line = model_temp.predict(plot_years)

plt.plot(X['year'], df_clean_yearly['teplota_avg_C'], label='Historická data')
plt.plot(plot_years, trend_line, color='red', linestyle='--', label='Lineární extrapolace (100 let)')
plt.title('Naivní lineární extrapolace průměrné teploty')
plt.xlabel('Rok')
plt.ylabel('Průměrná teplota (°C)')
plt.legend()
plt.grid(True)
plt.savefig('graf_predikce_linearni_csv.png')
print("Graf naivní predikce uložen do 'graf_predikce_linearni_csv.png'")

print("\nHotovo. Opět připomínám: Nezapomeň v PDF rozebrat, proč je ta predikce na 100+ let nesmysl!")
