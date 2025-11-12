import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io

# --- KROK 1: Vložení vašeho analytického skriptu do cachované funkce ---
# Toto zajistí, že se celý výpočet (včetně generování reportu)
# provede pouze jednou, a ne při každé interakci s aplikací.

@st.cache_data
def load_and_process_data():
    """
    Tato funkce obsahuje celý váš analytický skript.
    Spustí se jen jednou a vrátí klíčové výsledky.
    """
    
    # --- 1. Data Acquisition and Preprocessing ---
    # print('--- Krok 1: Zpracování a čištění dat ---\n') # Printy skryjeme pro Streamlit

    file_name = 'weather_data.csv'
    try:
        df = pd.read_csv(file_name)
        # print(f"Úspěšně načten soubor '{file_name}'.")
    except FileNotFoundError:
        # print(f"Chyba: Soubor '{file_name}' nebyl nalezen.")
        # print("Vytvářím fiktivní DataFrame pro demonstrační účely.")
        dates = pd.to_datetime(pd.date_range(start='2000-01-01', periods=100, freq='D'))
        np.random.seed(42)
        dummy_data = {
            'Date': dates,
            'Temperature': np.random.uniform(low=-10, high=35, size=100),
            'WindSpeed': np.random.uniform(low=0, high=30, size=100),
            'Precipitation': np.random.uniform(low=0, high=50, size=100)
        }
        df = pd.DataFrame(dummy_data)

    # Convert 'Date' column to datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        # print('Sloupec "Date" převeden na datetime a nastaven jako index.')

    # Ensure the index has frequency information
    df.index = pd.to_datetime(df.index)
    if df.index.freq is None:
        df = df.asfreq('D') # Set frequency to daily
        # print('Frekvence indexu nastavena na denní (D).')

    # Handle missing values (ffill then bfill)
    df = df.ffill().bfill()
    # print('Chybějící hodnoty ošetřeny pomocí ffill/bfill.')

    # Outlier Detection and Handling (IQR method - capping)
    columns_to_check = ['Temperature', 'WindSpeed', 'Precipitation']
    for col in columns_to_check:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    # print('Odlehlé hodnoty ošetřeny metodou IQR (zastropování).')

    # print('\n--- Krok 2: Vývoj predikčního modelu a generování předpovědí ---\n')

    # --- 2. Model Development and Predictions ---
    weather_variables = ['Temperature', 'WindSpeed', 'Precipitation']
    es_models = {}
    forecast_results = {}
    long_term_forecast_results = {
        '100_years': {},
        '1000_years': {}
    }
    long_term_forecast_periods = {
        '10_years': 365 * 10,
        '100_years': 365 * 100,
        '1000_years': 365 * 1000
    }

    for col in weather_variables:
        # print(f'Zpracovávám proměnnou: {col}...')
        series_to_forecast = df[col]

        # Fit ExponentialSmoothing model
        model = sm.tsa.ExponentialSmoothing(
            series_to_forecast,
            trend='add',
            seasonal='add',
            seasonal_periods=7, # Assuming weekly seasonality for short dummy data
            initialization_method="estimated"
        )
        es_fit = model.fit()
        es_models[col] = es_fit

        # Generate 10-year forecast
        forecast_periods_days_10y = long_term_forecast_periods['10_years']
        start_date_forecast_10y = series_to_forecast.index[-1] + pd.Timedelta(days=1)
        end_date_forecast_10y = start_date_forecast_10y + pd.Timedelta(days=forecast_periods_days_10y - 1)
        forecast_index_10y = pd.date_range(start=start_date_forecast_10y, end=end_date_forecast_10y, freq='D')
        forecast_10y = es_fit.forecast(steps=forecast_periods_days_10y)
        forecast_10y.index = forecast_index_10y
        forecast_results[col] = forecast_10y
        # print(f'  - 10letá předpověď pro {col} vygenerována.')

        # Generate 100-year forecast
        forecast_periods_days_100y = long_term_forecast_periods['100_years']
        start_date_forecast_100y = series_to_forecast.index[-1] + pd.Timedelta(days=1)
        end_date_forecast_100y = start_date_forecast_100y + pd.Timedelta(days=forecast_periods_days_100y - 1)
        forecast_index_100y = pd.date_range(start=start_date_forecast_100y, end=end_date_forecast_100y, freq='D')
        forecast_100y = es_fit.forecast(steps=forecast_periods_days_100y)
        forecast_100y.index = forecast_index_100y
        long_term_forecast_results['100_years'][col] = forecast_100y
        # print(f'  - 100letá předpověď pro {col} vygenerována.')

        # Generate 1000-year simulated forecast (due to Timestamp overflow limitation)
        forecast_periods_days_1000y = long_term_forecast_periods['1000_years']
        average_historical_value = series_to_forecast.mean()
        synthetic_forecast_values_daily_1000y = np.full(forecast_periods_days_1000y, average_historical_value)
        long_term_forecast_results['1000_years'][col] = synthetic_forecast_values_daily_1000y
        # print(f'  - 1000letá simulovaná předpověď (historický průměr) pro {col} vygenerována.')


    # --- 3. Quantify Uncertainties (Approximated Prediction Intervals) ---
    # (Tato část není v UI přímo použita, ale je nutná pro report)
    # print('\n--- Krok 3: Kvantifikace nejistot (přibližné predikční intervaly) ---\n')

    def get_approx_prediction_intervals(model_fit, point_forecast_series, historical_df_col):
        resid_std = 0.0
        if hasattr(model_fit, 'resid') and model_fit.resid is not None and len(model_fit.resid) > 1:
            resid_std = model_fit.resid.std()
        if resid_std == 0.0 or np.isnan(resid_std):
            resid_std = historical_df_col.std()
            if resid_std == 0.0 or np.isnan(resid_std):
                resid_std = 0.1
        lower_bound = point_forecast_series - 1.96 * resid_std
        upper_bound = point_forecast_series + 1.96 * resid_std
        pred_int_df = pd.DataFrame({
            'mean': point_forecast_series,
            'mean_ci_lower': lower_bound,
            'mean_ci_upper': upper_bound
        }, index=point_forecast_series.index)
        return pred_int_df

    prediction_intervals_10y = {}
    prediction_intervals_100y = {}
    for col in weather_variables:
        if col in es_models and col in forecast_results:
            model_fit = es_models[col]
            prediction_intervals_10y[col] = get_approx_prediction_intervals(model_fit, forecast_results[col], df[col])
        if col in es_models and col in long_term_forecast_results['100_years']:
            model_fit = es_models[col]
            prediction_intervals_100y[col] = get_approx_prediction_intervals(model_fit, long_term_forecast_results['100_years'][col], df[col])

    # --- 4. Generování reportu (Markdown) ---
    # print('\n--- Krok 4: Generování reportu (Markdown) ---\n')

    # Výpočet ročních průměrů pro 1000letou předpověď (nutné pro f-string)
    yearly_avg_temp = pd.Series(np.array([long_term_forecast_results['1000_years']['Temperature'][i*365 : (i+1)*365].mean() for i in range(long_term_forecast_periods['1000_years'] // 365)]))
    yearly_avg_wind = pd.Series(np.array([long_term_forecast_results['1000_years']['WindSpeed'][i*365 : (i+1)*365].mean() for i in range(long_term_forecast_periods['1000_years'] // 365)]))
    yearly_avg_precip = pd.Series(np.array([long_term_forecast_results['1000_years']['Precipitation'][i*365 : (i+1)*365].mean() for i in range(long_term_forecast_periods['1000_years'] // 365)]))

    # Celý váš Markdown report jako f-string
    report_content_markdown = f"""
# Předpověď počasí pro Brno: 10, 100 a 1000 let

## Shrnutí: Klíčová zjištění analýzy dat

* **Získávání dat**: Přímé automatické získávání historických dat o počasí pro Brno nebylo možné, proto byl pro demonstrační účely použit 100denní fiktivní datový soubor. Proces nastínil manuální kroky pro získání reálných dat uživateli.
* **Předzpracování a čištění dat**: Fiktivní datový soubor prošel čištěním, včetně řešení chybějících hodnot pomocí dopředného a zpětného vyplňování a detekce/omezení odlehlých hodnot metodou mezikvartilního rozpětí (IQR). Sloupec 'Date' byl správně formátován jako index typu datetime.
* **Explorační analýza dat (EDA)**: Byla provedena komplexní EDA fiktivních dat, vizualizace historických trendů, měsíčních průměrů, distribucí pomocí histogramů a identifikace potenciálních odlehlých hodnot pomocí box plotů pro teplotu, rychlost větru a srážky.
* **Výzvy při vývoji modelu**: Počáteční pokusy o použití knihovny `Prophet` selhaly kvůli přetrvávající chybě `AttributeError` (související se `stan_backend`), což si vyžádalo přechod na `statsmodels.tsa.ExponentialSmoothing`.
* **10letá předpověď (Exponential Smoothing)**:
    * **Teplota**: Předpovídán průměr 26.4°C (rozsah: 7.0°C až 45.5°C), což ukazuje silný vzestupný trend oproti historickému průměru.
    * **Rychlost větru**: Předpovídán průměr 54.1 m/s (rozsah: 10.7 m/s až 94.6 m/s), což také naznačuje významný vzestupný trend.
    * **Srážky**: Předpovídán průměr -833.1 mm (minimum: -1689.5 mm), což je fyzicky nerealistické a zdůrazňuje kritické omezení neomezeného aditivního modelu pro tuto proměnnou v tomto horizontu.
* **100letá předpověď (Exponential Smoothing)**: Extrapolace aditivního trendu a sezónnosti vedla k fyzicky nemožným předpovědím:
    * **Teplota**: Předpovídán průměr 159.3°C (maximum: 311.3°C).
    * **Rychlost větru**: Předpovídán průměr 397.3 m/s (maximum: 781.1 m/s).
    * **Srážky**: Předpovídán průměr -8506.5 mm (minimum: -17036.8 mm), což dále zdůrazňuje nedostatečnost modelu.
* **1000letá předpověď (simulovaná)**: Kvůli chybám přetečení `pd.Timestamp` a inherentním omezením `ExponentialSmoothing` pro takto dlouhé horizonty byla použita simulovaná předpověď s využitím historického průměru. Všechny proměnné se stabilizovaly na svých historických průměrech (např. Teplota ~ 11.16°C, rychlost větru ~ 14.93 m/s, srážky ~ 25.88 mm) s zanedbatelnou odchylkou, sloužící spíše jako zástupné hodnoty než jako skutečné předpovědi.
* **Kvantifikace nejistoty**: Pro 10leté a 100leté předpovědi byly vygenerovány přibližné 95% predikční intervaly založené na standardní odchylce reziduí, přičemž se uznává, že tato metoda pravděpodobně podceňuje skutečnou nejistotu kvůli rostoucí chybě na delších horizontech a předpokladům konstantní variance a normální distribuce chyb.
* **Omezení současného přístupu**: Analýza zdůraznila, že jednoduché statistické modely časových řad jsou zásadně nedostatečné pro robustní, fyzicky realistické dlouhodobé (100 až 1000 let) klimatické předpovědi bez doplnění o doménově specifické znalosti, fyzická omezení nebo nahrazení komplexními, fyzikálně založenými klimatickými modely.
* **Generování zprávy**: Byla úspěšně vygenerována komplexní zpráva ve formátu Markdown, shrnující celý proces, včetně analýzy historických dat, předzpracování, vývoje modelu, kvantifikovaných předpovědí s inherentními omezeními a podrobné diskuse o nejistotách a předpokladech.

### Závěry a další kroky

* **Využití reálných historických dat**: Získat a integrovat komplexní, vícedecenní historická data o počasí pro Brno, aby bylo možné smysluplnější a robustnější trénování a validaci modelu, čímž se překonají omezení fiktivních dat.
* **Použití pokročilých klimatických modelů pro dlouhodobé předpovědi**: Pro předpovědi přesahující 100 nebo 1000 let přejít od statistických modelů časových řad k fyzikálně založeným klimatickým modelům (např. modely všeobecné cirkulace), které mohou zahrnovat klimatické síly, fyzická omezení a poskytovat řadu věrohodných budoucích scénářů (např. scénáře IPCC SSP), namísto spoléhání se na neomezené extrapolace.

## Data Acquisition and Preprocessing

Given that direct historical weather data for Brno was not available for automated download, a **dummy dataset** was generated. This synthetic dataset allowed for the demonstration of data preprocessing and model development steps.

Key preprocessing steps included:
- **Missing Value Handling:** Missing values, if present in a real dataset, were addressed using a combination of forward-fill (`ffill()`) and backward-fill (`bfill()`) methods. This strategy assumes that missing data can be reasonably imputed from adjacent observations, which is common for time series data.
- **Outlier Detection and Handling:** Outliers were identified using the **Interquartile Range (IQR) method**. Data points falling outside 1.5 times the IQR from the first (Q1) and third (Q3) quartiles were considered outliers. These outliers were then **capped** at their respective lower or upper bounds to mitigate their undue influence on the model without removing data points entirely. For the dummy data, no outliers were detected after generation.
- **Data Consistency:** The 'Date' column was converted to datetime objects and set as the DataFrame index, ensuring proper time series functionality and frequency setting (daily, 'D'). This step ensures that time-based operations and seasonal analyses are performed correctly.

## Model Development

For long-term forecasting, two primary approaches were considered: `Prophet` and `statsmodels.tsa.ExponentialSmoothing`.

### Initial Attempt with Prophet:
The initial plan was to use Facebook's Prophet library due to its robustness with seasonality and holiday effects. However, an `AttributeError` related to `stan_backend` prevented its successful implementation. This issue typically arises from underlying dependency conflicts or environmental setup challenges with Prophet's C++ backend (Stan). Despite attempts to mitigate this by setting `mcmc_samples=0`, the error persisted, leading to the decision to pivot to an alternative model.

### Chosen Model: statsmodels.tsa.ExponentialSmoothing:
Given the technical difficulties with Prophet, `statsmodels.tsa.ExponentialSmoothing` was selected. This model is suitable for time series data exhibiting both trend and seasonality. Separate models were fitted for each weather variable: Temperature, Wind Speed, and Precipitation.
#### Model Configuration:
- **Trend Component:** An **additive trend** (`trend='add'`) was used. This assumes a linear increase or decrease over time. While simple, it can lead to unrealistic extrapolations over very long horizons if unconstrained.
- **Seasonal Component:** An **additive seasonality** (`seasonal='add'`) was applied. This implies that the seasonal fluctuations have a consistent magnitude irrespective of the series' overall level.
- **Seasonal Period:** For the short 100-day dummy dataset, a **weekly seasonal period** (`seasonal_periods=7`) was used to capture any potential weekly patterns. For a real, multi-year dataset, a yearly seasonality (`seasonal_periods=365`) would be more appropriate.
- **Initialization Method:** `initialization_method='estimated'` was used to allow the model to estimate optimal initial values.

This configuration allowed for the generation of 10-year and 100-year forecasts, with a special handling for the 1000-year horizon due to technical limitations as detailed later.
## Quantified Predictions

### 10-Year Forecast

The 10-year forecast, generated using the Exponential Smoothing model, shows the following characteristics:
- **Temperature:** Mean={forecast_results['Temperature'].describe()['mean']:.2f} (±{forecast_results['Temperature'].describe()['std']:.2f}), Range=[{forecast_results['Temperature'].describe()['min']:.2f}, {forecast_results['Temperature'].describe()['max']:.2f}].
  * The model predicts a significant upward trend for temperature, reaching higher values than historically observed.
- **WindSpeed:** Mean={forecast_results['WindSpeed'].describe()['mean']:.2f} (±{forecast_results['WindSpeed'].describe()['std']:.2f}), Range=[{forecast_results['WindSpeed'].describe()['min']:.2f}, {forecast_results['WindSpeed'].describe()['max']:.2f}].
  * Similarly, wind speeds show an increasing trend over the decade.
- **Precipitation:** Mean={forecast_results['Precipitation'].describe()['mean']:.2f} (±{forecast_results['Precipitation'].describe()['std']:.2f}), Range=[{forecast_results['Precipitation'].describe()['min']:.2f}, {forecast_results['Precipitation'].describe()['max']:.2f}].
  * Precipitation forecasts show physically unrealistic negative values, indicating the model's limitations for this variable over this horizon.

### 100-Year Forecast

Extrapolating the Exponential Smoothing model for 100 years reveals amplified trends:
- **Temperature:** Mean={long_term_forecast_results['100_years']['Temperature'].describe()['mean']:.2f} (±{long_term_forecast_results['100_years']['Temperature'].describe()['std']:.2f}), Range=[{long_term_forecast_results['100_years']['Temperature'].describe()['min']:.2f}, {long_term_forecast_results['100_years']['Temperature'].describe()['max']:.2f}].
  * Temperatures reach astronomically high and physically impossible values due to the unconstrained additive trend.
- **WindSpeed:** Mean={long_term_forecast_results['100_years']['WindSpeed'].describe()['mean']:.2f} (±{long_term_forecast_results['100_years']['WindSpeed'].describe()['std']:.2f}), Range=[{long_term_forecast_results['100_years']['WindSpeed'].describe()['min']:.2f}, {long_term_forecast_results['100_years']['WindSpeed'].describe()['max']:.2f}].
  * Wind speeds also escalate to highly unrealistic and impossible levels.
- **Precipitation:** Mean={long_term_forecast_results['100_years']['Precipitation'].describe()['mean']:.2f} (±{long_term_forecast_results['100_years']['Precipitation'].describe()['std']:.2f}), Range=[{long_term_forecast_results['100_years']['Precipitation'].describe()['min']:.2f}, {long_term_forecast_results['100_years']['Precipitation'].describe()['max']:.2f}].
  * Precipitation values are deeply negative, further highlighting the model's unsuitability for long-term unconstrained predictions.

### 1000-Year Forecast

For the 1000-year horizon, technical limitations (Timestamp overflow) prevented direct daily forecasting with `statsmodels`. Therefore, a **simulated forecast** based on the historical mean was used for visualization, with yearly averages presented:
- **Temperature (Yearly Averages):** Mean={yearly_avg_temp.describe()['mean']:.2f} (±{yearly_avg_temp.describe()['std']:.2e}), Range=[{yearly_avg_temp.describe()['min']:.2f}, {yearly_avg_temp.describe()['max']:.2f}].
  * These forecasts revert to the historical mean with negligible variance, serving as a placeholder due to model limitations for such extreme horizons.
- **WindSpeed (Yearly Averages):** Mean={yearly_avg_wind.describe()['mean']:.2f} (±{yearly_avg_wind.describe()['std']:.2e}), Range=[{yearly_avg_wind.describe()['min']:.2f}, {yearly_avg_wind.describe()['max']:.2f}].
  * These forecasts revert to the historical mean with negligible variance, serving as a placeholder due to model limitations for such extreme horizons.
- **Precipitation (Yearly Averages):** Mean={yearly_avg_precip.describe()['mean']:.2f} (±{yearly_avg_precip.describe()['std']:.2e}), Range=[{yearly_avg_precip.describe()['min']:.2f}, {yearly_avg_precip.describe()['max']:.2f}].
  * These forecasts revert to the historical mean with negligible variance, serving as a placeholder due to model limitations for such extreme horizons.

## Nejistoty, předpoklady a omezení

#### Nejistota v 1000letých předpovědích
1000leté předpovědi teploty, rychlosti větru a srážek byly zjednodušeny na historický průměr fiktivních dat kvůli technickým omezením. Konkrétně `statsmodels.ExponentialSmoothing` není navržen pro předpovědní horizonty tak extrémní, jako je 1000 let s denní granularitou, což vede k chybám přetečení `Timestamp` při pokusu o vytvoření denního `DateTimeIndex` pro takto dlouhé období. V důsledku toho nebylo možné generovat formální predikční intervaly, které se opírají o strukturu chyb modelu. Pro předpovědní horizont tisíciletí je nejistota obrovská a nelze ji kvantifikovat jednoduchými statistickými modely časových řad.

Konceptuální povaha nejistoty pro takto extrémní horizonty znamená, že jakákoli jednotlivá předpovědní hodnota je vysoce spekulativní. Budoucí klimatická dynamika je ovlivněna nesčetnými složitými, nelineárními interakcemi, vnějšími vlivy (např. sluneční aktivita, sopečné erupce) a změnami způsobenými člověkem (např. emise skleníkových plynů, změny ve využívání půdy), které statistický model založený pouze na historických vzorcích nemůže zachytit. Proto jsou 1000leté předpovědi pouze ilustrativní, odrážejí pouze průměr krátkých historických fiktivních dat a nepředstavují vědecky robustní dlouhodobou klimatickou projekci.

#### Učiněné předpoklady
Během zpracování dat a vývoje modelu byly učiněny následující předpoklady:

* **Předzpracování dat:**
    * **Chybějící hodnoty:** Předpokládalo se, že dopředné vyplňování (`ffill`) následované zpětným vyplňováním (`bfill`) je vhodnou strategií pro zpracování chybějících hodnot. To předpokládá, že chybějící datové body jsou nejlépe aproximovány nejnovějšími nebo nejbližšími dostupnými daty, což nemusí platit pro všechny typy meteorologických dat nebo pro dlouhé mezery.
    * **Odlehlé hodnoty:** Pro detekci odlehlých hodnot byla použita metoda **mezikvartilního rozpětí (IQR)** s násobitelem 1,5x IQR a odlehlé hodnoty byly **zastropovány** na jejich dolních nebo horních mezích. To předpokládá, že extrémní hodnoty jsou buď chyby, nebo že jejich dopad by měl být zmírněn jejich omezením, spíše než aby byly považovány za skutečné, byť vzácné, události nebo zcela odstraněny.
    * **Frekvence dat:** Předpokládalo se, že data mají denní frekvenci ('D'), která byla explicitně nastavena pro index DataFrame.

* **Vývoj modelu (ExponentialSmoothing):**
    * **Výběr modelu:** `ExponentialSmoothing` byl vybrán jako robustní univerzální model časových řad, schopný zachytit trend i sezónnost. To předpokládá, že budoucí vzorce budou obecně následovat prodloužení minulých vzorců.
    * **Trendová složka:** Byla předpokládána **aditivní trendová složka** (`trend='add'`). To znamená, že trendová složka přidává konstantní množství k předpovědi v každém období, což může vést k neomezenému a nerealistickému růstu nebo poklesu v dlouhých obdobích.
    * **Sezónní složka:** Byla použita **aditivní sezónnost** (`seasonal='add'`). To znamená, že sezónní výkyvy mají konstantní velikost bez ohledu na úroveň řady.
    * **Sezónní období:** Pro krátký 100denní fiktivní datový soubor bylo použito týdenní sezónní období (`seasonal_periods=7`). Pro reálná, víceletá meteorologická data by bylo typicky vhodnější roční sezónní období (`seasonal_periods=365`).
    * **Distribuce chyb:** Výpočet přibližných 95% predikčních intervalů předpokládá, že chyby předpovědi jsou normálně rozděleny a mají konstantní rozptyl, což je v reálných časových řadách často porušeno, zejména u delších horizontů, kde se nejistota typicky zvyšuje.

#### Zjištěná omezení
Během procesu bylo zjištěno několik významných omezení:

* **Chyba knihovny Prophet:** Chyba `AttributeError: 'Prophet' object has no attribute 'stan_backend'` zabránila použití modelu `Prophet`. To často poukazuje na problémy s prostředím nebo závislostmi, které nemohly být vyřešeny v rámci provádění notebooku, což si vynutilo přechod na `statsmodels.ExponentialSmoothing`.
* **`statsmodels.ExponentialSmoothing` pro dlouhodobé předpovědi:**
    * **Fyzicky nerealistické předpovědi:** Aditivní trendová složka v modelu `ExponentialSmoothing`, při extrapolaci na 10leté a zejména 100leté horizonty, vedla k fyzicky nerealistickým předpovědím. Například:
        * **Teplota a rychlost větru:** Hodnoty se staly nadměrně vysokými (např. teploty nad 150°C, rychlosti větru blízké rychlosti zvuku), což je nemožné.
        * **Srážky:** Předpovědi srážek se staly významně zápornými, což je fyzicky nemožné.
    * **Nedostatek omezení:** Model postrádá mechanismy pro uvalení fyzických omezení (např. teplotní meze, nezáporné srážky) během předpovědi, což jej činí nevhodným pro neomezenou dlouhodobou extrapolaci bez ručního dodatečného zpracování nebo sofistikovanějšího návrhu modelu.
* **Přetečení `pd.Timestamp` pro 1000leté `date_range`:** Pokus o vytvoření denního `pd.date_range` nebo objektů `pd.Timestamp` pro 1000leté období vedl k chybě přetečení. To naznačuje, že `DateTimeIndex` v knihovně pandas (který používá rozlišení nanosekund) má omezení pro extrémně vzdálená budoucí data.
* **Přibližné predikční intervaly:** Vzhledem k tomu, že API `statsmodels` přímo neposkytuje metodu `get_prediction` se standardním parametrem `alpha` pro `HoltWintersResultsWrapper` nebo `predict` s `return_everything=True`, musely být 95% predikční intervaly aproximovány pomocí směrodatné odchylky reziduí. Tato metoda je zjednodušení a obecně podceňuje skutečnou nejistotu, zejména s rostoucím predikčním horizontem.
* **Krátký fiktivní datový soubor:** Spoléhání se na 100denní fiktivní datový soubor významně omezilo schopnost robustně přizpůsobit modely s dlouhými sezónními obdobími (jako je roční sezónnost) a získat smysluplné dlouhodobé předpovědi. Reálná klimatická data se typicky rozprostírají přes mnoho desetiletí až staletí.

#### Dopad omezení na spolehlivost a interpretaci

Omezení vážně ovlivňují spolehlivost a interpretaci dlouhodobých předpovědí:

* **10letá předpověď:** Zatímco vykazuje věrohodné trendy, předpověď srážek stále ukazuje neplatné záporné hodnoty, což naznačuje, že i pro desetiletí se aditivní předpoklady modelu mohou pro určité proměnné zhroutit.
* **100letá předpověď:** Tyto předpovědi jsou vysoce nespolehlivé a fyzicky nemožné pro teplotu, rychlost větru a srážky. Jasně ukazují, že jednoduché statistické modely navržené pro krátkodobé až střednědobé předpovědi jsou zcela nedostatečné pro stoleté klimatické projekce bez zásadních úprav nebo integrace s fyzikálně založenou klimatologií.
* **1000letá předpověď:** Explicitní zjednodušení na historický průměr kvůli technickým omezením zbavuje tyto předpovědi skutečné prediktivní síly nad rámec odrazu průměru historického vstupu. Slouží spíše jako zástupný symbol, zdůrazňující extrémní obtížnost a potřebu zcela odlišných modelovacích paradigmat (např. modely systému Země) pro takové časové měřítko. Absence řádné kvantifikace nejistoty dále snižuje jejich užitečnost.

Závěrem, zatímco statistické modely časových řad, jako je Exponential Smoothing, mohou být užitečné pro krátkodobé předpovědi a identifikaci vzorců, jsou zásadně nedostatečné pro robustní a fyzicky realistické dlouhodobé (100 až 1000 let) klimatické předpovědi, pokud nejsou doplněny o znalosti specifické pro danou oblast, fyzická omezení nebo nahrazeny komplexními, fyzikálně založenými klimatickými modely.
"""

    # Vracíme klíčové objekty, které budeme potřebovat v UI
    return df, forecast_results, long_term_forecast_results, report_content_markdown

# --- Pomocná funkce pro generování PDF ---
def create_pdf_bytes(text):
    """
    Vytvoří PDF z textového řetězce a vrátí ho jako bytes.
    Používá základní kódování 'latin-1' s nahrazením znaků,
    aby se zabránilo chybám s českou diakritikou, která není v základu FPDF.
    Výsledné PDF nemusí zobrazit diakritiku správně, ale nespadne.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Převede text na kódování, které FPDF zvládne, nahradí neznámé znaky
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.multi_cell(0, 5, safe_text)
    return pdf.output() # Vrátí data jako bytes


# --- KROK 2: Vytvoření samotné Streamlit aplikace ---

st.set_page_config(page_title="Předpověď počasí", layout="wide")
st.title("Předpověď počasí pro Brno")

# Načtení dat (díky cache se to provede rychle)
try:
    with st.spinner("Provádím analýzu dat a trénuji modely..."):
        df_hist, fc_10y, fc_long, report_md = load_and_process_data()

    st.success("Analýza dokončena!")

    # --- Sekce 1: Interaktivní graf teplot ---
    st.header("📈 Interaktivní průzkum teplot (10letá předpověď)")

    # Spojení historických dat a 10leté předpovědi pro graf
    temp_hist = df_hist['Temperature']
    temp_fc = fc_10y['Temperature']
    full_temp_series = pd.concat([temp_hist, temp_fc])
    full_temp_series.name = "Teplota"

    # Výběr data
    min_date = full_temp_series.index.min().date()
    max_date = full_temp_series.index.max().date()
    
    # Výchozí rozsah: poslední rok historie + první 2 roky předpovědi
    default_start = temp_hist.index.max().date() - pd.DateOffset(years=1)
    default_end = temp_hist.index.max().date() + pd.DateOffset(years=2)

    selected_dates = st.date_input(
        "Vyberte časové období:",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date
    )

    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
        
        # Filtrování dat podle výběru
        filtered_data = full_temp_series.loc[start_date:end_date]
        
        st.subheader(f"Vývoj teploty od {start_date} do {end_date}")
        
        # Zobrazení grafu
        st.line_chart(filtered_data)
        
        # Zobrazení surových dat
        with st.expander("Zobrazit surová data pro vybrané období"):
            st.dataframe(filtered_data.to_frame())
    else:
        st.warning("Prosím, vyberte počáteční i konečné datum.")

    
    # --- Sekce 2: PDF Report ---
    st.markdown("---")
    st.header("📄 Závěrečný report a stažení PDF")

    # Vygenerování PDF v paměti
    pdf_data = create_pdf_bytes(report_md)
    
    # Tlačítko ke stažení
    st.download_button(
        label="Stáhnout kompletní report jako PDF",
        data=pdf_data,
        file_name="predpoved_pocasi_report.pdf",
        mime="application/pdf"
    )

    st.info("""
    **Poznámka:** Vygenerované PDF obsahuje surový text zprávy. Kvůli omezením základní knihovny FPDF 
    nemusí být česká diakritika v PDF souboru zobrazena správně. 
    Pro nejlepší zobrazení si pročtěte náhled reportu přímo zde v aplikaci.
    """)

    # Zobrazení náhledu Markdown reportu
    with st.expander("Zobrazit náhled reportu v aplikaci", expanded=True):
        st.markdown(report_md)

except Exception as e:
    st.error(f"Došlo k chybě při zpracování dat: {e}")
    st.exception(e)
