import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from io import BytesIO
from fpdf import FPDF
import os.path
from functools import reduce # Import pro jiný styl slučování

# --- Konfigurace ---
# Názvy souborů
TEMP_FILE = "mly-0-20000-0-11723-T.csv"
WIND_FILE = "mly-0-20000-0-11723-F.csv"
PRECIP_FILE = "mly-0-20000-0-11723-SRA.csv"

# Fonty pro PDF
FONT_NORMAL = "DejaVuSans.ttf"
FONT_BOLD = "DejaVuSans-Bold.ttf"

# Definice metrik pro grafy a PDF (přesunuto nahoru)
METRIC_DEFINITIONS = {
    'avg_temp': {'unit': '°C', 'label': 'Průměrná teplota'},
    'avg_wind': {'unit': 'm/s', 'label': 'Průměrná rychlost větru'},
    'sum_precip': {'unit': 'mm', 'label': 'Celkové roční srážky'}
}

# Ignorování varování
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# --- Datové Funkce ---

@st.cache_data
def load_and_filter_csv(file_path, time_filter, md_filter, output_col_name):
    """Načte a vyfiltruje jeden datový soubor."""
    try:
        raw_data = pd.read_csv(
            file_path, 
            usecols=['YEAR', 'MONTH', 'TIMEFUNCTION', 'MDFUNCTION', 'VALUE']
        )
        filtered_data = raw_data[
            (raw_data['TIMEFUNCTION'] == time_filter) & 
            (raw_data['MDFUNCTION'] == md_filter)
        ].copy()
        
        selected_data = filtered_data[['YEAR', 'MONTH', 'VALUE']]
        selected_data = selected_data.rename(columns={'VALUE': output_col_name})
        selected_data[output_col_name] = pd.to_numeric(selected_data[output_col_name], errors='coerce')
        return selected_data
        
    except FileNotFoundError:
        st.error(f"Chyba: Soubor '{file_path}' nebyl nalezen. Ujistěte se, že je přítomen v repozitáři.")
        return None
    except Exception as e:
        st.error(f"Neočekávaná chyba při zpracování '{file_path}': {e}")
        return None

@st.cache_data
def get_processed_data_and_models():
    """Hlavní funkce pro načtení dat, jejich zpracování a trénink modelů."""
    with st.spinner("Probíhá načítání a zpracování dat..."):
        
        temp_data = load_and_filter_csv(TEMP_FILE, 'AVG', 'AVG', 'avg_temp')
        wind_data = load_and_filter_csv(WIND_FILE, 'AVG', 'AVG', 'avg_wind')
        precip_data = load_and_filter_csv(PRECIP_FILE, '07:00', 'SUM', 'sum_precip')

        dataframes_to_merge = [temp_data, wind_data, precip_data]
        
        if any(df is None for df in dataframes_to_merge):
            st.error("Chyba při načítání jednoho nebo více datových souborů. Zkontrolujte logy.")
            return None, None, None, None

        # Slučování dat pomocí 'reduce' (jiný styl než původní kód)
        merged_monthly_data = reduce(lambda left, right: pd.merge(left, right, on=['YEAR', 'MONTH'], how='outer'), dataframes_to_merge)

        # Filtrace pouze kompletních roků
        year_completeness = merged_monthly_data.dropna().groupby('YEAR').size().reset_index(name='month_count')
        valid_years = year_completeness[year_completeness['month_count'] == 12]['YEAR']
        complete_monthly_data = merged_monthly_data[merged_monthly_data['YEAR'].isin(valid_years)]

        if complete_monthly_data.empty:
            st.error("Po filtraci na kompletní roky nezbyla žádná data.")
            return None, None, None, None

        # Agregace na roční data
        annual_summary = complete_monthly_data.groupby('YEAR').agg(
            avg_temp=('avg_temp', 'mean'),
            avg_wind=('avg_wind', 'mean'),
            sum_precip=('sum_precip', 'sum')
        ).reset_index().dropna()

        if annual_summary.empty:
            st.error("Nepodařilo se vytvořit roční souhrn.")
            return None, None, None, None

        # Trénink modelů
        regression_models = {}
        trend_analysis = {}
        years_array = annual_summary['YEAR'].values.reshape(-1, 1)
        
        metrics_to_regress = METRIC_DEFINITIONS.keys()

        for metric in metrics_to_regress:
            values_array = annual_summary[metric].values
            model = LinearRegression()
            model.fit(years_array, values_array)
            
            regression_models[metric] = model
            trend_analysis[metric] = {'slope': model.coef_[0], 'intercept': model.intercept_}
            
            # Přidání sloupce s trendem pro vykreslení
            annual_summary[f'{metric}_trend'] = model.predict(years_array)
            
        st.success("Data byla úspěšně zpracována a modely natrénovány.")
        return annual_summary, trend_analysis, regression_models, merged_monthly_data

# --- Funkce pro PDF a Grafy ---

def generate_trend_plot(metric, metric_info, annual_data, future_data, trends):
    """Vytvoří Matplotlib graf a vrátí ho jako in-memory buffer."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Historická data
    ax.scatter(annual_data['YEAR'], annual_data[metric], label=f'Roční data ({metric_info["label"]})', alpha=0.7, s=10)
    
    # Historický trend
    ax.plot(annual_data['YEAR'], annual_data[f'{metric}_trend'], color='red', linestyle='--', label=f'Lineární trend ({trends[metric]["slope"]:.4f} {metric_info["unit"]}/rok)')
    
    # Spojnice na predikci
    last_year_data = annual_data['YEAR'].max()
    last_val_data = annual_data.loc[annual_data['YEAR'] == last_year_data, f'{metric}_trend'].values[0]
    first_pred_year = future_data.index.min()
    first_pred_val = future_data.loc[first_pred_year, f'pred_{metric}']
    
    ax.plot([last_year_data, first_pred_year], [last_val_data, first_pred_val], color='red', linestyle=':', label='Extrapolace')
    
    # Budoucí predikované body
    ax.plot(future_data.index, future_data[f'pred_{metric}'], color='red', marker='o', linestyle=':', markersize=5)
    
    ax.set_title(f'Historický vývoj a extrapolace - {metric_info["label"]} (Brno)')
    ax.set_xlabel('Rok')
    ax.set_ylabel(f'{metric_info["label"]} ({metric_info["unit"]})')
    ax.legend()
    ax.grid(True)
    
    plot_stream = BytesIO()
    fig.savefig(plot_stream, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig) 
    plot_stream.seek(0)
    return plot_stream

def create_report_document(annual_data, trends, models, future_projections, metric_info_dict):
    """Sestaví kompletní PDF report."""
    
    if not os.path.isfile(FONT_NORMAL) or not os.path.isfile(FONT_BOLD):
        st.error(f"Kritická chyba: Chybí soubory fontů '{FONT_NORMAL}' nebo '{FONT_BOLD}'. PDF nelze vygenerovat.")
        return None

    try:
        report = FPDF(orientation='P', unit='mm', format='A4')
        
        report.add_font('DejaVu', '', FONT_NORMAL, uni=True)
        report.add_font('DejaVu', 'B', FONT_BOLD, uni=True)
        
        # --- Stránka 1: Úvod a Metodika ---
        report.add_page()
        content_width = report.w - report.l_margin - report.r_margin
        
        # Titulek
        report.set_font('DejaVu', 'B', 16)
        report.multi_cell(content_width, 10, 'Report o klimatických trendech: Stanice Brno 11723', 0, 'C', ln=1)
        report.ln(10)

        # Metodika
        report.set_font('DejaVu', 'B', 12)
        report.multi_cell(content_width, 10, '1. Metodologie', 0, 'L', ln=1)
        report.set_font('DejaVu', '', 10)
        report.multi_cell(content_width, 5, 
            "Analýza vychází z dat CSV souborů (T, F, SRA). Pro každou veličinu (teplota, vítr, srážky) "
            "byla provedena filtrace relevantních měsíčních dat (průměry, sumy). "
            "Data byla následně agregována na roční bázi. Do analýzy byly zahrnuty pouze roky "
            "s kompletními daty (12 měsíců), aby se předešlo zkreslení průměrů či sum.\n"
            "Pro identifikaci trendu byla použita lineární regrese, kde závislou proměnnou byl rok. "
            "Tento regresní model byl následně použit pro extrapolaci budoucích hodnot.",
            0, 'L', ln=1
        )
        report.ln(5)

        # Omezení
        report.set_font('DejaVu', 'B', 12)
        report.multi_cell(content_width, 10, '2. Důležitá omezení modelu', 0, 'L', ln=1)
        report.set_font('DejaVu', 'B', 10)
        report.multi_cell(content_width, 5, 
            "Je nutné zdůraznit, že tento model představuje POUZE lineární extrapolaci, nikoliv reálnou klimatickou predikci.",
            0, 'L', ln=1
        )
        report.set_font('DejaVu', '', 10)
        report.multi_cell(content_width, 5,
            "Hlavní omezení přístupu:\n"
            " - Linearita: Klima je nelineární systém. Předpoklad, že trend posledních dekád bude identicky pokračovat stovky let, je věcně nesprávný.\n"
            " - Chybějící fyzika: Model nebere v potaz žádné fyzikální mechanismy (vliv skleníkových plynů, cykly, body zvratu). Jde o čistě statistický přístup.\n"
            " - Horizont: Extrapolace na 10 let je nejistá, na 100 let je spekulativní a na 1000 let je spíše myšlenkovým cvičením, které demonstruje limity metody.\n"
            " - Lokální data: Data z jedné stanice mohou být ovlivněna lokálními faktory (např. městský tepelný ostrov), které nemusí odrážet globální trend.\n\n"
            "Závěr: Výsledky (zejména dlouhodobé) nelze považovat za prognózu. Slouží jako demonstrace lineárního trendu.",
            0, 'L', ln=1
        )
        
        # --- Stránka 2: Výsledky (Tabulky) ---
        report.add_page()
        content_width = report.w - report.l_margin - report.r_margin
        
        report.set_font('DejaVu', 'B', 12)
        report.multi_cell(content_width, 10, '3. Výsledky analýzy', 0, 'L', ln=1)
        report.ln(5)

        # Tabulka 1: Trendy
        report.set_font('DejaVu', 'B', 11)
        report.multi_cell(content_width, 10, 'Tabulka 1: Identifikované trendy (Sklon regrese)', 0, 'L', ln=1)
        report.set_font('DejaVu', '', 10)
        
        report.cell(60, 7, 'Měřeno', 1, 0)
        report.cell(60, 7, 'Roční změna (sklon)', 1, 1) # '1' na konci posune na další řádek
        
        report.cell(60, 7, 'Průměrná teplota', 1, 0)
        report.cell(60, 7, f"{trends['avg_temp']['slope']:.4f} °C / rok", 1, 1)
        
        report.cell(60, 7, 'Průměrný vítr', 1, 0)
        report.cell(60, 7, f"{trends['avg_wind']['slope']:.4f} m/s / rok", 1, 1)
        
        report.cell(60, 7, 'Roční srážky', 1, 0)
        report.cell(60, 7, f"{trends['sum_precip']['slope']:.4f} mm / rok", 1, 1)
        report.ln(10)

        # Tabulka 2: Predikce
        report.set_font('DejaVu', 'B', 11)
        report.multi_cell(content_width, 10, 'Tabulka 2: Projekce do budoucna (zaokrouhleno)', 0, 'L', ln=1)
        
        report.set_font('DejaVu', 'B', 10)
        col_width = 45
        report.cell(col_width, 7, 'Rok', 1, 0, 'C')
        report.cell(col_width, 7, 'Teplota (°C)', 1, 0, 'C')
        report.cell(col_width, 7, 'Vítr (m/s)', 1, 0, 'C')
        report.cell(col_width, 7, 'Srážky (mm)', 1, 1, 'C')

        report.set_font('DejaVu', '', 10)
        for year, row in future_projections.iterrows():
            report.cell(col_width, 7, str(year), 1, 0, 'C')
            report.cell(col_width, 7, f"{row['pred_avg_temp']:.1f}", 1, 0, 'C')
            report.cell(col_width, 7, f"{row['pred_avg_wind']:.1f}", 1, 0, 'C')
            report.cell(col_width, 7, f"{row['pred_sum_precip']:.0f}", 1, 1, 'C')
        
        # --- Stránky 3, 4, 5: Grafy ---
        for metric, info in metric_info_dict.items():
            report.add_page()
            content_width = report.w - report.l_margin - report.r_margin
            
            report.set_font('DejaVu', 'B', 12)
            report.multi_cell(content_width, 10, f"4. Vizualizace: {info['label']}", 0, 'L', ln=1)
            report.ln(5)
            
            plot_stream = generate_trend_plot(metric, info, annual_data, future_projections, trends)
            report.image(plot_stream, x=10, y=None, w=190)
            plot_stream.close()

        # Vrácení finálního PDF
        return bytes(report.output(dest='S'))

    except Exception as e:
        st.error(f"Došlo k chybě při generování PDF reportu: {e}")
        return None

# --- Hlavní Rozhraní Aplikace Streamlit ---

st.set_page_config(layout="wide", page_title="Analýza Klimatu Brno")
st.title("Lineární analýza klimatu - Stanice Brno")
st.caption("Nástroj pro analýzu historických dat a lineární extrapolaci trendů.")

# Zpracování dat
annual_data, trend_results, trained_models, monthly_data = get_processed_data_and_models()

# Zobrazí se, jen když je vše v pořádku
if annual_data is not None:
    
    # --- Postranní panel ---
    st.sidebar.header("Výsledky regrese")
    st.sidebar.write("Identifikovaný sklon trendu (jednotek/rok):")
    st.sidebar.json({
        "avg_temp_slope": f"{trend_results['avg_temp']['slope']:.4f} °C/rok",
        "avg_wind_slope": f"{trend_results['avg_wind']['slope']:.4f} m/s/rok",
        "sum_precip_slope": f"{trend_results['sum_precip']['slope']:.4f} mm/rok"
    })
    
    st.sidebar.header("Nastavení projekce")
    st.sidebar.info("Zvolte roky pro výpočet extrapolace.")
    current_year = datetime.now().year
    
    horizon_1 = st.sidebar.slider("Horizont 1 (roky)", 1, 50, 10)
    horizon_2 = st.sidebar.slider("Horizont 2 (roky)", 51, 500, 100)
    horizon_3 = st.sidebar.slider("Horizont 3 (roky)", 501, 2000, 1000)
    
    projection_years = [current_year + horizon_1, current_year + horizon_2, current_year + horizon_3]
    
    # --- Hlavní stránka ---
    
    # Dynamická predikce
    projections_data = {}
    for metric, model in trained_models.items():
        future_years_array = np.array(projection_years).reshape(-1, 1)
        future_predictions = model.predict(future_years_array)
        projections_data[f'pred_{metric}'] = future_predictions

    future_projections = pd.DataFrame(projections_data, index=projection_years)
    future_projections.index.name = 'Year'
    display_projections = future_projections.round(2)

    st.header("Tabulka budoucích scénářů")
    st.dataframe(display_projections, use_container_width=True)
    st.warning("Upozornění: Extrapolace na stovky let postrádá reálnou prediktivní hodnotu a slouží pouze jako demonstrace lineárního modelu.")

    # Ukázkový graf
    st.subheader("Ukázkový graf (teplota)")
    with st.spinner("Generuji náhled grafu..."):
        # METRIC_DEFINITIONS je již definováno nahoře
        fig_temp_plot = generate_trend_plot('avg_temp', METRIC_DEFINITIONS['avg_temp'], annual_data, future_projections, trend_results)
        st.image(fig_temp_plot, caption="Vizualizace vývoje a extrapolace průměrné teploty", use_column_width=True)
        
    st.divider()
    
    # --- Generování PDF ---
    st.header("Export do PDF")
    
    with st.spinner("Připravuji kompletní PDF report..."):
        pdf_bytes = create_report_document(annual_data, trend_results, trained_models, future_projections, METRIC_DEFINITIONS)

    if pdf_bytes:
        st.download_button(
            label="Stáhnout kompletní report (PDF)",
            data=pdf_bytes, 
            file_name=f"klima_analyza_brno_{current_year}.pdf",
            mime="application/pdf"
        )
        st.success("Report je připraven ke stažení.")
    else:
        st.error("Report se nepodařilo vygenerovat. Zkontrolujte chybová hlášení.")

else:
    # Zobrazí se, pokud selže načítání dat
    st.error("Načítání dat selhalo. Aplikaci nelze spustit. Zkontrolujte, zda jsou datové soubory přítomny.")
