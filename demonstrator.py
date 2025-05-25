import streamlit as st
import pandas as pd
import numpy as np
import xgboost
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Konfiguracja Nazw Plików ---
DATA_FILE = "vet_eye_crm_data_1000_PL.csv"
MODEL_UPSELL_FILE = "model_upsell.json"
MODEL_CHURN_FILE = "model_churn.json"

# --- Funkcje Pomocnicze ---

@st.cache_data # Cache'owanie wczytanych danych
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"BŁĄD: Nie znaleziono pliku danych '{file_path}'. Upewnij się, że plik jest w repozytorium.")
        return None

@st.cache_resource # Cache'owanie wczytanych modeli
def load_xgb_model(file_path):
    try:
        model = xgboost.XGBClassifier() # Tworzymy pusty model
        model.load_model(file_path)    # Wczytujemy stan z pliku
        return model
    except Exception as e:
        st.error(f"BŁĄD: Nie można wczytać modelu z '{file_path}'. Szczegóły: {e}")
        return None

def get_preprocessor(df_for_fitting):
    """
    Definiuje i dopasowuje preprocessor (ColumnTransformer) na podstawie dostarczonego DataFrame.
    Zakładamy, że df_for_fitting to DataFrame z surowymi cechami (przed one-hot encoding).
    """
    categorical_features = ['segment']
    # Sprawdzenie, czy kolumna 'segment' istnieje
    if not all(feature in df_for_fitting.columns for feature in categorical_features):
        st.error("Błąd: W danych brakuje kolumny 'segment' potrzebnej do preprocessingu.")
        return None
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' 
    )
    # Dopasowujemy preprocessor do danych (np. całego wczytanego CSV lub jego części)
    # aby 'nauczył się' kategorii dla 'segment'
    try:
        # Usuwamy kolumny, których nie ma w X_source_global użytym do treningu
        cols_to_drop_for_fit = ['client_id', 'buy_label', 'churn_label']
        df_fit_features = df_for_fitting.drop(columns=[col for col in cols_to_drop_for_fit if col in df_for_fitting.columns])
        preprocessor.fit(df_fit_features)
        return preprocessor
    except Exception as e:
        st.error(f"Błąd podczas dopasowywania preprocessora: {e}")
        return None

def get_recommendations(buy_score, churn_score, segment):
    """Generuje proste rekomendacje na podstawie scoringów i segmentu."""
    recs = []
    buy_prob = buy_score * 100
    churn_prob = churn_score * 100

    # Rekomendacje Upsell
    if buy_prob > 75:
        recs.append(f"🔥 **Wysoki Potencjał Zakupowy ({buy_prob:.0f}%)!** Rozważ pilny kontakt z ofertą premium.")
        if segment == "szpital":
            recs.append("   - Zaproponuj najnowszy model USG VetEye ProMax z pakietem głowic specjalistycznych.")
            recs.append("   - Zaoferuj dedykowane szkolenie dla personelu szpitala.")
        elif segment == "klinika":
            recs.append("   - Zaproponuj rozszerzenie obecnego sprzętu o dodatkową głowicę TU2-Cardio.")
            recs.append("   - Poinformuj o nowej wersji platformy TU2 AI z zaawansowaną analizą obrazu.")
        else: # mobilny
            recs.append("   - Zaproponuj lekki i wytrzymały model VetEye Portable 15 z nową baterią.")
    elif buy_prob > 50:
        recs.append(f"💡 **Średni Potencjał Zakupowy ({buy_prob:.0f}%)!** Dobry moment na ofertę rozszerzającą.")
        recs.append("   - Wyślij e-mail z informacją o promocji na materiały eksploatacyjne lub akcesoria.")
    else:
        recs.append(f"ℹ️ Niski Potencjał Zakupowy ({buy_prob:.0f}%). Skup się na budowaniu relacji.")

    # Rekomendacje Antychurn
    if churn_prob > 70:
        recs.append(f"🚨 **Bardzo Wysokie Ryzyko Rezygnacji ({churn_prob:.0f}%)!** Wymagane natychmiastowe działania utrzymaniowe!")
        recs.append("   - Osobisty telefon od opiekuna klienta w celu zdiagnozowania problemu.")
        recs.append("   - Zaproponuj specjalne warunki odnowienia subskrypcji lub dodatkowe wsparcie techniczne.")
    elif churn_prob > 40:
        recs.append(f"⚠️ **Podwyższone Ryzyko Rezygnacji ({churn_prob:.0f}%)!** Zaplanuj działania prewencyjne.")
        recs.append("   - Zaproś na bezpłatny webinar dotyczący nowych funkcji platformy TU2.")
        recs.append("   - Przeanalizuj historię zgłoszeń serwisowych i zaoferuj proaktywne wsparcie.")
    else:
        recs.append(f"👍 Niskie Ryzyko Rezygnacji ({churn_prob:.0f}%). Monitoruj standardowo.")
    
    recs.append("\n*Pamiętaj, to są sugestie systemu AI. Ostateczna decyzja i forma działania należą do Ciebie.*")
    return recs

# --- Główna Aplikacja Streamlit ---
st.set_page_config(page_title="Vet-Eye AI Scoring POC", layout="wide")
st.title("🐾 Vet-Eye S.A. - Demonstrator Systemu Scoringowego AI")
st.markdown("Wersja Proof of Concept wspierająca decyzje handlowe.")

# Wczytanie danych i modeli
df_crm = load_data(DATA_FILE)
model_upsell = load_xgb_model(MODEL_UPSELL_FILE)
model_churn = load_xgb_model(MODEL_CHURN_FILE)

# Inicjalizacja i dopasowanie preprocessora na wczytanych danych CRM
# Robimy to raz po wczytaniu danych.
if df_crm is not None:
    # Przygotowujemy DataFrame do dopasowania preprocessora (tylko cechy, bez client_id i targetów)
    df_features_for_preprocessor_fit = df_crm.drop(columns=['client_id', 'buy_label', 'churn_label'], errors='ignore')
    preprocessor = get_preprocessor(df_features_for_preprocessor_fit)
else:
    preprocessor = None

if df_crm is not None and model_upsell is not None and model_churn is not None and preprocessor is not None:
    st.sidebar.header("Wybierz Klienta do Analizy")
    
    # Tworzenie czytelniejszych etykiet dla listy wyboru klienta
    # Zakładamy, że chcemy pokazać ID i np. segment dla ułatwienia
    client_display_list = [f"{row.client_id} ({row.segment}, {row.clinic_size} os.)" for index, row in df_crm.iterrows()]
    
    selected_client_display = st.sidebar.selectbox(
        "Klient:",
        options=client_display_list,
        index=0 # Domyślnie pierwszy klient
    )
    
    # Pobranie client_id z wybranej opcji
    selected_client_id = selected_client_display.split(" ")[0]
    
    client_data_row = df_crm[df_crm['client_id'] == selected_client_id].iloc[0:1] # Pobieramy jako DataFrame (1 wiersz)

    if not client_data_row.empty:
        # Przygotowanie danych wybranego klienta do predykcji
        # Usuwamy kolumny, które nie są cechami modelu i nie były użyte do fitowania preprocessora
        client_features_for_prediction = client_data_row.drop(columns=['client_id', 'buy_label', 'churn_label'], errors='ignore')
        
        # Upewnienie się, że kolejność kolumn jest taka sama jak podczas treningu preprocessora
        # (preprocessor.fit() zapamiętuje kolejność kolumn, które widział)
        # df_features_for_preprocessor_fit to DataFrame użyty do fitowania preprocessora
        # Jego kolejność kolumn jest referencyjna.
        # Jednak ColumnTransformer sam sobie poradzi z kolejnością, jeśli nazwy kolumn się zgadzają.
        # Ważne, żeby `client_features_for_prediction` miał te same kolumny co `df_features_for_preprocessor_fit`
        # (poza tymi, które celowo usunęliśmy, jak client_id, buy_label, churn_label)
        
        try:
            client_data_processed = preprocessor.transform(client_features_for_prediction)
            
            # Predykcje
            buy_score_proba = model_upsell.predict_proba(client_data_processed)[0][1]  # Prawdopodobieństwo klasy 1
            churn_score_proba = model_churn.predict_proba(client_data_processed)[0][1] # Prawdopodobieństwo klasy 1

            # Prezentacja Danych Klienta i Wyników
            st.subheader(f"Analiza Klienta: {client_data_row['client_id'].iloc[0]}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Potencjał Zakupowy (30 dni)", value=f"{buy_score_proba*100:.1f}%")
                if buy_score_proba > 0.75:
                    st.success("Bardzo wysoki potencjał!")
                elif buy_score_proba > 0.5:
                    st.info("Średni potencjał.")
                else:
                    st.warning("Niski potencjał.")
            
            with col2:
                st.metric(label="Ryzyko Rezygnacji (Churn)", value=f"{churn_score_proba*100:.1f}%")
                if churn_score_proba > 0.7:
                    st.error("Bardzo wysokie ryzyko!")
                elif churn_score_proba > 0.4:
                    st.warning("Podwyższone ryzyko.")
                else:
                    st.success("Niskie ryzyko.")

            st.markdown("---")
            st.subheader("Sugerowane Działania (System AI Vet-Eye):")
            recommendations = get_recommendations(buy_score_proba, churn_score_proba, client_data_row['segment'].iloc[0])
            for rec in recommendations:
                st.markdown(rec)

            st.markdown("---")
            st.subheader("Dane Klienta (z symulowanego CRM):")
            
            # Przygotowanie danych do wyświetlenia z "marketingowymi" nazwami
            display_data = {
                "ID Klienta": client_data_row['client_id'].iloc[0],
                "Segment Rynku": client_data_row['segment'].iloc[0],
                "Wielkość Kliniki (personel)": client_data_row['clinic_size'].iloc[0],
                "Liczba Posiadanych Urządzeń Vet-Eye": client_data_row['devices_owned'].iloc[0],
                "Dni od Ostatniego Zakupu": client_data_row['last_purchase_days_ago'].iloc[0],
                "Łączna Liczba Zakupów": client_data_row['purchase_count'].iloc[0],
                "Średnia Wartość Zakupu (PLN)": f"{client_data_row['avg_purchase_value'].iloc[0]:.2f}",
                "Aktywna Subskrypcja TU2": "Tak" if client_data_row['tu2_active'].iloc[0] == 1 else "Nie",
                "Liczba Sesji TU2 (ost. 30 dni)": client_data_row['tu2_sessions_last_30d'].iloc[0],
                "Wykorzystanie Modułów AI w TU2 (%)": f"{client_data_row['ai_usage_ratio'].iloc[0]*100:.0f}%",
                "Dni od Ostatniego Kontaktu Handlowego": client_data_row['last_contact_days_ago'].iloc[0],
                "Wskaźnik Otwarć E-maili (%)": f"{client_data_row['open_rate'].iloc[0]*100:.0f}%",
                "Wskaźnik Kliknięć w E-mailach (%)": f"{client_data_row['click_rate'].iloc[0]*100:.0f}%",
                "Liczba Zgłoszeń Serwisowych (ost. 6 m-cy)": client_data_row['support_tickets_last_6m'].iloc[0]
            }
            # Wyświetlanie jako lista lub w ładniejszym formacie
            for key, value in display_data.items():
                st.markdown(f"**{key}:** {value}")

        except Exception as e:
            st.error(f"Błąd podczas przetwarzania danych wybranego klienta lub predykcji: {e}")
            st.error("Upewnij się, że preprocessor został poprawnie dopasowany i dane wejściowe są prawidłowe.")

else:
    st.warning("Nie udało się wczytać danych lub modeli. Sprawdź, czy pliki znajdują się w repozytorium.")

st.sidebar.markdown("---")
st.sidebar.info("To jest aplikacja demonstracyjna (POC) systemu scoringowego AI firmy Vet-Eye S.A.")