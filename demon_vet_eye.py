import streamlit as st
import pandas as pd
import numpy as np
import xgboost
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import clone # Potrzebne do klonowania preprocessora

# --- Konfiguracja Nazw Plików ---
DATA_FILE = "vet_eye_crm_data_1000_PL.csv"
MODEL_UPSELL_FILE = "model_upsell.json"
MODEL_CHURN_FILE = "model_churn.json"

# --- Funkcje Pomocnicze ---

@st.cache_data # Cache'owanie wczytanych danych
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Sprawdzenie, czy kluczowe kolumny istnieją
        required_cols = ['client_id', 'segment', 'buy_label', 'churn_label'] # Dodaj inne, jeśli są krytyczne
        if not all(col in df.columns for col in required_cols):
            st.error(f"BŁĄD: W pliku danych '{file_path}' brakuje jednej z wymaganych kolumn: {required_cols}.")
            return None
        return df
    except FileNotFoundError:
        st.error(f"BŁĄD: Nie znaleziono pliku danych '{file_path}'. Upewnij się, że plik jest w repozytorium.")
        return None
    except Exception as e:
        st.error(f"BŁĄD: Nieoczekiwany problem przy wczytywaniu danych z '{file_path}'. Szczegóły: {e}")
        return None


@st.cache_resource # Cache'owanie wczytanych modeli
def load_xgb_model(file_path):
    try:
        model = xgboost.XGBClassifier()
        model.load_model(file_path)
        return model
    except Exception as e:
        st.error(f"BŁĄD: Nie można wczytać modelu z '{file_path}'. Szczegóły: {e}")
        return None

@st.cache_resource # Cache'owanie dopasowanego preprocessora
def get_and_fit_preprocessor(_df_for_fitting): # Podkreślenie, że df jest tylko do fitowania
    """
    Definiuje i dopasowuje preprocessor (ColumnTransformer) na podstawie dostarczonego DataFrame.
    """
    categorical_features = ['segment']
    if not all(feature in _df_for_fitting.columns for feature in categorical_features):
        st.error("Błąd: W danych brakuje kolumny 'segment' potrzebnej do preprocessingu.")
        return None
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' 
    )
    try:
        # Używamy kopii df_for_fitting do fitowania, aby uniknąć modyfikacji oryginału, jeśli byłby używany gdzieś indziej
        # Usuwamy kolumny, które nie są cechami modelu (zakładamy, że są obecne w _df_for_fitting)
        df_fit_features = _df_for_fitting.drop(columns=['client_id', 'buy_label', 'churn_label'], errors='ignore')
        preprocessor.fit(df_fit_features)
        return preprocessor
    except Exception as e:
        st.error(f"Błąd podczas dopasowywania preprocessora: {e}")
        return None

def get_simplified_feature_influence(client_features_processed_df, model, top_n=3):
    """
    Zwraca uproszczony opis wpływu cech na podstawie ogólnej ważności cech modelu
    i wartości cech danego klienta. To NIE jest SHAP, tylko heurystyka.
    """
    try:
        importances = model.feature_importances_
        # Zakładamy, że client_features_processed_df ma nazwy kolumn odpowiadające kolejności w importances
        # lub że model został wytrenowany z feature_names.
        # Dla uproszczenia, jeśli model nie ma feature_names_in_, spróbujemy użyć kolumn z przetworzonego df.
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        elif client_features_processed_df is not None and hasattr(client_features_processed_df, 'columns'):
             feature_names = client_features_processed_df.columns
        else: # Fallback, jeśli nie mamy nazw cech
            feature_names = [f"Cecha_{i}" for i in range(len(importances))]


        feature_importance_map = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        
        influential_features_text = []
        for feature_name, importance_score in feature_importance_map[:top_n]:
            # Mapowanie przetworzonej nazwy cechy na bardziej czytelną, jeśli to możliwe
            # To wymagałoby dodatkowej logiki mapowania (np. z one-hot na oryginalną kategorię)
            # Na razie użyjemy przetworzonej nazwy
            
            # Sprawdzenie wartości tej cechy dla klienta
            client_value_text = ""
            if client_features_processed_df is not None and feature_name in client_features_processed_df.columns:
                client_val = client_features_processed_df[feature_name].iloc[0]
                # Prosta interpretacja (można rozbudować)
                if client_val > 0.5 and client_val <=1: # Dla cech binarnych lub znormalizowanych
                    client_value_text = " (wysoka wartość u klienta)"
                elif client_val == 0 or (client_val <0.5 and client_val >=0) : # Dla cech binarnych lub znormalizowanych
                    client_value_text = " (niska wartość u klienta)"
                elif client_val > 1: # Dla cech numerycznych
                     client_value_text = f" (wartość: {client_val:.2f})"


            influential_features_text.append(f"- **{feature_name.replace('onehot__segment_', 'Segment: ').replace('remainder__', '')}**{client_value_text}")
        
        if influential_features_text:
            return "Prawdopodobne kluczowe czynniki wpływające na scoring (ogólna ważność cech):\n" + "\n".join(influential_features_text)
        else:
            return "Analiza wpływu poszczególnych cech nie jest dostępna."

    except Exception as e:
        # st.warning(f"Nie można było wygenerować uproszczonego wpływu cech: {e}")
        return "Analiza wpływu poszczególnych cech nie jest obecnie dostępna."


def get_recommendations(buy_score_proba, churn_score_proba, segment, client_id=""):
    """Generuje proste rekomendacje na podstawie scoringów i segmentu."""
    recs = []
    buy_prob_percent = buy_score_proba * 100
    churn_prob_percent = churn_score_proba * 100

    recs.append(f"**Klient {client_id}:**")
    # Rekomendacje Upsell
    if buy_prob_percent > 70: # Zmieniony próg dla większej selektywności
        recs.append(f"🔥 **Wysoki Potencjał Zakupowy ({buy_prob_percent:.0f}%)!** Rozważ pilny kontakt z ofertą premium.")
        if segment == "szpital":
            recs.append("   - Zaproponuj najnowszy model USG VetEye ProMax z pakietem głowic specjalistycznych.")
            recs.append("   - Zaoferuj dedykowane szkolenie dla personelu szpitala.")
        elif segment == "klinika":
            recs.append("   - Zaproponuj rozszerzenie obecnego sprzętu o dodatkową głowicę TU2-Cardio lub TU2-Endo.")
            recs.append("   - Poinformuj o nowej wersji platformy TU2 AI z zaawansowaną analizą obrazu i możliwością konsultacji zdalnych.")
        else: # mobilny
            recs.append("   - Zaproponuj lekki i wytrzymały model VetEye Portable 15 z nową, wydajniejszą baterią i specjalnym etui.")
    elif buy_prob_percent > 45: # Zmieniony próg
        recs.append(f"💡 **Średni Potencjał Zakupowy ({buy_prob_percent:.0f}%)!** Dobry moment na ofertę rozszerzającą lub budowanie relacji.")
        recs.append("   - Wyślij e-mail z informacją o promocji na materiały eksploatacyjne, akcesoria lub nadchodzący webinar tematyczny.")
        recs.append("   - Zaproponuj krótką rozmowę telefoniczną w celu zbadania aktualnych potrzeb.")
    else:
        recs.append(f"ℹ️ Niski Potencjał Zakupowy ({buy_prob_percent:.0f}%). Skup się na utrzymaniu dobrej relacji i monitoringu.")

    # Rekomendacje Antychurn
    if churn_prob_percent > 65: # Zmieniony próg
        recs.append(f"🚨 **Bardzo Wysokie Ryzyko Rezygnacji ({churn_prob_percent:.0f}%)!** Wymagane natychmiastowe działania utrzymaniowe!")
        recs.append("   - Osobisty telefon od opiekuna klienta w celu zdiagnozowania problemu i okazania wsparcia.")
        recs.append("   - Zaproponuj specjalne, indywidualne warunki odnowienia subskrypcji lub pakiet dodatkowego wsparcia technicznego/szkoleniowego.")
        recs.append("   - Przeanalizuj dokładnie historię kontaktów i zgłoszeń, aby zrozumieć przyczynę.")
    elif churn_prob_percent > 35: # Zmieniony próg
        recs.append(f"⚠️ **Podwyższone Ryzyko Rezygnacji ({churn_prob_percent:.0f}%)!** Zaplanuj działania prewencyjne.")
        recs.append("   - Zaproś na bezpłatny webinar dotyczący nowych funkcji platformy TU2 lub warsztat podnoszący umiejętności.")
        recs.append("   - Wyślij wartościowe materiały edukacyjne lub case studies pokazujące korzyści z produktów/usług Vet-Eye.")
        recs.append("   - Rozważ proaktywny kontakt w celu zebrania feedbacku i zaoferowania pomocy.")
    else:
        recs.append(f"👍 Niskie Ryzyko Rezygnacji ({churn_prob_percent:.0f}%). Utrzymuj standardowy, dobry kontakt.")
    
    recs.append("\n*Pamiętaj, to są sugestie systemu AI oparte na danych. Ostateczna decyzja i forma działania należą do Ciebie.*")
    return recs

# --- Główna Aplikacja Streamlit ---
st.set_page_config(page_title="Vet-Eye AI Scoring POC", layout="wide")
st.title("🐾 Vet-Eye S.A. - Demonstrator Systemu Scoringowego AI")
st.markdown("Wersja Proof of Concept wspierająca decyzje handlowe.")

# Wczytanie danych i modeli
df_crm_global = load_data(DATA_FILE)
model_upsell_global = load_xgb_model(MODEL_UPSELL_FILE)
model_churn_global = load_xgb_model(MODEL_CHURN_FILE)
preprocessor_global_fitted = None # Zostanie zainicjalizowany po wczytaniu df_crm_global

if df_crm_global is not None:
    df_features_for_preprocessor_fit = df_crm_global.drop(columns=['client_id', 'buy_label', 'churn_label'], errors='ignore')
    if not df_features_for_preprocessor_fit.empty:
         preprocessor_global_fitted = get_and_fit_preprocessor(df_features_for_preprocessor_fit.copy()) # Używamy kopii

# --- Definicja Zakładek ---
if df_crm_global is not None and model_upsell_global is not None and model_churn_global is not None and preprocessor_global_fitted is not None:
    
    # Obliczanie scoringów dla wszystkich klientów (do sortowania)
    # Przygotowujemy dane wszystkich klientów do predykcji
    df_clients_features = df_crm_global.drop(columns=['client_id', 'buy_label', 'churn_label'], errors='ignore')
    
    # Sprawdzenie, czy preprocessor został poprawnie dopasowany
    try:
        clients_data_processed = preprocessor_global_fitted.transform(df_clients_features)
        
        # Dodanie nazw kolumn do przetworzonych danych, jeśli to możliwe
        # To pomoże w funkcji get_simplified_feature_influence, jeśli model nie ma feature_names_in_
        try:
            processed_cols = list(preprocessor_global_fitted.get_feature_names_out())
            clients_data_processed_df = pd.DataFrame(clients_data_processed, columns=processed_cols, index=df_clients_features.index)
        except: # Fallback
            clients_data_processed_df = pd.DataFrame(clients_data_processed, index=df_clients_features.index)


        df_crm_global['buy_score_proba'] = model_upsell_global.predict_proba(clients_data_processed)[:, 1]
        df_crm_global['churn_score_proba'] = model_churn_global.predict_proba(clients_data_processed)[:, 1]
        
        # Dodanie czytelnych etykiet scoringu
        def get_buy_score_label(score):
            if score > 0.7: return "Wysoki Potencjał"
            if score > 0.45: return "Średni Potencjał"
            return "Niski Potencjał"
        
        def get_churn_score_label(score):
            if score > 0.65: return "Wysokie Ryzyko"
            if score > 0.35: return "Średnie Ryzyko"
            return "Niskie Ryzyko"

        df_crm_global['buy_score_label'] = df_crm_global['buy_score_proba'].apply(get_buy_score_label)
        df_crm_global['churn_score_label'] = df_crm_global['churn_score_proba'].apply(get_churn_score_label)


        tab1_title = "📈 Moduł Sprzedażowy (Upsell)"
        tab2_title = "🛡️ Moduł Antychurnowy"
        tab3_title = "👤 Szczegóły Klienta"

        tab1, tab2, tab3 = st.tabs([tab1_title, tab2_title, tab3_title])

        with tab1:
            st.header(tab1_title)
            st.markdown("Lista klientów posortowana według potencjału na dosprzedaż (najwyższy na górze).")
            df_upsell_sorted = df_crm_global.sort_values(by='buy_score_proba', ascending=False)
            st.dataframe(df_upsell_sorted[['client_id', 'segment', 'clinic_size', 'buy_score_proba', 'buy_score_label', 'churn_score_proba', 'churn_score_label']].rename(columns={
                'client_id': 'ID Klienta', 'segment': 'Segment', 'clinic_size': 'Wielkość Kliniki',
                'buy_score_proba': 'Potencjał Zakupowy (%)', 'buy_score_label': 'Ocena Potencjału',
                'churn_score_proba': 'Ryzyko Churn (%)', 'churn_score_label': 'Ocena Ryzyka Churn'
            }).style.format({'Potencjał Zakupowy (%)': '{:.1%}', 'Ryzyko Churn (%)': '{:.1%}'}), height=500)

        with tab2:
            st.header(tab2_title)
            st.markdown("Lista klientów posortowana według ryzyka rezygnacji (najwyższe na górze).")
            df_churn_sorted = df_crm_global.sort_values(by='churn_score_proba', ascending=False)
            st.dataframe(df_churn_sorted[['client_id', 'segment', 'clinic_size', 'churn_score_proba', 'churn_score_label', 'buy_score_proba', 'buy_score_label']].rename(columns={
                'client_id': 'ID Klienta', 'segment': 'Segment', 'clinic_size': 'Wielkość Kliniki',
                'churn_score_proba': 'Ryzyko Churn (%)', 'churn_score_label': 'Ocena Ryzyka Churn',
                'buy_score_proba': 'Potencjał Zakupowy (%)', 'buy_score_label': 'Ocena Potencjału'
            }).style.format({'Ryzyko Churn (%)': '{:.1%}', 'Potencjał Zakupowy (%)': '{:.1%}'}), height=500)

        with tab3:
            st.header(tab3_title)
            st.markdown("Wybierz klienta z poniższej listy, aby zobaczyć szczegółową analizę i rekomendacje.")
            
            client_id_list = df_crm_global['client_id'].tolist()
            selected_client_id_tab3 = st.selectbox(
                "Wybierz ID Klienta:",
                options=client_id_list,
                index=0, # Domyślnie pierwszy klient
                key="client_selector_tab3"
            )
            
            client_data_row = df_crm_global[df_crm_global['client_id'] == selected_client_id_tab3].iloc[0:1]

            if not client_data_row.empty:
                buy_score_proba_selected = client_data_row['buy_score_proba'].iloc[0]
                churn_score_proba_selected = client_data_row['churn_score_proba'].iloc[0]
                segment_selected = client_data_row['segment'].iloc[0]

                col1_detail, col2_detail = st.columns(2)
                with col1_detail:
                    st.metric(label="Potencjał Zakupowy (30 dni)", value=f"{buy_score_proba_selected*100:.1f}%")
                    st.markdown(f"**Ocena:** {client_data_row['buy_score_label'].iloc[0]}")
                
                with col2_detail:
                    st.metric(label="Ryzyko Rezygnacji (Churn)", value=f"{churn_score_proba_selected*100:.1f}%")
                    st.markdown(f"**Ocena:** {client_data_row['churn_score_label'].iloc[0]}")

                st.markdown("---")
                st.subheader("Sugerowane Działania (System AI Vet-Eye):")
                recommendations = get_recommendations(buy_score_proba_selected, churn_score_proba_selected, segment_selected, selected_client_id_tab3)
                for rec in recommendations:
                    st.markdown(rec)

                st.markdown("---")
                st.subheader("Uproszczona Analiza Wpływu Cech:")
                # Przygotowanie danych klienta do analizy wpływu cech (potrzebuje przetworzonych danych)
                client_features_for_influence = df_clients_features[df_clients_features.index == client_data_row.index[0]]
                # Używamy tej samej logiki, co przy predykcjach dla wszystkich klientów
                if hasattr(preprocessor_global_fitted, 'transform'): # Sprawdzenie czy preprocessor jest gotowy
                    client_data_processed_single_df = pd.DataFrame(
                        preprocessor_global_fitted.transform(client_features_for_influence),
                        columns=clients_data_processed_df.columns # Używamy kolumn z przetworzonego zbioru
                    )
                    
                    st.markdown("##### Dla Potencjału Zakupowego:")
                    st.markdown(get_simplified_feature_influence(client_data_processed_single_df, model_upsell_global))
                    st.markdown("##### Dla Ryzyka Rezygnacji:")
                    st.markdown(get_simplified_feature_influence(client_data_processed_single_df, model_churn_global))
                else:
                    st.warning("Preprocessor nie został poprawnie zainicjalizowany do analizy wpływu cech.")


                st.markdown("---")
                st.subheader("Dane Klienta (z symulowanego CRM):")
                display_data = {
                    "ID Klienta": client_data_row['client_id'].iloc[0],
                    "Segment Rynku": segment_selected,
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
                for key, value in display_data.items():
                    st.markdown(f"**{key}:** {value}")

    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania danych dla wszystkich klientów lub predykcji: {e}")
        st.error("Sprawdź logi i konfigurację preprocessora.")

else:
    st.error("Nie udało się wczytać danych CRM lub modeli. Aplikacja nie może kontynuować.")
    st.warning("Upewnij się, że pliki `vet_eye_crm_data_1000_PL.csv`, `model_upsell.json` oraz `model_churn.json` znajdują się w głównym katalogu repozytorium GitHub.")

st.sidebar.markdown("---")
st.sidebar.info("To jest aplikacja demonstracyjna (POC) systemu scoringowego AI firmy Vet-Eye S.A. Wersja ulepszona z zakładkami.")