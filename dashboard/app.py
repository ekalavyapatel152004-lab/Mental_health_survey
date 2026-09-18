import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mental Health in Tech Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        :root {
            --indigo: #4C6EF5;
            --coral: #FF6B6B;
            --teal: #20C997;
            --amber: #F59F00;
            --violet: #845EF7;
            --cyan: #15AABF;
            --slate: #172554;
        }

        .main {
            background-color: #f7f9fc;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* These titles sit directly on the app's own dark background
           (not on a white card), so they need a light color, not the
           navy (#172554) used for text inside white cards. The old
           dark-on-dark combination was nearly unreadable. */
        .dashboard-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #E8ECFF;
            margin-bottom: 0.2rem;
        }

        .dashboard-subtitle {
            color: #AEB7D6;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 750;
            color: #E8ECFF;
            margin-top: 1rem;
            margin-bottom: 0.7rem;
        }

        /* ---- KPI / metric cards ---- */
        /* Force equal card height across a row, regardless of how much
           text each card holds (fixes uneven-height KPI cards).
           Covers both older and newer Streamlit testid names, since
           relying on a single nested chain breaks across versions. */
        div[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            display: flex !important;
            flex-direction: column !important;
        }
        div[data-testid="stColumn"] > div,
        div[data-testid="stColumn"] [data-testid="stVerticalBlock"],
        div[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stColumn"] [data-testid="stElementContainer"],
        div[data-testid="stColumn"] div[data-testid="stMarkdown"],
        div[data-testid="stColumn"] div[data-testid="stMarkdownContainer"] {
            height: 100% !important;
            flex: 1 1 auto;
        }

        .kpi-card {
            /* color-scheme: light pins this card to light-mode rendering
               and readable default text color, even when the surrounding
               app is using Streamlit's dark theme. */
            color-scheme: light;
            background: white;
            color: #172554;
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            border: 1px solid #e2e8f0;
            border-left: 4px solid var(--accent, var(--indigo));
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.06);
            width: 100%;
            height: 100%;
            min-height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0.5rem;
            box-sizing: border-box;
        }

        .kpi-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.5rem;
        }

        .kpi-icon {
            flex-shrink: 0;
            width: 32px;
            height: 32px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            background: rgba(76, 110, 245, 0.12);
        }

        .kpi-label {
            color: #64748b;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .kpi-value {
            color: #172554;
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .kpi-description {
            color: #94a3b8;
            font-size: 0.76rem;
            line-height: 1.2;
            /* Reserve space for 2 lines always, so a card with a
               one-line description (e.g. "Filtered survey records")
               is exactly as tall as a neighboring card whose
               description wraps to 2 lines (e.g. "Percentage among
               valid treatment responses"). Without this, cards were
               only as tall as their own content and the row-stretch
               CSS wasn't reliably equalizing them across all
               Streamlit versions. */
            min-height: 2.4em;
        }

        .kpi-delta {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            color: #15803d;
            background: #dcfce7;
            border-radius: 999px;
            padding: 0.15rem 0.6rem;
            width: fit-content;
        }

        /* These three cards previously had no explicit text color, so
           their text silently inherited Streamlit's page-level color.
           Under the app's dark theme that default is near-white, which
           made the text unreadable on these light card backgrounds.
           color-scheme: light + an explicit color fixes that for good. */
        .insight-card {
            color-scheme: light;
            background: white;
            color: #172554;
            border-left: 5px solid #2563eb;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        .warning-card {
            color-scheme: light;
            background: #fff7ed;
            color: #7c2d12;
            border-left: 5px solid #f97316;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.8rem;
        }

        .success-card {
            color-scheme: light;
            background: #f0fdf4;
            color: #14532d;
            border-left: 5px solid #16a34a;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.8rem;
        }

        .insight-card b, .warning-card b, .success-card b { color: inherit; }

        div[data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #e2e8f0;
            padding: 1rem;
            border-radius: 14px;
        }

        [data-testid="stSidebar"] {
            background-color: #0f172a;
        }

        [data-testid="stSidebar"] * {
            color: white;
        }

        .stDownloadButton button {
            width: 100%;
        }

        /* ---- Chart cards: premium colored-border wrapper ----
           st.container(border=True, key="chart_<color>_<n>") renders a
           div carrying a class "st-key-chart_<color>_<n>". We target that
           substring to give every chart a colored top border and card
           styling, cycling through the palette by key prefix. */
        div[class*="st-key-chart_"] {
            color-scheme: light !important;
            border: 1px solid #e9ecff !important;
            border-radius: 16px !important;
            padding: 0.6rem 0.9rem 1rem 0.9rem !important;
            box-shadow: 0 3px 14px rgba(15, 23, 42, 0.06) !important;
            background: white !important;
            border-top: 4px solid var(--indigo) !important;
        }
        div[class*="st-key-chart_indigo_"] { border-top-color: var(--indigo) !important; }
        div[class*="st-key-chart_teal_"]   { border-top-color: var(--teal) !important; }
        div[class*="st-key-chart_coral_"]  { border-top-color: var(--coral) !important; }
        div[class*="st-key-chart_violet_"] { border-top-color: var(--violet) !important; }
        div[class*="st-key-chart_amber_"]  { border-top-color: var(--amber) !important; }
        div[class*="st-key-chart_cyan_"]   { border-top-color: var(--cyan) !important; }
    </style>
    """,
    unsafe_allow_html=True
)



# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    candidate_paths = [
        os.path.join(base_dir, "..", "data", "survey_cleaned.csv"),
        os.path.join(base_dir, "data", "survey_cleaned.csv"),
        os.path.join(base_dir, "survey_cleaned.csv"),
    ]

    for path in candidate_paths:
        path = os.path.abspath(path)
        if os.path.exists(path):
            return pd.read_csv(path)

    st.error(
        "Dataset not found. Expected 'survey_cleaned.csv' in one of these locations:\n\n"
        + "\n".join(f"- {os.path.abspath(p)}" for p in candidate_paths)
    )
    st.stop()



# ============================================================
# DATA PREPARATION
# ============================================================

def _clean_text(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    )


def prepare_data(data):
    data = data.copy()

    # Age
    if "age" in data.columns:
        data["Age"] = pd.to_numeric(data["age"], errors="coerce")
        data.loc[~data["Age"].between(18, 100), "Age"] = np.nan
        data["Age_Group"] = pd.cut(
            data["Age"],
            bins=[17, 24, 34, 44, 54, np.inf],
            labels=["18-24", "25-34", "35-44", "45-54", "55+"],
            include_lowest=True
        )

    # Gender
    if "gender" in data.columns:
        gender_mapping = {
            "m": "Male",
            "male": "Male",
            "man": "Male",
            "male-ish": "Male",
            "maile": "Male",
            "mal": "Male",
            "f": "Female",
            "female": "Female",
            "woman": "Female",
            "femake": "Female",
            "female (cis)": "Female",
            "female (trans)": "Female",
            "trans-female": "Female",
            "trans female": "Female",
            "non-binary": "Other",
            "nonbinary": "Other",
            "queer": "Other",
            "genderqueer": "Other",
            "androgyne": "Other",
            "agender": "Other",
            "something else": "Other",
            "nah": "Other",
            "p": "Other",
            "other": "Other",
        }

        cleaned_gender = _clean_text(data["gender"]).replace(gender_mapping)
        data["Gender_Clean"] = cleaned_gender.fillna("Unknown")

    # Timestamp
    if "timestamp" in data.columns:
        data["Timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

    # Company size
    if "no_employees" in data.columns:
        company_size_mapping = {
            "1-5": "Small",
            "6-25": "Small",
            "26-100": "Medium",
            "100-500": "Large",
            "500-1000": "Large",
            "more than 1000": "Enterprise",
        }

        employee_band = _clean_text(data["no_employees"])
        data["Company_Size_Group"] = (
            employee_band
            .replace(company_size_mapping)
            .fillna("Unknown")
        )

    # Treatment: always derive the analytical flag from the source field
    if "treatment" in data.columns:
        treatment_clean = _clean_text(data["treatment"])
        data["Treatment_Label"] = treatment_clean.map(
            {"yes": "Yes", "no": "No"}
        )
        data["Treatment_Flag"] = treatment_clean.map(
            {"yes": 1, "no": 0}
        )

    # Binary analytical flags
    flag_sources = {
        "Family_History_Flag": "family_history",
        "Remote_Work_Flag": "remote_work",
        "Tech_Company_Flag": "tech_company",
    }

    for flag_name, source_column in flag_sources.items():
        if source_column in data.columns:
            cleaned = _clean_text(data[source_column])
            data[flag_name] = cleaned.map({"yes": 1, "no": 0})

    # Work interference score
    if "work_interfere" in data.columns:
        work_mapping = {
            "never": 0,
            "rarely": 1,
            "sometimes": 2,
            "often": 3,
        }

        data["Work_Interference_Score"] = (
            _clean_text(data["work_interfere"])
            .map(work_mapping)
        )

    # Workplace openness index
    openness_mapping = {
        "no": 0,
        "some of them": 1,
        "yes": 2,
    }

    openness_scores = []

    for column in ["coworkers", "supervisor"]:
        if column in data.columns:
            openness_scores.append(
                _clean_text(data[column]).map(openness_mapping)
            )

    if openness_scores:
        openness_df = pd.concat(openness_scores, axis=1)
        data["Workplace_Openness_Index"] = openness_df.mean(
            axis=1,
            skipna=True
        )

    # Support-related response score.
    # Each answered item contributes Yes=1 / No=0.
    # A minimum of 3 answered items is required before calculating
    # the composite score.
    support_mappings = {
        "benefits": {"yes": 1, "no": 0},
        "care_options": {"yes": 1, "no": 0},
        "wellness_program": {"yes": 1, "no": 0},
        "seek_help": {"yes": 1, "no": 0},
        "anonymity": {"yes": 1, "no": 0},
    }

    support_scores = []

    for column, mapping in support_mappings.items():
        if column in data.columns:
            support_scores.append(
                _clean_text(data[column]).map(mapping)
            )

    if support_scores:
        support_df = pd.concat(support_scores, axis=1)
        answered_count = support_df.notna().sum(axis=1)

        data["Support_Items_Answered"] = answered_count
        data["Support_Availability_Score"] = support_df.mean(
            axis=1,
            skipna=True
        )
        data.loc[
            data["Support_Items_Answered"] < 3,
            "Support_Availability_Score"
        ] = np.nan

    return data


raw_df = load_data()
ORIGINAL_COLUMN_COUNT = len(raw_df.columns)
df = prepare_data(raw_df)



# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_rate(data, column, value=None):
    if "Treatment_Flag" not in data.columns or len(data) == 0:
        return np.nan

    valid = data["Treatment_Flag"].notna()

    if value is not None and column in data.columns:
        value_mask = (
            data[column]
            .astype("string")
            .str.strip()
            .str.lower()
            == str(value).strip().lower()
        )
        valid &= value_mask

    selected = data.loc[valid, "Treatment_Flag"]

    if selected.empty:
        return np.nan

    return selected.mean() * 100


def kpi_card(label, value, icon="\U0001F4CA", accent="#4C6EF5", description=None, delta=None):
    extra_html = ""
    if description:
        extra_html += f'<div class="kpi-description">{description}</div>'
    if delta:
        extra_html += f'<div class="kpi-delta">{delta}</div>'
    return f"""
    <div class="kpi-card" style="--accent:{accent};">
        <div class="kpi-top">
            <div class="kpi-label">{label}</div>
            <div class="kpi-icon" style="--accent:{accent}; background: color-mix(in srgb, {accent} 14%, white); color:{accent};">{icon}</div>
        </div>
        <div class="kpi-value">{value}</div>
        {extra_html}
    </div>
    """


def create_kpi(label, value, description=""):
    # Kept for backward compatibility with any remaining call sites.
    return kpi_card(label, value, description=description)


def style_chart(fig, height=450):
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=40),
        title_font=dict(size=19, color="#172554"),
        font=dict(family="Arial", color="#334155"),
        legend_title_text=""
    )
    return fig


def treatment_rate_table(data, group_column):
    if group_column not in data.columns or "Treatment_Flag" not in data.columns:
        return pd.DataFrame(
            columns=[group_column, "Respondents", "Treatment_Rate"]
        )

    work = data[[group_column, "Treatment_Flag"]].copy()

    # Treatment rate denominator contains only valid treatment responses.
    work = work.dropna(subset=["Treatment_Flag"])

    if work.empty:
        return pd.DataFrame(
            columns=[group_column, "Respondents", "Treatment_Rate"]
        )

    work["_Group"] = work[group_column].astype("object")
    work["_Group"] = work["_Group"].where(
        work["_Group"].notna(),
        "Unknown"
    )

    result = (
        work.groupby("_Group", dropna=False)
        .agg(
            Respondents=("Treatment_Flag", "count"),
            Treatment_Rate=("Treatment_Flag", "mean")
        )
        .reset_index()
        .rename(columns={"_Group": group_column})
    )

    result["Treatment_Rate"] = result["Treatment_Rate"] * 100

    return result


def run_chi_square(data, column):
    if column not in data.columns or "Treatment_Label" not in data.columns:
        return None, None, None, pd.DataFrame(), None

    test_data = data[[column, "Treatment_Label"]].dropna()

    if test_data.empty:
        return None, None, None, pd.DataFrame(), None

    table = pd.crosstab(
        test_data[column],
        test_data["Treatment_Label"]
    )

    if table.shape[0] < 2 or table.shape[1] < 2:
        return None, None, None, table, None

    chi2, p_value, dof, expected = chi2_contingency(table)

    low_expected_cells = int((expected < 5).sum())
    total_expected_cells = int(expected.size)

    assumption_warning = None
    if total_expected_cells > 0:
        low_expected_pct = low_expected_cells / total_expected_cells

        if low_expected_pct > 0.20 or expected.min() < 1:
            assumption_warning = (
                "Chi-square expected-frequency assumptions may be weak "
                f"({low_expected_cells}/{total_expected_cells} expected cells below 5)."
            )

    return chi2, p_value, dof, table, assumption_warning


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR adjustment for multiple p-values."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)

    valid_mask = np.isfinite(p_values)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_p = p_values[valid_mask]
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    n = len(sorted_p)

    sorted_adjusted = np.empty(n, dtype=float)

    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = sorted_p[i] * n / rank
        running_min = min(running_min, value)
        sorted_adjusted[i] = min(running_min, 1.0)

    adjusted_valid = np.empty(n, dtype=float)
    adjusted_valid[order] = sorted_adjusted

    adjusted[valid_indices] = adjusted_valid
    return adjusted



def cramers_v(x, y):
    """Calculate Cramér's V for two categorical variables."""
    pair = pd.DataFrame({"x": x, "y": y}).dropna()

    if pair.empty:
        return np.nan

    table = pd.crosstab(pair["x"], pair["y"])

    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan

    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    rows, cols = table.shape
    denominator = min(rows - 1, cols - 1)

    if denominator <= 0:
        return np.nan

    return float(np.sqrt(phi2 / denominator))


# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource
def train_model(data):
    model_data = data.copy()

    if "treatment" not in model_data.columns:
        return {"error": "The dataset does not contain the 'treatment' column."}

    model_data["Treatment_Target"] = (
        _clean_text(model_data["treatment"])
        .map({"yes": 1, "no": 0})
    )

    features = [
        "Age",
        "Gender_Clean",
        "family_history",
        "work_interfere",
        "remote_work",
        "no_employees",
        "mental_health_consequence",
        "benefits",
        "care_options",
        "wellness_program",
        "seek_help",
        "anonymity",
        "coworkers",
        "supervisor",
        "tech_company"
    ]

    available_features = [
        feature
        for feature in features
        if feature in model_data.columns
    ]

    model_data = model_data.dropna(subset=["Treatment_Target"])

    if not available_features:
        return {"error": "No model features are available in the dataset."}

    if model_data["Treatment_Target"].nunique() < 2:
        return {"error": "The treatment target contains fewer than two classes."}

    class_counts = model_data["Treatment_Target"].value_counts()

    if class_counts.min() < 2:
        return {
            "error": (
                "There are too few observations in one treatment class "
                "to create a stratified train/test split."
            )
        }

    X = model_data[available_features]
    y = model_data["Treatment_Target"].astype(int)

    numeric_features = [
        feature
        for feature in ["Age"]
        if feature in available_features
    ]

    categorical_features = [
        feature
        for feature in available_features
        if feature not in numeric_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                )
            )
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_transformer, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_transformer, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = (
        roc_auc_score(y_test, y_prob)
        if y_test.nunique() == 2
        else np.nan
    )

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "F1-Score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "ROC-AUC": roc_auc
    }

    # Cross-validation on the training dataset.
    cv_metrics = None
    min_class_count = int(y_train.value_counts().min())

    if min_class_count >= 3:
        n_splits = min(5, min_class_count)
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        cv_metrics = {
            "Accuracy Mean": cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="accuracy"
            ),
            "F1 Mean": cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="f1"
            ),
            "ROC-AUC Mean": cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc"
            )
        }

    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    coefficients = model.named_steps[
        "classifier"
    ].coef_[0]

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    feature_importance["Absolute_Coefficient"] = (
        feature_importance["Coefficient"].abs()
    )

    feature_importance = feature_importance.sort_values(
        "Absolute_Coefficient",
        ascending=False
    )

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "cv_metrics": cv_metrics
    }



# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <h1 style="color:white;">🧠 Mental Health</h1>
    <p style="color:#cbd5e1;">Tech Survey Analytics</p>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    [
        "🏠 Overview",
        "📊 Treatment Analysis",
        "🏢 Workplace Support",
        "👥 Demographic Analysis",
        "🔍 Advanced Insights",
        "🤖 Predictive Model",
        "🗃️ Data Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

filtered_df = df.copy()

# Country — included because geographic variation is one of the project
# questions and is also part of the EDA findings.
if "country" in df.columns:
    country_options = sorted(
        df["country"].dropna().astype(str).str.strip().unique().tolist()
    )

    selected_country = st.sidebar.multiselect(
        "Country",
        country_options,
        default=country_options
    )

    if selected_country:
        filtered_df = filtered_df[
            filtered_df["country"].astype(str).isin(selected_country)
        ]
    else:
        filtered_df = filtered_df.iloc[0:0]

# Gender
if "Gender_Clean" in df.columns:
    gender_options = sorted(
        df["Gender_Clean"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_gender = st.sidebar.multiselect(
        "Gender",
        gender_options,
        default=gender_options
    )

    if selected_gender:
        filtered_df = filtered_df[
            filtered_df["Gender_Clean"].astype(str).isin(selected_gender)
        ]
    else:
        filtered_df = filtered_df.iloc[0:0]

# Company size — use the original survey bands for filtering.
if "no_employees" in df.columns:
    company_size_options = [
        value
        for value in [
            "1-5",
            "6-25",
            "26-100",
            "100-500",
            "500-1000",
            "More than 1000"
        ]
        if value in df["no_employees"].astype(str).unique()
    ]

    selected_company_size = st.sidebar.multiselect(
        "Company Size",
        company_size_options,
        default=company_size_options
    )

    if selected_company_size:
        filtered_df = filtered_df[
            filtered_df["no_employees"].astype(str).isin(selected_company_size)
        ]
    else:
        filtered_df = filtered_df.iloc[0:0]

# Age
if "Age" in df.columns:
    valid_ages = df["Age"].dropna()

    if len(valid_ages) > 0:
        min_age = int(valid_ages.min())
        max_age = int(valid_ages.max())

        age_range = st.sidebar.slider(
            "Age Range",
            min_age,
            max_age,
            (min_age, max_age)
        )

        filtered_df = filtered_df[
            filtered_df["Age"].between(
                age_range[0],
                age_range[1]
            )
        ]

# Remote work
if "remote_work" in df.columns:
    remote_options = sorted(
        df["remote_work"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_remote = st.sidebar.multiselect(
        "Remote Work",
        remote_options,
        default=remote_options
    )

    if selected_remote:
        filtered_df = filtered_df[
            filtered_df["remote_work"].astype(str).isin(selected_remote)
        ]
    else:
        filtered_df = filtered_df.iloc[0:0]

st.sidebar.markdown("---")

st.sidebar.info(
    "This dashboard presents statistical associations "
    "from survey data. It is not a diagnostic tool."
)

# Prevent downstream selectboxes/charts from failing when filters
# return zero records.
if filtered_df.empty:
    st.warning(
        "No records match the selected filters. "
        "Please select at least one value in each active filter."
    )
    st.stop()



# ============================================================
# COMMON VALUES
# ============================================================

total_records = len(filtered_df)

valid_treatment = filtered_df["Treatment_Flag"].notna()

treatment_response_count = int(valid_treatment.sum())
treatment_count = int(
    filtered_df.loc[valid_treatment, "Treatment_Flag"].sum()
)

no_treatment_count = treatment_response_count - treatment_count

treatment_rate = (
    treatment_count / treatment_response_count * 100
    if treatment_response_count > 0
    else np.nan
)

family_history_rate = (
    filtered_df["family_history"]
    .astype("string")
    .str.strip()
    .str.lower()
    .eq("yes")
    .mean() * 100
    if "family_history" in filtered_df.columns and len(filtered_df) > 0
    else np.nan
)

work_interference_rate = (
    calculate_rate(filtered_df, "work_interfere", "Often")
    if "work_interfere" in filtered_df.columns
    else np.nan
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="dashboard-title">Mental Health in Tech Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Interactive analysis of mental-health treatment patterns "
        "among technology-sector respondents."
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Executive Overview</div>',
        unsafe_allow_html=True
    )

    kpi_columns = st.columns(4)

    with kpi_columns[0]:
        st.markdown(
            kpi_card(
                "Total Respondents",
                f"{total_records:,}",
                icon="\U0001F465",
                accent="#4C6EF5",
                description="Filtered survey records"
            ),
            unsafe_allow_html=True
        )

    with kpi_columns[1]:
        st.markdown(
            kpi_card(
                "Received Treatment",
                f"{treatment_count:,}",
                icon="\U0001F48A",
                accent="#20C997",
                description="Respondents answering Yes"
            ),
            unsafe_allow_html=True
        )

    with kpi_columns[2]:
        st.markdown(
            kpi_card(
                "Treatment Rate",
                f"{treatment_rate:.2f}%",
                icon="\U0001F4C8",
                accent="#FF6B6B",
                description="Percentage among valid treatment responses"
            ),
            unsafe_allow_html=True
        )

    with kpi_columns[3]:
        st.markdown(
            kpi_card(
                "Family History",
                f"{family_history_rate:.2f}%" if pd.notna(family_history_rate) else "N/A",
                icon="\U0001F9EC",
                accent="#845EF7",
                description="Respondents answering Yes"
            ),
            unsafe_allow_html=True
        )

    st.markdown("")

    left, right = st.columns(2)

    with left:
        treatment_counts = (
            filtered_df["Treatment_Label"]
            .value_counts()
            .reset_index()
        )

        treatment_counts.columns = [
            "Treatment",
            "Count"
        ]

        fig = px.pie(
            treatment_counts,
            names="Treatment",
            values="Count",
            hole=0.55,
            title="Treatment Distribution"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        with st.container(border=True, key="chart_indigo_1"):
            st.plotly_chart(
                style_chart(fig, 420),
                use_container_width=True
            )

    with right:
        if "Gender_Clean" in filtered_df.columns:
            gender_distribution = (
                filtered_df["Gender_Clean"]
                .value_counts(normalize=True)
                .mul(100)
                .round(1)
                .reset_index()
            )

            gender_distribution.columns = [
                "Gender",
                "Percentage"
            ]

            fig = px.bar(
                gender_distribution,
                x="Gender",
                y="Percentage",
                text="Percentage",
                title="Gender Distribution (%)",
                labels={
                    "Gender": "Gender",
                    "Percentage": "Percentage (%)"
                }
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_yaxes(range=[0, 100])

            with st.container(border=True, key="chart_coral_2"):
                st.plotly_chart(
                    style_chart(fig, 420),
                    use_container_width=True
                )

            st.caption(
                "Gender percentages represent the distribution of "
                "respondents in the filtered dataset and add up to 100%."
            )

    st.markdown(
        '<div class="section-title">Key Business Insights</div>',
        unsafe_allow_html=True
    )

    work_insight = "Work-interference treatment rates are unavailable."
    if "work_interfere" in filtered_df.columns:
        work_summary = treatment_rate_table(filtered_df, "work_interfere")
        if not work_summary.empty:
            highest_work = work_summary.loc[
                work_summary["Treatment_Rate"].idxmax()
            ]
            work_insight = (
                f"The highest observed treatment rate is "
                f"{highest_work['Treatment_Rate']:.1f}% for "
                f"{highest_work['work_interfere']} work interference "
                f"(n={int(highest_work['Respondents'])})."
            )

    family_insight = "Family-history treatment rates are unavailable."
    if "family_history" in filtered_df.columns:
        family_summary = treatment_rate_table(
            filtered_df,
            "family_history"
        )
        if not family_summary.empty:
            family_insight = "Treatment rates by family history are shown in the Treatment Analysis page."

    st.markdown(
        f"""
        <div class="insight-card">
        <b>Work interference:</b> {work_insight}
        </div>

        <div class="insight-card">
        <b>Family history:</b> {family_insight}
        </div>

        <div class="insight-card">
        <b>Interpretation:</b> These are descriptive associations in the
        filtered survey data; they do not establish causation.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        filtered_df.head(10),
        use_container_width=True,
        height=300
    )


# ============================================================
# PAGE 2 — TREATMENT ANALYSIS
# ============================================================

elif page == "📊 Treatment Analysis":

    st.markdown(
        '<div class="dashboard-title">Treatment Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Explore how treatment rates differ across major survey factors."
        "</div>",
        unsafe_allow_html=True
    )

    tabs = st.tabs(
        [
            "Work Interference",
            "Family History",
            "Mental-health Consequence",
            "Statistical Tests"
        ]
    )

    with tabs[0]:
        st.subheader("Treatment Rate by Work Interference")

        work_table = treatment_rate_table(
            filtered_df,
            "work_interfere"
        )

        work_order = [
            "Never",
            "Rarely",
            "Sometimes",
            "Often",
            "Not Applicable / Unknown"
        ]

        work_table["Sort_Order"] = work_table[
            "work_interfere"
        ].map(
            {
                value: index
                for index, value in enumerate(work_order)
            }
        )

        work_table = work_table.sort_values(
            "Sort_Order"
        )

        fig = px.bar(
            work_table,
            x="work_interfere",
            y="Treatment_Rate",
            text="Treatment_Rate",
            color="Treatment_Rate",
            title="Treatment Rate by Work Interference",
            labels={
                "work_interfere": "Work Interference",
                "Treatment_Rate": "Treatment Rate (%)"
            },
            color_continuous_scale="Blues"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_yaxes(range=[0, 100])

        with st.container(border=True, key="chart_teal_3"):
            st.plotly_chart(
                style_chart(fig, 500),
                use_container_width=True
            )

        st.dataframe(
            work_table.drop(columns=["Sort_Order"]),
            use_container_width=True
        )

    with tabs[1]:
        st.subheader("Treatment Rate by Family History")

        family_table = treatment_rate_table(
            filtered_df,
            "family_history"
        )

        fig = px.bar(
            family_table,
            x="family_history",
            y="Treatment_Rate",
            text="Treatment_Rate",
            color="family_history",
            title="Family History and Treatment",
            labels={
                "family_history": "Family History",
                "Treatment_Rate": "Treatment Rate (%)"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_yaxes(range=[0, 100])

        with st.container(border=True, key="chart_violet_4"):
            st.plotly_chart(
                style_chart(fig, 450),
                use_container_width=True
            )

        st.dataframe(
            family_table,
            use_container_width=True
        )

    with tabs[2]:
        st.subheader(
            "Treatment Rate by Mental-health Consequence"
        )

        consequence_table = treatment_rate_table(
            filtered_df,
            "mental_health_consequence"
        )

        fig = px.bar(
            consequence_table,
            x="mental_health_consequence",
            y="Treatment_Rate",
            text="Treatment_Rate",
            color="mental_health_consequence",
            title="Perceived Mental-health Consequences",
            labels={
                "mental_health_consequence":
                    "Mental-health Consequence",
                "Treatment_Rate":
                    "Treatment Rate (%)"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_yaxes(range=[0, 100])

        with st.container(border=True, key="chart_amber_5"):
            st.plotly_chart(
                style_chart(fig, 450),
                use_container_width=True
            )

        st.dataframe(
            consequence_table,
            use_container_width=True
        )


    with tabs[3]:
        st.subheader("Chi-square Statistical Tests")

        test_columns = [
            "work_interfere",
            "family_history",
            "Gender_Clean",
            "mental_health_consequence",
            "Age_Group",
            "remote_work",
            "Company_Size_Group"
        ]

        test_results = []
        assumption_warnings = []

        for column in test_columns:
            if column not in filtered_df.columns:
                continue

            chi2, p_value, dof, table, assumption_warning = run_chi_square(
                filtered_df,
                column
            )

            if p_value is not None:
                test_results.append(
                    {
                        "Variable": column,
                        "Chi-square": round(chi2, 4),
                        "P-value": p_value,
                        "Degrees of Freedom": dof
                    }
                )

                if assumption_warning:
                    assumption_warnings.append(
                        f"{column}: {assumption_warning}"
                    )

        test_results_df = pd.DataFrame(test_results)

        if not test_results_df.empty:
            test_results_df["Adjusted P-value"] = benjamini_hochberg(
                test_results_df["P-value"].values
            )

            test_results_df["Significant (FDR < 0.05)"] = np.where(
                test_results_df["Adjusted P-value"] < 0.05,
                "Yes",
                "No"
            )

            display_results = test_results_df.copy()

            display_results["P-value"] = display_results[
                "P-value"
            ].apply(
                lambda value: (
                    "< 0.001"
                    if value < 0.001
                    else f"{value:.4f}"
                )
            )

            display_results["Adjusted P-value"] = display_results[
                "Adjusted P-value"
            ].apply(
                lambda value: (
                    "< 0.001"
                    if value < 0.001
                    else f"{value:.4f}"
                )
            )

            st.dataframe(
                display_results,
                use_container_width=True
            )

        if assumption_warnings:
            st.warning(
                "Some chi-square tests have small expected frequencies. "
                "Interpret those results with caution."
            )
            for warning in assumption_warnings:
                st.caption(warning)

        st.info(
            "P-values test for an association between the selected variable "
            "and treatment response. The adjusted p-value applies a "
            "Benjamini-Hochberg false-discovery-rate correction across the "
            "tests shown. Statistical significance does not prove causation."
        )


# ============================================================
# PAGE 3 — WORKPLACE SUPPORT
# ============================================================

elif page == "🏢 Workplace Support":

    st.markdown(
        '<div class="dashboard-title">Workplace Support Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Analyze benefits, care options, anonymity, openness, and "
        "workplace support systems."
        "</div>",
        unsafe_allow_html=True
    )

    support_columns = [
        "benefits",
        "care_options",
        "wellness_program",
        "seek_help",
        "anonymity"
    ]

    available_support = [
        column for column in support_columns
        if column in filtered_df.columns
    ]

    if available_support:
        selected_support = st.selectbox(
            "Select workplace support variable",
            available_support
        )

        support_table = treatment_rate_table(
            filtered_df,
            selected_support
        )

        if not support_table.empty:
            fig = px.bar(
                support_table,
                x=selected_support,
                y="Treatment_Rate",
                text="Treatment_Rate",
                color="Treatment_Rate",
                color_continuous_scale="Teal",
                title=f"Treatment Rate by {selected_support}"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_yaxes(range=[0, 100])

            with st.container(border=True, key="chart_cyan_6"):
                st.plotly_chart(
                    style_chart(fig, 500),
                    use_container_width=True
                )

            st.dataframe(
                support_table,
                use_container_width=True
            )
    else:
        st.info("No workplace support variables are available in the dataset.")

    st.markdown(
        '<div class="section-title">Workplace Openness</div>',
        unsafe_allow_html=True
    )

    openness_columns = [
        column for column in ["coworkers", "supervisor"]
        if column in filtered_df.columns
    ]

    if openness_columns:
        openness_choice = st.selectbox(
            "Select openness variable",
            openness_columns
        )

        openness_table = treatment_rate_table(
            filtered_df,
            openness_choice
        )

        if not openness_table.empty:
            fig = px.bar(
                openness_table,
                x=openness_choice,
                y="Treatment_Rate",
                text="Treatment_Rate",
                color=openness_choice,
                title=f"Treatment Rate by {openness_choice}"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_yaxes(range=[0, 100])

            with st.container(border=True, key="chart_indigo_7"):
                st.plotly_chart(
                    style_chart(fig, 450),
                    use_container_width=True
                )
    else:
        st.info("No workplace openness variables are available in the dataset.")

    st.markdown(
        '<div class="section-title">Support-related Response Score</div>',
        unsafe_allow_html=True
    )

    if "Support_Availability_Score" in filtered_df.columns:
        fig = px.histogram(
            filtered_df.dropna(subset=["Support_Availability_Score"]),
            x="Support_Availability_Score",
            color="Treatment_Label",
            nbins=10,
            barmode="overlay",
            opacity=0.75,
            title="Distribution of Support-related Response Score",
            labels={
                "Support_Availability_Score": "Average Support-related Score (0–1)",
                "Treatment_Label": "Treatment"
            }
        )

        with st.container(border=True, key="chart_coral_8"):
            st.plotly_chart(
                style_chart(fig, 450),
                use_container_width=True
            )

        st.caption(
            "Score is the average of answered support-related items. "
            "At least 3 answered items are required."
        )


# ============================================================
# PAGE 4 — DEMOGRAPHIC ANALYSIS
# ============================================================

elif page == "👥 Demographic Analysis":

    st.markdown(
        '<div class="dashboard-title">Demographic Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Explore treatment patterns across gender, age, company size, "
        "remote work, and technology companies."
        "</div>",
        unsafe_allow_html=True
    )

    demographic_tabs = st.tabs(
        [
            "Age",
            "Gender",
            "Country",
            "Company Size",
            "Remote Work",
            "Tech Company"
        ]
    )

    with demographic_tabs[0]:
        if "Age_Group" in filtered_df.columns:
            age_table = treatment_rate_table(
                filtered_df,
                "Age_Group"
            )

            fig = px.bar(
                age_table,
                x="Age_Group",
                y="Treatment_Rate",
                text="Treatment_Rate",
                color="Treatment_Rate",
                color_continuous_scale="Viridis",
                title="Treatment Rate by Age Group"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_yaxes(range=[0, 100])

            with st.container(border=True, key="chart_teal_9"):
                st.plotly_chart(
                    style_chart(fig, 450),
                    use_container_width=True
                )

            st.dataframe(
                age_table,
                use_container_width=True
            )

    with demographic_tabs[1]:
        st.subheader("Gender Distribution")

        gender_distribution = (
            filtered_df["Gender_Clean"]
            .value_counts(dropna=False)
            .rename_axis("Gender")
            .reset_index(name="Respondents")
        )

        total_gender_respondents = gender_distribution["Respondents"].sum()

        if total_gender_respondents > 0:
            gender_distribution["Percentage"] = (
                gender_distribution["Respondents"]
                / total_gender_respondents
                * 100
            ).round(1)
        else:
            gender_distribution["Percentage"] = 0.0

        fig = px.bar(
            gender_distribution,
            x="Gender",
            y="Percentage",
            text="Percentage",
            color="Gender",
            title="Gender Distribution (%)",
            labels={
                "Gender": "Gender",
                "Percentage": "Percentage (%)"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_yaxes(range=[0, 100])

        with st.container(border=True, key="chart_violet_10"):
            st.plotly_chart(
                style_chart(fig, 450),
                use_container_width=True
            )

        st.caption(
            "Gender percentages represent the distribution of respondents "
            "in the filtered dataset and add up to 100%."
        )

        st.dataframe(
            gender_distribution,
            use_container_width=True
        )

    with demographic_tabs[2]:
        st.subheader("Country Analysis")

        if "country" in filtered_df.columns:
            country_counts = (
                filtered_df["country"]
                .value_counts()
                .head(10)
                .rename_axis("Country")
                .reset_index(name="Respondents")
            )

            if not country_counts.empty:
                fig = px.bar(
                    country_counts.sort_values("Respondents"),
                    x="Respondents",
                    y="Country",
                    orientation="h",
                    text="Respondents",
                    title="Top 10 Countries by Respondent Count"
                )

                fig.update_traces(textposition="outside")
                with st.container(border=True, key="chart_amber_11"):
                    st.plotly_chart(
                        style_chart(fig, 500),
                        use_container_width=True
                    )

            country_rate = treatment_rate_table(
                filtered_df,
                "country"
            )

            country_rate = country_rate[country_rate["Respondents"] >= 20].copy()

            if not country_rate.empty:
                country_rate = country_rate.sort_values(
                    "Treatment_Rate",
                    ascending=True
                )

                fig = px.bar(
                    country_rate,
                    x="Treatment_Rate",
                    y="country",
                    orientation="h",
                    text="Treatment_Rate",
                    title="Treatment Rate by Country (n ≥ 20 respondents)",
                    labels={
                        "country": "Country",
                        "Treatment_Rate": "Treatment Rate (%)"
                    }
                )

                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside"
                )
                fig.update_xaxes(range=[0, 100])

                with st.container(border=True, key="chart_cyan_12"):
                    st.plotly_chart(
                        style_chart(fig, 600),
                        use_container_width=True
                    )

                st.dataframe(
                    country_rate.sort_values("Treatment_Rate", ascending=False),
                    use_container_width=True
                )
            else:
                st.info(
                    "No country has at least 20 respondents under the current filters."
                )
        else:
            st.info("Country data is not available.")

    with demographic_tabs[3]:
        if "Company_Size_Group" in filtered_df.columns:
            company_table = treatment_rate_table(
                filtered_df,
                "Company_Size_Group"
            )

            company_order = [
                "Small",
                "Medium",
                "Large",
                "Enterprise",
                "Unknown"
            ]
            company_table["Company_Size_Group"] = pd.Categorical(
                company_table["Company_Size_Group"],
                categories=company_order,
                ordered=True
            )
            company_table = company_table.sort_values("Company_Size_Group")

            fig = px.bar(
                company_table,
                x="Company_Size_Group",
                y="Treatment_Rate",
                text="Treatment_Rate",
                color="Company_Size_Group",
                title="Treatment Rate by Company Size"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_yaxes(range=[0, 100])

            with st.container(border=True, key="chart_indigo_13"):
                st.plotly_chart(
                    style_chart(fig, 450),
                    use_container_width=True
                )

            st.dataframe(
                company_table,
                use_container_width=True
            )

    with demographic_tabs[4]:
        if "remote_work" in filtered_df.columns:
            remote_table = treatment_rate_table(
                filtered_df,
                "remote_work"
            )

            if not remote_table.empty:
                fig = px.bar(
                    remote_table,
                    x="remote_work",
                    y="Treatment_Rate",
                    text="Treatment_Rate",
                    color="remote_work",
                    title="Treatment Rate by Remote Work"
                )

                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside"
                )

                fig.update_yaxes(range=[0, 100])

                with st.container(border=True, key="chart_coral_14"):
                    st.plotly_chart(
                        style_chart(fig, 450),
                        use_container_width=True
                    )

                st.dataframe(
                    remote_table,
                    use_container_width=True
                )
        else:
            st.info("Remote-work data is not available.")

    with demographic_tabs[5]:
        if "tech_company" in filtered_df.columns:
            tech_table = treatment_rate_table(
                filtered_df,
                "tech_company"
            )

            if not tech_table.empty:
                fig = px.bar(
                    tech_table,
                    x="tech_company",
                    y="Treatment_Rate",
                    text="Treatment_Rate",
                    color="tech_company",
                    title="Treatment Rate by Technology Company"
                )

                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside"
                )

                fig.update_yaxes(range=[0, 100])

                with st.container(border=True, key="chart_teal_15"):
                    st.plotly_chart(
                        style_chart(fig, 450),
                        use_container_width=True
                    )

                st.dataframe(
                    tech_table,
                    use_container_width=True
                )
        else:
            st.info("Technology-company data is not available.")



# ============================================================
# PAGE 5 — ADVANCED INSIGHTS
# ============================================================

elif page == "🔍 Advanced Insights":

    st.markdown(
        '<div class="dashboard-title">Advanced Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Explore correlations, statistical relationships, and advanced "
        "analytical patterns."
        "</div>",
        unsafe_allow_html=True
    )

    advanced_tabs = st.tabs(
        [
            "Association Heatmap",
            "Treatment Relationships",
            "Age vs Treatment",
            "Advanced Summary"
        ]
    )

    with advanced_tabs[0]:
        st.subheader("Association Heatmap — Cramér's V")

        association_variables = [
            "treatment",
            "family_history",
            "work_interfere",
            "benefits",
            "care_options",
            "wellness_program",
            "seek_help",
            "anonymity",
            "mental_health_consequence",
            "obs_consequence"
        ]

        available_association = [
            column
            for column in association_variables
            if column in filtered_df.columns
        ]

        if len(available_association) >= 2:
            association_matrix = pd.DataFrame(
                index=available_association,
                columns=available_association,
                dtype=float
            )

            for row_var in available_association:
                for col_var in available_association:
                    association_matrix.loc[row_var, col_var] = cramers_v(
                        filtered_df[row_var],
                        filtered_df[col_var]
                    )

            fig = px.imshow(
                association_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Blues",
                title="Association Strength Between Key Workplace Variables",
                zmin=0,
                zmax=1
            )

            with st.container(border=True, key="chart_violet_16"):
                st.plotly_chart(
                    style_chart(fig, 780),
                    use_container_width=True
                )

            st.caption(
                "Cramér's V measures association strength between categorical "
                "variables. Values closer to 1 indicate stronger association; "
                "association does not establish causation."
            )
        else:
            st.info("Not enough categorical variables are available.")

    with advanced_tabs[1]:
        st.subheader("Treatment Rate Comparison")

        comparison_variables = [
            "family_history",
            "work_interfere",
            "Gender_Clean",
            "mental_health_consequence",
            "remote_work",
            "Company_Size_Group"
        ]

        available_comparison = [
            column
            for column in comparison_variables
            if column in filtered_df.columns
        ]

        if available_comparison:
            selected_variable = st.selectbox(
                "Select comparison variable",
                available_comparison
            )

            comparison_table = treatment_rate_table(
                filtered_df,
                selected_variable
            )

            if not comparison_table.empty:
                fig = px.scatter(
                    comparison_table,
                    x="Respondents",
                    y="Treatment_Rate",
                    size="Respondents",
                    color=selected_variable,
                    text=selected_variable,
                    title="Respondent Count vs Treatment Rate",
                    labels={
                        "Respondents": "Number of Respondents",
                        "Treatment_Rate": "Treatment Rate (%)"
                    }
                )

                fig.update_traces(
                    textposition="top center"
                )

                with st.container(border=True, key="chart_amber_17"):
                    st.plotly_chart(
                        style_chart(fig, 500),
                        use_container_width=True
                    )

                st.dataframe(
                    comparison_table,
                    use_container_width=True
                )
        else:
            st.info("No comparison variables are available.")

    with advanced_tabs[2]:
        st.subheader("Age Distribution by Treatment")

        if "Age" in filtered_df.columns and "Treatment_Label" in filtered_df.columns:
            age_data = filtered_df.dropna(
                subset=["Age", "Treatment_Label"]
            )

            if not age_data.empty:
                fig = px.histogram(
                    age_data,
                    x="Age",
                    color="Treatment_Label",
                    nbins=25,
                    marginal="box",
                    barmode="overlay",
                    opacity=0.75,
                    title="Age Distribution for Treatment Groups"
                )

                with st.container(border=True, key="chart_cyan_18"):
                    st.plotly_chart(
                        style_chart(fig, 550),
                        use_container_width=True
                    )

                fig = px.box(
                    age_data,
                    x="Treatment_Label",
                    y="Age",
                    color="Treatment_Label",
                    title="Age Distribution by Treatment Status"
                )

                with st.container(border=True, key="chart_indigo_19"):
                    st.plotly_chart(
                        style_chart(fig, 450),
                        use_container_width=True
                    )
            else:
                st.info("No valid age and treatment responses are available.")
        else:
            st.info("Age or treatment data is not available.")

    with advanced_tabs[3]:
        st.subheader("Advanced Analytical Summary")

        summary_rows = []
        assumption_warnings = []

        for variable in [
            "work_interfere",
            "family_history",
            "Gender_Clean",
            "mental_health_consequence",
            "Age_Group",
            "remote_work",
            "Company_Size_Group"
        ]:
            if variable not in filtered_df.columns:
                continue

            chi2, p_value, dof, table, assumption_warning = run_chi_square(
                filtered_df,
                variable
            )

            if p_value is not None:
                summary_rows.append({
                    "Variable": variable,
                    "Chi-square": chi2,
                    "P-value": p_value,
                    "Degrees of Freedom": dof
                })

                if assumption_warning:
                    assumption_warnings.append(
                        f"{variable}: {assumption_warning}"
                    )

        summary_df = pd.DataFrame(summary_rows)

        if not summary_df.empty:
            summary_df["Adjusted P-value"] = benjamini_hochberg(
                summary_df["P-value"].values
            )

            summary_df["Significant (FDR < 0.05)"] = np.where(
                summary_df["Adjusted P-value"] < 0.05,
                "Yes",
                "No"
            )

            display_summary = summary_df.copy()
            display_summary["Chi-square"] = display_summary[
                "Chi-square"
            ].round(4)

            display_summary["P-value"] = display_summary[
                "P-value"
            ].apply(
                lambda value: (
                    "< 0.001"
                    if value < 0.001
                    else f"{value:.4f}"
                )
            )

            display_summary["Adjusted P-value"] = display_summary[
                "Adjusted P-value"
            ].apply(
                lambda value: (
                    "< 0.001"
                    if value < 0.001
                    else f"{value:.4f}"
                )
            )

            st.dataframe(
                display_summary,
                use_container_width=True
            )

            significant = summary_df.loc[
                summary_df["Adjusted P-value"] < 0.05,
                "Variable"
            ].tolist()

            not_significant = summary_df.loc[
                summary_df["Adjusted P-value"] >= 0.05,
                "Variable"
            ].tolist()

            st.markdown(
                f"""
                <div class="success-card">
                <b>Variables with FDR-adjusted p-value &lt; 0.05:</b><br>
                {', '.join(significant) if significant else 'None identified in the available filtered data.'}
                </div>

                <div class="warning-card">
                <b>Variables with FDR-adjusted p-value ≥ 0.05:</b><br>
                {', '.join(not_significant) if not_significant else 'None identified in the available filtered data.'}
                </div>

                <div class="insight-card">
                <b>Interpretation caution:</b><br>
                These tests evaluate statistical association only. Results can
                change when global filters are applied.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No valid chi-square tests could be calculated.")

        if assumption_warnings:
            st.warning(
                "Some tests have small expected frequencies and should be "
                "interpreted cautiously."
            )
            for warning in assumption_warnings:
                st.caption(warning)



# ============================================================
# PAGE 6 — PREDICTIVE MODEL
# ============================================================

elif page == "🤖 Predictive Model":

    st.markdown(
        '<div class="dashboard-title">Predictive Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Logistic regression model for predicting treatment responses."
        "</div>",
        unsafe_allow_html=True
    )

    st.info(
        "The model is trained on the full cleaned dataset; the global dashboard "
        "filters are not applied to model training."
    )

    with st.spinner("Training logistic regression model..."):
        model_results = train_model(df)

    if "error" in model_results:
        st.error(model_results["error"])
    else:
        metrics = model_results["metrics"]

        metric_columns = st.columns(5)

        metric_meta = [
            ("Accuracy", "\U0001F3AF", "#4C6EF5"),
            ("Precision", "\U0001F52C", "#20C997"),
            ("Recall", "\U0001F9ED", "#FF6B6B"),
            ("F1-Score", "\u2696\uFE0F", "#F59F00"),
            ("ROC-AUC", "\U0001F4C8", "#845EF7"),
        ]

        for index, (metric_name, icon, accent) in enumerate(metric_meta):
            with metric_columns[index]:
                metric_value = metrics[metric_name]

                display_value = (
                    f"{metric_value:.2%}"
                    if pd.notna(metric_value)
                    else "N/A"
                )

                st.markdown(
                    kpi_card(metric_name, display_value, icon=icon, accent=accent),
                    unsafe_allow_html=True
                )

        cv_metrics = model_results.get("cv_metrics")

        if cv_metrics:
            st.markdown(
                '<div class="section-title">Cross-Validation Summary</div>',
                unsafe_allow_html=True
            )

            cv_cols = st.columns(3)

            cv_items = [
                ("Accuracy", "Accuracy Mean", "\U0001F3AF", "#4C6EF5"),
                ("F1-Score", "F1 Mean", "\u2696\uFE0F", "#20C997"),
                ("ROC-AUC", "ROC-AUC Mean", "\U0001F4C8", "#FF6B6B"),
            ]

            for col, (label, key, icon, accent) in zip(cv_cols, cv_items):
                values = cv_metrics[key]

                with col:
                    st.markdown(
                        kpi_card(
                            f"CV {label}",
                            f"{np.mean(values):.2%}",
                            icon=icon,
                            accent=accent,
                            delta=f"\u00B1 {np.std(values):.2%}"
                        ),
                        unsafe_allow_html=True
                    )

            st.caption(
                "Cross-validation is calculated on the training data only. "
                "The held-out test set above remains separate for final evaluation."
            )
        else:
            st.info(
                "Cross-validation was not run because the training data "
                "does not contain enough observations in every class."
            )

        st.markdown("")

        model_tabs = st.tabs(
            [
                "Feature Importance",
                "Confusion Matrix",
                "ROC Curve",
                "Classification Report"
            ]
        )

        with model_tabs[0]:
            st.subheader(
                "Strongest Logistic Regression Associations"
            )

            feature_importance = model_results[
                "feature_importance"
            ].copy()

            feature_importance["Display_Feature"] = (
                feature_importance["Feature"]
                .str.replace(
                    "categorical__",
                    "",
                    regex=False
                )
                .str.replace(
                    "numeric__",
                    "",
                    regex=False
                )
                .str.replace(
                    "x0_",
                    "",
                    regex=False
                )
            )

            top_features = feature_importance.head(20).copy()
            top_features = top_features.sort_values("Coefficient")

            fig = px.bar(
                top_features,
                x="Coefficient",
                y="Display_Feature",
                orientation="h",
                color="Coefficient",
                color_continuous_scale="RdBu",
                title="Top 20 Model Coefficients",
                labels={
                    "Coefficient": "Model Coefficient",
                    "Display_Feature": "Feature"
                }
            )

            fig.add_vline(
                x=0,
                line_dash="dash"
            )

            with st.container(border=True, key="chart_coral_20"):
                st.plotly_chart(
                    style_chart(fig, 700),
                    use_container_width=True
                )

            st.dataframe(
                feature_importance[
                    [
                        "Display_Feature",
                        "Coefficient",
                        "Absolute_Coefficient"
                    ]
                ],
                use_container_width=True
            )

            st.info(
                "Positive coefficients indicate higher predicted treatment "
                "likelihood relative to a reference category. Negative "
                "coefficients indicate lower predicted likelihood. These "
                "coefficients represent model associations, not causation."
            )

        with model_tabs[1]:
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                model_results["y_test"],
                model_results["y_pred"],
                labels=[0, 1]
            )

            fig = px.imshow(
                cm,
                text_auto=True,
                x=["Predicted No", "Predicted Yes"],
                y=["Actual No", "Actual Yes"],
                color_continuous_scale="Blues",
                title="Confusion Matrix"
            )

            fig.update_xaxes(side="bottom")

            with st.container(border=True, key="chart_teal_21"):
                st.plotly_chart(
                    style_chart(fig, 450),
                    use_container_width=True
                )

            st.write(
                "Rows represent actual values. Columns represent predicted values."
            )

        with model_tabs[2]:
            st.subheader("ROC Curve")

            if model_results["y_test"].nunique() == 2:
                fpr, tpr, thresholds = roc_curve(
                    model_results["y_test"],
                    model_results["y_prob"]
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"ROC-AUC = {metrics['ROC-AUC']:.3f}"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random Classifier",
                        line=dict(dash="dash")
                    )
                )

                fig.update_layout(
                    title="Receiver Operating Characteristic Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate"
                )

                with st.container(border=True, key="chart_violet_22"):
                    st.plotly_chart(
                        style_chart(fig, 500),
                        use_container_width=True
                    )
            else:
                st.info(
                    "ROC curve cannot be calculated because the test set "
                    "contains only one class."
                )

        with model_tabs[3]:
            st.subheader("Classification Report")

            report = classification_report(
                model_results["y_test"],
                model_results["y_pred"],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df,
                use_container_width=True
            )

            st.warning(
                "This model is intended for analytical exploration only. "
                "It should not be used for diagnosis or individual mental-health "
                "decisions."
            )


# ============================================================
# PAGE 7 — DATA EXPLORER
# ============================================================

elif page == "🗃️ Data Explorer":

    st.markdown(
        '<div class="dashboard-title">Data Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Inspect, filter, search, and download the cleaned survey dataset."
        "</div>",
        unsafe_allow_html=True
    )

    st.caption(
        "All Data Explorer records reflect the active Gender, Age, and Remote Work filters."
    )

    explorer_tabs = st.tabs(
        [
            "Dataset",
            "Missing Values",
            "Column Summary",
            "Download"
        ]
    )

    with explorer_tabs[0]:
        st.subheader("Filtered Dataset")

        search_text = st.text_input(
            "Search within the dataset",
            placeholder="Enter a value to search..."
        )

        explorer_df = filtered_df.copy()

        if search_text:
            mask = explorer_df.astype(str).apply(
                lambda column: column.str.contains(
                    search_text,
                    case=False,
                    na=False,
                    regex=False
                )
            ).any(axis=1)

            explorer_df = explorer_df[mask]

        st.write(
            f"Showing {len(explorer_df):,} records"
        )

        st.dataframe(
            explorer_df,
            use_container_width=True,
            height=600
        )

    with explorer_tabs[1]:
        st.subheader("Missing Value Analysis")

        missing_table = pd.DataFrame({
            "Column": filtered_df.columns,
            "Missing Values": filtered_df.isna().sum().values
        })

        missing_table["Missing Percentage"] = (
            missing_table["Missing Values"]
            / len(filtered_df)
            * 100
            if len(filtered_df) > 0
            else 0
        )

        missing_table = missing_table.sort_values(
            "Missing Values",
            ascending=False
        )

        fig = px.bar(
            missing_table.head(15),
            x="Missing Values",
            y="Column",
            orientation="h",
            text="Missing Values",
            title="Top Columns with Missing Values"
        )

        with st.container(border=True, key="chart_amber_23"):
            st.plotly_chart(
                style_chart(fig, 550),
                use_container_width=True
            )

        st.dataframe(
            missing_table,
            use_container_width=True
        )

    with explorer_tabs[2]:
        st.subheader("Column Summary")

        summary_table = pd.DataFrame({
            "Column": filtered_df.columns,
            "Data Type": filtered_df.dtypes.astype(str).values,
            "Unique Values": [
                filtered_df[column].nunique(
                    dropna=True
                )
                for column in filtered_df.columns
            ],
            "Missing Values": [
                filtered_df[column].isna().sum()
                for column in filtered_df.columns
            ]
        })

        st.dataframe(
            summary_table,
            use_container_width=True,
            height=600
        )

    with explorer_tabs[3]:
        st.subheader("Download Data")

        csv_data = filtered_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Filtered Analytical CSV",
            data=csv_data,
            file_name="mental_health_filtered_data.csv",
            mime="text/csv"
        )

        full_csv = raw_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Original Cleaned Dataset",
            data=full_csv,
            file_name="survey_cleaned.csv",
            mime="text/csv"
        )

        st.success(
            "Your filtered analytical data and original cleaned dataset are ready for download."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.85rem;">
        Mental Health in Tech Analytics Dashboard |
        Built with Python, Pandas, Plotly, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
