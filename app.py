import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Global Superstore Profit Prediction", page_icon="📊", layout="wide")

# Professional dashboard theme
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #F4F7FB;
        color: #172B4D;
    }
    /* Fix main page text visibility */
.stApp p,
.stApp label,
.stApp span,
.stApp [data-testid="stMarkdownContainer"] {
    color: #172B4D;
}
    /* Main page headings */
    h1 {
        color: #14243A !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        color: #23415A !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #14243A;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DFE7F1;
        border-left: 5px solid #18A999;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(20, 36, 58, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #52657A;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #14243A;
        font-weight: 800;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background-color: #18A999;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 9px 18px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #128577;
        color: #FFFFFF;
        border: none;
    }

    /* Input fields */
    [data-testid="stSelectbox"],
    [data-testid="stNumberInput"],
    [data-testid="stMultiSelect"] {
        border-radius: 8px;
    }

    /* Data tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #DFE7F1;
        border-radius: 10px;
    }

    /* Subtle horizontal separators */
    hr {
        border-color: #DCE5EF;
    }
    
    /* Make warning messages readable */
    [data-testid="stAlert"] {
        color: #14243A !important;
        background-color: #FFF4D6 !important;
        border: 1px solid #E6B84C !important;
        border-radius: 10px;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] span {
        color: #14243A !important;
    }
    
    /* ===== PREMIUM SIDEBAR DESIGN ===== */

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101D33 0%, #172E49 55%, #0E2438 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.10);
    }

    /* Sidebar spacing */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #F3F7FC;
    }

    /* Sidebar headings */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        letter-spacing: 0.3px;
    }

    /* Radio navigation options */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 8px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 5px;
        transition: all 0.2s ease;
    }

    /* Hover effect */
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(24, 169, 153, 0.18);
        border-color: #18A999;
        transform: translateX(3px);
    }

    /* Selected navigation option */
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, #18A999, #138A91);
        border: 1px solid #55D6C2;
        box-shadow: 0 4px 12px rgba(24, 169, 153, 0.20);
    }

    /* Radio circles */
    [data-testid="stSidebar"] [role="radio"] {
        accent-color: #18A999;
    }

    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.16);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Sidebar captions */
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #B8C9DC !important;
        font-size: 0.82rem;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(24, 169, 153, 0.12);
        border: 1px solid #18A999;
        color: #FFFFFF !important;
        border-radius: 9px;
        font-weight: 700;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #18A999;
        color: #FFFFFF !important;
    }
    
    /* ===== SIDEBAR NAVIGATION POLISH ===== */

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 8px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 10px 12px;
        margin: 4px 0;
        transition: background 0.2s ease, border-color 0.2s ease;
    }

    /* Hover effect */
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(24, 169, 153, 0.18);
        border-color: #36CFC0;
    }

    

    /* Keep navigation text readable */
    [data-testid="stSidebar"] [role="radiogroup"] label p {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    /* ===== SIDEBAR SPACING & TYPOGRAPHY ===== */

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.7rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        padding-top: 6px;
        padding-bottom: 6px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        min-height: 42px;
        display: flex;
        align-items: center;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #B8C9DC !important;
        font-size: 0.78rem;
        line-height: 1.6;
    }

    [data-testid="stSidebar"] hr {
        margin-top: 14px;
        margin-bottom: 14px;
        border-color: rgba(255, 255, 255, 0.15);
    }
    /* Fix text visibility when text is selected */
::selection {
    background-color: #b7d7ff !important;
    color: #172033 !important;
}

::-moz-selection {
    background-color: #b7d7ff !important;
    color: #172033 !important;
}

</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Global Superstore.csv")

TARGET = "Profit"
NUMERIC_FEATURES = ["Discount", "Quantity", "Sales", "Shipping Cost"]
CATEGORICAL_FEATURES = [
    "Category", "Sub-Category", "Region", "Market", "Segment",
    "Ship Mode", "Order Priority", "State"
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MISSING_ASSIGNMENT_COLUMNS = [
    "Customer_Age", "Customer_Gender", "Customer_Rating", "Returned", "Salesperson",
    "Unit_Price", "Delivery_Days"
]

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Dataset not found. Put Global Superstore.csv inside the project's data folder."
        )
    df = pd.read_csv(DATA_PATH)
    # Standardize column names by trimming spaces only; preserve the source's column names.
    df.columns = [str(c).strip() for c in df.columns]
    return df

def clean_data(raw):
    df = raw.copy()
    # The uploaded source has dates stored as the literal text '00:00.0'.
    # Convert to datetime only if valid dates exist; do not invent replacement dates.
    for col in ["Order Date", "Ship Date"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                df[col] = parsed
            else:
                df[col] = pd.NaT
    for col in ["Discount", "Profit", "Quantity", "Sales", "Shipping Cost", "Year", "weeknum"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # No automatic deletion of unusual values; report them to the user.
    return df

@st.cache_data
def get_model_results(df):
    # Train on a sample only if the file grows substantially; current dataset is ~51k rows.
    usable = df.dropna(subset=[TARGET]).copy()
    features = [c for c in MODEL_FEATURES if c in usable.columns]
    numeric = [c for c in NUMERIC_FEATURES if c in features]
    categorical = [c for c in CATEGORICAL_FEATURES if c in features]
    X = usable[features].copy()
    y = usable[TARGET].astype(float)
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in categorical:
        X[col] = X[col].astype("string").fillna("Unknown").astype(str)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical)
        ],
        remainder="drop"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=14, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=80, max_depth=18, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )
    }
    fitted = {}
    predictions = {}
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        predictions[name] = pred
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mean_squared_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred)
        })
   
    metrics = pd.DataFrame(rows).sort_values("RMSE")

    # Get feature importance from the trained Random Forest model
    rf_pipeline = fitted["Random Forest"]
    rf_model = rf_pipeline.named_steps["model"]
    preprocessor = rf_pipeline.named_steps["preprocess"]

    feature_names = preprocessor.get_feature_names_out()
    importance_values = rf_model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    }).sort_values("Importance", ascending=False)

    return {
        "models": fitted,
        "predictions": predictions,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "feature_importance": feature_importance
    }


def money(value):
    return f"${value:,.2f}"

def show_missing_columns_note():
    absent = [c for c in MISSING_ASSIGNMENT_COLUMNS if c not in df.columns]
    if absent:
        st.warning(
            "Your assignment describes some columns that are not in this teacher-provided dataset: "
            + ", ".join(absent)
            + ". This app will not invent these values. The related analyses are marked unavailable."
        )

raw_df = load_data()
df = clean_data(raw_df)



st.sidebar.markdown("---")

st.sidebar.markdown("# 📊")
st.sidebar.markdown("## SUPERSTORE")
st.sidebar.markdown("**PROFIT ANALYTICS**")

st.sidebar.caption("Business Intelligence Dashboard")
st.sidebar.markdown(
    "Explore data. Discover insights. Predict profit."
)



st.sidebar.markdown("---")

st.sidebar.markdown(
    "<p style='color:#36CFC0; font-size:11px; "
    "font-weight:800; letter-spacing:1.5px; "
    "margin-bottom:8px;'>MAIN MENU</p>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Choose a section",
    [
        "1. Home",
        "2. Dataset Explorer",
        "3. Sales Dashboard",
        "4. Product & Regional Analysis",
        "5. Customer & Delivery Analysis",
        "6. Model Evaluation",
        "7. Profit Prediction"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")





if page == "1. Home":
    st.title("Global Superstore Sales Analytics & Profit Prediction")
    st.subheader("Project overview")
    st.write(
        "This project explores retail sales, identifies products and regions associated with profit, "
        "and trains regression models to estimate order profit from order characteristics."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns)}")
    c3.metric("Target variable", "Profit")
    st.markdown("### Business problem")
    st.write(
        "A global retailer wants to understand which products, markets, customer segments and shipping "
        "choices are associated with stronger sales and profit, and wants a model to estimate profit."
    )
    st.markdown("### Project objectives")
    st.markdown(
        "- Inspect and clean the provided dataset.\n"
        "- Explore sales, profit, discounts, products, regions and customer segments.\n"
        "- Engineer useful features without inventing missing data.\n"
        "- Train and compare Linear Regression, Decision Tree and Random Forest models.\n"
        "- Provide an interactive profit prediction form.\n"
        "- Translate findings into practical business recommendations."
    )
    st.markdown("### Dataset description")
    st.write(
        "The file contains 51,290 rows and 27 columns, including Sales, Profit, Quantity, Discount, "
        "Category, Sub-Category, Region, Market, Segment and shipping information."
    )
    
    show_missing_columns_note()
    st.info(
        "Data-quality limitation: Order Date and Ship Date in this file contain '00:00.0' rather than "
        "usable dates. The app does not fabricate dates, monthly trends, or delivery duration."
    )

elif page == "2. Dataset Explorer":
    st.title("Dataset Explorer")
    st.write("Explore the original dataset and download a filtered subset.")
    st.markdown("### Dataset summary")
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", len(df.columns))
    b.metric("Missing cells", f"{raw_df.isna().sum().sum():,}")
    d.metric("Duplicate rows", f"{raw_df.duplicated().sum():,}")
    with st.expander("Column names and data types"):
        st.dataframe(pd.DataFrame({"Column": df.columns, "Data type": df.dtypes.astype(str).values}), use_container_width=True)
    with st.expander("Descriptive statistics"):
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
    with st.expander("First five rows"):
        st.dataframe(raw_df.head(), use_container_width=True)
    with st.expander("Last five rows"):
        st.dataframe(raw_df.tail(), use_container_width=True)

    filtered = df.copy()
    r1, r2, r3 = st.columns(3)
    if "Region" in filtered:
        regions = sorted(filtered["Region"].dropna().astype(str).unique())
        chosen_regions = r1.multiselect("Region filter", regions, default=regions)
        if chosen_regions:
            filtered = filtered[filtered["Region"].astype(str).isin(chosen_regions)]
    if "Category" in filtered:
        cats = sorted(filtered["Category"].dropna().astype(str).unique())
        chosen_cats = r2.multiselect("Category filter", cats, default=cats)
        if chosen_cats:
            filtered = filtered[filtered["Category"].astype(str).isin(chosen_cats)]
    if "Product Name" in filtered:
        products = sorted(filtered["Product Name"].dropna().astype(str).unique())
        chosen_products = r3.multiselect("Product filter (optional)", products)
        if chosen_products:
            filtered = filtered[filtered["Product Name"].astype(str).isin(chosen_products)]
    st.caption(f"Showing {len(filtered):,} filtered rows.")
    rows_to_show = st.selectbox("Number of rows to preview", [5, 10, 25, 50, 100], index=2)
    st.dataframe(filtered.head(rows_to_show), use_container_width=True)
    st.download_button(
        "Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_global_superstore.csv",
        mime="text/csv"
    )
    st.markdown("### Data quality checks")
    st.write("Missing values by column")
    st.dataframe(raw_df.isna().sum().rename("Missing values").to_frame(), use_container_width=True)
    st.write("Date columns: the source values are unusable, so actual date filtering is not available.")

elif page == "3. Sales Dashboard":
    st.title("Sales Dashboard")
    
    # Interactive dashboard filters
    st.markdown("### Filter your analysis")

    f1, f2, f3 = st.columns(3)

    years = sorted(df["Year"].dropna().unique().tolist())
    regions = sorted(df["Region"].dropna().unique().tolist())
    categories = sorted(df["Category"].dropna().unique().tolist())

    selected_years = f1.multiselect(
        "Select year",
        years,
        default=years
    )

    selected_regions = f2.multiselect(
        "Select region",
        regions,
        default=regions
    )

    selected_categories = f3.multiselect(
        "Select category",
        categories,
        default=categories
    )

    # Apply the selected filters
    dashboard_df = df[
        df["Year"].isin(selected_years)
        & df["Region"].isin(selected_regions)
        & df["Category"].isin(selected_categories)
    ].copy()

    if dashboard_df.empty:
        st.warning("No data matches these filters. Please select at least one option in each filter.")
        st.stop()

    
    total_sales = dashboard_df["Sales"].sum()
    total_profit = dashboard_df["Profit"].sum()
    orders = dashboard_df["Order ID"].nunique()
    avg_order_value = dashboard_df.groupby("Order ID")["Sales"].sum().mean()
    avg_profit = dashboard_df.groupby("Order ID")["Profit"].sum().mean()
    profit_margin = (
        total_profit / total_sales * 100
        if total_sales != 0 else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", money(total_sales))
    c2.metric("Total Profit", money(total_profit))
    c3.metric("Total Orders", f"{orders:,}")
    c4.metric("Average Order Value", money(avg_order_value))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Average Profit per Order", money(avg_profit))
    c6.metric("Average Discount", f"{df['Discount'].mean():.1%}")
    c7.metric("Average Shipping Cost", money(df["Shipping Cost"].mean()))
    c8.metric("Customers", f"{df['Customer ID'].nunique():,}")
    left, right = st.columns(2)
    with left:
        cat_sales = dashboard_df.groupby("Category")["Sales"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        cat_sales.plot(kind="barh", ax=ax)
        ax.set_title("Sales by Category"); ax.set_xlabel("Sales"); ax.set_ylabel("Category")
        st.pyplot(fig); plt.close(fig)
    with right:
        cat_profit = dashboard_df.groupby("Category")["Profit"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        cat_profit.plot(kind="barh", ax=ax)
        ax.set_title("Profit by Category"); ax.set_xlabel("Profit"); ax.set_ylabel("Category")
        st.pyplot(fig); plt.close(fig)
    st.markdown("### Sales and profit by region")
    region_stats = dashboard_df.groupby("Region")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)
    st.dataframe(region_stats.style.format({"Sales": "${:,.2f}", "Profit": "${:,.2f}"}), use_container_width=True)
    st.bar_chart(region_stats[["Sales", "Profit"]])

elif page == "4. Product & Regional Analysis":
    st.title("Product & Regional Analysis")
    left, right = st.columns(2)
    with left:
        top_sales = df.groupby("Product Name")["Sales"].sum().nlargest(10).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        top_sales.plot(kind="barh", ax=ax)
        ax.set_title("Top 10 Products by Sales"); ax.set_xlabel("Sales"); ax.set_ylabel("Product")
        st.pyplot(fig); plt.close(fig)
    with right:
        top_profit = df.groupby("Product Name")["Profit"].sum().nlargest(10).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        top_profit.plot(kind="barh", ax=ax)
        ax.set_title("Top 10 Products by Profit"); ax.set_xlabel("Profit"); ax.set_ylabel("Product")
        st.pyplot(fig); plt.close(fig)
    st.markdown("### Category and sub-category performance")
    sub = df.groupby(["Category", "Sub-Category"])[["Sales", "Profit"]].sum().sort_values("Profit", ascending=False)
    st.dataframe(sub.style.format({"Sales": "${:,.2f}", "Profit": "${:,.2f}"}), use_container_width=True)
    left, right = st.columns(2)
    with left:
        top_markets = df.groupby("Market")["Sales"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        top_markets.plot(kind="bar", ax=ax)
        ax.set_title("Sales by Market"); ax.set_xlabel("Market"); ax.set_ylabel("Sales")
        ax.tick_params(axis="x", rotation=35)
        st.pyplot(fig); plt.close(fig)
    with right:
        top_regions = df.groupby("Region")["Profit"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        top_regions.plot(kind="bar", ax=ax)
        ax.set_title("Profit by Region"); ax.set_xlabel("Region"); ax.set_ylabel("Profit")
        ax.tick_params(axis="x", rotation=35)
        st.pyplot(fig); plt.close(fig)
    st.markdown("### Salesperson performance")
    st.info("A Salesperson column is not present in the supplied dataset, so salesperson rankings cannot be calculated.")
    st.markdown("### Products with high sales but low profit")
    product_stats = df.groupby("Product Name")[["Sales", "Profit"]].sum()
    sales_cutoff = product_stats["Sales"].quantile(0.75)
    low_profit_cutoff = product_stats["Profit"].quantile(0.25)
    candidates = product_stats[(product_stats["Sales"] >= sales_cutoff) & (product_stats["Profit"] <= low_profit_cutoff)].sort_values("Sales", ascending=False).head(15)
    st.dataframe(candidates.style.format({"Sales": "${:,.2f}", "Profit": "${:,.2f}"}), use_container_width=True)

elif page == "5. Customer & Delivery Analysis":
    st.title("Customer & Delivery Analysis")
    st.markdown("### Customer segment analysis")
    left, right = st.columns(2)
    with left:
        segment_sales = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        segment_sales.plot(kind="bar", ax=ax)
        ax.set_title("Sales by Customer Segment"); ax.set_xlabel("Segment"); ax.set_ylabel("Sales")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig); plt.close(fig)
    with right:
        segment_profit = df.groupby("Segment")["Profit"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        segment_profit.plot(kind="bar", ax=ax)
        ax.set_title("Profit by Customer Segment"); ax.set_xlabel("Segment"); ax.set_ylabel("Profit")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig); plt.close(fig)
    st.markdown("### Top customers by sales")
    top_customers = df.groupby(["Customer ID", "Customer Name"])[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False).head(10)
    st.dataframe(top_customers.style.format({"Sales": "${:,.2f}", "Profit": "${:,.2f}"}), use_container_width=True)
    st.markdown("### Shipping analysis")
    left, right = st.columns(2)
    with left:
        ship_counts = df["Ship Mode"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ship_counts.plot(kind="bar", ax=ax)
        ax.set_title("Orders by Ship Mode"); ax.set_xlabel("Ship Mode"); ax.set_ylabel("Rows")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig); plt.close(fig)
    with right:
        ship_profit = df.groupby("Ship Mode")["Profit"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ship_profit.plot(kind="bar", ax=ax)
        ax.set_title("Average Profit by Ship Mode"); ax.set_xlabel("Ship Mode"); ax.set_ylabel("Average Profit")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig); plt.close(fig)
    st.info(
        "The supplied file does not contain Customer_Age, Customer_Gender, Customer_Rating, Returned, "
        "or Delivery_Days. Ship Date and Order Date are unusable in this copy, so delivery duration, "
        "late-delivery returns, ratings, and return-rate analyses cannot be honestly calculated."
    )

elif page == "6. Model Evaluation":
    st.title("Model Evaluation")
    st.write("Models predict Profit using order characteristics available in this dataset.")
    with st.spinner("Training three regression models. The first run may take a little time..."):
        results = get_model_results(df)
        
        st.markdown("### Random Forest: Top 10 Important Features")

        importance = results["feature_importance"].head(10).copy()
        importance = importance.sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(importance["Feature"], importance["Importance"])
        ax.set_title("Top 10 Features for Profit Prediction")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Model comparison")
    st.dataframe(results["metrics"].style.format({
        "MAE": "${:,.2f}", "MSE": "{:,.2f}", "RMSE": "${:,.2f}", "R²": "{:.4f}"
    }), use_container_width=True)
    best_name = results["metrics"].iloc[0]["Model"]
    st.success(f"Best model by lowest RMSE on this test split: {best_name}")
    st.caption("MAE/RMSE are lower-is-better; R² is generally better when closer to 1. Metrics are estimates on a held-out test set.")
    model_choice = st.selectbox("Choose model for diagnostic charts", list(results["models"].keys()), index=list(results["models"].keys()).index(best_name))
    pred = results["predictions"][model_choice]
    actual = results["y_test"].to_numpy()
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(actual, pred, alpha=0.35, s=12)
        low = min(np.min(actual), np.min(pred)); high = max(np.max(actual), np.max(pred))
        ax.plot([low, high], [low, high], linestyle="--")
        ax.set_title(f"Actual vs Predicted Profit — {model_choice}")
        ax.set_xlabel("Actual Profit"); ax.set_ylabel("Predicted Profit")
        st.pyplot(fig); plt.close(fig)
    with right:
        residuals = actual - pred
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.histplot(residuals, bins=40, kde=True, ax=ax)
        ax.set_title("Residual Distribution"); ax.set_xlabel("Actual − Predicted Profit")
        st.pyplot(fig); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_metrics = results["metrics"].set_index("Model")[["RMSE"]]
    plot_metrics.plot(kind="bar", ax=ax, legend=False)
    ax.set_title("RMSE Comparison (lower is better)"); ax.set_ylabel("RMSE"); ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig); plt.close(fig)
    selected_pipe = results["models"][model_choice]
    model_obj = selected_pipe.named_steps["model"]
    prep = selected_pipe.named_steps["preprocess"]
    if hasattr(model_obj, "feature_importances_"):
        try:
            feature_names = prep.get_feature_names_out()
            importances = pd.Series(model_obj.feature_importances_, index=feature_names).nlargest(15).sort_values()
            fig, ax = plt.subplots(figsize=(8, 5))
            importances.plot(kind="barh", ax=ax)
            ax.set_title(f"Top Feature Importances — {model_choice}")
            ax.set_xlabel("Importance"); ax.set_ylabel("Feature")
            st.pyplot(fig); plt.close(fig)
        except Exception:
            st.info("Feature importance is not available for this model/preprocessing setup.")
    else:
        st.info("Feature importance is available for the Decision Tree and Random Forest models, not Linear Regression.")

elif page == "7. Profit Prediction":
    st.title("Profit Prediction")
    st.write("Enter the details of a hypothetical order. The model estimates profit; it does not guarantee the actual outcome.")
    with st.spinner("Preparing models for prediction..."):
        results = get_model_results(df)
    model_name = st.selectbox("Prediction model", list(results["models"].keys()), index=list(results["models"].keys()).index(results["metrics"].iloc[0]["Model"]))
    c1, c2, c3 = st.columns(3)
    category = c1.selectbox("Category", sorted(df["Category"].dropna().unique()))
    subcategory_options = sorted(df.loc[df["Category"] == category, "Sub-Category"].dropna().unique())
    subcategory = c2.selectbox("Sub-Category", subcategory_options)
    region = c3.selectbox("Region", sorted(df["Region"].dropna().unique()))
    c4, c5, c6 = st.columns(3)
    market = c4.selectbox("Market", sorted(df["Market"].dropna().unique()))
    segment = c5.selectbox("Segment", sorted(df["Segment"].dropna().unique()))
    ship_mode = c6.selectbox("Ship Mode", sorted(df["Ship Mode"].dropna().unique()))
    c7, c8, c9, c10 = st.columns(4)
    priority = c7.selectbox("Order Priority", sorted(df["Order Priority"].dropna().unique()))
    state = c8.selectbox("State / Province", sorted(df["State"].dropna().unique()))
    quantity = c9.number_input("Quantity", min_value=1, max_value=1000, value=3, step=1)
    discount = c10.number_input("Discount (0 to 1)", min_value=0.0, max_value=1.0, value=0.0, step=0.05, help="For example, 0.20 means 20% discount.")
    c11, c12 = st.columns(2)
    sales = c11.number_input("Sales ($)", min_value=0.0, value=100.0, step=10.0)
    shipping_cost = c12.number_input("Shipping Cost ($)", min_value=0.0, value=5.0, step=1.0)
    if st.button("Predict Profit", type="primary"):
        new_order = pd.DataFrame([{
            "Discount": discount, "Quantity": quantity, "Sales": sales, "Shipping Cost": shipping_cost,
            "Category": category, "Sub-Category": subcategory, "Region": region, "Market": market,
            "Segment": segment, "Ship Mode": ship_mode, "Order Priority": priority, "State": state
        }])
        model = results["models"][model_name]
        prediction = float(model.predict(new_order[results["features"]])[0])
        
        st.metric("Predicted Profit", money(prediction))

        profit_margin = (prediction / sales * 100) if sales > 0 else 0
        st.metric("Predicted Profit Margin", f"{profit_margin:.2f}%")

        if prediction < 0:
            st.error("Loss warning: this order is predicted to make a loss. Review discount, cost, and product mix.")
            profit_category = "Loss"
        elif prediction < 25:
            profit_category = "Low Profit"
        elif prediction < 100:
            profit_category = "Medium Profit"
        else:
            profit_category = "High Profit"
        st.write(f"Estimated profit category: **{profit_category}**")
        st.caption("Profit categories use simple example thresholds and can be adjusted for the business.")
        st.warning("Prediction quality depends on the dataset. Do not treat the estimate as a guarantee.")
