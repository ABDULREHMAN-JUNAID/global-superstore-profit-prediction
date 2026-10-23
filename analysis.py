"""Run standalone exploratory analysis and export a text summary plus charts."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Global Superstore.csv"
OUT = ROOT / "analysis_outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
print("SHAPE:", df.shape)
print("\nCOLUMNS AND TYPES:\n", df.dtypes)
print("\nMISSING VALUES:\n", df.isna().sum())
print("\nDUPLICATE ROWS:", df.duplicated().sum())
print("\nDESCRIPTIVE STATISTICS:\n", df.describe(include="all").transpose().to_string())
print("\nTOTAL SALES:", df["Sales"].sum())
print("TOTAL PROFIT:", df["Profit"].sum())
print("UNIQUE ORDERS:", df["Order ID"].nunique())
print("UNIQUE CUSTOMERS:", df["Customer ID"].nunique())
print("UNIQUE PRODUCTS:", df["Product Name"].nunique())
print("UNIQUE BRANDS: Not available; dataset has no Brand column.")
print("ORDER DATE EXAMPLE:", df["Order Date"].head().tolist())
print("NOTE: Order Date / Ship Date values are unusable ('00:00.0'); no monthly trend or delivery duration is calculated.")

# Export useful aggregate tables
df.groupby("Category")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False).to_csv(OUT/"category_performance.csv")
df.groupby("Region")[["Sales", "Profit"]].sum().sort_values("Profit", ascending=False).to_csv(OUT/"region_performance.csv")
df.groupby("Sub-Category")[["Sales", "Profit"]].sum().sort_values("Profit", ascending=False).to_csv(OUT/"subcategory_performance.csv")
df.groupby(["Customer ID", "Customer Name"])[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False).head(20).to_csv(OUT/"top_customers.csv")

sns.set_theme(style="whitegrid")
for col, title, filename in [
    ("Category", "Sales by Category", "sales_by_category.png"),
    ("Region", "Profit by Region", "profit_by_region.png"),
    ("Sub-Category", "Profit by Sub-Category", "profit_by_subcategory.png"),
    ("Ship Mode", "Rows by Ship Mode", "ship_mode_distribution.png"),
]:
    value_col = "Sales" if "Sales" in title else ("Profit" if "Profit" in title else None)
    if value_col:
        values = df.groupby(col)[value_col].sum().sort_values()
    else:
        values = df[col].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    values.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(value_col or "Row count")
    ax.set_ylabel(col)
    fig.tight_layout()
    fig.savefig(OUT/filename, dpi=150)
    plt.close(fig)

print(f"\nAnalysis outputs saved to: {OUT}")
