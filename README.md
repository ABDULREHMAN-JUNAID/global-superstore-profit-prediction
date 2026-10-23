# Global Superstore Sales Analytics & Profit Prediction

Beginner-friendly Data Science and Machine Learning project using the teacher-provided `Global Superstore.csv`.

## What's included
- `app.py`: seven-section interactive Streamlit app.
- `analysis.py`: standalone data inspection, summary tables and chart export.
- `data/Global Superstore.csv`: source dataset provided for this project.
- `requirements.txt`: Python dependencies.
- `analysis_outputs/`: created after running `analysis.py`.

## Important dataset limitation
The supplied CSV has 51,290 rows and 27 columns. `Order Date` and `Ship Date` contain the unusable value `00:00.0` rather than dates. The dataset also does not include `Customer_Age`, `Customer_Gender`, `Customer_Rating`, `Returned`, `Salesperson`, `Brand`, `Unit_Price`, or `Delivery_Days`. The app does not invent these values. Monthly date trends, return analysis, age/gender analysis, ratings, delivery duration, and salesperson analysis are therefore marked unavailable or adapted.

The dataset does include `Year` and `weeknum`, but because the date fields are corrupted, this project does not claim to recover exact order or ship dates from them.

## 1. Install Python
Install Python 3.10 or newer from https://www.python.org/downloads/ . During Windows installation, select "Add Python to PATH".

## 2. Open the project folder
Extract the ZIP file. Open the extracted `Global_Superstore_Profit_Prediction_Project` folder in VS Code.

## 3. Create a virtual environment (recommended)
Open the VS Code terminal and run:

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install libraries
```bash
pip install -r requirements.txt
```

## 5. Run the data analysis script
```bash
python analysis.py
```
It prints dataset information and creates aggregate CSV tables and charts in `analysis_outputs/`.

## 6. Run the Streamlit app
```bash
streamlit run app.py
```
A browser tab should open. Use the left sidebar to visit all seven sections. The first model-training page may take a little time because three models are trained.

## 7. What the models do
The target is `Profit`. The app trains:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

It compares MAE, MSE, RMSE and R² on a held-out test split. The input features include Discount, Quantity, Sales, Shipping Cost, Category, Sub-Category, Region, Market, Segment, Ship Mode, Order Priority and State. Identifier/name columns are excluded. `Profit` is the target, so it is never included as an input feature.

## 8. Push to GitHub
Create a new repository on GitHub. In the project folder, run:
```bash
git init
git add app.py analysis.py requirements.txt README.md data/Global\ Superstore.csv
git commit -m "Build Global Superstore profit prediction project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```
On Windows PowerShell, if the escaped filename command is inconvenient, use:
```bash
git add .
```

## 9. Deploy on Streamlit Community Cloud
1. Push the project to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository and branch `main`.
4. Set the main file path to `app.py`.
5. Deploy. Streamlit installs packages from `requirements.txt`.
6. Open the public app URL and test every page and the prediction form.

## Business recommendations
Use the actual output of the dashboard before finalizing recommendations. Suitable evidence-based themes include:
1. Prioritize categories/sub-categories with strong total profit.
2. Review products with high sales but low/negative profit.
3. Investigate the relationship between discount and profit before increasing discounts.
4. Compare regional profit and tailor inventory/pricing to regional demand.
5. Review shipping modes using profit and shipping-cost summaries.
6. Use the best-performing model's test metrics, not training performance, to judge prediction quality.

For the final presentation, state clearly that date, return, rating, gender/age, delivery-duration and salesperson analyses are limited by missing/corrupted columns in the provided file.
