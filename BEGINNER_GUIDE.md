# Beginner Guide: Run the Project

## What is this project?
It summarizes the Global Superstore sales dataset and trains three machine-learning models to estimate profit.

## What to click first
1. Open the app.
2. Visit **Home** and read the data limitations.
3. Visit **Dataset Explorer** to inspect the columns and filter rows.
4. Visit **Sales Dashboard** to see totals and category/region charts.
5. Visit **Product & Regional Analysis** to find top products and low-profit products.
6. Visit **Customer & Delivery Analysis** to analyze customer segments and shipping modes.
7. Visit **Model Evaluation** and compare the three models.
8. Visit **Profit Prediction**, fill in order details, and click **Predict Profit**.

## Important ML vocabulary
- **Feature/input:** information given to a model, such as Sales, Discount, Category or Quantity.
- **Target:** the value the model learns to predict. Here, the target is Profit.
- **Train set:** rows used to teach the model.
- **Test set:** separate rows used to estimate performance on unseen examples.
- **MAE:** average absolute prediction error; lower is better.
- **MSE:** average squared prediction error; lower is better.
- **RMSE:** square root of MSE, in the same unit as Profit; lower is better.
- **R²:** how well the model explains variation in the target; closer to 1 is generally better, but it can be negative.

## How to explain the date problem
"The file provided for this project contains the literal value 00:00.0 in Order Date and Ship Date. I checked the columns and did not fabricate dates. Therefore, monthly trends and delivery-duration analysis cannot be computed reliably from this copy of the dataset."

## How to explain missing assignment columns
"The assignment template mentions age, gender, ratings, returns, delivery days and salesperson, but these columns are absent from the provided Global Superstore file. I used the available customer segment and shipping-mode fields as related analyses and documented the unavailable requirements."
