Model 1: Profit Prediction — Random Forest Regressor
Objective: Predict how much profit an order or product is expected to generate.

For example, suppose a business wants to sell a product with the following characteristics:
- Product category: Electronics
- Product price: ₹2,000
- Product cost: ₹1,400
- Freight expense: ₹150
- Discount: ₹100
The model estimates the expected profit using patterns learned from historical data.

Target variable: profit

Recommended algorithms:
1. Linear Regression — baseline model.
2. Decision Tree Regressor — interpretable model.
3. Random Forest Regressor — recommended starting point.
4. XGBoost Regressor — another model worth comparing.
Features you could use:
- price
- product_cost
- freight_value
- discount
- return_loss
- product_category_name_english
- product_weight_g
- product_length_cm
- product_height_cm
- product_width_cm
- Seller location and product category.
How this helps the business
The business can estimate expected profitability before deciding which products to promote or discount.

Important limitation: Your dataset already contains profit, product_cost, discount, and return_loss. If profit is calculated directly from these variables, using them as model inputs could make the prediction artificially easy.

For a realistic model, first verify the profit formula. Exclude variables that directly determine the target if they would not be available at prediction time. Compare the model against a simple formula-based baseline.

Also, if your goal is to predict profit before an order is placed, you must not use information that becomes available only after delivery.





##


Complete Machine Learning Project: Profit Prediction Using Random Forest Regressor
We will build your model using Python and your cleaned Cleaned_master_Olist_Data.csv dataset.

The goal is to train a Machine Learning model that predicts the expected profit of an e-commerce order or product item.

We will complete these steps:
1. Load your cleaned dataset.
2. Inspect the columns and understand how profit is calculated.
3. Clean the data and prepare features.
4. Select the target variable (profit).
5. Split the dataset into training and testing sets.
6. Train a Linear Regression model as a baseline.
7. Train a Random Forest Regressor.
8. Evaluate both models using MAE, RMSE, and R².
9. Compare actual profit with predicted profit.
10. Predict profit for a new example.
11. Save the trained model so you can use it again.
One important point: Your dataset already contains the profit column. We must avoid data leakage, where the model receives information that directly reveals the answer it is supposed to predict.

For the first version, we'll predict profit using product, order, and seller characteristics. We'll inspect the dataset before deciding which columns to exclude.



######


Step 1: Set up your Python environment
You can run this project in VS Code, Jupyter Notebook, or Google Colab.
Install the required libraries
Open the terminal and run:
pip install pandas numpy scikit-learn matplotlib seaborn joblib

Keep your files organized
Create a folder named Olist_Profit_Prediction with this structure:
Olist_Profit_Prediction/
│
├── Cleaned_master_Olist_Data.csv
├── profit_prediction.py
├── profit_prediction_model.pkl
└── actual_vs_predicted.png

Put your cleaned CSV file inside this folder. The Python code below will generate the model file and graph when you run it.


######

Step 2: Understand the prediction approach
Your dataset contains these columns:
- price
- product_cost
- freight_value
- discount
- return_loss
- profit
- Product category and dimensions
- Customer and seller locations
- Order purchase timestamp
The first five financial columns are important because they may determine the profit directly.

For example, if the dataset calculates profit as:
\[
\text{Profit}=\text{Price}-\text{Product Cost}-\text{Freight}-\text{Discount}-\text{Return Loss}
\]
then using all five financial values to predict profit would amount to reconstructing a formula rather than learning a meaningful pattern.

Therefore, our primary ML model will exclude product_cost, freight_value, discount, and return_loss. It will use product characteristics, price, location, and purchase timing to estimate expected historical profit.

This model will estimate profit from historical patterns; it will not know the exact product cost or shipping expense for a new transaction.

If you already know all the financial components, calculate profit directly using your verified business formula instead of using ML.



######



Step 3: Complete Python code
Create a file named profit_prediction.py and copy the following code into it.

The script performs data loading, preprocessing, model training, evaluation, visualization, model saving, and prediction for a new example.
# ============================================================# E-COMMERCE PROFIT PREDICTION# Algorithms: Linear Regression and Random Forest Regressor# Dataset: Cleaned_master_Olist_Data.csv# ============================================================import osimport warningsimport joblibimport numpy as npimport pandas as pdimport matplotlib.pyplot as pltfrom sklearn.compose import ColumnTransformerfrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import OneHotEncoder, StandardScalerfrom sklearn.impute import SimpleImputerfrom sklearn.linear_model import LinearRegressionfrom sklearn.ensemble import RandomForestRegressorfrom sklearn.metrics import (    mean_absolute_error,    mean_squared_error,    r2_score)warnings.filterwarnings("ignore")# ============================================================# 1. LOAD DATASET# ============================================================FILE_PATH = "Cleaned_master_Olist_Data.csv"if not os.path.exists(FILE_PATH):    raise FileNotFoundError(        f"Dataset not found: {FILE_PATH}\n"


######


Step 4: Run the project
Open your terminal in the project folder.

Run:
python profit_prediction.py

The program will automatically execute all the steps.
Expected output
Your exact results will depend on the data and the trained models. The terminal should display results in this format:
============================================================
DATASET INFORMATION
============================================================

Dataset shape: (114835, 44)

============================================================
TRAINING: Linear Regression
============================================================

MAE  : [calculated value]
RMSE : [calculated value]
R2   : [calculated value]

============================================================
TRAINING: Random Forest
============================================================

MAE  : [calculated value]
RMSE : [calculated value]
R2   : [calculated value]

============================================================
MODEL COMPARISON
============================================================

                       MAE       RMSE        R2
Linear Regression       ...        ...        ...
Random Forest           ...        ...        ...

Best model by test MAE: [calculated model]

These are output placeholders, not actual measured results. You will obtain your real metrics when you execute the script.





######




Step 5: Understand the evaluation metrics
Three metrics will help you assess your model.
Metric	Meaning	What you want
MAE	Average absolute difference between actual and predicted profit	Lower
RMSE	Penalizes larger prediction errors more heavily	Lower
R²	Measures how well the model explains variation in the target	Generally higher


Example of MAE
Suppose the model produces an MAE of 12.

This means its predictions differ from actual profit by an average absolute amount of approximately 12 units of the target currency on the test dataset.

It does not mean every prediction is wrong by exactly 12.
Example of R²
- R² = 1.0: Perfect predictions on the evaluation dataset.
- R² = 0.0: No improvement over predicting the test-set mean, under the standard R² definition.
- R² < 0: Worse than that mean-based baseline.
A negative R² is possible, especially when predicting future records from historical data. It does not automatically mean your code is incorrect.


######



Step 6: Understand the actual-versus-predicted graph
After training, the program creates actual_vs_predicted.png.

The graph compares the real profit values with the predictions made by your model.

Interpret it as follows:
- Points close to the diagonal: Predictions are close to actual profit.
- Points far from the diagonal: The model makes larger errors.
- Points systematically above or below the diagonal: The model may consistently overpredict or underpredict profit in those regions.
The graph helps you visually evaluate the model rather than relying only on a single metric.




######




Step 7: Predict profit for your own product
At the bottom of the script, find the new_order dictionary.
new_order = {    "price": 150.0,    "product_weight_g": 1200.0,    "product_length_cm": 30.0,    "product_height_cm": 15.0,    "product_width_cm": 20.0,    "product_name_lenght": 40.0,    "product_description_lenght": 300.0,    "product_photos_qty": 3.0,    "purchase_month": 10,    "purchase_day_of_week": 2,    "purchase_hour": 14,    "product_category_name_english": "housewares",    "seller_state": "SP",    "customer_state": "SP"}


Change the values to describe your product or order.

For example, change the price, product dimensions, category, and locations.

The model will return its estimated profit.

Important: This is an estimate based on historical patterns. Because the model does not receive actual product cost, freight expense, discount, or return loss, it cannot calculate the exact profit for your new order.

Also, the example's category and location values should be checked against your dataset's actual category and location values. Unknown categories are supported by the preprocessing pipeline, but predictions for unfamiliar products or locations may be less reliable.




######



Step 8: Files generated by the project
After successful execution, your folder will contain:
File	Purpose
profit_prediction_model.pkl	Saved trained ML model and preprocessing pipeline
model_comparison.csv	Comparison of Linear Regression and Random Forest
actual_vs_predicted.csv	Actual profit, predicted profit, and prediction errors
actual_vs_predicted.png	Visualization of model performance


You can use the saved model for future predictions without retraining it.

Keep the saved model and its preprocessing pipeline together, and use compatible versions of scikit-learn when loading it.




######




Step 9: How to explain this project in your presentation
You can explain your methodology in these five points:
1. Data preparation: Loaded the cleaned e-commerce dataset, removed repeated order-item records, handled missing values, and extracted purchase-time features.
2. Feature selection: Selected product characteristics, price, category, seller location, customer location, and purchase timing. Excluded direct profit components to reduce target leakage.
3. Model development: Trained Linear Regression as a baseline and Random Forest Regressor as a nonlinear model.
4. Model evaluation: Compared the models using MAE, RMSE, and R² on a chronologically held-out test set.
5. Business application: Estimated expected historical profit patterns to support product evaluation and profitability analysis.
One final recommendation
For your case study, do not select Random Forest as the best model just because it is more advanced. Choose it only if its evaluation results justify doing so.

The next important step is to inspect the actual profit formula and test-set metrics. If your objective is to estimate profit before a sale, this model is a reasonable starting experiment. If your objective is to calculate exact profit from known price, cost, freight, discount, and return-loss values, a verified financial formula will be more accurate and easier to maintain.









##### IMP

| Metric | Meaning | What you want |
|---|---|---|
| MAE | Average absolute difference between actual and predicted profit | Lower |
| RMSE | Penalizes larger prediction errors more heavily | Lower |
| R² | Measures how well the model explains variation in the target | Generally higher |





Example of MAE
Suppose the model produces an MAE of 12.

This means its predictions differ from actual profit by an average absolute amount of approximately 12 units of the target currency on the test dataset.

It does not mean every prediction is wrong by exactly 12.
Example of R²
- R² = 1.0: Perfect predictions on the evaluation dataset.
- R² = 0.0: No improvement over predicting the test-set mean, under the standard R² definition.
- R² < 0: Worse than that mean-based baseline.
A negative R² is possible, especially when predicting future records from historical data. It does not automatically mean your code is incorrect.