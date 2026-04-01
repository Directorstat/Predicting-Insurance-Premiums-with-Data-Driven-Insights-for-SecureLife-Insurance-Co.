# Predicting Insurance Premiums with Data-Driven Insights
# SecureLife Insurance Co.

**Project Overview**

To develop a regression model to predict the Premium Amount based on the data provided. 

**The key objectives are:**

Clean and preprocess the dataset.

Explore feature importance and relationships.

Build and evaluate a robust predictive model.

Interpret results and provide actionable insights.

**The goal:**

Estimate Premium_Amount using demographic, financial, and behavioral features

Support better pricing decisions for SecureLife Insurance Co.

Identify key drivers of insurance risk

Dataset size: 278,860 records and 20 features (mixed numerical and categorical)

**Tools and Environment**

The model was carried out in Google Colab using:

pandas for data handling

numpy for numerical operations

matplotlib and seaborn for visualization

scikit-learn for preprocessing and modelling

# Step 1: Data Loading and Inspection

You loaded the dataset:

df = pd.read_csv("/content/Insurance Premium Prediction Dataset.csv")

Then, Inspected the structure: Used df.head() to preview rows and Used df.info() to check data types and missing values

Key observation: Many columns had missing values, and Data types included float, int, and object

**Handling Missing Values**

Custom missing indicators were defined:

missing_vals = ['n/a', ' ', '-', '?']

Reloaded the dataset with these treated as NaN.

Numerical Variables: Selected numeric columns and applied mean imputation: SimpleImputer(strategy='mean')

Why: Keeps the dataset size intact and is suitable when the data is not heavily skewed

Categorical Variables: Applied the most frequent imputation: SimpleImputer(strategy='most_frequent')

Why: Preserves category distribution and is simple and effective for large datasets

Result: All missing values removed, and the dataset is ready for analysis

**Data Cleaning**

Duplicate Check: df.duplicated().sum()

Result: 0 duplicates

Outlier Detection: Boxplots were used for all numerical features.

Findings: Outliers in: Annual_Income, Health_Score, Previous_Claims and Premium_Amount

**Outlier Treatment**

Applied the Interquartile Range (IQR):

Q1 = 25th percentile, Q3 = 75th percentile

IQR = Q3 - Q1

Rule: Remove values outside

Lower bound = Q1 - 1.5 × IQR, Upper bound = Q3 + 1.5 × IQR

Result: Dataset reduced to 242,492 rows, Cleaner distribution and Less noise for modelling.

# Step 2: Exploratory Data Analysis (EDA)

**Univariate Analysis**

Numerical: Histograms showed distribution patterns and identified skewness in income and claims

<img width="1486" height="985" alt="image" src="https://github.com/user-attachments/assets/8ebe4872-f16b-44ba-be0f-81d39aefc957" />

Categorical: Count plots showed frequency distribution and helped understand dominant categories.

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/069d878d-721e-4300-bd13-a5f066e29f79" />
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/2652cc7b-b922-43ee-9fee-31794256263c" />
<img width="630" height="469" alt="image" src="https://github.com/user-attachments/assets/4dfaab07-d68d-4f70-89a8-97f49e41e793" />
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/7ca1855c-d3ec-4887-bab8-b1396c9aabc6" />
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/86f60f5a-4880-42cf-88c6-f4b43296dc55" />
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/64d3dbae-7982-45a7-b597-697fae299045" />
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/24a6b698-2d26-4c3d-b578-0f98ac90190d" />
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/9e17f81f-b33b-410d-a972-7dcaced68026" />
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/58180cd0-37c5-4a54-b0e9-b1687e7f1ed2" />

**Bivariate Analysis**

Analysed the relationship with target (Premium_Amount): Scatter plots for each numerical variable

Observation: Weak visible relationships and no strong linear trends

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/a35f3754-b34d-47d0-8042-9c2e23710f48" />

**Multivariate Analysis**

Correlation heatmaps: sns.heatmap(df.corr())

Key insight:

All correlations with Premium_Amount were near zero

<img width="1050" height="835" alt="image" src="https://github.com/user-attachments/assets/aaad52bb-9d76-413b-b11b-c123487ff4f2" />

Age correlation is 0.0047, and others are even lower

This explains poor model performance later.

# Step 3: Feature Engineering

Date Transformation

df1['Policy_Start_Date'] = pd.to_datetime(...)

Then dropped it: No extracted features like year or duration could be improved

**Encoding Categorical Variables**

Label Encoding: LabelEncoder()

Applied to: Gender, Marital_Status, Education_Level, Occupation, Location, Policy_Type, Customer_Feedback, Smoking_Status, Exercise_Frequency, Property_Type

Why: Converts text to numeric and is required for machine learning models

Limitation:  Imposes artificial order and Not ideal for all models

**Train-Test Split**

train_test_split(test_size=0.2)

Result: 80% training and 20% testing

Ensures:  Model generalization and Fair evaluation

# Step 4: Model Development (Linear Models)

Experiment with different regression algorithms ( LinearRegression, Ridge, Lasso, RandomForestRegressor and GradientBoostingRegressor )

    Model     MAE        MSE      R²
    
0  Linear Regression  567.11  487010.93 -0.0004

1   Ridge Regression  567.11  487010.93 -0.0004

2   Lasso Regression  567.10  487002.57 -0.0004

**Interpretation:** The provided metrics suggest that the current regression models are performing very poorly, essentially failing to capture any meaningful patterns in the data. 
The  R2  values are near zero (and slightly negative), which indicates that the models are not performing any better than a "naive" model that simply guesses the average premium for every single customer. 
While the MAE (Mean Absolute Error) of 567.11 shows that the predictions are off by roughly 567 units on average, the high MSE and stagnant performance across Linear, Ridge, and Lasso suggest that the relationship between the features and the premium amount is likely non-linear. 
From a company perspective, these results imply that simple linear models are insufficient for the pricing strategy; the "understanding" here is that the underlying data likely contains complex interactions such as the nonlinear impact of age or specific risk combinations, which these basic models are mathematically unable to see.

# Build an MLR model using sklearn

**Feature Selection using Recursive Feature Elimination (RFE)**

**Predict on the test data using the RFE-trained regressor**

**Feature Selection (RFE)**

Applied Recursive Feature Elimination: RFE(LinearRegression(), n_features_to_select=10)

Result:

Evaluation Metrics for RFE Model: MAE: 567.14, MSE: 487050.67 and R²: -0.0005

**Conclusion:
Evaluation Metrics for RFE Model**

The metrics for the RFE (Recursive Feature Elimination) model indicate that the feature selection process has not yet resolved the core issue: 
the model is currently unable to explain any of the variance in premium amounts. An  R2  of -0.0005 is statistically equivalent to zero, meaning the model is performing no better than a horizontal line drawn at the average premium value. 
While RFE has likely narrowed down the variables, the MAE of 567.14 remains high relative to the lack of predictive power, confirming that the relationship between the selected features and the target is either non-linear or that critical predictive information is still missing from the dataset.

**Residual Analysis**

<img width="875" height="552" alt="image" src="https://github.com/user-attachments/assets/335c0e67-f229-4d27-909a-80dee9165eea" />

Residuals scattered randomly and No pattern

Interpretation: Model not capturing structure, and data lacks predictive signal.

Since the Linear, Ridge, and Lasso are not performing better, we will look at other models.
I want to move toward the non-linear models (Random Forest and Gradient Boosting), 
which can detect the "steps" and "curves" in insurance risk that linear models simply miss.

# DEFINE AND TRAIN MODELS ON RANDOM FOREST AND GRADIENT BOOST REGRESSOR

Random Forest
Gradient Boosting

Using pipeline: Pipeline([("scaler", StandardScaler()), ("model", ...)])

     Model     MAE        MSE    RMSE      R²
     
0      Random Forest  567.03  487020.90  697.87 -0.0004

1  Gradient Boosting  567.32  487547.56  698.25 -0.0015

Conclusion: Even advanced models failed

**Root Cause Analysis**

Computed feature correlation with the target:

corr_matrix['Premium_Amount']

<img width="918" height="835" alt="image" src="https://github.com/user-attachments/assets/2cd094dc-a438-47c8-9e9a-317da766b618" />

Key finding: All features have near-zero correlation

Implication: The dataset lacks predictive power, and the features do not explain the premium.

**Final Insights**

Models are not the problem

Data is the problem

Main issue: No strong relationship between features and target.

The output clearly shows that all features in the dataset have extremely low absolute correlation values with the target variable, Premium_Amount. 
The highest correlation, apart from Premium_Amount with itself, is Age at 0.004706, which is very close to zero. This lack of correlation is the primary reason why all the models (Linear, Ridge, Lasso, Random Forest, and Gradient Boosting) have performed so poorly, resulting in near-zero or negative R² scores. 
These models cannot find a strong relationship if one doesn't exist in the data they are given. 

**Business Interpretation**

For SecureLife Insurance Co: Current data cannot support premium prediction and Pricing decisions need better data

Possible missing factors: Risk profile, Claims severity, Lifestyle details, Medical history, Geographic risk indicators
The next cell will now filter out these features.

**Key Skills Demonstrated:**

Data cleaning and preprocessing

Handling missing values

Outlier detection and treatment

Exploratory data analysis

Feature engineering

Model building and evaluation

Root cause analysis

# Conclusion

Built a complete machine learning workflow.

The outcome matters: Identified why models failed

Traced the issue to weak data

This strengthens my skills in Shows technical depth, Shows critical thinking and Shows real-world problem diagnosis.
