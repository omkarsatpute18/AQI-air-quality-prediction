🌍 AQI Prediction Using Machine Learning — Complete Project Summary
1. Project Objective
The main objective of the project is to build a machine learning regression system that predicts the Air Quality Index (AQI) based on different air-pollution measurements.
Instead of manually calculating or estimating AQI from individual pollutants, we trained an ML model to learn the relationship between pollutant concentrations and the final AQI value.
Problem
Air quality is affected by multiple pollutants simultaneously. The relationship between these pollutants and AQI can be complex and nonlinear.
Solution
We developed an end-to-end system:
Raw Air Quality Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Selection
        ↓
Data Preprocessing
        ↓
Train/Test Split
        ↓
ML Model Training
        ↓
Model Evaluation
        ↓
Final Model Selection
        ↓
Save Model
        ↓
Streamlit Application
        ↓
User enters pollutant values
        ↓
Predicted AQI

2. Dataset Import
We used the city_day.csv dataset containing historical air-quality observations.
The dataset contains pollutant measurements and the corresponding AQI.
We loaded it using Pandas:
import pandas as pddf = pd.read_csv("city_day.csv")


Then we inspected the dataset using operations such as:
df.head()df.shapedf.info()df.describe()


Why?
This helps us understand:
- Number of observations
- Number of features
- Data types
- Numerical columns
- Missing values
- Target variable
- Statistical distribution
3. Understanding the Dataset
Our target variable is:
AQI

The model uses 12 pollutant features as inputs:
Feature	Meaning
PM2.5	Fine particulate matter
PM10	Coarse particulate matter
NO	Nitric oxide
NO2	Nitrogen dioxide
NOx	Nitrogen oxides
NH3	Ammonia
CO	Carbon monoxide
SO2	Sulfur dioxide
O3	Ozone
Benzene	Benzene concentration
Toluene	Toluene concentration
Xylene	Xylene concentration


So conceptually:
12 pollutant measurements
          ↓
     ML model
          ↓
       AQI value

4. Data Quality Analysis
Before training a model, we checked the quality of the dataset.
We looked for:
Missing values
df.isnull().sum()


Air-quality datasets commonly contain missing pollutant measurements.
For example:
PM2.5      missing values
PM10       missing values
NO         missing values
...

Missing values cannot simply be passed directly to many ML algorithms.
Therefore, preprocessing was necessary.
Duplicate records
We also checked:
df.duplicated().sum()


This helps identify repeated observations.
Data types
We inspected:
df.dtypes


to make sure the model features were numerical.
5. Exploratory Data Analysis — EDA
EDA was performed to understand the relationship between pollution variables and AQI before blindly training a model.
We created meaningful visualizations such as:
AQI distribution
This helps understand whether AQI values are:
- concentrated around a particular range
- normally distributed
- highly skewed
- containing extreme values
Pollutant vs AQI relationships
We examined how pollutant concentrations relate to AQI.
For example:
PM2.5 → AQI
PM10  → AQI
NO2   → AQI
CO    → AQI
O3    → AQI

This gives engineering meaning to the model.
Correlation analysis
We also examined correlations between numerical variables.
The purpose was to understand:
- Which pollutants are strongly related to AQI
- Which pollutants are correlated with each other
- Whether multicollinearity may exist
For example, nitrogen-related pollutants such as:
NO
NO2
NOx

can have strong relationships with each other.
6. Feature and Target Separation
After understanding the data, we separated:
Features — X
PM2.5
PM10
NO
NO2
NOx
NH3
CO
SO2
O3
Benzene
Toluene
Xylene

Target — y
AQI

Conceptually:
X = df[features]y = df["AQI"]


The ML model learns:
X → y

or:
Pollutant concentrations → AQI

7. Train-Test Split
We then divided the data into training and testing datasets.
Conceptually:
Dataset
   │
   ├── Training Data
   │      ↓
   │    Model learns
   │
   └── Testing Data
          ↓
       Evaluation

The important reason for doing this is that we should evaluate the model on data it did not see during training.
This provides a better estimate of how the model will perform on new pollution measurements.
8. Preprocessing — Missing Value Handling
One of the important preprocessing steps in our project was missing-value imputation.
We used:
SimpleImputer(strategy="median")


Why median?
Suppose:
PM2.5 = 20
PM2.5 = 25
PM2.5 = missing
PM2.5 = 30
PM2.5 = 1000

Using the mean could be strongly affected by the extreme value 1000.
Median is more robust to outliers.
So the missing value is replaced using the median of the corresponding feature.
9. Creating the ML Pipeline
Instead of doing preprocessing separately and then training the model, we combined preprocessing and the ML model into a scikit-learn Pipeline.
Conceptually:
Input Data
    ↓
SimpleImputer
    ↓
Decision Tree Regressor
    ↓
Predicted AQI

This is very important because the exact same preprocessing is automatically applied when a new user enters data in the Streamlit application.
Our final saved model is essentially:
SimpleImputer(strategy="median")
                ↓
       DecisionTreeRegressor

10. Model Training
We experimented with different regression approaches during the project and ultimately selected the Decision Tree Regressor as the final model.
The problem is a regression problem because AQI is a numerical value.
For example:
Input pollutants
      ↓
AQI = 189

rather than:
Good / Moderate / Poor

Therefore, a regression algorithm is appropriate.
11. Why Decision Tree?
A Decision Tree Regressor learns relationships through a series of decision rules.
Conceptually:
              PM2.5 < threshold?
                  /       \
                Yes        No
                /           \
          NO2 condition    PM10 condition
             /                 \
           ...                 ...
                    ↓
                 AQI

Advantages for this project:
- Can model nonlinear relationships
- Easy to interpret conceptually
- Works well with numerical data
- Does not require feature scaling
- Suitable for relationships between pollutant concentrations and AQI
12. Model Evaluation
After training, we evaluated the model using regression metrics.
We used:
MAE — Mean Absolute Error
Your final model achieved approximately:
MAE = 29.319

This means the average absolute prediction error is about 29 AQI points.
MSE — Mean Squared Error
MSE = 3211.757

MSE penalizes larger errors more heavily.
RMSE — Root Mean Squared Error
RMSE = 56.672

RMSE is in the same unit as AQI, making it easier to interpret than MSE.
R² Score
R² = 0.8246

This means the model explains approximately 82.46% of the variation in AQI on the evaluated data.
So the final result was approximately:
Metric	Result
MAE	29.319
MSE	3211.757
RMSE	56.672
R²	0.8246


These are the results we established for your final Decision Tree model.
13. Final Model Selection
After evaluating the candidate approaches, we selected:
Decision Tree Regressor

as the final model.
The important point is that we didn't select a model simply because it was more complicated.
The project requirement emphasizes that the model should be justified, evaluated and useful for the engineering problem.
14. Saving the Trained Model
After selecting the final model, we saved it using Joblib.
The saved file is:
final_model.pkl

This is extremely important.
Instead of retraining the model every time the application starts, Streamlit can simply load:
final_model.pkl

and use it for prediction.
Conceptually:
Training
   ↓
Trained Pipeline
   ↓
final_model.pkl

Then:
Streamlit
   ↓
Load final_model.pkl
   ↓
Predict

15. Building the Streamlit Application
After completing the ML model, we created a graphical interface using Streamlit.
The purpose was to convert our notebook-based ML project into an actual usable application.
The user doesn't need to run Python code.
Instead, the user sees input fields for the pollutants.
For example:
PM2.5     [ 80  ]
PM10      [ 120 ]
NO        [ 20  ]
NO2       [ 35  ]
NOx       [ 45  ]
NH3       [ 10  ]
CO        [ 0.8 ]
SO2       [ 15  ]
O3        [ 30  ]
Benzene   [ 2   ]
Toluene   [ 5   ]
Xylene    [ 1   ]

        [ Predict AQI ]

16. How Streamlit Prediction Works
When the user clicks:
Predict AQI

the application collects the 12 values.
It creates a Pandas DataFrame with exactly the same feature names used during training:
PM2.5
PM10
NO
NO2
NOx
NH3
CO
SO2
O3
Benzene
Toluene
Xylene

Then:
predicted_aqi = model.predict(input_data)[0]


The saved pipeline automatically performs:
User Input
    ↓
Pandas DataFrame
    ↓
SimpleImputer
    ↓
Decision Tree
    ↓
Predicted AQI

17. Local Testing
We tested the model independently before relying on Streamlit.
We entered:
PM2.5 = 80
PM10 = 120
NO = 20
NO2 = 35
NOx = 45
NH3 = 10
CO = 0.8
SO2 = 15
O3 = 30
Benzene = 2
Toluene = 5
Xylene = 1

All 12 values were correctly recognized as float64.
The model successfully returned:
Prediction: [189.]

This was an important debugging step because it proved that:
- The model works.
- The .pkl file works.
- The pipeline works.
- The imputer works.
- The input feature names are correct.
- The input data types are correct.
18. GitHub Repository
We then organized the project into a GitHub repository:
AQI-air-quality-prediction GitHub repository
The important project components include:
AQI-air-quality-prediction/
│
├── city_day.csv
├── main.ipynb
├── final_model.pkl
├── app.py
├── requirements.txt
└── README.md

The notebook contains the data analysis and model development, while app.py is responsible for the deployed user interface.
19. Deployment Using Streamlit Cloud
We then deployed the application using Streamlit Community Cloud.
The basic deployment architecture is:
GitHub Repository
       ↓
Streamlit Cloud
       ↓
Install dependencies
       ↓
Load app.py
       ↓
Load final_model.pkl
       ↓
Web Application

Your application URL was:
AQI Prediction System
20. Deployment Error and Debugging
Initially, the deployed application produced an error when clicking Predict AQI.
The traceback showed that the error occurred inside:
model.predict()
      ↓
Pipeline
      ↓
SimpleImputer

We then tested the exact same input locally.
It worked:
Prediction: [189.]

So we determined that the problem wasn't the prediction logic itself.
21. Dependency Compatibility Problem
We checked your local environment:
Python       3.13.2
scikit-learn 1.7.1
NumPy        2.2.4
pandas       2.2.3
joblib       1.5.1

Your requirements.txt was then pinned to these versions:
streamlit
pandas==2.2.3
numpy==2.2.4
scikit-learn==1.7.1
joblib==1.5.1

The reason was to make the deployment environment match the environment in which the model was successfully trained/tested.
22. Second Deployment Problem
After that, Streamlit's deployment logs gave us the exact issue.
Streamlit was using:
Python 3.14.8

and was trying to install:
scikit-learn==1.7.1

which required SciPy.
The deployment then attempted to build SciPy from source and failed because the required Fortran compiler wasn't available:
Unknown compiler(s)
gfortran
flang
...

The logs show that dependency installation failed before the application could properly start.    logs-omkarsatpute18-aqi-air-qua…    logs-omkarsatpute18-aqi-air-qua…
So the current deployment problem is not an ML-model problem.
It's an environment/dependency compatibility problem.
23. Current Deployment Solution
Our target deployment environment is:
Python 3.13
scikit-learn 1.7.1
NumPy 2.2.4
pandas 2.2.3
joblib 1.5.1

This matches the environment where your model successfully produced:
AQI = 189

The Streamlit deployment therefore needs to use a compatible Python version rather than Python 3.14.
24. Overall Architecture of Your Project
The entire project can be understood in one diagram:
                 ┌─────────────────────┐
                 │   city_day.csv      │
                 │   Raw AQI Dataset   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Understanding  │
                 │ & Data Cleaning     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │       EDA           │
                 │ Graphs/Correlation  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Feature Selection   │
                 │ 12 Pollutant Inputs │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Train/Test Split   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    ML Pipeline      │
                 │                     │
                 │ SimpleImputer       │
                 │       ↓             │
                 │ Decision Tree       │
                 │    Regressor        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Evaluation       │
                 │ MAE / MSE / RMSE    │
                 │ R² = 0.8246         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  final_model.pkl    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Streamlit App    │
                 │      app.py         │
                 └──────────┬──────────┘
                            │
                   User enters 12
                   pollutant values
                            │
                            ▼
                 ┌─────────────────────┐
                 │    AQI Prediction    │
                 │       = 189          │
                 │   for test input     │
                 └─────────────────────┘

25. In Simple Words — What You Actually Built
If someone asks you "What did you do in this project?", you can explain it like this:
I developed an end-to-end machine learning system for predicting Air Quality Index from pollutant concentrations. I first loaded and analyzed a real-world air-quality dataset, performed data-quality checks and exploratory data analysis, and selected 12 pollutant features as inputs. I handled missing values using median imputation and created a machine-learning pipeline containing a SimpleImputer and Decision Tree Regressor. I trained and evaluated the model using MAE, MSE, RMSE and R², achieving an R² of 0.8246. I then saved the trained pipeline as a .pkl file and developed a Streamlit web application where users can enter pollutant values and receive an AQI prediction. Finally, I uploaded the project to GitHub and deployed the application through Streamlit Community Cloud.

The key concepts you should understand
For your viva/interview, the most important things are:
1. Why AQI prediction is a regression problem
2. What each of the 12 pollutant features represents
3. Why missing values needed to be handled
4. Why median imputation was used
5. What a Pipeline does
6. Why Decision Tree Regressor was selected
7. Difference between MAE, MSE, RMSE and R²
8. What R² = 0.8246 means
9. Why we split training and testing data
10. Why the model was saved as final_model.pkl
11. How Streamlit loads the model and makes predictions
12. Why the deployment failed — Python/dependency compatibility, not the ML prediction itself
This gives you the complete story from raw dataset → EDA → preprocessing → ML → evaluation → model serialization → Streamlit → deployment.
