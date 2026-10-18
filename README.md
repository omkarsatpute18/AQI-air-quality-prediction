# 🌍 Air Quality Index (AQI) Prediction Using Machine Learning

A machine-learning based engineering project that predicts the **Air Quality Index (AQI)** from daily air-pollutant measurements.

The project follows a complete engineering ML workflow:

**Problem Definition → Dataset Understanding → Data Quality Analysis → EDA → Preprocessing → Feature Selection → Model Training → Evaluation → Interpretation → Prediction → Demonstration**

---

## 📌 Project Overview

Air pollution is an important environmental engineering problem because pollutant concentrations vary across locations and time.

This project develops a supervised machine-learning system that estimates **continuous AQI** using pollutant measurements. The predicted AQI can be used as an application-level monitoring and decision-support signal.

The project is designed according to the mini-project requirements provided by the **Department of Electronics and Telecommunication Engineering, Pune Vidyarthi Griha’s College of Engineering, Technology and Management, Pune-09**.

> **Important:** This system is a machine-learning prediction/decision-support system. It is not intended to replace official regulatory AQI measurements.

---

## 🖥️ Streamlit Application

A Streamlit GUI was created to make the model usable without directly interacting with the notebook.

### 🌐 Live Demo

The application is deployed and available online:

👉 **[Open the AQI Prediction Application](https://airqualityindexprediction123.streamlit.app/)**

The user enters pollutant measurements, and the application uses the trained machine-learning pipeline to predict AQI.

## 🎯 Objectives

- Use a real-world air-quality dataset.
- Understand the structure and quality of the dataset.
- Analyze missing values, duplicates, invalid values, outliers, and correlations.
- Determine an appropriate ML formulation for AQI prediction.
- Perform justified preprocessing and feature selection.
- Compare multiple regression algorithms.
- Evaluate models using appropriate regression metrics.
- Interpret the final model and its important features.
- Provide a simple interface for predicting AQI from new pollutant measurements.
- Translate the ML output into an engineering/application interpretation.
- Document limitations and recommendations for future improvement.

---

## 🏭 Engineering Domain

**Environmental Engineering + Machine Learning / Data Science**

The system can support:

- Air-quality monitoring
- Environmental analysis
- Pollution monitoring
- Decision-support systems
- Smart-city/environmental applications
- Identification of important pollution variables

---

## 📊 Dataset

### Dataset Used

**Air Quality Data in India – `city_day.csv`**

The dataset contains daily city-level air-quality observations from Indian cities.

### Dataset Characteristics

| Property | Description |
|---|---|
| Dataset | Air Quality Data in India |
| File | `city_day.csv` |
| Observations | 29,531 |
| Columns | 16 |
| Geographic coverage | 26 Indian cities |
| Target | `AQI` |
| Alternative target | `AQI_Bucket` |
| Granularity | Daily city-level observations |

### Dataset Source

The dataset is publicly available through Kaggle and is associated with air-quality records compiled from CPCB-related data.

---

## 🧪 Features Used

The final primary predictor set contains the following 12 pollutant variables:

```text
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
```

### Target Variable

```text
AQI
```

AQI is treated as a **continuous numerical target**, so the primary ML problem is formulated as **regression**.

---

## 🚫 Target Leakage Prevention

`AQI_Bucket` is **not used as an input feature**.

This is important because `AQI_Bucket` represents an AQI category derived from AQI information. Using it while predicting AQI could provide target-related information to the model and produce misleadingly optimistic results.

Therefore:

```text
Pollutant Measurements
        ↓
Machine Learning Model
        ↓
Predicted AQI
```

---

## 🔎 Exploratory Data Analysis

The notebook performs systematic data exploration, including:

- Dataset shape
- First and last rows
- Column names
- Data types
- Statistical summaries
- Unique values
- Target distribution
- Missing-value analysis
- Duplicate analysis
- Invalid-value checks
- Outlier analysis
- Correlation analysis
- Pollutant distributions
- Pollutant vs AQI relationships
- City-wise AQI analysis
- Temporal AQI analysis

### Important Visualizations

1. AQI distribution
2. AQI category distribution
3. Pollutant distributions
4. Correlation heatmap
5. Pollutant vs AQI scatter plots
6. City-wise AQI comparison
7. Time-based AQI trend
8. Actual vs predicted AQI
9. Residual/error analysis
10. Prediction-error distribution

---

## 🧹 Data Preprocessing

The project uses justified preprocessing rather than blindly modifying the dataset.

### Missing Target Values

Rows without an `AQI` value are excluded from supervised modeling because a target value is required for training.

### Missing Predictor Values

Missing pollutant values are handled using **median imputation**.

The imputer is included inside the scikit-learn pipeline so that the imputation parameters are learned from the training data.

### Outliers

Extreme pollution values are **not automatically removed**.

The reason is that extreme pollution measurements may represent genuine environmental events. Removing them without evidence could remove important information from the dataset.

### Scaling

Scaling is used for Linear Regression:

```text
Median Imputation
        ↓
StandardScaler
        ↓
Linear Regression
```

Tree-based models do not require feature scaling:

```text
Median Imputation
        ↓
Decision Tree Regressor
```

and

```text
Median Imputation
        ↓
Random Forest Regressor
```

---

## 🧠 Feature Selection

Feature selection considers:

- Data availability
- Missingness
- Correlation with AQI
- Feature-to-feature correlation
- Domain relevance
- Possible redundancy
- Model-based importance

Special attention is given to:

```text
NO
NO2
NOx
```

because these nitrogen-related variables can be correlated.

Features are not removed solely because they are correlated; their usefulness is considered together with domain reasoning and model behavior.

---

## 🤖 Machine Learning Models

Three regression models are considered.

### 1. Linear Regression

Used as an interpretable baseline.

**Advantages:**
- Simple
- Easy to understand
- Useful baseline

**Limitation:**
- May not capture nonlinear relationships.

---

### 2. Decision Tree Regressor

Used to model nonlinear relationships and interactions between pollutant variables.

**Advantages:**
- Captures nonlinear patterns
- Easy to visualize conceptually
- Does not require feature scaling

**Limitation:**
- Can overfit without appropriate constraints.

---

### 3. Random Forest Regressor

Uses multiple decision trees and combines their predictions.

**Advantages:**
- Captures nonlinear relationships
- More stable than a single tree in many situations
- Handles interactions between variables

**Limitations:**
- More complex
- Less directly interpretable than a single decision tree

---

## 📐 Train/Test Split

The project uses:

```python
test_size = 0.20
random_state = 42
```

Therefore:

- **80%** of the data is used for training.
- **20%** is used for testing.

The same test set is used to compare the candidate models fairly.

---

## 📏 Evaluation Metrics

The regression models are evaluated using:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted AQI.

**Lower is better.**

### MSE — Mean Squared Error

Squares the prediction errors, giving greater weight to large errors.

**Lower is better.**

### RMSE — Root Mean Squared Error

The square root of MSE, expressed on the AQI scale.

**Lower is better.**

### R² — R-squared

Measures how much variation in AQI is explained by the model relative to a baseline.

**Higher is generally better.**

---

## 🏆 Final Model

The saved project model is:

```text
Decision Tree Regressor
```

inside a scikit-learn Pipeline:

```text
Input Pollutants
       ↓
SimpleImputer
       ↓
DecisionTreeRegressor
       ↓
Predicted AQI
```

The saved model file is:

```text
final_model.pkl
```

### Important

The exact MAE, MSE, RMSE, R² values should be taken directly from the completed notebook. No metric values are fabricated in this README.

---

## 📊 Model Results

Insert the exact values from the final notebook before submission:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | Add notebook value | Add notebook value | Add notebook value | Add notebook value |
| Decision Tree Regressor | Add notebook value | Add notebook value | Add notebook value | Add notebook value |
| Random Forest Regressor | Add notebook value | Add notebook value | Add notebook value | Add notebook value |

---

## 🔍 Feature Importance

The final Decision Tree model can be used to obtain feature importance.

The exact ranking should be copied from the notebook.

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Add notebook value | Add notebook value |
| 2 | Add notebook value | Add notebook value |
| 3 | Add notebook value | Add notebook value |
| 4 | Add notebook value | Add notebook value |
| 5 | Add notebook value | Add notebook value |

> Feature importance represents the contribution of a variable to model predictions. It does **not** prove that the variable causes AQI changes.

---

## 🖥️ Streamlit Application

A Streamlit GUI was created to make the model usable without directly interacting with the notebook.

The user enters:

- PM2.5
- PM10
- NO
- NO2
- NOx
- NH3
- CO
- SO2
- O3
- Benzene
- Toluene
- Xylene

The application then returns:

```text
Predicted AQI
```

### Application Flow

```text
User enters pollutant values
            ↓
Streamlit interface
            ↓
Input DataFrame
            ↓
Saved ML Pipeline
            ↓
AQI Prediction
            ↓
Predicted AQI displayed
```

---

## 📁 Project Structure

Recommended final project structure:

```text
AQI-air-quality-prediction/
│
├── main.ipynb
├── app.py
├── final_model.pkl
├── city_day.csv
├── requirements.txt
├── README.md
│
└── AQI_ML_Engineering_Final_Report.pdf
```

### File Description

| File | Purpose |
|---|---|
| `main.ipynb` | Complete ML implementation and analysis |
| `app.py` | Streamlit prediction interface |
| `final_model.pkl` | Saved trained ML pipeline |
| `city_day.csv` | Dataset |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `AQI_ML_Engineering_Final_Report.pdf` | Engineering report |

---

## ▶️ Run the Project Locally

### 1. Clone/download the project

Open the project folder in VS Code.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The Streamlit application will open in the browser.

---

## 📦 Requirements

Recommended `requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

---

## 💡 Engineering Interpretation

The predicted AQI can be treated as an application-level monitoring signal.

For example:

```text
Pollutant Measurements
        ↓
Predicted AQI
        ↓
Engineering Interpretation
        ↓
Monitoring / Alert / Decision Support
```

A high predicted AQI can indicate that the observed pollutant conditions deserve additional monitoring or pollution-management attention.

The system should not be interpreted as a medical diagnosis or as an official regulatory AQI measurement.

---

## ⚠️ Limitations

1. **Dataset dependency**  
   Model performance depends on the quality and representativeness of the historical dataset.

2. **Geographic variation**  
   The relationship between pollutants and AQI may differ across cities and regions.

3. **Temporal variation**  
   The current project performs AQI estimation from pollutant measurements and is not a dedicated future-day forecasting system.

4. **Missing data**  
   Median imputation provides an approximation when measurements are unavailable.

5. **Extreme values**  
   Genuine pollution events can influence model behavior.

6. **Sensor/data quality**  
   Poor-quality measurements can lead to inaccurate predictions.

7. **Official AQI distinction**  
   The ML prediction should not be presented as a replacement for official AQI calculation or regulatory monitoring.

8. **Out-of-distribution inputs**  
   Predictions may be unreliable when pollutant combinations are very different from the training data.

9. **Feature importance**  
   Model importance does not imply causation.

10. **Deployment limitations**  
    The deployed application requires compatible Python dependencies and the same input feature definitions used during training.

---

## 🚀 Future Improvements

Possible improvements include:

- Cross-validation
- Hyperparameter tuning
- More recent air-quality data
- More cities and geographic information
- Meteorological variables such as temperature, humidity, wind speed, and pressure
- Time-aware forecasting
- City-specific models
- Model drift monitoring
- Automated data-quality validation
- Comparison with official monitoring stations
- More advanced deployment infrastructure
- Real-time air-quality data integration

These improvements should only be added when justified by the project objective and available data.

---

## 🎓 Viva Preparation

### Why did you choose this problem?

Air pollution is a real-world environmental engineering problem. Predicting AQI from pollutant measurements can support monitoring and decision-making.

### Why is this an engineering problem?

The ML model is part of a monitoring and decision-support system. Its output can be used to identify conditions requiring additional attention.

### Why did you choose regression?

AQI is a continuous numerical value, and the project objective is to estimate that value.

### Why didn't you use AQI_Bucket?

Using AQI_Bucket as an input would introduce target leakage because it contains category information derived from AQI.

### Why use median imputation?

Median imputation provides a simple way to handle missing numerical pollutant measurements and is less sensitive to extreme values than mean imputation.

### Why didn't you automatically remove outliers?

Extreme pollution values may represent genuine environmental events. Removing them without evidence could remove meaningful information.

### Why compare multiple models?

Comparing multiple suitable algorithms provides evidence for model selection instead of choosing an algorithm blindly.

### Why use MAE, MSE, RMSE and R²?

Each metric provides different information about model performance, allowing a more complete evaluation.

### What is the final model?

The saved final model is a Decision Tree Regressor inside a preprocessing Pipeline.

### What does the application do?

It accepts 12 pollutant measurements and uses the trained model to predict AQI.

### What is the engineering use of the prediction?

The predicted AQI can be used as a monitoring and decision-support signal to identify potentially high-pollution conditions.

---

## 📋 Requirement Compliance

The project addresses the major requirements of the college mini-project guideline:

- ✅ Engineering problem defined
- ✅ Objective defined
- ✅ Expected output defined
- ✅ Real-world dataset selected
- ✅ Dataset described
- ✅ Data exploration performed
- ✅ Data quality analyzed
- ✅ Missing values investigated
- ✅ Duplicate/invalid-value analysis performed
- ✅ Outlier analysis performed
- ✅ Meaningful visualizations created
- ✅ Preprocessing justified
- ✅ Target leakage considered
- ✅ Feature selection performed
- ✅ Regression technique selected
- ✅ Multiple ML models compared
- ✅ Evaluation metrics used
- ✅ Model interpretation performed
- ✅ Engineering interpretation provided
- ✅ Limitations documented
- ✅ Recommendations provided
- ✅ Final prediction interface created
- ✅ Engineering report prepared
- ✅ Viva questions prepared

---

## 📚 References

1. Pune Vidyarthi Griha’s College of Engineering, Technology and Management, Department of Electronics and Telecommunication Engineering — Mini Project Guidelines.
2. Rohan Rao — Air Quality Data in India dataset, Kaggle.
3. Central Pollution Control Board (CPCB), Government of India — underlying air-quality data source context.
4. Python ecosystem documentation: pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

---

## 👨‍💻 Author

**Omkar Satpute**

Machine Learning / Data Science Engineering Mini Project

**Project:** Air Pollution / Air Quality Index (AQI) Prediction Using Machine Learning
