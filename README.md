# Solar Flare Classification Using Machine Learning

**🚀 Live Dashboard:** [https://solarflareclassification.streamlit.app/](https://solarflareclassification.streamlit.app/)

## Overview
Solar flares are sudden bursts of electromagnetic radiation released from the Sun’s atmosphere. These high-energy events can significantly affect Earth's technological infrastructure, including satellite communications, navigation systems, and power grids. Understanding and predicting solar flare activity is therefore crucial for space weather monitoring and risk mitigation.

This project applies machine learning techniques to classify solar flare events and analyze the key factors influencing flare intensity using historical solar observation data.

The analysis explores patterns in solar flare activity and evaluates the performance of machine learning models for classification tasks.

---

## Objectives
The main objectives of this project are:

- Classify solar flare intensity levels using machine learning models  
- Identify the most influential features contributing to solar flare classification  
- Explore patterns and trends in solar flare activity over time  
- Provide insights that could support space weather prediction systems

---

## Dataset
The dataset contains **solar flare observation records from 1981 to 2017**. It includes multiple attributes describing solar flare events and their characteristics.

Key variables in the dataset include:

| Feature | Description |
|-------|-------------|
| `flare_number` | Unique identifier for each solar flare |
| `start_time` | Start time of the flare |
| `end_time` | End time of the flare |
| `peak_time` | Time when the flare reached peak intensity |
| `region` | Solar region where the flare occurred |
| `class` | Solar flare classification (A, B, C, M, X) |
| `intensity` | Measured flare intensity |

The original dataset was provided in **text format** and converted into a structured dataset for further analysis.

---

## Methodology

### 1. Data Preprocessing
- Assigned descriptive column names to the raw dataset
- Converted time-related variables into datetime format
- Cleaned and structured the dataset to ensure consistency

### 2. Feature Engineering
Additional features were created to improve analysis and model performance:

- **Flare Duration** – Time difference between flare start and end
- **Time to Peak** – Time taken for a flare to reach maximum intensity

These features help capture important characteristics of flare behavior.

### 3. Machine Learning Models & MLOps Pipeline
A robust MLOps pipeline was implemented to evaluate multiple algorithms and select the best model for deploying a classification system for the most impactful intensity categories (C, M, and X classes):

- **Random Forest Classifier** (Selected for Deployment)
- **Decision Tree Classifier**
- **Gradient Boosting Classifier**
- **Logistic Regression**

**Data Transformation:** To handle the severe class imbalance (where C-class flares vastly outnumber M and X-class flares), the **SMOTE (Synthetic Minority Over-sampling Technique)** algorithm was applied during the transformation phase to ensure the model learns minority patterns effectively.

**Model Selection:** The models were evaluated using a comparative pipeline. The **Random Forest Classifier** outperformed the others and was automatically selected by the `ModelTrainer` component for final artifact creation and Streamlit deployment.

---

## Results

Key results from the analysis and multi-model evaluation include:

- The **Random Forest model** achieved the highest overall classification accuracy at approximately **69%**.
- The model performed exceptionally well on the **majority class (C-Class)** with high precision and recall, but minority classes (**M and X-Class**) remain challenging to classify perfectly despite SMOTE due to the inherent complexity and extreme imbalance of the data.
- Feature importance analysis (using **SHAP**) revealed that **intensity, flare duration, and time-to-peak** are the strongest predictors for flare classification.

---

## Data Visualization

The project includes several visual analyses to explore solar flare patterns:

- Distribution of solar flare classes
- Correlation heatmap between important features
- Temporal trends in solar flare activity (1981–2017)
- Distribution of daily flare occurrences
- Analysis of **"All-Clear Days"** with no flare activity
- Multi-model evaluation metrics (Precision, Recall, F1-Score)

These visualizations provide insights into long-term patterns and fluctuations in solar activity.

---

## Technologies Used

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, SMOTE (imbalanced-learn)
- **Interpretability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Deployment & Web App:** Streamlit
- **Environment:** Jupyter Notebook, Modular MLOps Architecture

---

## Applications

The insights generated from this project can support:

- Space weather forecasting systems
- Early warning systems for solar storms
- Protection of satellite communication infrastructure
- Research on solar activity patterns and solar cycles

---

## Future Improvements

Possible future improvements include:

- Applying deep learning models for improved classification performance
- Addressing class imbalance using advanced techniques
- Developing predictive models for forecasting future solar flare events
- Integrating additional solar observation datasets

---

## Author

**Himel Das**

MSc in Applied Artificial Intelligence and Data Analytics  
University of Bradford

---

