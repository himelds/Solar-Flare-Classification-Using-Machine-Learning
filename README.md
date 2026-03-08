# Solar Flare Classification Using Machine Learning

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

### 3. Machine Learning Models
Two classification models were implemented and evaluated:

- **Logistic Regression**
- **Random Forest**

These models were trained to classify solar flare intensity categories.

---

## Results

Key results from the analysis include:

- Logistic Regression achieved approximately **65% classification accuracy**
- Random Forest achieved approximately **64% classification accuracy**
- Both models performed well on **majority classes (B and C)**
- Minority classes such as **A, M, and X** were more difficult to classify due to class imbalance
- Feature importance analysis showed that **intensity, flare duration, and time-to-peak** are key predictors for classification

---

## Data Visualization

The project includes several visual analyses to explore solar flare patterns:

- Distribution of solar flare classes
- Correlation heatmap between important features
- Temporal trends in solar flare activity (1981–2017)
- Distribution of daily flare occurrences
- Analysis of **"All-Clear Days"** with no flare activity

These visualizations provide insights into long-term patterns and fluctuations in solar activity.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Project Structure

solar-flare-classification/
│
├── data/
│ └── solar_flare_dataset.csv
│
├── notebooks/
│ └── solar_flare_analysis.ipynb
│
├── visualizations/
│ └── plots/
│
├── results/
│ └── model_performance.png
│
└── README.md
