# Solar-Flare-Classification-Using-Machine-Learning

## Overview
Solar flares are sudden bursts of electromagnetic radiation released from the Sun’s atmosphere. Strong solar flare events can disrupt satellite communication, navigation systems, and power grids on Earth. Accurate classification and prediction of solar flares are therefore important for space weather monitoring and early warning systems.

This project applies machine learning techniques to classify solar flare events and analyze the key factors influencing flare intensity using historical solar observation data.

---

## Objectives
The main goals of this project are:

- Classify solar flare intensity using machine learning models
- Identify the most influential features contributing to flare classification
- Analyze patterns and trends in solar flare activity over time

---

## Dataset
The dataset contains solar flare observations from **1981 to 2017** and includes multiple attributes describing flare activity.

Key variables include:

- `flare_number` – Unique identifier of each solar flare  
- `start_time` – Start time of the flare  
- `end_time` – End time of the flare  
- `peak_time` – Time when the flare reached maximum intensity  
- `region` – Solar region where the flare occurred  
- `class` – Solar flare classification (A, B, C, M, X)  
- `intensity` – Measured flare intensity  

The raw dataset was initially stored in a text format and later converted into a structured dataset for analysis.

---

## Methodology

### Data Preprocessing
- Assigned descriptive column names to the dataset
- Converted time-related fields into datetime format
- Cleaned and structured the dataset for analysis

### Feature Engineering
To improve the dataset for modeling, additional features were created:

- **Flare Duration** – Time difference between `end_time` and `start_time`
- **Time to Peak** – Time required for a flare to reach its maximum intensity

These engineered features provide additional information for understanding solar flare behavior.

### Machine Learning Models
Two classification algorithms were implemented:

- Logistic Regression
- Random Forest

The models were trained to classify solar flare intensity levels based on the available features.

---

## Results

Key findings from the analysis include:

- Logistic Regression achieved approximately **65% classification accuracy**
- Random Forest achieved approximately **64% classification accuracy**
- Both models performed well on **majority classes (B and C)**
- Minority classes such as **A, M, and X** were more difficult to classify
- **Intensity**, **flare duration**, and **time-to-peak** were identified as the most important features influencing flare classification

---

## Data Visualization
Several visualization techniques were used to explore solar flare patterns:

- Distribution of solar flare classes
- Feature correlation heatmap
- Temporal analysis of flare activity
- Daily flare frequency analysis
- Analysis of "All-Clear Days" (days with no flare activity)

These visualizations help reveal long-term patterns and cyclical trends in solar activity.

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
