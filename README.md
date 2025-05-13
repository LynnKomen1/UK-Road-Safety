# UK Road Safety: Casualty Severity Prediction

This project builds and compares multiple machine learning models to predict casualty severity in road accidents using the UK Road Safety dataset.

## Objective
- Predict whether a casualty is likely to be *slight* or *serious* based on accident-related features
- Improve emergency response and inform healthcare and transport policy

## Tools Used
- Python (pandas, scikit-learn, xgboost)
- Jupyter Notebook
- GridSearchCV for hyperparameter tuning

## Models Compared
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- AdaBoost

## Best Performing Model
- **Logistic Regression** (C=0.1) with **95% test accuracy** and strong generalisation

## Key Features
- Feature engineering: interaction terms (e.g., `age_band_of_casualty` + `casualty_class`)
- Chi-square and Phi-K correlation for variable selection
- One-hot encoding and model tuning with cross-validation

## Visualisations
- Stacked bar charts for severity distribution
- Confusion matrices and classification reports for evaluation

## Possible Improvements
- Use real-time data feeds
- Add contextual variables such as geospatial data (e.g., location of accidents)
- Add weather conditions (e.g., rain, fog, or lighting), which can improve model accuracy and realism
- Implement ensemble stacking

---

📁 `UK-Road-Safety.ipynb`: Full code and results  
📂 `my_models/`: Saved trained models (`.joblib` format)
