# Employee Attrition Prediction

This repository contains code and data for predicting employee attrition. The project aims to identify key factors that contribute to employee turnover and develop predictive models to 
identify at-risk employees.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Methodology](#methodology)
- [Files Description](#files-description)
- [Dependencies](#dependencies)
- [Installation and Usage](#installation-and-usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

Employee attrition is a significant concern for organizations, impacting productivity and incurring costs. This project utilizes machine learning to analyze employee data and predict the likelihood of attrition, enabling proactive retention strategies.

## Data Description

The dataset comprises various employee attributes, including demographic, job-related, and satisfaction metrics. It is structured into raw, engineered, and processed versions for training and validation.

-   `data/raw/train.csv`: Original training data.
-   `data/raw/validation.csv`: Original validation data.
-   `data/processed/train_engineered.csv`: Training data after feature engineering.
-   `data/processed/train_processed.csv`: Training data after preprocessing.
-   `data/processed/validation_engineered.csv`: Validation data after feature engineering.
-   `data/processed/validation_processed.csv`: Validation data after preprocessing.
-   `test_engineered.csv`: Engineered test data.

## Methodology

This project employs a multi-faceted approach:

1.  **Data Preprocessing:** Handling missing values, encoding categorical variables, and scaling numerical features.
2.  **Feature Engineering:** Creating interaction features and transforming existing variables.
3.  **Exploratory Data Analysis (EDA):** Visualizing data distributions and correlations.
4.  **Model Building:** Training Logistic Regression, Gradient Boosting, Random Forest, and XGBoost models.
5.  **Model Evaluation:** Using metrics like accuracy, precision, recall, F1-score, and ROC AUC.
6.  **Hyperparameter Tuning:** Optimizing models using GridSearchCV.
7.  **Segmented Analysis:** Training and evaluating models for different job level segments.
8.  **Subsegment Analysis:** Creating and analysing results of smaller subsegments.
9.  **Feature Importance Analysis:** Identifying key predictors for each model.
10. **Test Data Application:** Generating predictions on the `test_engineered.csv` dataset.

## Files Description

-   `data/raw/`: Contains the original raw datasets.
-   `data/processed/`: Contains the engineered and processed datasets.
-   `segment.py`: Python script for segmented XGBoost modeling and detailed analysis.
-   `AttritionPrediction.ipynb`: Jupyter notebook for ensemble model implementation.
-   `Logistic Regression.ipynb`: Jupyter notebook for Logistic Regression model implementation.
-   `test_engineered.csv`: Test dataset with engineered features.

## Dependencies

-   Python (>=3.6)
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost
-   Matplotlib
-   Seaborn
-   Imbalanced-learn

## Installation and Usage

1. Clone the repository:

  git clone [https://github.com/Kvnhooman/Employee-Attrition-Prediction_AAI-501.git](https://github.com/Kvnhooman/Employee-Attrition-Prediction_AAI-501.git)

2. Navigate to the project directory:
  cd Employee-Attrition-Prediction_AAI-501

3.Install dependencies:

  pip install -r requirements.txt

4. Run the scripts and notebooks:
  python segment.py
  jupyter notebook AttritionPrediction.ipynb
  jupyter notebook Logistic\ Regression.ipynb

## Results

XGBoost and ensemble models showed strong performance, with ROC AUC scores above 0.8 on the validation set.
Key features influencing attrition include (List of key features here after running the models).
Segmented analysis revealed variations in feature importance across job levels, highlighting the need for tailored retention strategies.
The gradient boosting and logistic regression models have been implemented, and the test data can be run through them to produce the final results. (add the test result performance here)

## Future Improvements

Refine hyperparameter tuning for all models.
Explore additional feature engineering techniques to improve model performance.
Implement a cost-sensitive analysis to evaluate the financial impact of attrition.
Develop a web application or API for real-time attrition risk assessment.
Add more detailed documentation, including a data dictionary.


## Contributing

Contributions are welcome. Please submit pull requests or open issues for suggestions and improvements.

## Contact

Team at university of San Diego
