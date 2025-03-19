# Employee Attrition Prediction_AAI 501
This is gonna predict if your employee is gonna leave you!

Run "python3 preprocessing_pipeline1.py" or "python3 preprocessing_pipeline2.py" in the terminal to preprocess training and validation dataset.

🔹 Preprocessing Pipeline 1 (Before Feature Selection & Scaling)

✔ Missing data handling

✔ Initial feature extraction & selection

✔ Categorical Encoding (One-Hot / Ordinal Encoding)

🔹 Preprocessing Pipeline 2 (After Encoding, Before Model Training)

✔ Feature scaling

✔ Dimensionality reduction (feature selection, PCA, etc.)

✔ Final data transformation before model training

"encoded_columns.json" is used to track One-Hot Encoded columns because feature scales only apply to numerical features, ignoring encoded categorical ones.
