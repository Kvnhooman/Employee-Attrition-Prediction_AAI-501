# Employee Attrition Prediction_AAI 501
This is gonna predict if your employee is gonna leave you!

Run "python3 preprocessing_pipeline.py" in the terminal to preprocess training and validation dataset.
In this pipeline, you can update configurations to decide which operation to apply. For example, scale features are not required in all of the ML models, which can be skipped by setting the value to False.
config = {
    "impute_missing": True,
    "encode_categorical": True,
    "scale_features": False,
    "scaling_method": "standard",
    "feature_selection": False,
    "handle_class_imbalance": False
}

