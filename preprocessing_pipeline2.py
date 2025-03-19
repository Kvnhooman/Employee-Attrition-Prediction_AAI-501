import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self, config):
        """
        Initializes the preprocessor with configuration options.
        """
        self.config = config
        self.encoders = {}

    def load_data(self, train_file, val_file):
        """Loads datasets from files."""
        self.train_df = pd.read_csv(train_file)
        self.val_df = pd.read_csv(val_file)
        
        # Separate features & target
        self.X_train, self.y_train = self.train_df.drop(columns=["Attrition"]), self.train_df["Attrition"]
        self.X_val, self.y_val = self.val_df.drop(columns=["Attrition"]), self.val_df["Attrition"]

    def apply_feature_selection(self):
        """TODO: Applies feature selection to keep only the top K features."""
        if self.config["feature_selection"]:
            print("Feature selection applied.")
            
    def apply_feature_scaling(self):
        """
            Applies feature scaling if enabled.
            Models like Logistic Regression, SVM, KNN, and Neural Networks require scaling since they are sensitive to feature magnitudes.
            Tree-based models (Random Forest, Decision Trees, XGBoost, etc.) do NOT require scaling because they are scale-invariant.     
        """
        if self.config["scale_features"]:
            print(f"Applying {self.config['scaling_method']} scaling...")
            
            scaler = StandardScaler() if self.config["scaling_method"] == "standard" else MinMaxScaler()
            
            # Load encoded column names from JSON file
            try:
                with open("encoded_columns.json", "r") as file:
                    # Load previously encoded column names
                    encoded_cols = json.load(file)
            except FileNotFoundError:
                # Default to empty list if file is missing
                encoded_cols = []

            # Select numerical columns (excluding One-Hot Encoded categorical features)
            num_cols = [col for col in self.X_train.columns if col not in encoded_cols]

            num_cols = self.X_train.select_dtypes(include=np.number).columns

            # Comment out to let code throws exception with NaN, which will help to debug likely due to encoding
            # Check for NaN or Infinite values before scaling
            # self.X_train[num_cols] = self.X_train[num_cols].replace([np.inf, -np.inf], np.nan)
            # self.X_val[num_cols] = self.X_val[num_cols].replace([np.inf, -np.inf], np.nan)

            # Fill NaN values with the mean of each column (or use another imputation method)
            # self.X_train[num_cols] = self.X_train[num_cols].fillna(self.X_train[num_cols].mean())
            # self.X_val[num_cols] = self.X_val[num_cols].fillna(self.X_train[num_cols].mean())  # Use train mean for consistency

            # Apply scaling
            self.X_train[num_cols] = scaler.fit_transform(self.X_train[num_cols])
            self.X_val[num_cols] = scaler.transform(self.X_val[num_cols])

            self.encoders["scaler"] = scaler

            print("Feature scaling applied successfully.")


    def handle_class_imbalance(self):
        """Handles class imbalance using SMOTE (Synthetic Oversampling)."""
        if self.config["handle_class_imbalance"]:
            print("Applying SMOTE to balance classes...")
            
            smote = SMOTE(sampling_strategy="auto", random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            
            print("Class imbalance handled.")

    def save_preprocessed_data(self, output_train, output_val):
        """Saves the processed training and validation datasets."""
        train_final = pd.concat([pd.DataFrame(self.X_train), self.y_train.reset_index(drop=True)], axis=1)
        val_final = pd.concat([pd.DataFrame(self.X_val), self.y_val.reset_index(drop=True)], axis=1)
        
        train_final.to_csv(output_train, index=False)
        val_final.to_csv(output_val, index=False)

        print(f"Processed datasets saved: {output_train}, {output_val}")

    def run_pipeline(self, train_file, val_file, output_train, output_val):
        """Executes the full preprocessing pipeline."""
        self.load_data(train_file, val_file)
        self.apply_feature_selection()
        self.apply_feature_scaling()
        self.handle_class_imbalance()
        self.save_preprocessed_data(output_train, output_val)


# Configuration
config = {
    "scale_features": True,
    "scaling_method": "standard",
    "feature_selection": False,
    "handle_class_imbalance": False
}

# Run the Preprocessing Pipeline
if __name__ == "__main__":
    preprocessor = DataPreprocessor(config)
    preprocessor.run_pipeline(
        train_file="data/processed/train_processed.csv",
        val_file="data/processed/validation_processed.csv",
        output_train="data/processed/train_scaled.csv",
        output_val="data/processed/validation_scaled.csv"
    )
