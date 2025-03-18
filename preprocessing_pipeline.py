import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self, config):
        """
        Initializes the preprocessor with configuration options.
        """
        self.config = config
        self.encoders = {}  # Store fitted encoders/scalers for later use

    def load_data(self, train_file, val_file):
        """Loads datasets from files."""
        self.train_df = pd.read_csv(train_file)
        self.val_df = pd.read_csv(val_file)
        
        # Separate features & target
        self.X_train, self.y_train = self.train_df.drop(columns=["Attrition"]), self.train_df["Attrition"]
        self.X_val, self.y_val = self.val_df.drop(columns=["Attrition"]), self.val_df["Attrition"]

    def handle_missing_values(self):
        """Handles missing values in numerical and categorical columns."""
        if self.config["impute_missing"]:
            print("Handling missing values...")
            
            # Numerical imputation
            num_imputer = SimpleImputer(strategy="mean")
            num_cols = self.X_train.select_dtypes(include=np.number).columns
            self.X_train[num_cols] = num_imputer.fit_transform(self.X_train[num_cols])
            self.X_val[num_cols] = num_imputer.transform(self.X_val[num_cols])
            
            # Categorical imputation
            cat_imputer = SimpleImputer(strategy="most_frequent")
            cat_cols = self.X_train.select_dtypes(include="object").columns
            self.X_train[cat_cols] = cat_imputer.fit_transform(self.X_train[cat_cols])
            self.X_val[cat_cols] = cat_imputer.transform(self.X_val[cat_cols])
            
            print("Missing values handled.")

    def encode_categorical_features(self):
        """Applies One-Hot Encoding for nominal features and Label Encoding for ordinal features."""
        if self.config["encode_categorical"]:
            print("Encoding categorical features...")
            
            # Define ordinal features and mappings
            ordinal_mappings = {
                "Work-Life Balance": {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3},
                "Job Satisfaction": {"Low": 0, "Medium": 1, "High": 2, "Very High": 3},
                "Performance Rating": {"Low": 0, "Below Average": 1, "Average": 2, "High": 3},
                "Education Level": {"High School": 0, "Associate Degree": 1, "Bachelor’s Degree": 2, "Master’s Degree": 3, "PhD": 4},
                "Job Level": {"Entry": 0, "Mid": 1, "Senior": 2},
                "Company Size": {"Small": 0, "Medium": 1, "Large": 2},
                "Company Reputation": {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3},
                "Employee Recognition": {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}             
            }
        
            # Apply ordinal mapping for ordinal features
            for col, mapping in ordinal_mappings.items():
                if col in self.X_train.columns:
                    self.X_train[col] = self.X_train[col].map(mapping)
                if col in self.X_val.columns:
                    self.X_val[col] = self.X_val[col].map(mapping)

                # Store mappings for later use (for test data)
                self.encoders[col] = mapping

            # Identify categorical columns BEFORE encoding
            all_categorical_cols = self.X_train.select_dtypes(include="object").columns.tolist()

            # Identify nominal categorical columns (only the ones NOT in ordinal_mappings)
            nominal_cols = [col for col in all_categorical_cols if col not in ordinal_mappings]

            # One-Hot Encoding for nominal features
            if len(nominal_cols) > 0:
                encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

                # Apply One-Hot Encoding only to nominal columns
                X_train_nominal = encoder.fit_transform(self.X_train[nominal_cols])
                X_val_nominal = encoder.transform(self.X_val[nominal_cols])

                # Convert to DataFrame with proper column names
                nominal_col_names = encoder.get_feature_names_out(nominal_cols)
                
                X_train_nominal = pd.DataFrame(X_train_nominal, columns=nominal_col_names, index=self.X_train.index)
                X_val_nominal = pd.DataFrame(X_val_nominal, columns=nominal_col_names, index=self.X_val.index)

                # Drop original nominal columns and replace them with encoded versions
                self.X_train = self.X_train.drop(columns=nominal_cols)
                self.X_val = self.X_val.drop(columns=nominal_cols)
                self.X_train = pd.concat([self.X_train, X_train_nominal], axis=1)
                self.X_val = pd.concat([self.X_val, X_val_nominal], axis=1)

                # Store the encoder for later use on test data
                self.encoders["onehot_encoder"] = encoder

            print("Categorical encoding done.")

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
            num_cols = self.X_train.select_dtypes(include=np.number).columns

            # Comment out to let code throws exception with NaN, likely due to encoding
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
        self.handle_missing_values()
        self.encode_categorical_features()
        self.apply_feature_selection()
        self.apply_feature_scaling()
        self.handle_class_imbalance()
        self.save_preprocessed_data(output_train, output_val)


# Configuration
config = {
    "impute_missing": True,
    "encode_categorical": True,
    "scale_features": False,
    "scaling_method": "standard",
    "feature_selection": False,
    "handle_class_imbalance": False
}

# Run the Preprocessing Pipeline
if __name__ == "__main__":
    preprocessor = DataPreprocessor(config)
    preprocessor.run_pipeline(
        train_file="data/processed/train.csv",
        val_file="data/processed/validation.csv",
        output_train="data/processed/train_processed.csv",
        output_val="data/processed/validation_processed.csv"
    )
