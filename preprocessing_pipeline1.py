import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class DataPreprocessor:
    def __init__(self):
        """
        Initializes the preprocessor.
        """
        self.encoders = {}

    def load_data(self, train_file, val_file):
        """Loads datasets from files."""
        self.train_df = pd.read_csv(train_file)
        self.val_df = pd.read_csv(val_file)
        
         # Drop EmployeeID column
        if "Employee ID" in self.train_df.columns:
            self.train_df.drop(columns=["Employee ID"], inplace=True)
            self.val_df.drop(columns=["Employee ID"], inplace=True)
        
        # Convert Attrition to binary values (Yes -> 1, No -> 0)
        self.train_df["Attrition"] = self.train_df["Attrition"].map({"Left": 1, "Stayed": 0})
        self.val_df["Attrition"] = self.val_df["Attrition"].map({"Left": 1, "Stayed": 0})

        
        # Separate features & target
        self.X_train, self.y_train = self.train_df.drop(columns=["Attrition"]), self.train_df["Attrition"]
        self.X_val, self.y_val = self.val_df.drop(columns=["Attrition"]), self.val_df["Attrition"]

    def handle_missing_values(self):
        """Handles missing values in numerical and categorical columns."""
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
        
    def remove_outliers(self):
        """Removes rows with Z-score greater than threshold on any numeric column."""
        print("Removing outliers...")

        # Apply outlier removal and preserve index
        mask_train = (zscore(self.X_train.select_dtypes(include='number')).astype(float) < 3).all(axis=1)
        mask_val = (zscore(self.X_val.select_dtypes(include='number')).astype(float) < 3).all(axis=1)

        # Apply masks to both features and targets
        original_train_len = len(self.X_train)
        original_val_len = len(self.X_val)

        self.X_train = self.X_train[mask_train].reset_index(drop=True)
        self.y_train = self.y_train[mask_train].reset_index(drop=True)

        self.X_val = self.X_val[mask_val].reset_index(drop=True)
        self.y_val = self.y_val[mask_val].reset_index(drop=True)

        print(f"Outliers removed: {original_train_len - len(self.X_train)} from training set")
        print(f"Outliers removed: {original_val_len - len(self.X_val)} from validation set")
            
    def auto_encode(self, df, threshold = 2):
        df_encoded = df.copy()
        for col in df.select_dtypes(include=["object", "category"]).columns:
            unique_vals = df[col].nunique()

            if unique_vals <= threshold:
                # Ordinal: Label Encoding
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col])
                self.encoders[col] = ("label", le)
            else:
                # Nominal: One-Hot Encoding
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                transformed = ohe.fit_transform(df[[col]])
                ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                df_ohe = pd.DataFrame(transformed, columns=ohe_cols, index=df.index)

                # Drop original and add encoded
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, df_ohe], axis=1)

                self.encoders[col] = ("onehot", ohe)

        return df_encoded
    
    def encode_categorical_features(self):
        """Applies One-Hot Encoding for nominal features and Label Encoding for ordinal features."""
        
        print("Encoding categorical features...")
        
        self.encoders = {}  # Reset encoders before training
        self.X_train = self.auto_encode(self.X_train)
        self.X_val = self.auto_encode(self.X_val)
        
        print("Categorical encoding done.")
        print(f"Training data NaN: {self.X_train.isnull().sum()}")
        print(f"Validation data NaN: {self.X_val.isnull().sum()}")

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
        self.remove_outliers()
        self.encode_categorical_features()
        self.save_preprocessed_data(output_train, output_val)

# Run the Preprocessing Pipeline
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline(
        train_file="data/processed/train.csv",
        val_file="data/processed/validation.csv",
        output_train="data/processed/train_processed.csv",
        output_val="data/processed/validation_processed.csv"
    )
