import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

class DataPreprocessor:
    def __init__(self, preserve_columns_for_grouping = None):
        """
        Initializes the preprocessor.
        """
        self.encoders = {}
        self.preserve_columns_for_grouping = preserve_columns_for_grouping or []

    def load_data(self, train_file, val_file):
        """Loads datasets from files."""
        self.train_df = pd.read_csv(train_file)
        self.val_df = pd.read_csv(val_file)
        
         # Drop EmployeeID column
        if "Employee ID" in self.train_df.columns:
            self.train_df.drop(columns=["Employee ID"], inplace=True)
            self.val_df.drop(columns=["Employee ID"], inplace=True)
            
        if "Company Tenure" in self.train_df.columns:
            self.train_df.drop(columns=["Company Tenure"], inplace=True)
            self.val_df.drop(columns=["Company Tenure"], inplace=True)
        
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
            
    def feature_engineering(self):
        """Creates new features from existing columns."""
        print("Adding engineered features...")

        for df in [self.X_train, self.X_val]:
            # Avoid divide-by-zero
            df['Promotions_per_Year'] = df['Number of Promotions'] / (df['Years at Company'] + 1)
            df['Income_per_Year'] = (df['Monthly Income'] * 12) / (df['Years at Company'] + 1)
            df['First_Promotion_Delay'] = df['Years at Company'] / (df['Number of Promotions'] + 1)

            # Stress index
            df['Overtime_Flag'] = (df['Overtime'] == 'Yes').astype(int)
            df['WLB_Poor_Flag'] = (df['Work-Life Balance'] == 'Poor').astype(int)
            df['Distance_Level'] = pd.qcut(df['Distance from Home'], 3, labels=[0, 1, 2]).astype(int)
            df['Stress_Index'] = df['Overtime_Flag'] + df['WLB_Poor_Flag'] + df['Distance_Level']

            # Tenure stage
            df['Tenure_Stage'] = pd.cut(df['Years at Company'],
                                        bins=[0, 3, 10, df['Years at Company'].max()],
                                        labels=['Early', 'Mid', 'Senior'])

            # Dependents ratio
            df['Dependents_Ratio'] = df['Number of Dependents'] / (df['Age'] + 1)

            # NEW FEATURES
            # Has promotion flag
            df['Has_Promotion'] = (df['Number of Promotions'] > 0).astype(int)

            # Promotion speed
            df['Promotion_Speed'] = df['Years at Company'] / (df['Number of Promotions'] + 1)

            # Ordinal encoding for Work-Life Balance
            wlb_map = {
                'Poor': 0,
                'Fair': 1,
                'Good': 2,
                'Excellent': 3
            }
            df['Work_Life_Balance_Level'] = df['Work-Life Balance'].map(wlb_map)

            # Binned Distance from Home
            df['Distance_Bin'] = pd.cut(df['Distance from Home'],
                                        bins=[-1, 5, 15, df['Distance from Home'].max()],
                                        labels=['Near', 'Medium', 'Far'])

            # Income vs. Job Level average (optional depending on data availability)
            if 'Job Level' in df.columns:
                df['Income_vs_JobLevel'] = df['Monthly Income'] / (df.groupby('Job Level')['Monthly Income'].transform('mean') + 1)

        print("Feature engineering done.")

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
            
    def auto_encode(self, df, preserve_columns_for_grouping=None, threshold=2):
        if preserve_columns_for_grouping is None:
            preserve_columns_for_grouping = []

        df_encoded = df.copy()

        for col in df.select_dtypes(include=["object", "category"]).columns:
            unique_vals = df[col].nunique()

            if col in self.encoders:
                encoder_type, encoder = self.encoders[col]

                if encoder_type == "label":
                    df_encoded[col] = encoder.transform(df[col])
                elif encoder_type == "onehot":
                    transformed = encoder.transform(df[[col]])
                    ohe_cols = encoder.get_feature_names_out([col])
                    df_ohe = pd.DataFrame(transformed, columns=ohe_cols, index=df.index)
                    if col not in preserve_columns_for_grouping:
                        df_encoded.drop(columns=[col], inplace=True)
                    df_encoded = pd.concat([df_encoded, df_ohe], axis=1)

            else:
                if unique_vals <= threshold:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df[col])
                    self.encoders[col] = ("label", le)

                else:
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                    transformed = ohe.fit_transform(df[[col]])
                    ohe_cols = ohe.get_feature_names_out([col])
                    df_ohe = pd.DataFrame(transformed, columns=ohe_cols, index=df.index)
                    if col not in preserve_columns_for_grouping:
                        df_encoded.drop(columns=[col], inplace=True)
                    df_encoded = pd.concat([df_encoded, df_ohe], axis=1)
                    self.encoders[col] = ("onehot", ohe)

        return df_encoded

    def encode_categorical_features(self):
        """Applies One-Hot Encoding for nominal features and Label Encoding for ordinal features."""
        
        print("Encoding categorical features...")
        
        # Reset encoders before training
        self.encoders = {}
        self.X_train = self.auto_encode(self.X_train, preserve_columns_for_grouping=self.preserve_columns_for_grouping )
        self.X_val = self.auto_encode(self.X_val, preserve_columns_for_grouping = self.preserve_columns_for_grouping)
        
        print("Categorical encoding done.")
        print(f"Training data NaN: {self.X_train.isnull().sum()}")
        print(f"Validation data NaN: {self.X_val.isnull().sum()}")

    def process_by_segment(self, group_column):
        """
        Splits the data by a group column and processes each segment separately.
        Removes outliers and encodes features independently per segment.
        Returns a dictionary with training and validation data per group.
        """
        print(f"Processing data by segments in '{group_column}'...")

        groups = self.X_train[group_column].unique()
        segment_data = {}

        for group in groups:
            print(f"  Segment: {group}")
            
            # Filter by group
            mask_train = self.X_train[group_column] == group
            mask_val = self.X_val[group_column] == group

            X_train_grp = self.X_train[mask_train].copy()
            y_train_grp = self.y_train[mask_train].copy()
            X_val_grp = self.X_val[mask_val].copy()
            y_val_grp = self.y_val[mask_val].copy()

            # Outlier removal within segment
            num_cols = X_train_grp.select_dtypes(include='number').columns
            if not num_cols.empty:
                z_scores = zscore(X_train_grp[num_cols])
                mask = (np.abs(z_scores) < 3).all(axis=1)
                X_train_grp = X_train_grp[mask]
                y_train_grp = y_train_grp[mask]

            # Drop the group column after segmentation (unless preserving it)
            if group_column not in self.preserve_columns_for_grouping:
                X_train_grp.drop(columns=[group_column], inplace=True)
                X_val_grp.drop(columns=[group_column], inplace=True)

            # Encode each group independently
            X_train_encoded = self.auto_encode(X_train_grp, preserve_columns_for_grouping=[])
            X_val_encoded = self.auto_encode(X_val_grp, preserve_columns_for_grouping=[])

            # Save per-group data
            segment_data[group] = {
                "X_train": X_train_encoded.reset_index(drop=True),
                "y_train": y_train_grp.reset_index(drop=True),
                "X_val": X_val_encoded.reset_index(drop=True),
                "y_val": y_val_grp.reset_index(drop=True)
            }

        print("Segmentation-based processing complete.")
        return segment_data

    def process_by_kmeans(self, n_clusters=4, features_for_clustering=None):
        """
        Segments the data using KMeans clustering and returns per-cluster datasets
        for training individual models.
        """
        print(f"Clustering with KMeans (k={n_clusters})...")

        # Use only numeric features if not specified
        if features_for_clustering is None:
            features_for_clustering = self.X_train.select_dtypes(include='number').columns.tolist()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train[features_for_clustering])
        X_val_scaled = scaler.transform(self.X_val[features_for_clustering])

        # Fit KMeans on training data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels_train = kmeans.fit_predict(X_train_scaled)
        cluster_labels_val = kmeans.predict(X_val_scaled)

        # Add cluster ID to data
        self.X_train["Cluster"] = cluster_labels_train
        self.X_val["Cluster"] = cluster_labels_val

        # Create per-cluster splits
        segment_data = {}

        for cluster_id in range(n_clusters):
            print(f"  Preparing cluster: {cluster_id}")

            mask_train = self.X_train["Cluster"] == cluster_id
            mask_val = self.X_val["Cluster"] == cluster_id

            X_train_cluster = self.X_train[mask_train].copy()
            y_train_cluster = self.y_train[mask_train].copy()
            X_val_cluster = self.X_val[mask_val].copy()
            y_val_cluster = self.y_val[mask_val].copy()

            # Optional: Drop 'Cluster' if not needed as a feature
            X_train_cluster.drop(columns=["Cluster"], inplace=True)
            X_val_cluster.drop(columns=["Cluster"], inplace=True)

            # Encode separately for each cluster (recommended)
            X_train_encoded = self.auto_encode(X_train_cluster)
            X_val_encoded = self.auto_encode(X_val_cluster)

            segment_data[cluster_id] = {
                "X_train": X_train_encoded.reset_index(drop=True),
                "y_train": y_train_cluster.reset_index(drop=True),
                "X_val": X_val_encoded.reset_index(drop=True),
                "y_val": y_val_cluster.reset_index(drop=True),
            }

        print("KMeans-based segmentation complete.")
        return segment_data

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
        # self.feature_engineering()
        self.remove_outliers()
        self.encode_categorical_features()
        self.save_preprocessed_data(output_train, output_val)

preserve_columns_for_grouping=[""]

# Run the Preprocessing Pipeline
if __name__ == "__main__":
    preprocessor = DataPreprocessor(preserve_columns_for_grouping)
    preprocessor.run_pipeline(
        train_file="data/processed/train.csv",
        val_file="data/processed/validation.csv",
        output_train="data/processed/train_processed.csv",
        output_val="data/processed/validation_processed.csv"
    )
