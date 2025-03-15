from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(): 

    # Load train.csv
    train_data = pd.read_csv("data/raw/train.csv")

    # Separate features & target variable
    X = train_data.drop(columns=["Attrition"])
    y = train_data["Attrition"]

    # Split into 80% training, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save updated train data to a new file
    updated_train_data = pd.concat([X_train, y_train], axis=1)
    updated_train_data.to_csv("data/processed/train.csv", index=False)
    
    # Save the validation set separately
    validation_data = pd.concat([X_val, y_val], axis=1)
    validation_data.to_csv("data/processed/validation.csv", index=False)
    
    print("Data Split Complete!")
    
# Ensure the function runs only if script is executed directly
# if __name__ == "__main__":
    # split_data()