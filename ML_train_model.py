import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model.logistic import train as log_train
from model.decision_tree import train as dt_train
from model.knn import train as knn_train
from model.naive_bayes import train as nb_train
from model.random_forest import train as rf_train
from model.xgboost_model import train as xgb_train

# Load dataset
ds = pd.read_csv("data/heart.csv")
ds.columns = ds.columns.str.strip()               # Remove any leading/trailing whitespace from column names
print(ds.columns)                                 # Print column names to verify they are correct

X = ds.drop("target", axis=1)                     # Separate features (X) by dropping the target column
y = ds["target"]                                 # Separate target variable (y) from the dataset 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42                 # Split data into training and testing sets (80% train, 20% test
)

# Scaling
scaler = StandardScaler()                       # Initialize scaler
X_train = scaler.fit_transform(X_train)         # Fit scaler on training data and transform it
X_test = scaler.transform(X_test)               # Transform test data using the same scaler (do not fit again)

# Train models
models = {
    "logistic": log_train(X_train, y_train),       # Train logistic regression model
    "decision_tree": dt_train(X_train, y_train),   # Train decision tree model
    "knn": knn_train(X_train, y_train),            # Train KNN model
    "naive_bayes": nb_train(X_train, y_train),     # Train Naive Bayes model
    "random_forest": rf_train(X_train, y_train),   # Train random forest model
    "xgboost": xgb_train(X_train, y_train)         # Train XGBoost model
}

# Save models
for name, model in models.items():
    joblib.dump(model, f"models_saved/{name}.pkl")  # Save each model with its name

joblib.dump(scaler, "models_saved/scaler.pkl")    # Save the scaler separately

print("All models and scaler saved successfully") 
