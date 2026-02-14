import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

st.title("❤️ Heart Disease Prediction")            # Set the page title and layout for the Streamlit app
st.markdown(
    "<p style='font-size:18px;'>Comparison of Multiple Machine Learning Models</p>",
    unsafe_allow_html=True
)

# Load scaler and models
scaler = joblib.load("models_saved/scaler.pkl")

# Load all trained machine learning models
# Stored as a dictionary for easy selection
models = {
    "Logistic Regression": joblib.load("models_saved/logistic.pkl"),     # Load Logistic Regression model from the specified file path
    "Decision Tree": joblib.load("models_saved/decision_tree.pkl"),      # Load Decision Tree model from the specified file path
    "KNN": joblib.load("models_saved/knn.pkl"),                          # Load K-Nearest Neighbors model from the specified file path
    "Naive Bayes": joblib.load("models_saved/naive_bayes.pkl"),          # Load Naive Bayes model from the specified file path
    "Random Forest": joblib.load("models_saved/random_forest.pkl"),      # Load Random Forest model from the specified file path
    "XGBoost": joblib.load("models_saved/xgboost.pkl"),                  # Load XGBoost model from the specified file path
}


# ---------------- Dataset Selection ----------------
st.subheader("Select Dataset")

data_option = st.radio(
    "Choose Dataset Option:",
    ("Use Default Dataset (heart.csv)", "Upload a New Dataset")
)

if data_option == "Use Default Dataset (heart.csv)":    # If the user selects the option to use the default dataset, attempt to load it
    try:
        df = pd.read_csv("data/heart.csv")                      # Load the default dataset from the specified file path
        st.success("Default dataset loaded successfully !")
    except:
        st.error("Default dataset not found! Place heart.csv inside data folder.")
        st.stop()
else:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])    # Provide a file uploader for the user to upload a new dataset in CSV format
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)                                 # Load the uploaded dataset into a DataFrame
        st.success("Custom dataset uploaded successfully !")
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
model_name = st.selectbox("Select Model", list(models.keys()))        # Dropdown menu for selecting a machine learning model   
model = models[model_name]   # Get the selected model from the dictionary
# Separate features (X) and target variable (y)
X = df.drop("target", axis=1)
y = df["target"]

X_scaled = scaler.transform(X)       # Scale feature values using the trained scaler

y_pred = model.predict(X_scaled)                 # Predict class labels (0 or 1)
y_prob = model.predict_proba(X_scaled)[:, 1]     # Predict probability values for ROC-AUC calculation

# Display evaluation metrics
st.subheader("Evaluation Metrics")

st.write("Accuracy:", accuracy_score(y, y_pred))          # Calculate and display accuracy score
st.write("AUC:", roc_auc_score(y, y_prob))                # Calculate and display ROC-AUC score using predicted probabilities
st.write("Precision:", precision_score(y, y_pred))        # Calculate and display precision score
st.write("Recall:", recall_score(y, y_pred))              # Calculate and display recall score
st.write("F1 Score:", f1_score(y, y_pred))                # Calculate and display F1 score
st.write("MCC:", matthews_corrcoef(y, y_pred))            # Calculate and display Matthews Correlation Coefficient (MCC) score

# Display confusion matrix
st.subheader("Confusion Matrix")


cm = confusion_matrix(y, y_pred)
# Create a matplotlib figure with reduced size

fig, ax = plt.subplots(figsize=(3, 3))    
im = ax.imshow(cm, cmap="viridis")         # Display the confusion matrix as an image
# Label axes
ax.set_xlabel("Predicted", fontsize=8)      # Set x-axis label to "Predicted" with smaller font size
ax.set_ylabel("Actual", fontsize=8)         # Set y-axis label to "Actual" with smaller font size
# ax.set_title("Confusion Matrix", fontsize=9)

# Display values inside each confusion matrix cell
for i in range(cm.shape[0]):             # Loop through each row of the confusion matrix
    for j in range(cm.shape[1]):         # Loop through each column of the confusion matrix
        ax.text(j, i, cm[i, j],          # Add text annotation for the value in the cell
                ha="center", va="center",   # Center the text horizontally and vertically
                fontsize=8, color="black")  # Set font size and color for the text annotation

fig.colorbar(im, fraction=0.04, pad=0.03)         # Add color bar for better interpretation
plt.tight_layout()
st.pyplot(fig)