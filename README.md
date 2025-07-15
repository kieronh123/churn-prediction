# 🧠 Customer Churn Prediction

This project predicts whether a customer is likely to churn using machine learning. It uses models like Logistic Regression, Random Forest, and XGBoost — with a full preprocessing pipeline and SMOTE to handle class imbalance. A REST API (FastAPI) serves predictions, and everything is containerized with Docker.

---


## 🚀 Features

- 🔍 Exploratory Data Analysis (EDA)
- 🧹 Preprocessing pipeline (scikit-learn)
- 🔄 SMOTE for balancing classes
- 🧠 Models: Logistic Regression, Random Forest, XGBoost
- 🧪 Hyperparameter tuning
- 📈 Evaluation with accuracy, precision, recall, F1
- ⚙️ REST API with FastAPI
- 🐳 Dockerized for deployment
- 💡 GitHub Actions CI/CD

---

## 🧪 Local Development Setup

### 1. Clone the Repository


git clone https://github.com/kieronh123/churn-prediction.git
cd churn-prediction
2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
3. Install Dependencies

pip install -r requirements.txt
---


## 🐳 Running the API with Docker
1. Build the Docker Image
bash
Copy
Edit
docker build -t churn-api .
2. Run the Container
bash
Copy
Edit
docker run -p 8000:80 churn-api
3. Open in Browser
Visit: http://localhost:8000/docs

This opens the Swagger UI where you can try out the /predict endpoint.
---

## 📥 API Example
### Endpoint
POST /predict

### Example Input
json

{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 845.5
}
### Example Response

{
  "churn_prediction": 1,
  "churn_probability": 0.8723
}

---



⚙️ GitHub Actions CI/CD
GitHub Actions automatically:

Installs dependencies

Lints code with black

Builds the Docker image

Config: .github/workflows/docker.yml

📂 Project Structure

churn-prediction/
├── app/                        # FastAPI app
│   └── main.py
├── models/                     # Trained model + pipeline
│   └── logistic_model_tuned.joblib
│   └── preprocessing_pipeline.joblib
├── src/                        # ML logic
│   ├── model.py
│   └── preprocessing.py
├── notebooks/                  # Optional Jupyter notebooks
├── Dockerfile
├── requirements.txt
├── README.md
└── .dockerignore
License
This project is licensed under the MIT License.


---

