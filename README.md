# Customer Churn Prediction

This project predicts whether a customer is likely to churn using machine learning. It uses models like Logistic Regression, Random Forest, and XGBoost — with a full preprocessing pipeline and SMOTE to handle class imbalance. A REST API (FastAPI) serves predictions, and everything is containerized with Docker.

---

## Features

- Exploratory Data Analysis (EDA)
- Preprocessing pipeline using scikit-learn
- SMOTE for class imbalance
- Models: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning
- Evaluation using accuracy, precision, recall, and F1
- REST API via FastAPI
- Dockerized deployment
- GitHub Actions CI/CD integration

---

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kieronh123/churn-prediction.git
cd churn-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the API with Docker

### 1. Build the Docker Image

```bash
docker build -t churn-api .
```

### 2. Run the Container

```bash
docker run -p 8000:80 churn-api
```

### 3. Access the Swagger UI

Open your browser and visit: http://localhost:8000/docs

---

## API Example

### Endpoint

**POST** `/predict`

### Example Input

```json
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
```

### Example Response

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8723
}
```

---

## GitHub Actions CI/CD

GitHub Actions automatically:

- Installs dependencies
- Lints code with Black
- Runs unit tests
- Builds the Docker image

Workflow file: `.github/workflows/docker.yml`

---

## Project Structure

```
churn-prediction/
├── app/
│   └── main.py
├── models/
│   ├── logistic_model_tuned.joblib
│   └── preprocessing_pipeline.joblib
├── src/
│   ├── model.py
│   └── preprocessing.py
├── notebooks/
├── tests/
├── Dockerfile
├── requirements.txt
├── README.md
└── .dockerignore
```

---

## License

This project is licensed under the MIT License.