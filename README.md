
# 🎓 Student Success Predictor

Predict whether a student will **pass or fail** based on features like Study Hours, Attendance, Past Scores, Internet access, and Sleep Hours using Machine Learning.

---

## 📌 Project Overview

This project provides two versions of the same ML workflow:

1. **`edu_success_predictor.py`** – Python script version to **run the entire pipeline at once**.
2. **`student_success.ipynb`** – Jupyter Notebook version for **step-by-step execution with detailed explanations, visualizations, and outputs**.

Both files produce the **same model and predictions**, but the notebook is more interactive and educational.

---

## 🧰 Features

- Load & explore student data
- Handle missing values and encode categorical variables
- Feature scaling & preprocessing
- Train/test split (stratified)
- Train baseline models: Logistic Regression, Decision Tree
- Evaluate models using **accuracy, precision, recall, F1-score**
- Visualize feature importance, confusion matrix, and model predictions
- Save and load trained model using `joblib`
- Optional improvements: hyperparameter tuning, ensemble models

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/student-success-predictor.git
cd student-success-predictor

2️⃣ Create a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/Mac

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the project

Python script:

python edu_success_predictor.py


Jupyter Notebook:

jupyter notebook
# open student_success.ipynb and run cells

📁 Files

data.csv – Dataset for training the model

edu_success_predictor.py – End-to-end script version

student_success.ipynb – Detailed notebook version

requirements.txt – All Python dependencies

.gitignore – Ignore environment & temporary files

📈 Example Outputs

Confusion matrix, classification report

Feature importance plots

Prediction of new student data

💡 Notes

Ensure the Python environment matches your installed packages (matplotlib, seaborn, scikit-learn, xgboost, joblib, imbalanced-learn, pandas, numpy).

The notebook preserves outputs in cells, ideal for demonstrations or learning purposes.

The script is useful for quick execution and automation.

📌 Author

Mohammad Warish

B.Tech CSE Student | RKDF University Ranchi

GitHub: your-username

📝 License

MIT License


---

### 3️⃣ .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.pyc

# Virtual Environment
.venv/
env/
venv/

# Jupyter Notebook
.ipynb_checkpoints/

# Logs and reports
*.log
*.csv

# OS files
.DS_Store
Thumbs.db

# Joblib model files
*.pkl