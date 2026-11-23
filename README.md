
# AI Email Spam Classifier

An end-to-end Machine Learning project that classifies email messages as Spam or Not Spam using Natural Language Processing (NLP) and supervised learning.

## Project Overview

The objective of this project is to build a machine learning application that identifies unwanted emails by learning patterns from previously labelled email messages.

The application allows users to enter an email message and receive a spam classification with estimated prediction probabilities.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Complement Naive Bayes
- Logistic Regression
- Streamlit
- Joblib

## Dataset

Dataset: Apache SpamAssassin Public Corpus.

The dataset contains:
- Legitimate emails (Ham)
- Spam emails

The collected dataset contained 3,000 usable email records:
- 2,501 Ham emails
- 499 Spam emails

## Project Workflow

1. Downloaded and extracted the email dataset.
2. Parsed raw email messages.
3. Cleaned and preprocessed email text.
4. Converted text into numerical features using TF-IDF.
5. Split the dataset into training and testing sets.
6. Trained multiple machine learning classifiers.
7. Compared model performance.
8. Selected Logistic Regression based on experimental results.
9. Developed an interactive Streamlit application.

## Machine Learning Algorithms

The following algorithms were compared:

- Multinomial Naive Bayes
- Complement Naive Bayes
- Logistic Regression

Hyperparameter tuning was performed using GridSearchCV and five-fold cross-validation.

## Model Evaluation

Observed Logistic Regression test results:

| Metric | Ham | Spam |
|---|---:|---:|
| Precision | 0.99 | 0.95 |
| Recall | 0.99 | 0.97 |
| F1-score | 0.99 | 0.96 |

Observed test accuracy: 98.67%.

Note: These are experimental results from the current 600-email test split. The same split was used during model experimentation, so independent validation on a fresh dataset is still recommended.

## Application Features

- Email text input
- Spam and Not Spam classification
- Estimated prediction probabilities
- Interactive Streamlit interface

## Project Structure

```text
AI-Email-Spam-Classifier/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── spam_pipeline_comparison.pkl
│
├── app.py
├── download_dataset.py
├── explore_dataset.py
├── preprocess_data.py
├── tfidf_features.py
├── train_model.py
├── tune_model.py
├── compare_models.py
├── test_predictions.py
├── requirements.txt
└── README.md
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd AI-Email-Spam-Classifier
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

Make sure the trained model file is available in the models directory before launching the application.

## Future Improvements

- Evaluate on a completely independent email dataset.
- Experiment with additional NLP techniques.
- Improve spam detection for previously unseen email patterns.
- Add email file upload functionality.
- Explore advanced machine learning and deep learning approaches.

## Author

Karthik Madhagoni
