# 🚢 Titanic Survival Prediction with Scikit-learn Pipeline

This project implements a complete machine learning pipeline to predict Titanic passenger survival using Scikit-learn. It includes data preprocessing, model training with Random Forest, hyperparameter tuning, and performance evaluation.

## 📂 Dataset

- **Source**: [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Features**: Passenger demographics and ticket information
- **Target**: `Survived` (0 = No, 1 = Yes)

## 🧰 Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn

## 🔄 Pipeline Overview

1. **Data Loading**: Import the dataset from the provided URL.
2. **Exploratory Data Analysis (EDA)**:
   - Display dataset head and info.
   - Visualize survival distribution by gender and class.
3. **Data Preprocessing**:
   - Drop irrelevant columns: `Name`, `Ticket`, `Cabin`, `PassengerId`.
   - Identify numerical and categorical features.
   - Handle missing values using `SimpleImputer`.
   - Scale numerical features with `StandardScaler`.
   - Encode categorical features with `OneHotEncoder`.
4. **Model Training**:
   - Split data into training and testing sets.
   - Train a `RandomForestClassifier` within a `Pipeline`.
5. **Model Evaluation**:
   - Calculate accuracy score.
   - Generate confusion matrix and classification report.
   - Plot ROC curve.
6. **Hyperparameter Tuning**:
   - Use `GridSearchCV` to find the best parameters.
   - Evaluate the optimized model.
7. **Feature Importance**:
   - Extract and plot feature importances from the trained model.

## 📈 Results

- **Baseline Accuracy**: ~[Insert Accuracy]
- **Optimized Accuracy**: ~[Insert Accuracy]
- **ROC AUC**: ~[Insert AUC Score]

## 📊 Visualizations

- Survival distribution by gender and class.
- Confusion matrix heatmap.
- ROC curve.
- Feature importance bar chart.

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

Developed by [Peterson Chiquetto](https://github.com/petersonchiquetto)

