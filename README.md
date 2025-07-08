# Heart Disease Prediction Project 🫀

This project uses various machine learning models to predict the presence of heart disease in patients, based on their medical attributes. It was built as part of my training with the [Zero to Mastery](https://zerotomastery.io) program.

## 📌 Problem Statement

> Given clinical parameters about a patient, can we predict whether or not they have heart disease?

## 🎯 Project Goal

The initial goal was to achieve **95% accuracy** in predicting heart disease as a proof of concept.

The best model achieved **88.52% accuracy**, which is promising but leaves room for further improvement.

## 📊 Dataset

The data is from the [Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), also available on [Kaggle](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland).

## 🔧 Tools and Technologies

- Python
- Pandas, NumPy, Matplotlib
- scikit-learn (Logistic Regression, KNN, RandomForest)
- Jupyter Notebook

## 🧠 Modeling Process

1. Explored relationships between features and the target variable
2. Visualized trends using bar charts, histograms, correlation matrix, and more
3. Trained and compared 3 models:
   - Logistic Regression ✅ (Best: **88.52% accuracy**)
   - K-Nearest Neighbors (KNN)
   - Random Forest Classifier
4. Improved best model using cross-validation and hyperparameter tuning
5. Evaluated performance using:
   - Accuracy
   - Precision, Recall, F1 Score using cross-validation
   - ROC Curve
   - Confusion Matrix
   - Feature Importance

## 📈 Visualizations

> All plots are saved in the `images/` folder and include:

- Heart disease frequency by sex
- Age vs. Max Heart Rate
- Age distribution
- Chest pain types vs. heart disease
- Correlation heatmap
- ROC Curve
- Feature importance

## 🧪 Experimentation and Next Steps

Although our best model achieved 88.52% accuracy (short of the 95% target), future improvements could include:

- Collecting more data
- Testing advanced models like XGBoost or CatBoost
- Testing the removal of low-importance features (e.g., `age`, `chol`) to reduce dataset dimensionality and potentially improve model performance
- More extensive hyperparameter tuning

## 🚀 Getting Started

To run this project:
1. Clone this repo
2. Set up a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## 📁 Files

- `end-to-end-heart-disease-classification.ipynb`: Main notebook
- `images/`: Visualizations
- `heart-disease (1).csv`: Dataset
- `heart_disease_model.joblib`: Trained model
- `.gitignore`: Excludes env/ and cache files

## 📬 Contact

Feel free to reach out if you have any questions!

> Arafat Mahmood  
> [GitHub](https://github.com/ArafatMahmood45)
> [LinkedIn](https://www.linkedin.com/in/arafat-mahmood-3b0208213/)
