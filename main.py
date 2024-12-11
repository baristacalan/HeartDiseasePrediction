# import libraries
# Load dataset and Exploratory Data Analysis
# Handle missing ones, outliers etc.
# Train-Test split
# Standardization
#Modellization
#CM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

def main():
    df = pd.read_csv('heart_disease_uci.csv')
    print(df.info())
    describe = df.describe()
    df.drop(columns=["id", "ca"], inplace=True)
    sns.countplot(data=df, x="num")
    plt.show()
    # trestbps - chol - fbs - restecg - thalch - exang - oldpeak - slope - thal

    null_columns = df.columns[df.isnull().any()].tolist()
    for col in null_columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col].fillna(value=df[col].mean(), inplace=True)
        else:
            df[col].fillna(value=df[col].mode()[0], inplace=True)

    """df['trestbps'].fillna(value=df['trestbps'].mean(), inplace=True)
    df['chol'].fillna(value=df['chol'].mean(), inplace=True)
    df['fbs'].fillna(value=df['fbs'].mode()[0], inplace=True)
    df['restecg'].fillna(value=df['restecg'].mode()[0], inplace=True)
    df['thalch'].fillna(value=df['thalch'].mean(), inplace=True)
    df['exang'].fillna(value=df['exang'].mode()[0], inplace=True)
    df['oldpeak'].fillna(value=df['oldpeak'].mean(), inplace=True)
    df['slope'].fillna(value=df['slope'].mode()[0], inplace=True)
    df['thal'].fillna(value=df['thal'].mode()[0], inplace=True)"""


    X = df.drop(columns=["num"], axis=1)
    Y = df['num']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

    sns.pairplot(df, vars=numerical_features, hue="num")
    plt.show()

    x_train_num = x_train[numerical_features]
    x_test_num = x_test[numerical_features]

    scaler = StandardScaler()
    x_train_num_scaled = scaler.fit_transform(x_train_num)
    x_test_num_scaled = scaler.transform(x_test_num)

    x_train_cat = x_train[categorical_features]
    x_test_cat = x_test[categorical_features]

    encoder = OneHotEncoder(sparse_output=False, drop="first")
    x_train_cat_encoded = encoder.fit_transform(x_train_cat)
    x_test_cat_encoded = encoder.transform(x_test_cat)

    x_train_transformed = np.hstack((x_train_num_scaled, x_train_cat_encoded))
    x_test_transformed = np.hstack((x_test_num_scaled, x_test_cat_encoded))

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    knn = KNeighborsClassifier()

    voting_clf = VotingClassifier(
        estimators=[("rf", rf), ("knn", knn)],
        voting="soft"
    )
    voting_clf.fit(x_train_transformed, y_train)

    y_pred = voting_clf.predict(x_test_transformed)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    sns.heatmap(data=cm, cmap="inferno")
    plt.show()

    return 0

if __name__ == '__main__':
    main()