
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV

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
    rf_params = {
        "n_estimators" : [100, 200, 300],
        "max_depth" : [None, 10, 20, 30],
        "min_samples_split" : [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4]
    }
    gcv = GridSearchCV(estimator=rf, param_grid=rf_params, cv=5, scoring="accuracy", n_jobs=-1)
    gcv.fit(x_train_transformed, y_train)
    best_rf = gcv.best_estimator_
    print("Best Params: ", gcv.best_params_)
    knn = KNeighborsClassifier()
    rfe = RFE(estimator=rf, n_features_to_select=5)
    rfe.fit(X=x_train_transformed, y=y_train)

    voting_clf = VotingClassifier(
        estimators=[("rf", best_rf), ("knn", knn), ("rfe", rfe)],
        voting="soft"
    )
    voting_clf.fit(x_train_transformed, y_train)

    y_pred = voting_clf.predict(x_test_transformed)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    sns.heatmap(data=cm, cmap="Blues", annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    return 0

if __name__ == '__main__':
    main()