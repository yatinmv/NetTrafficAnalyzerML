import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from numpy import std
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix



def clean_dataset(df):
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def getAccuracy(model,X,y):

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report the model performance
    print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


def logisticRegressionModel():
    # Code Here
    print("Logistic Regression Code")


def kNNModel():
    # Code Here
    print("KNN Code")

def randomForestModel(X,y):
    # Code Here
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
    classifier.fit(X, y)
    # Predicting the Test set results
    ypred = classifier.predict(X)
    getAccuracy(classifier,X,y)

def main():
    df = pd.read_csv('Combined_VPN_NonVPN.csv')
    # df = clean_dataset(df)
    X = df.iloc[:,range(58)]
    y =df['Label'] 
    randomForestModel(X,y)   

    
if __name__ == "__main__":
    main()