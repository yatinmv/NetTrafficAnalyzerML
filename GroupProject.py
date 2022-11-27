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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from matplotlib.colors import ListedColormap
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')



def clean_dataset(df):
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def plotConfusionMatrix(y_test,y_pred):
    fig = plt.figure(figsize=(10,5), dpi=100)
    ax1 = fig.add_subplot(111)
    cm = confusion_matrix(y_test, ypred)#, labels= target_names)
    sns.heatmap(cm, annot = True, cbar = False, fmt = "d", linewidths = .5, cmap = "Blues")
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted class")
    ax1.set_ylabel("Actual class")
    target_names = set(y)
    ax1.set_xticklabels(target_names)
    ax1.set_yticklabels(target_names)

    plt.show()

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

def printScores(y_test,y_pred):
    print('Accuracy score: %.2f%%' %(accuracy_score(y_test, y_pred)*100))  
    print('Precision score: %.2f%%' % (precision_score(y_test, y_pred, average= 'weighted')*100))
    print('Recall score: %.2f%%' % (recall_score(y_test, y_pred, average= 'weighted')*100))

def randomForestCrossValidation1(X_train,y_train):
    sns.set(rc={'figure.figsize':(9,6)})
    maxFeatures = [10,20,30,50,58]
    nEstimators = [10,20,50,75,100,150,200,250]
    for p in maxFeatures:
        mean_array = []
        std_array = []
        for c in nEstimators:
            clf = RandomForestClassifier(n_estimators = c, max_features = p, criterion = 'entropy', random_state = 42)
            scores = cross_val_score(clf, X_train, y_train, cv=5,scoring='f1_micro')
            mean_array.append(np.array(scores).mean())
            std_array.append(np.array(scores).std())

        plt.errorbar(nEstimators, mean_array, yerr=std_array, label="MaxFeatures = {0}".format(p))
    plt.xlabel("nEstimators Value")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs nEstimators for different values of MaxFeaures")

    plt.legend( loc='best',fontsize=15, bbox_to_anchor=(1.38, 1))
    plt.show()

def randomForestCrossValidation2(X_train,y_train):

    sns.set(rc={'figure.figsize':(9,6)})
    maxDepth = [1,2,5,10,15,20]
    nEstimators = [10,20,50,75,100,150,200,250]
    for p in maxDepth:
        mean_array = []
        std_array = []
        for c in nEstimators:
            clf = RandomForestClassifier(n_estimators = c, max_depth= p, criterion = 'entropy', random_state = 42)
            scores = cross_val_score(clf, X_train, y_train, cv=5,scoring='f1_micro')
            mean_array.append(np.array(scores).mean())
            std_array.append(np.array(scores).std())

        plt.errorbar(nEstimators, mean_array, yerr=std_array, label="maxDepth = {0}".format(p))
    plt.xlabel("nEstimators Value")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs nEstimators for different values of maxDepth")

    plt.legend( loc='best',fontsize=15, bbox_to_anchor=(1.38, 1))
    plt.show()


def randomForestModel(X_train,y_train,X_test,y_test):
    # Creating the Training and Test set from data

    classifier = RandomForestClassifier(n_estimators = 100, max_features = 50, criterion = 'entropy', random_state = 42,max_depth=5)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # getAccuracy(classifier,X,y)
    printScores(y_test,y_pred)



def main():
    df = pd.read_csv('Combined_VPN_NonVPN.csv')
    # df = clean_dataset(df)
    X = df.iloc[:,range(58)]
    y =df['Label'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)


    # Random Forest
    randomForestCrossValidation1(X_train,y_train)
    randomForestCrossValidation2(X_train,y_train)
    
    randomForestModel(X_train,y_train,X_test,y_test)   
    

    
if __name__ == "__main__":
    main()