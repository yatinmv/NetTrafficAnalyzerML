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
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
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
    print('F1 score: %.2f%%' % (f1_score(y_test, y_pred, average= 'weighted')*100))

def randomForestROCCurve(X,y):
    classes = ['VPN','No VPN','Tor']
    y = label_binarize(y, classes=classes)
    n_classes = 3
    

    # Add noisy features to make the problem harder
    # random_state = np.random.RandomState(0)
    # n_samples, n_features = X.shape
    # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators = 150, max_features = 'sqrt', criterion = 'entropy', random_state = 42,max_depth=5))
    y_score = classifier.fit(X_train, y_train).predict(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                    ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def randomForestCrossValidation1(X_train,y_train):
    # sns.set(rc={'figure.figsize':(9,6)})
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

    # sns.set(rc={'figure.figsize':(9,6)})
    plt.figure()
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])

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
    plt.legend( loc='lower right')
    plt.show()


def randomForestModel(X_train,y_train,X_test,y_test):
    # Creating the Training and Test set from data

    classifier = RandomForestClassifier(n_estimators = 150, max_features = 'sqrt', criterion = 'entropy', random_state = 42,max_depth=5)
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
    # randomForestCrossValidation1(X_train,y_train)
    randomForestCrossValidation2(X_train,y_train)
    
    randomForestModel(X_train,y_train,X_test,y_test)   
    randomForestROCCurve(X,y)
    

    
if __name__ == "__main__":
    main()