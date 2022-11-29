import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from numpy import std
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from matplotlib.colors import ListedColormap
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


def clean_dataset(df):
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def remove_unwanted_columns(df):
    df.drop(
        [
            "Flow ID",
            "Src IP",
            "Src Port",
            "Dst IP",
            "Dst Port",
            "Protocol",
            "Timestamp",
            "Fwd Byts/b Avg",
            "Fwd Pkts/b Avg",
            "Fwd Blk Rate Avg",
            "Bwd Byts/b Avg",
            "FIN Flag Cnt",
            "SYN Flag Cnt",
            "RST Flag Cnt",
            "URG Flag Cnt",
            "CWE Flag Count",
            "ECE Flag Cnt",
            "Fwd PSH Flags",
            "Bwd PSH Flags",
            "Fwd URG Flags",
            "Bwd URG Flags",
        ],
        axis=1,
        inplace=True,
    )
    return df


def plotConfusionMatrix(y_test, ypred):
    fig = plt.figure(figsize=(10, 5), dpi=100)
    ax1 = fig.add_subplot(111)
    cm = confusion_matrix(y_test, ypred)  # , labels= target_names)
    sns.heatmap(cm, annot=True, cbar=False, fmt="d", linewidths=0.5, cmap="Blues")
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted class")
    ax1.set_ylabel("Actual class")
    target_names = set(ypred)
    ax1.set_xticklabels(target_names)
    ax1.set_yticklabels(target_names)

    plt.show()


def getAccuracy(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    # report the model performance
    print("Mean Accuracy: %.3f (%.3f)" % (mean(n_scores), std(n_scores)))


def logisticRegressionModel():
    # Code Here
    print("Logistic Regression Code")


def printScores(y_test, y_pred):
    print("Accuracy score: %.2f%%" % (accuracy_score(y_test, y_pred) * 100))
    print(
        "Precision score: %.2f%%"
        % (precision_score(y_test, y_pred, average="weighted") * 100)
    )
    print(
        "Recall score: %.2f%%"
        % (recall_score(y_test, y_pred, average="weighted") * 100)
    )

def printScores(y_test,y_pred):
    print('Accuracy score: %.2f%%' %(accuracy_score(y_test, y_pred)*100))
    print('Precision score: %.2f%%' % (precision_score(y_test, y_pred, average='weighted')*100))
    print('Recall score: %.2f%%' % (recall_score(y_test, y_pred, average='weighted')*100))
    print('F1 score: %.2f%%' % (f1_score(y_test, y_pred, average='weighted')*100))

def randomForestROCCurve(X,y):
    classes = ['Normal Traffic', 'VPN', 'Tor']
    y = label_binarize(y, classes=classes)
    n_classes = 3
    
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators = 100, max_features = 10, criterion = 'entropy', random_state = 42,max_depth=20))
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
    plt.figure()
    plt.rc('font', size=10)
    maxFeatures = [10,20,30,50,58]
    nEstimators = [10,20,50,75,100,150,200,250]
    for p in maxFeatures:
        mean_array = []
        std_array = []
        for c in nEstimators:
            clf = RandomForestClassifier(
                n_estimators=c, max_features=p, criterion="entropy", random_state=42
            )
            scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_micro")
            mean_array.append(np.array(scores).mean())
            std_array.append(np.array(scores).std())

        plt.errorbar(
            nEstimators, mean_array, yerr=std_array, label="MaxFeatures = {0}".format(p)
        )
    plt.xlabel("nEstimators Value")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs nEstimators for different values of MaxFeaures")

    plt.legend(loc="lower right")
    plt.show()


def randomForestCrossValidation2(X_train, y_train):
    plt.figure()
    plt.rc('font', size=10)
    maxDepth = [1, 2, 5, 10, 15, 20]
    nEstimators = [10, 20, 50, 75, 100, 150, 200, 250]
    for p in maxDepth:
        mean_array = []
        std_array = []
        for c in nEstimators:
            clf = RandomForestClassifier(
                n_estimators=c, max_depth=p, criterion="entropy", random_state=42
            )
            scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_micro")
            mean_array.append(np.array(scores).mean())
            std_array.append(np.array(scores).std())

        plt.errorbar(
            nEstimators, mean_array, yerr=std_array, label="maxDepth = {0}".format(p)
        )
    plt.xlabel("nEstimators Value")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs nEstimators for different values of maxDepth")

    plt.legend(loc="lower right")
    plt.show()


def randomForestModel(X_train, y_train, X_test, y_test):
    # Creating the Training and Test set from data

    classifier = RandomForestClassifier(n_estimators = 100, max_features = 10, criterion = 'entropy', random_state = 42,max_depth=20)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # getAccuracy(classifier,X,y)
    plotConfusionMatrix(y_test,y_pred)
    printScores(y_test, y_pred)


def knn_performance(X, y, ki_range):
    kf = KFold(n_splits=5)
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    mean_error_f1 = []
    std_error_f1 = []

    for ki in ki_range:
        model = KNeighborsClassifier(n_neighbors=ki, weights='uniform')
        model.fit(X, y)
        temp_f1 = []
        temp_accuracy = []
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            temp_f1.append(f1_score(y[test], ypred, average="micro"))
            temp_accuracy.append(accuracy_score(y[test], ypred))
        mean_error_f1.append(np.array(temp_f1).mean())
        std_error_f1.append(np.array(temp_f1).std())

    print("Mean of F1 Score : ", mean_error_f1)
    print("Standard Deviation of F1 Score : ", std_error_f1)
    plt.errorbar(
        ki_range, mean_error_f1, yerr=std_error_f1, linewidth=3, label="kNN Model"
    )
    plt.xlabel("k")
    plt.ylabel("Mean F1 Score")
    plt.title("Graph of F1 Score and k")
    plt.tight_layout()
    plt.show()


def knn_roc(X, y):
    y_roc = label_binarize(y, classes=["Normal Traffic", "VPN", "Tor"])

    n_classes = 3

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_roc, test_size=0.25, random_state=0
    )

    # classifier
    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3, weights='uniform'))
    y_score = clf.fit(X_train, y_train).predict(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        if i == 0:
            plt.plot(
                fpr[i],
                tpr[i],
                label="ROC curve for Normal Traffic (area = %0.2f)" % roc_auc[i],
            )
        if i == 1:
            plt.plot(
                fpr[i], tpr[i], label="ROC curve for VPN (area = %0.2f)" % roc_auc[i]
            )
        if i == 2:
            plt.plot(
                fpr[i], tpr[i], label="ROC curve for Tor (area = %0.2f)" % roc_auc[i]
            )

    plt.plot([0, 1], [0, 1], "k--", label="ROC Curve for Baseline Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize='xx-small')
    plt.title("ROC Curve for all Classes")
    plt.show()


def knn_model(X_train, X_test, y_train, y_test, X, y):

    knn = KNeighborsClassifier(n_neighbors=3, weights='uniform').fit(X_train, y_train)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    print("------------kNN Model-------------")
    y_pred = knn.predict(X_test)
    printScores(y_test, y_pred)
    print("---------Baseline Model-----------")
    y_pred_baseline = dummy.predict(X_test)
    printScores(y_test, y_pred_baseline)
    ki_range = [1, 3, 5, 7, 9]
    print("Set of Nearest Neighbors Considered", ki_range)
    knn_performance(X, y, ki_range)
    knn_roc(X, y)
    plotConfusionMatrix(y_test, knn.predict(X_test))


def main():
    df = pd.read_csv("Final_Dataset.csv")
    df = remove_unwanted_columns(df)
    # df = clean_dataset(df)
    X = df.iloc[:, range(62)]
    y = df["Label"]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = np.nan_to_num(X)
    df.drop(df[df['Flow Byts/s'] == 'Infinity'].index, inplace=True)
    df.drop(df[df['Flow Pkts/s'] == 'Infinity'].index, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=21
    )

    # kNN Model

    knn_model(X_train, X_test, y_train, y_test, X, y)

    # Random Forest
    # randomForestCrossValidation1(X_train,y_train)
    # randomForestCrossValidation2(X_train,y_train)
    
    randomForestModel(X_train,y_train,X_test,y_test)
    randomForestROCCurve(X,y)

    
if __name__ == "__main__":
    main()
