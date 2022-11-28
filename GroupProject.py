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
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
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
            "Flow Byts/s",
            "Flow Pkts/s",
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


def randomForestCrossValidation1(X_train, y_train):
    sns.set(rc={"figure.figsize": (9, 6)})
    maxFeatures = [10, 20, 30, 50, 58]
    nEstimators = [10, 20, 50, 75, 100, 150, 200, 250]
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

    plt.legend(loc="best", fontsize=15, bbox_to_anchor=(1.38, 1))
    plt.show()


def randomForestCrossValidation2(X_train, y_train):
    sns.set(rc={"figure.figsize": (9, 6)})
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

    plt.legend(loc="best", fontsize=15, bbox_to_anchor=(1.38, 1))
    plt.show()


def randomForestModel(X_train, y_train, X_test, y_test):
    # Creating the Training and Test set from data

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_features=50,
        criterion="entropy",
        random_state=42,
        max_depth=5,
    )
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # getAccuracy(classifier,X,y)
    printScores(y_test, y_pred)


def knn_performance(X, y, ki_range):
    kf = KFold(n_splits=10)
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    mean_error_f1 = []
    std_error_f1 = []

    for ki in ki_range:
        model = KNeighborsClassifier(n_neighbors=ki)
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
    plt.title("Error Bar Graph of F1 Score and k")
    plt.legend(bbox_to_anchor=(0, 1), loc="upper left", ncol=1)
    plt.tight_layout()
    plt.show()


def knn_roc(X, y):
    y_roc = label_binarize(y, classes=["Normal Traffic", "VPN", "Tor"])

    n_classes = 3

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_roc, test_size=0.33, random_state=0
    )

    # classifier
    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))
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

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title("ROC Curve for all Classes")
    plt.show()


def knn_model(X_train, X_test, y_train, y_test, X, y):

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = np.nan_to_num(X)

    knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    print("------------kNN Model-------------")
    y_pred = knn.predict(X_test)
    printScores(y_test, y_pred)
    print("---------Baseline Model-----------")
    y_pred_baseline = dummy.predict(X_test)
    printScores(y_test, y_pred_baseline)
    ki_range = [3, 5, 7]
    print("Set of Nearest Neighbours Considered", ki_range)
    knn_performance(X, y, ki_range)
    knn_roc(X, y)
    plotConfusionMatrix(y, knn.predict(X))


def main():
    df = pd.read_csv("Final_Dataset.csv")
    df = remove_unwanted_columns(df)
    # df = clean_dataset(df)
    X = df.iloc[:, range(60)]
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=21
    )

    # kNN Model

    knn_model(X_train, X_test, y_train, y_test, X, y)

    # Random Forest
    randomForestCrossValidation1(X_train, y_train)
    randomForestCrossValidation2(X_train, y_train)
    randomForestModel(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
