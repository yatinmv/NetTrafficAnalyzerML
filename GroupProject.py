import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from numpy import std
from numpy import mean
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix,f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def clean_dataset(df):
    #  Removing
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    df.dropna(inplace=True)
    return df

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
    # Printing Confusion Matrix
    print(cm)
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

def tag_output_values(lable_value):
    if (lable_value == "Normal Traffic"):
        return 1
    elif (lable_value == "Tor"):
        return 0
    else:
        return -1

def SVM_ModelSelection():
    data_set = pd.read_csv("Final_Dataset.csv")
    data_set = clean_dataset(data_set)
    data_set = remove_unwanted_columns(data_set)
    data_set.dropna(inplace=True)
    data_set['Out_put'] = data_set['Label'].apply(tag_output_values)
    X = data_set[['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
                  'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
                  'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
                  'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
                  'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
                  'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std',
                  'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean',
                  'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Header Len',
                  'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min',
                  'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
                  'PSH Flag Cnt', 'ACK Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
                  'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Bwd Pkts/b Avg',
                  'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts',
                  'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
                  'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
                  'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
                  'Idle Std', 'Idle Max', 'Idle Min']]
    y = data_set["Out_put"]
    degree_1 = PolynomialFeatures(degree=1, include_bias=False)
    degree_2 = PolynomialFeatures(degree=2, include_bias=False)
    X_std_1 = degree_1.fit_transform(X)
    X_std_2 = degree_2.fit_transform(X)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_std_1)
    X_scalar = scaler.fit_transform(X_std_2)
    X_train_std, X_test, y_train, y_test = train_test_split(X_std, y, random_state=0, train_size=0.8)
    X_sc_train, x_2_test, y_train_2, y_test_2 = train_test_split(X_scalar, y, random_state=0, train_size=0.8)
    param = {'C': [0.01, 0.1, 1, 10, 100]}
    lr_model = LinearSVC(multi_class="ovr")
    gs_model_1 = GridSearchCV(estimator=lr_model, param_grid=param,n_jobs=-1,scoring="f1_macro",return_train_score=True,cv=5)
    gs_model_1.fit(X_train_std, y_train)
    gs_model_2 = GridSearchCV(estimator=lr_model, param_grid=param, n_jobs=-1, scoring="f1_macro", return_train_score=True,
                          cv=5)
    gs_model_2.fit(X_sc_train, y_train_2)
    print("best_parameter for degree 1")
    print(gs_model_1.best_params_)
    print("best_parameter for degree 2")
    print(gs_model_2.best_params_)
    # Train a LR model with best parameters
    model_1 = LinearSVC(**gs_model_1.best_params_,verbose = 1,multi_class="ovr")
    model_2 = LinearSVC(**gs_model_2.best_params_, verbose=1, multi_class="ovr")
    model_1.fit(X_train_std, y_train)
    model_2.fit(X_sc_train, y_train_2)
    y_score = model_1.predict(X_test)
    res=pd.DataFrame(gs_model_1.cv_results_)
    res_2 = pd.DataFrame(gs_model_2.cv_results_)
    y_predict_train_1=model_1.predict(X_train_std)
    y_predict_train_2 = model_2.predict(X_sc_train)
    y_test_predict_1=model_1.predict(X_test)
    y_test_predict_2 = model_2.predict(x_2_test)
    plt.errorbar(
            res['param_C'], res['mean_train_score'], yerr=res['std_train_score'], label="C values with degree 1"
        )
    plt.errorbar(
    res_2['param_C'], res_2['mean_train_score'], yerr=res_2['std_train_score'], label="C values with degree 2"
    )
    plt.xlabel("C")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Different C Values ")
    plt.legend(loc="lower right")
    plt.show()
    print("TRAIN Scores For degree1")
    printScores(y_train, y_predict_train_1)
    print("TEST SCORE FOR DEGREE 1")
    printScores(y_test, y_test_predict_1)
    print("TRAIN Scores For degree2")
    printScores(y_train_2, y_predict_train_2)
    print("TEST SCORE FOR DEGREE 2")
    printScores(y_test_2, y_test_predict_2)



def svm_scores(df):
    # Constructing SVM Model with c=1 and degree =2
    print("SVM Regression Model")
    df['Out_put'] = df['Label'].apply(tag_output_values)
    y = df['Label']
    necessary_col = list(df.columns)
    necessary_col.remove("Out_put")
    necessary_col.remove("Label")
    X = df[necessary_col]
    degree = PolynomialFeatures(degree=2, include_bias=False)
    x_degree_2 = degree.fit_transform(X)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(x_degree_2)
    X_train_std, X_test, y_train, y_test = train_test_split(X_std, y, random_state=0, test_size=0.25)
    model = LinearSVC(C=1, max_iter=10000)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train_std)
    print("Scores on Test Data")
    printScores(y_test, y_pred)
    plotConfusionMatrix(y_test, y_pred)
    print("Scores on Train Data")
    printScores(y_train, y_pred_train)
    plotConfusionMatrix(y_train, y_pred_train)
    print("SVM Regression Code Done")
def SVM_ROC_Curve(df):
    classes = ['Normal Traffic', 'VPN', 'Tor']
    n_classes = 3
    # shuffle and split training and test sets
    df['Out_put'] = df['Label'].apply(tag_output_values)
    y = df['Label']
    y = label_binarize(y, classes=classes)
    necessary_col = list(df.columns)
    necessary_col.remove("Out_put")
    necessary_col.remove("Label")
    X = df[necessary_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.80)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    degree = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_features = degree.fit_transform(X_train)
    X_test_features = degree.fit_transform(X_test)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(LinearSVC(C=1, max_iter=10000))
    y_score = classifier.fit(X_poly_features, y_train).predict(X_test_features)

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
    y_pred_train = classifier.predict(X_train)
    # getAccuracy(classifier,X,y)
    plotConfusionMatrix(y_test,y_pred)
    print("TEST Scores")
    printScores(y_test, y_pred)
    print("Train Scores")
    printScores(y_train, y_pred_train)
    plotConfusionMatrix(y_train, y_pred_train)


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
    print(" test scores")
    printScores(y_test, y_pred)
    print("Train")
    y_pred_train = knn.predict(X_train)
    printScores(y_train, y_pred_train)
    print("---------Baseline Model-----------")
    y_pred_baseline = dummy.predict(X_test)
    printScores(y_test, y_pred_baseline)
    ki_range = [1, 3, 5, 7, 9]
    print("Set of Nearest Neighbors Considered", ki_range)
    knn_performance(X, y, ki_range)
    knn_roc(X, y)
    plotConfusionMatrix(y_test, knn.predict(X_test))
    plotConfusionMatrix(y_train, knn.predict(X_train))

def baseLineModel(df):
    # Constructing Base Line Predictor Model
    print("Base Line Predictor ")
    df['Out_put'] = df['Label'].apply(tag_output_values)
    y = df['Out_put']
    necessary_col = list(df.columns)
    necessary_col.remove("Out_put")
    necessary_col.remove("Label")
    X = df[necessary_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)
    dummy=DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    y_pred_baseline = dummy.predict(X_test)
    y_pred_baseline_train = dummy.predict(X_train)
    print("Base Line Predictor Score")
    printScores(y_test, y_pred_baseline)
    # print("BaseLine Confusion Matrix test")
    # plotConfusionMatrix(y_test, y_pred_baseline)
    # print("Basline confusion Matrix train ")
    # plotConfusionMatrix(y_train,y_pred_baseline_train)

def combinedROCCurve(df):
    classes = ['Normal Traffic', 'VPN', 'Tor']
    n_classes = 3
    # shuffle and split training and test sets
    df['Out_put'] = df['Label'].apply(tag_output_values)
    y = df['Label']
    y = label_binarize(y, classes=classes)
    necessary_col = list(df.columns)
    necessary_col.remove("Out_put")
    necessary_col.remove("Label")
    X = df[necessary_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)
    degree = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_features = degree.fit_transform(X_train)
    X_test_features = degree.fit_transform(X_test)
    scaler = StandardScaler()
    X_poly_features = scaler.fit_transform(X_poly_features)
    X_test_features=scaler.fit_transform(X_test_features)

    # Learn to predict each class against the other
    classifier_lin = OneVsRestClassifier(LinearSVC(C=1, verbose=1, max_iter=10000, multi_class="ovr"))
    y_score_lin = classifier_lin.fit(X_poly_features, y_train).predict(X_test_features)
    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3, weights='uniform'))
    y_score_knn = clf.fit(X_train, y_train).predict(X_test)
    classifier_for = OneVsRestClassifier(
        RandomForestClassifier(n_estimators=100, max_features=10, criterion='entropy', random_state=42, max_depth=20))
    y_score_for = classifier_for.fit(X_train, y_train).predict(X_test)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    y_pred_baseline = dummy.predict(X_test)
    y_pred_baseline_train = dummy.predict(X_train)

    # Compute ROC curve and ROC area for each class
    fpr_svm = dict()
    tpr_svm = dict()
    roc_auc_svm = dict()
    fpr_knn = dict()
    tpr_knn = dict()
    roc_auc_knn = dict()
    fpr_for = dict()
    tpr_for = dict()
    roc_auc_for = dict()
    fpr_base = dict()
    tpr_base= dict()
    roc_auc_base = dict()
    for i in range(n_classes):
        fpr_svm[i], tpr_svm[i], _ = roc_curve(y_test[:, i], y_score_lin[:, i])
        roc_auc_svm[i] = auc(fpr_svm[i], tpr_svm[i])
        fpr_knn[i], tpr_knn[i], _ = roc_curve(y_test[:, i], y_score_knn[:, i])
        roc_auc_knn[i] = auc(fpr_knn[i], tpr_knn[i])
        fpr_for[i], tpr_for[i], _ = roc_curve(y_test[:, i], y_score_for[:, i])
        roc_auc_for[i] = auc(fpr_for[i], tpr_for[i])
        fpr_base[i], tpr_base[i], _ = roc_curve(y_test[:, i], y_pred_baseline[:, i])
        roc_auc_base[i] = auc(fpr_base[i], tpr_base[i])


    # Compute micro-average ROC curve and ROC area
    fpr_svm["micro"], tpr_svm["micro"], _ = roc_curve(y_test.ravel(), y_score_lin.ravel())
    roc_auc_svm["micro"] = auc(fpr_svm["micro"], tpr_svm["micro"])
    fpr_knn["micro"], tpr_knn["micro"], _ = roc_curve(y_test.ravel(), y_score_knn.ravel())
    roc_auc_knn["micro"] = auc(fpr_knn["micro"], tpr_knn["micro"])
    fpr_for["micro"], tpr_for["micro"], _ = roc_curve(y_test.ravel(), y_score_for.ravel())
    roc_auc_for["micro"] = auc(fpr_for["micro"], tpr_for["micro"])
    fpr_base["micro"], tpr_base["micro"], _ = roc_curve(y_test.ravel(), y_pred_baseline.ravel())
    roc_auc_base["micro"] = auc(fpr_base["micro"], tpr_base["micro"])

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_svm["micro"], tpr_svm["micro"],
             label='SVM micro-average (area = {0:0.2f})'
                   ''.format(roc_auc_svm["micro"]))
    plt.plot(fpr_knn["micro"], tpr_knn["micro"],
             label='KNN micro-average  (area = {0:0.2f})'
                   ''.format(roc_auc_knn["micro"]))
    plt.plot(fpr_for["micro"], tpr_for["micro"],
             label='Random Forest micro-average (area = {0:0.2f})'
                   ''.format(roc_auc_for["micro"]))
    plt.plot(fpr_base["micro"], tpr_base["micro"],'k--',
             label='BaseLine micro-average (area = {0:0.2f})'
                   ''.format(roc_auc_base["micro"]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(' ROC Curves ')
    plt.legend(loc="lower right")
    plt.show()

def main():
    df = pd.read_csv("Final_Dataset.csv")
    df = remove_unwanted_columns(df)
    X = df.iloc[:, range(62)]
    y = df["Label"]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = np.nan_to_num(X)
    df.drop(df[df['Flow Byts/s'] == 'Infinity'].index, inplace=True)
    df.drop(df[df['Flow Pkts/s'] == 'Infinity'].index, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=21
    )

    #  Base Line Model details
    baseLineModel(df)
    # SVM Model Hyperparameter Selection
    SVM_ModelSelection()
    # SVM Model Training
    svm_model()

    # kNN Model
    knn_model(X_train, X_test, y_train, y_test, X, y)

    #Random Forest
    # randomForestCrossValidation1(X_train,y_train)
    # randomForestCrossValidation2(X_train,y_train)

    randomForestModel(X_train,y_train,X_test,y_test)
    randomForestROCCurve(X,y)

def svm_model():
    df = pd.read_csv("Final_Dataset.csv")
    df = remove_unwanted_columns(df)
    df = clean_dataset(df)
    svm_scores(df)
    SVM_ROC_Curve(df)


    
if __name__ == "__main__":
    main()
