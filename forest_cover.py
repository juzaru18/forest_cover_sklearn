# Author : Juan Zamorano Ruiz
# Copyright: MIT License
# e-mail: juzaru18@gmail.com


# General import for python and sklearn
import pandas as pd
import seaborn as sns
import numpy as np
import time as t
import matplotlib.pyplot as plt
import itertools

import os
import warnings


from matplotlib.colors import ListedColormap
from time import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, accuracy_score, log_loss, recall_score, precision_score, confusion_matrix, classification_report

# Import classifier to test


from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier


def preprocess(dataset):
    """
    Given train dataset from Forest Cover Type dataset, it returns
    two dataframes, one with just labels and one with train dataset with all
    atritubes except CoverType atribute
    """

    labels = dataset.Cover_Type.values

    dataset= dataset.drop(['Cover_Type'], axis=1)

    return dataset, labels


def standardize (train):
    """
    given a forest cover type dataset, this function transform the data
    to StandardScaler [-1,1] scale
    Returns just first 10 atributes on tha scale concatenate with the other
    atributes
    """

    train1 = train

    Size = 10

    X_temp1 = train1.iloc[:, :Size]

    # Transform to StandardScaler
    X_temp1 = StandardScaler().fit_transform(X_temp1)

    r, c = train.shape
    X_train1 = np.concatenate((X_temp1, train1.iloc[:, Size:c - 1]), axis=1)  

    return X_train1


#########################################################################################
# Reference:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.PuRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = ['Spruce/Fir', 'Lodgepole', 'Ponderosa', 
               'Cotton/Willow', 'Aspen', 'Douglas-fir', 'Krummholz',]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("matriz de confusion normalizada")
    else:
        print('matriz de confusion sin normalizar')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

###########################################################################################################

def compare_classifiers(classifiers, X_train, y_train, X_test, y_test, conf_matrix=False):
    """Given a list o classifiers and datasets, it returns a dataframe with values of fit for each classifier, if
    conf_matrix is set to True then it will plot a figure with a confusion matrix for each classifier in the list"""

    log_cols = ["Classifier", "Accuracy", "Precision Score", 
                "Recall Score", "F1 Score","Log Loss", "Time to fit"]
    log = pd.DataFrame(columns=log_cols)
    
    for clf in classifiers:

        print("=" * 30)
        print("Fit for classifier --> ", clf.__class__.__name__)

        start = time()

        clf.fit(X_train, y_train)


        time_spent = (time() - start)
        print(" took %.2f seconds" % (time() - start))

        print("=" * 30)
        print(clf.__class__.__name__)

        print("***** Results *****")
        time_init_pred = time()
        y_pred = clf.predict(X_test)
        time_predict = time() - time_init_pred
        print(" took %.2f seconds to predict on test dataset" % time_predict)
        
        acc = accuracy_score(y_test, y_pred)

        ############# Precision #########################
        # The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives
        # The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
        precision = precision_score(y_test, y_pred, average='weighted')

        ############# Recall Score ######################
        # The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives
        # The recall is intuitively the ability of the classifier to find all the positive samples
        recall = recall_score(y_test, y_pred, average='weighted')

        f1 = f1_score(y_test, y_pred, average ='weighted')
        print("Accuracy of the classifier: {:.3%}".format(acc))
        print("Precision score of the classifier: {:.3%}".format(precision))
        print("F1 Score of the classifier: {:.3%}".format(f1))
        
        class_report = classification_report(y_test, y_pred)
        print ("Classification report: ")
        #print(class_report)
        if hasattr(clf, "predict_proba"):
            train_predictions = clf.predict_proba(X_test)
        else:
            conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
            train_predictions = clf.decision_function(X_test)
        
        #plot confusion matrix if variable conf_matrix is set to True
        if conf_matrix:
            cnf_matrix = confusion_matrix(y_test, y_pred)
            np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, title='matriz de confusion sin normalizacion')

            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, normalize=True,
                                  title='matriz de confusion con normalizacion')
            plt.show()


        ll = log_loss(y_test, train_predictions)
        print("Compute recall of the classifier: {:.3%}".format(recall))
        print("Log Loss: {}".format(ll))
        entry = pd.DataFrame([[clf.__class__.__name__, acc * 100,
                               precision * 100, recall*100, f1 * 100, ll, time_spent]], columns=log_cols)
        log = log.append(entry)
        print("=" * 30)
    return log



#########################################Show graph values #############################################################

def show_graphs(table):
    """
    Given a panda dataframe with different values, show different graphs for comparasion of different classifiers
    accuracy, log loss, Time spent, Precision Score, Recall Score, time to predict
    :param table: a panda dataframe with some values of some classifiers
    :return: some seaborn plots with graphical info
    Uncomment to show the plots
    """
    values_sorted = table.sort_values(["Exactitud(%)", "Recuperacion(%)"], ascending=False)
    time_sorted = table.sort_values(["Tiempo de entrenamiento (en segundos)"])
    log_sorted = table.sort_values(["Log Loss"])
    precision_sorted = table.sort_values(["Precision(%)"], ascending=False)
    F1_sorted = table.sort_values(["Valor F(%)"], ascending=False)
   
    
    


    sns.set(style="white", context="talk")

    #sns.set_palette("Reds",10)
    sns.set()

    g = sns.barplot(x='Clasificador', y='Exactitud(%)', data=values_sorted, label="Exactitud",palette="coolwarm")
    plt.xticks(rotation=60)
    plt.title("Mejor cuanto mas cerca de 100")
    plt.show(g)

    g = sns.barplot(x='Clasificador', y='Recuperacion(%)', data=values_sorted, label="Recuperacion", palette="coolwarm")
    plt.xticks(rotation=60)
    plt.title("Mejor cuanto mas cerca de 100, %")
    plt.show(g)

    g = sns.barplot(x='Clasificador', y='Log Loss', data=log_sorted, label="Log Loss",palette="coolwarm")
    plt.xticks(rotation=60)
    plt.title("Mejor cuanto mas cercano a 0")
    plt.show(g)

    g = sns.barplot(x='Clasificador', y='Precision(%)', data=precision_sorted, label="Precision", palette="coolwarm")
    plt.xticks(rotation=60)
    plt.title("Mejor cuanto mas cerca de 100")
    plt.show(g)

    g = sns.barplot(x='Clasificador', y='Valor F(%)', data=F1_sorted, label="Valor F", palette="coolwarm")
    plt.xticks(rotation=60)
    plt.title("Mejor cuanto mas cerca de 100")
    plt.show(g)

    g = sns.barplot(x='Clasificador', y='Tiempo de entrenamiento (en segundos)', data=time_sorted, label="Tiempo de ajuste",palette="coolwarm")
    plt.xticks(rotation=60)
    plt.title("Mejor cuanto menor")
    plt.show(g) 
    
    
    sns.despine(bottom=True)



"""List of classifiers and their hyperparameters after some tunning using GridSearch. This list of classifiers
will be used to fit with X_train and y_train datasets and check which one is the best fitting for classifying"""

list_multilayerperc = [MLPClassifier(warm_start=False, shuffle=True, nesterovs_momentum=True, hidden_layer_sizes=(1024, 512, 256, 128),
                       validation_fraction=0.333,
                       solver = 'adam',
                       learning_rate='constant', max_iter=162, batch_size=200, random_state=1,
                       momentum=0.11593,
                       tol=0.081977,
                       alpha=0.01, activation='relu', early_stopping=False)]

list_perceptron = [Perceptron(alpha=0.001, fit_intercept= True, penalty='l1', random_state= 10,
                                shuffle= True)]
        
list_linear_model = [LogisticRegression(warm_start=True, fit_intercept=True, solver='newton-cg',
                                        multi_class='multinomial', random_state=1, max_iter=500, n_jobs=2)]
        
list_neighbors = [KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', p=1)]

list_trees = [DecisionTreeClassifier(splitter="random", max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, criterion='entropy', random_state=5,
                                     max_features=None,
                                     max_depth=None)]
list_ensemble= [ExtraTreesClassifier(bootstrap=False, criterion='entropy', max_depth=None, max_leaf_nodes=None,
                                    min_samples_leaf=1,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=5, warm_start=False)
]

list_total_classifiers = [list_perceptron,
                          list_multilayerperc, 
                          list_neighbors, 
                          list_trees, 
                          list_ensemble]


if __name__ == "__main__":

    init = time()

    # Import datasets for training and test
    start = time()

    print("Reading dataset")
    
    train = pd.read_csv('trees.csv')
    
    #Uncomment next 2 lines to read metric values from CSV file and show them in plots
    #graph = pd.read_csv('example.csv')
    #show_graphs(graph)
    

    print("Read took %.2f seconds for %d samples."
          % (time() - start, len(train)))
    print("+" * 30)


    start = time()
    

    X, y = preprocess(train)
    X1 = standardize(X)

    sss = StratifiedShuffleSplit(10, test_size=0.3, random_state=10)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    #iterate over each classifier
    for n in range (0, len(list_total_classifiers)):
        log = compare_classifiers(list_total_classifiers[n], X_train, y_train, X_test, y_test,conf_matrix=False)
        #Uncomment next line if you want to save the results stored in log variable to a csv file
    #    log.to_csv('example.csv', mode='a', sep=',')
    
    #Cross validation for X values (first 10 atributes) after standardization
    for train_index, test_index in sss.split(X1, y):
        X_train, X_test = X1[train_index], X1[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    for n in range (0, len(list_total_classifiers)):
        log = compare_classifiers(list_total_classifiers[n], X_train, y_train, X_test, y_test,conf_matrix=False)
        #Uncomment next line if you want to save the results stored in log variable to a csv file
        # add header=False if we do not want the headers
    #    log.to_csv('example.csv', mode='a', sep=',')
    
    end = time()
    print("script duration: %r seconds" % (end - start))
