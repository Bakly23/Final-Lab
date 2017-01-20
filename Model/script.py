import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import itertools
import warnings
warnings.filterwarnings('ignore')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "%2.1f" % (cm[i, j] * 100) + "%",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.grid(False)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title + ".png", bbox_inches='tight')
    print("Confusion Matrix is saved  to {}".format(title + ".png"))


def measure_classifier_test_train(classifier, classifier_name, data, target, target_names):
    train_data, test_data, train_labels, test_labels = train_test_split(data, target, test_size=0.3, stratify = target,
                                                                        random_state=42)
    classifier.fit(train_data, train_labels)
    print("Classification report(train) for the {}".format(classifier_name))
    print(classification_report(train_labels, classifier.predict(train_data), labels=target_names,
                                target_names=target_names))
    pred_labels=classifier.predict(test_data)
    print("Classification report(test) for the {}".format(classifier_name))
    print(classification_report(test_labels, pred_labels, labels=target_names,
                                target_names=target_names))
    cv_score = cross_val_score(classifier, data, target, scoring='f1_weighted', cv=10).mean()
    print("Cross validated weighted f1 score for the {} is {:1.3f}".format(classifier_name, cv_score))
    cnf_matrix = confusion_matrix(test_labels, pred_labels, labels=target_names)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix for ' + classifier_name,
                          normalize=True, cmap=plt.cm.Blues)
    return cv_score


def train(data, labels):
    estimator = RandomForestClassifier(max_features='sqrt', max_depth=10, n_estimators=100, min_samples_leaf=1,
                                       min_samples_split=2)
    measure_classifier_test_train(estimator, 'pre set random forest classifier', data, labels,
                                  ['bad', 'ok', 'fine', 'good'])
    return estimator


def resetandtrain(data, labels):
    parameters_grid = {
        'max_features': ['sqrt', 'log2'],
        'n_estimators': [10, 50, 100, 200, 300],
        'max_depth': [2, 5, 10, 15, 20],
        'min_samples_split': [2, 3, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10]
    }

    cv = StratifiedShuffleSplit(labels, n_iter=3, test_size=0.3, random_state=0)
    grid_cv = GridSearchCV(RandomForestClassifier(), parameters_grid, scoring='f1_weighted', cv=cv)
    grid_cv.fit(data, labels)
    best_estimator = grid_cv.best_estimator_
    measure_classifier_test_train(best_estimator, 'reset random forest classifier', data, labels,
                                  ['bad', 'ok', 'fine', 'good'])
    return best_estimator


def predictandsave(estimator, data, path):
    labels = estimator.predict(StandardScaler().fit_transform(data))
    pd.DataFrame(labels).to_csv(path)


def launch(data_path, train_data_path, resetup, result_path):
    red_wine_df = pd.read_csv(train_data_path, header=0, sep=',')
    red_wine_df_features = StandardScaler().fit_transform(red_wine_df.drop(['quality'], axis=1))
    class_mapping = {1: 'bad', 2: 'bad', 3: 'bad', 4: 'bad', 5: 'ok', 6: 'fine', 7: 'good', 8: 'good', 9: 'good',
                     10: 'good'}
    new_labels = red_wine_df['quality'].map(lambda x: class_mapping[x])
    data_to_learn = pd.read_csv(data_path, header=0, sep=',')
    if resetup:
        estimator = resetandtrain(red_wine_df_features, new_labels)
    else:
        estimator = train(red_wine_df_features, new_labels)
    predictandsave(estimator, data_to_learn, result_path)


def main(argv):
    train_data_path = argv[2] if len(argv) > 2 else 'winequality-red-train.csv'
    resetup = argv[3] if len(argv) > 3  else False
    result_path = argv[4] if len(argv) > 4 else 'result.csv'
    launch(argv[1], train_data_path, resetup, result_path)

if __name__ == '__main__':
    main(sys.argv)
