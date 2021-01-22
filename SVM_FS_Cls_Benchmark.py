import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from collections import Counter
from sklearn.utils import safe_sqr

class ClassifierBenchmark:
    def __init__(self):
        # Load the data
        self.data = pd.read_csv("/Normalized_Mock.txt", sep="\t")

    # split the data into training and testing
    def splitData(self, x):
        #split data into test and training
        self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=x, stratify=self.data[['Cluster']])
        self.X, self.y = self.train.iloc[:, :-1], self.train['Cluster']
        self.featureNames = self.X.columns.tolist()
        self.X = self.X.to_numpy()
        self.y = self.y.values

        self.X_val, self.y_val = self.test.iloc[:, :-1], self.test['Cluster']
        self.X_val = self.X_val.to_numpy()
        self.y_val = self.y_val.values

        return self.X, self.y, self.X_val, self.y_val, self.featureNames
	
    # Run SVM-RFE and get the 200 most important features
    def svmFC(self, x, step):
        self.step = step
        self.X, self.y, self.X_val, self.y_val, self.featureNames = self.splitData(x = x)
        self.features = self.featureNames

        self.j = 0
        while self.X.shape[1] > 201:
            self.j += 1
            self.svc = SVC(kernel='linear')
            self.Cs = np.array([0.5, 1.0, 10, 100])

            #  get the hyperparamaters
            self.clf = GridSearchCV(estimator=self.svc,
                               param_grid=dict(C=self.Cs),
                               cv=5,
                               return_train_score=True,
                               n_jobs=20)
            self.clf.fit(self.X, self.y)

            # do 5-fold cross validation
            self.cv_test_error = []
            self.skf = StratifiedKFold(n_splits=5, random_state=self.j, shuffle=True)
            for trn, tst in self.skf.split(self.X, self.y):
                self.train_train, self.train_test = self.X[trn], self.X[tst]
                self.train_clstrs, self.test_clstrs = self.y[trn], self.y[tst]
                self.val_clf = SVC(C=list(self.clf.best_params_.values())[0], kernel="linear")
                self.val_clf.fit(self.train_train, self.train_clstrs)
                self.cv_test_error.append(self.val_clf.score(self.train_test, self.test_clstrs))
            self.mean_cv_test_error = np.array(self.cv_test_error).mean()

            ## train classification for RFE

            self.rfe_clf = SVC(C=list(self.clf.best_params_.values())[0], kernel="linear")
            self.rfe_clf.fit(self.X, self.y)

            # get coeffs
            self.coefs = self.rfe_clf.coef_

            # get ranks
            if self.coefs.ndim > 1:
                self.ranks = np.argsort(safe_sqr(self.coefs).sum(axis=0))
            else:
                self.ranks = np.argsort(safe_sqr(self.coefs))

            # remove the X least important features from the array
            self.to_remove_index = []

            for r in range(self.step):
                self.to_remove_index.append(self.ranks[r])
            self.to_remove_index.sort(reverse=True)

            # remove from largest index to smallest
            for f in self.to_remove_index:
                self.X = np.delete(self.X, f, axis=1)
                self.X_val = np.delete(self.X_val, f, axis=1)
                del self.features[f]

        return self.X, self.y, self.X_val, self.y_val, self.features


    def svmFSOutput(self, x):
        self.train_X, self.y, self.val_X, self.y_val, self.proteins_feature = self.svmFC(x = x, step=1)
        return self.train_X, self.y, self.val_X, self.y_val, self.proteins_feature

    # random forest
    def randomForest(self, x):
        self.X, self.y, self.X_val, self.y_val, self.proteins_feature = self.svmFSOutput(x = x)
        self.rfc = RandomForestClassifier(random_state=42)
        self.param_grid = {
            'n_estimators': [250, 500, 750],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 6, 7, 8],
            'criterion': ['gini', 'entropy']
        }

        self.CV_rfcFeature = GridSearchCV(estimator=self.rfc, param_grid=self.param_grid, n_jobs=20)
        self.CV_rfcFeature.fit(self.X, self.y)
        #self.CV_rfcFeature.score(self.X_val, self.y_val)

        return (self.CV_rfcFeature.best_score_, self.CV_rfcFeature.score(self.X_val, self.y_val))

    # xgboost
    def xgboost(self, x):
        self.X, self.y, self.X_val, self.y_val, self.proteins_feature = self.svmFSOutput(x = x)
        self.xgboostModel = XGBClassifier(objective="multi:softprob", random_state=42)
        #self.kfold = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

        self.n_estimators = range(50, 300, 50)  # tune number of decision trees
        self.max_depth = range(1, 5, 2)  # size of decision trees
        self.learning_rate = [0.05, 0.1, 0.2]  # learning rate

        self.param_grid = dict(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth)
        self.kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

        self.grid_search_xgboost = GridSearchCV(self.xgboostModel, self.param_grid, scoring="accuracy", cv=self.kfold, n_jobs=20)
        self.grid_search_xgboost = self.grid_search_xgboost.fit(self.X, self.y)

        #print(self.grid_search_xgboost.score(self.X_val, self.y_val))

        return (self.grid_search_xgboost.best_score_, self.grid_search_xgboost.score(self.X_val, self.y_val))

    # Extremely Randomized Trees
    def extreeTree(self, x):
        self.X, self.y, self.X_val, self.y_val, self.proteins_feature = self.svmFSOutput(x = x)
        self.etreeclassifier = ExtraTreesClassifier(random_state=0)
        self.kfold = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

        self.param_grid = {
            'n_estimators': [100, 250, 500, 750],
            'max_features': ['auto'],
            'max_depth': [4, 5, 6, 7],
            'criterion': ['gini']
        }

        self.CV_exTeeCls = GridSearchCV(estimator=self.etreeclassifier, param_grid=self.param_grid, cv=self.kfold, n_jobs=20)
        self.CV_exTeeCls.fit(self.X, self.y)
        self.CV_exTeeCls.score(self.X_val, self.y_val)

        return (self.CV_exTeeCls.best_score_, self.CV_exTeeCls.score(self.X_val, self.y_val))

    # logistic regression with L1~ ridge regression
    def l1Logistic(self, x):
        self.X, self.y, self.X_val, self.y_val, self.proteins_feature = self.svmFSOutput(x = x)
        self.l1Log = LogisticRegression(penalty='l1',solver='liblinear')
        self.kfold = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

        self.param_grid = {
            "C" : np.logspace(-3,3,7)
        }

        self.CV_l1Log = GridSearchCV(estimator=self.l1Log, param_grid=self.param_grid, cv=self.kfold, n_jobs=20)
        self.CV_l1Log.fit(self.X, self.y)
        self.CV_l1Log.score(self.X_val, self.y_val)

        return (self.CV_l1Log.best_score_, self.CV_l1Log.score(self.X_val, self.y_val))

    # logsitic regression with L2 ~ lasso
    def l2logistic(self, x):
        self.X, self.y, self.X_val, self.y_val, self.proteins_feature = self.svmFSOutput(x = x)
        self.l2Log = LogisticRegression(penalty='l2',solver='liblinear')
        self.kfold = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
        self.param_grid = {
            "C": np.logspace(-3, 3, 7)
        }

        self.CV_l2Log = GridSearchCV(estimator=self.l2Log, param_grid=self.param_grid, cv=self.kfold, n_jobs=20)
        self.CV_l2Log.fit(self.X, self.y)
        self.CV_l2Log.score(self.X_val, self.y_val)

        return (self.CV_l2Log.best_score_, self.CV_l2Log.score(self.X_val, self.y_val))

    # SVM linear
    def svmLinear(self, x ):
        self.X, self.y, self.X_val, self.y_val, self.proteins_feature = self.svmFSOutput(x = x)
        self.svmLin = SVC(kernel='linear')
        self.kfold = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
        self.param_grid = {
            "C": np.logspace(-3, 3, 7)
        }

        self.CV_svmLin = GridSearchCV(estimator=self.svmLin, param_grid=self.param_grid, cv=self.kfold, n_jobs=20)
        self.CV_svmLin.fit(self.X, self.y)
        self.CV_svmLin.score(self.X_val, self.y_val)

        return (self.CV_svmLin.best_score_, self.CV_svmLin.score(self.X_val, self.y_val))

    # SVM linear with rbf
    def svmKernel(self, x):
        self.X, self.y, self.X_val, self.y_val, self.proteins_feature = self.svmFSOutput(x = x)
        self.svmKer = SVC(kernel='rbf')
        self.kfold = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
        self.param_grid = {
            "C": np.logspace(-3, 3, 7),
            'gamma': np.logspace(-3, 3, 7)
        }

        self.CV_svmKer = GridSearchCV(estimator=self.svmKer, param_grid=self.param_grid, cv=self.kfold, n_jobs=20)
        self.CV_svmKer.fit(self.X, self.y)
        self.CV_svmKer.score(self.X_val, self.y_val)

        return (self.CV_svmKer.best_score_, self.CV_svmKer.score(self.X_val, self.y_val))


    # Gaussian naive bayes
    def gaussianNB(self, x):
        self.X, self.y, self.X_val, self.y_val, self.proteins_feature = self.svmFSOutput(x = x)
        self.gnb = GaussianNB()
        self.kfold = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

        self.CV_gnb = GridSearchCV(estimator=self.gnb, param_grid={}, cv=self.kfold, n_jobs=20)
        self.CV_gnb.fit(self.X, self.y)
        self.CV_gnb.score(self.X_val, self.y_val)

        return (self.CV_gnb.best_score_, self.CV_gnb.score(self.X_val, self.y_val))

    # bagging classification with decision tree
    def baggingCls(self, x):
        self.X, self.y, self.X_val, self.y_val, self.proteins_feature = self.svmFSOutput(x = x)
        self.bagging = BaggingClassifier(n_estimators = 10, random_state = 1)
        self.kfold = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

        self.param_grid = {
            'max_samples' : [0.50, 0.75, 1],
            'max_features' : [0.50, 0.75, 1]
        }

        self.CV_bagging = GridSearchCV(estimator=self.bagging, param_grid={}, cv=self.kfold, n_jobs=20)
        self.CV_bagging.fit(self.X, self.y)
        self.CV_bagging.score(self.X_val, self.y_val)

        return (self.CV_bagging.best_score_, self.CV_bagging.score(self.X_val, self.y_val))

    # bagging classification with SVM
    def baggingClsSvm(self, x):
        self.X, self.y, self.X_val, self.y_val, self.proteins_feature = self.svmFSOutput(x = x)
        self.baggingsvm = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state = 1)
        self.kfold = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

        self.param_grid = {
            'max_samples' : [0.75, 1],
            'max_features' : [1]
        }

        self.CV_baggingsvm = GridSearchCV(estimator=self.bagging, param_grid={}, cv=self.kfold, n_jobs=20)
        self.CV_baggingsvm.fit(self.X, self.y)
        self.CV_baggingsvm.score(self.X_val, self.y_val)

        return (self.CV_baggingsvm.best_score_, self.CV_baggingsvm.score(self.X_val, self.y_val))

    def applyAll(self):
        # iterate the classifier 100 times
        self.result_array = np.array([])
        self.trainingArray = np.empty((0, 10), float)
        self.testingArray = np.empty((0, 10), float)

        for r in range(1) :
            self.X, self.y, self.X_val, self.y_val, self.proteins_feature = self.svmFSOutput(x = r)
            self.rftraing, self.rftesting = self.randomForest(x = r)
            self.xgboostTraining, self.xgboostTesting = self.xgboost(x = r)
            self.extreeTreeTraining, self.extreeTreeTesting = self.extreeTree(x = r)
            self.l1logTraining, self.l1logTesting = self.l1Logistic(x = r)
            self.l2logTraining, self.l2logTesting = self.l2logistic(x = r)
            self.svmlinTraining, self.svmlinTesting = self.svmLinear(x = r)
            self.svmKerTraining, self.svmKerTesting = self.svmKernel(x = r)
            self.gNBTraining, self.gNBTesting = self.gaussianNB(x = r)
            self.baggingTraining, self.baggingTesting = self.baggingCls(x = r)
            self.baggingSVMTraining, self.baggingSVMTesting = self.baggingClsSvm(x = r)

            #add them to arrays
            self.trainingArray = np.vstack((self.trainingArray,
                                            np.array([self.rftraing, self.xgboostTraining, self.extreeTreeTraining,
                                                      self.l1logTraining, self.l2logTraining, self.svmlinTraining,
                                                      self.svmKerTraining, self.gNBTraining, self.baggingTraining,
                                                      self.baggingSVMTraining])))

            self.testingArray = np.vstack((self.testingArray,
                                            np.array([self.rftesting, self.xgboostTesting, self.extreeTreeTesting,
                                                      self.l1logTesting, self.l2logTesting, self.svmlinTesting,
                                                      self.svmKerTesting, self.gNBTesting, self.baggingTesting,
                                                      self.baggingSVMTesting])))

            self.result_array = np.append(self.result_array, np.array(self.proteins_feature), axis=0)
            self.counts = Counter(self.result_array)

        return (self.trainingArray, self.testingArray, self.result_array, self.counts)


    def exportArrays(self, fileName):
        # export the data
        self.train, self.test, self.result_array, self.counts = self.applyAll()
        np.savetxt('SVM_Training_Score.txt', self.train, delimiter = '\t')
        np.savetxt('SVM_Testing_Score.txt', self.test, delimiter = '\t')
        np.savetxt("SVM_result_array.txt", self.result_array, fmt='%s')

        # export dictionary to txt file
        self.path = fileName + "/svm_feature_frequency.txt"

        with open(self.path, 'w') as f:
            for key, value in self.counts.items():
                self.string = "{}\t{}".format(key, value)
                f.write("%s\n" % self.string)

        return

cls = ClassifierBenchmark()
