
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict
from IPython.display import display
from scipy import interp
from matplotlib.colors import ListedColormap

from sklearn.svm import LinearSVC, SVC
from sklearn import ensemble, datasets, metrics, preprocessing, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import auc, mean_squared_error, roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import datetime

get_ipython().magic('matplotlib inline')


# In[2]:

train = pd.read_csv('data/data.csv')
train.head()


# In[3]:

vali = pd.read_csv('data/Validation_25.csv')
test = pd.read_csv('data/Testing 75.csv')


# In[4]:

y = train['Grant.Status']
X = train.drop('Grant.Status', axis=1)
X.head()


# In[5]:

# convert the first 25 string to numerical features
X['Sponsor.Code'] = pd.get_dummies(X['Sponsor.Code'], dummy_na=True).values.argmax(1)
X['Grant.Category.Code'] = pd.get_dummies(X['Grant.Category.Code'], dummy_na=True).values.argmax(1)
X['Contract.Value.Band...see.note.A'] = pd.get_dummies(X['Contract.Value.Band...see.note.A'], dummy_na=True).values.argmax(1)
X['RFCD.Code.1'] = pd.get_dummies(X['RFCD.Code.1'], dummy_na=True).values.argmax(1)
X['RFCD.Percentage.1'] = pd.get_dummies(X['RFCD.Percentage.1'], dummy_na=True).values.argmax(1)
X['RFCD.Code.2'] = pd.get_dummies(X['RFCD.Code.2'], dummy_na=True).values.argmax(1)
X['RFCD.Percentage.2'] = pd.get_dummies(X['RFCD.Percentage.2'], dummy_na=True).values.argmax(1)
X['RFCD.Code.3'] = pd.get_dummies(X['RFCD.Code.3'], dummy_na=True).values.argmax(1)
X['RFCD.Percentage.3'] = pd.get_dummies(X['RFCD.Percentage.3'], dummy_na=True).values.argmax(1)
X['RFCD.Code.4'] = pd.get_dummies(X['RFCD.Code.4'], dummy_na=True).values.argmax(1)
X['RFCD.Percentage.4'] = pd.get_dummies(X['RFCD.Percentage.4'], dummy_na=True).values.argmax(1)
X['RFCD.Code.5'] = pd.get_dummies(X['RFCD.Code.5'], dummy_na=True).values.argmax(1)
X['RFCD.Percentage.5'] = pd.get_dummies(X['RFCD.Percentage.5'], dummy_na=True).values.argmax(1)
X['SEO.Code.1'] = pd.get_dummies(X['SEO.Code.1'], dummy_na=True).values.argmax(1)
X['SEO.Percentage.1'] = pd.get_dummies(X['SEO.Percentage.1'], dummy_na=True).values.argmax(1)
X['SEO.Code.2'] = pd.get_dummies(X['SEO.Code.2'], dummy_na=True).values.argmax(1)
X['SEO.Percentage.2'] = pd.get_dummies(X['SEO.Percentage.2'], dummy_na=True).values.argmax(1)
X['SEO.Code.3'] = pd.get_dummies(X['SEO.Code.3'], dummy_na=True).values.argmax(1)
X['SEO.Percentage.3'] = pd.get_dummies(X['SEO.Percentage.3'], dummy_na=True).values.argmax(1)
X['SEO.Code.4'] = pd.get_dummies(X['SEO.Code.4'], dummy_na=True).values.argmax(1)
X['SEO.Percentage.4'] = pd.get_dummies(X['SEO.Percentage.4'], dummy_na=True).values.argmax(1)
X['SEO.Code.5'] = pd.get_dummies(X['SEO.Code.5'], dummy_na=True).values.argmax(1)
X['SEO.Percentage.5'] = pd.get_dummies(X['SEO.Percentage.5'], dummy_na=True).values.argmax(1)


# In[6]:

# convert string features of 15 people to numerics
col = 26
for i in range(15):
    X[X.columns[col]] = pd.get_dummies(X[X.columns[col]], dummy_na=True).values.argmax(1)
    X[X.columns[col + 2]] = pd.get_dummies(X[X.columns[col + 2]], dummy_na=True).values.argmax(1)
    X[X.columns[col + 3]] = pd.get_dummies(X[X.columns[col + 3]], dummy_na=True).values.argmax(1)
    X[X.columns[col + 6]] = pd.get_dummies(X[X.columns[col + 6]], dummy_na=True).values.argmax(1)
    X[X.columns[col + 7]] = pd.get_dummies(X[X.columns[col + 7]], dummy_na=True).values.argmax(1)
    col = col + 15


# In[7]:

# split to day, month, and year
def c_Day(x): return x.split("/")[0]
def c_Month(x): return x.split("/")[1]
def c_Year(x): return x.split("/")[2]
def c_Weekday(x):
    d = datetime.datetime(year = int(c_Year(x)), month = int(c_Month(x)), day = int(c_Day(x)))
    return d.weekday()
def c_Season(x):
    month = int(c_Month(x))
    day = int(c_Day(x))
    if (month in [1,2]):
        return 3 # "winter"
    else:
        if (month in [4,5]):
            return 0 # "spring"
        else:
            if (month in [7,8]):
                return 1 # "summer"
            else:
                if (month in [10,11]):
                    return 2 # "autumn"
                else:
                    if (month == 3):
                        if (day <= 21):
                            return 3 # "winter"
                        else:
                            return 0 # "spring"
                    else:
                        if (month == 6):
                            if (day <= 21):
                                return 0 # "spring"
                            else:
                                return 1 # "summer"
                        else:
                            if (month == 9):
                                if (day <= 21):
                                    return 1 # "summer"
                                else:
                                    return 2 # "autumn"
                            else:
                                if (month == 12):
                                    if (day <= 21):
                                        return 2 # "autumn"
                                    else:
                                        return 3 # "winter"
    return -1


# In[8]:

# assigning the split results to data columns
X['Start.day'] = X['Start.date'].apply(c_Day)
X['Start.month'] = X['Start.date'].apply(c_Month)
X['Start.year'] = X['Start.date'].apply(c_Year)
X['Start.weekday'] = X['Start.date'].apply(c_Weekday)
X['Start.season'] = X['Start.date'].apply(c_Season)
X = X.drop('Start.date', axis=1)
X.head(10)


# In[9]:

#daria selecting train, validation, and test
#select train set 
X_train = X.loc[~X['Grant.Application.ID'].isin(vali['ids']) & ~X['Grant.Application.ID'].isin(test['ids'])]
y_train = y.loc[~X['Grant.Application.ID'].isin(vali['ids']) & ~X['Grant.Application.ID'].isin(test['ids'])]

#select validation set 
X_vali =X.loc[X['Grant.Application.ID'].isin(vali['ids'])]
y_vali =y.loc[X['Grant.Application.ID'].isin(vali['ids'])]

#select test set for predictions
X_test =X.loc[X['Grant.Application.ID'].isin(test['ids'])]
y_test =y.loc[X['Grant.Application.ID'].isin(test['ids'])]


# In[10]:

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X.values)
X_train = imp.transform(X_train)
X_vali = imp.transform(X_vali)
X_test = imp.transform(X_test)


# In[11]:

#Daria check no of lines
display(X_train.shape)
display(y_train.shape)
display(X_vali.shape)
display(y_vali.shape)
display(X_test.shape)
display(y_test.shape)


# In[12]:

rfc = RandomForestClassifier(max_depth=200, n_estimators=1250)
rfc.fit(X_train, y_train)
print(rfc.score(X_vali, y_vali))
print(rfc.score(X_test, y_test))


# In[13]:

# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(y_test, rfc.predict_proba(X_test)[:,1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)
print('ROC AUC: %0.6f' % roc_auc)

# Plot of a ROC curve for a specific class
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[14]:

y_label = rfc.predict(X_test)
print("Classification report:")
print()
print(metrics.classification_report(y_test, y_label))
print()
print("Confusion matrix")
print(metrics.confusion_matrix(y_test, y_label))


# In[15]:

# Fit classifier with out-of-bag estimates
params = {'n_estimators': 250, 'max_depth': 10, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
gbm = ensemble.GradientBoostingClassifier(**params)

gbm.fit(X_train, y_train)
print(gbm.score(X_vali, y_vali))
print(gbm.score(X_test, y_test))


# In[16]:

# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(y_test, gbm.predict_proba(X_test)[:,1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)
print('ROC AUC: %0.6f' % roc_auc)

# Plot of a ROC curve for a specific class
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[17]:

y_label = gbm.predict(X_test)
print("Classification report:")
print()
print(metrics.classification_report(y_test, y_label))
print()
print("Confusion matrix")
print(metrics.confusion_matrix(y_test, y_label))


# In[18]:

names = ["Nearest Neighbors", "Decision Tree", "AdaBoost", "Naive Bayes", 
         "Lin Discr Analysis", "Quad Discr Analysis"]
classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name + ": " + str(score))
    y_label = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_label))
    print()
    print("Confusion matrix")
    print(metrics.confusion_matrix(y_test, y_label))
    print()


# In[19]:

svc = svm.SVC()
svc.fit(X_train, y_train)
print(svc.score(X_vali, y_vali))
print(svc.score(X_test, y_test))


# In[20]:

y_label = svc.predict(X_test)
print("Classification report:")
print()
print(metrics.classification_report(y_test, y_label))
print()
print("Confusion matrix")
print(metrics.confusion_matrix(y_test, y_label))


# In[ ]:



