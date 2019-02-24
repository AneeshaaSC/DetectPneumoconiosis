
# Scatterplot Matrix
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np


#Format the display a little bit
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format', lambda x:'%f'%x)

# read data from all different tabs in excel 
comb=pd.read_excel('combined.xlsx')
comb_2=comb.copy()
combcp=comb[['PatientNumMasked','Zone']]
combcp=pd.DataFrame((combcp))
comb.drop(['PatientNumMasked', 'Label','Zone'], axis=1,inplace=True)

numoru=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightUpper')
rucp=numoru[['PatientNumMasked']]
numorm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightMiddle')
numorl=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightLower')

numoll=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftLower')
llcp=numoll[['PatientNumMasked']]
numolm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftMiddle')
numolu=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftUpper')


from sklearn.ensemble import IsolationForest

# fit the model
def iso(dfname):
    clf = IsolationForest(max_samples=100, random_state=123)
    clf.fit(dfname)
    y_pred_train = clf.predict(dfname)
    num_normal = (y_pred_train == 1).sum()
    num_outliers = (y_pred_train == -1).sum()
    print(' ')
    print(' ')
    print('Number of normal observations: ',num_normal)
    print('Number of outliers: ',num_outliers)
    print(' ')
    print(' ')
    y_pred_train=pd.DataFrame((y_pred_train))
    y_pred_train.columns=["Outlierornot"]
    y_pred_train.reset_index(inplace=True)
    return y_pred_train

l=iso(comb)
#ruout=iso(numoru)
combcp.reset_index(inplace=True)
#rucp.reset_index(inplace=True)
#ruout=pd.merge(ruout, rucp, on='index')
l=pd.merge(l, combcp, on='index')
#llout=iso(numoll)
#llcp.reset_index(inplace=True)
#llout=pd.merge(llout, llcp, on='index')


#llout=llout[['PatientNumMasked','Outlierornot']]

value_list = [-1]
#Grab DataFrame rows where column has certain values

#outlist=(llout[llout.Outlierornot.isin(value_list)])
outlist=(l[l.Outlierornot.isin(value_list)])
#print(outlist)
#outlist.drop_duplicates(inplace=True)
print(' ')
print(' ')
print(outlist["Outlierornot"].value_counts())
#outlist=outlist.PatientNumMasked.unique()
#print(outlist)
#Outliers.drop_duplicates(inplace=True)

print(' ')
print(' ')
#print('Lung Zone    No. of Outliers')
#print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#l=outlist["PatientNumMasked"]
#print(llout.shape[0])
#llout = llout[~llout['PatientNumMasked'].isin(l)]
#print(llout.shape[0])

outpatlist=outlist["PatientNumMasked"]

print(comb_2.shape)

combnoout=comb_2[~comb_2['PatientNumMasked'].isin(outpatlist)]
combnoouty=combnoout[['Label']]

print(combnoout.shape)
combnoout.drop(['PatientNumMasked', 'Label','Zone'], axis=1,inplace=True)

from sklearn.ensemble import ExtraTreesClassifier    
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import train_test_split

from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

def drop_useless_features(predictors,target):
    #split data set into training and test sets
    pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, target, test_size=.3)
    classifier=ExtraTreesClassifier()
    #fit model on training set alone to perform feature selection
    classifier=classifier.fit(pred_train,tar_train)
    #importances=classifier.feature_importances_ 
    sfm = SelectFromModel(classifier,prefit=True)
    predictors = sfm.transform(predictors)
    return predictors 



combnoout=drop_useless_features(combnoout,combnoouty) 
print(combnoout.shape)
from sklearn.metrics import accuracy_score

def model_build_etc(dfpred,dftar):
    etc = ExtraTreesClassifier()
    pred_train, pred_test, tar_train, tar_test  = train_test_split(dfpred, dftar, test_size=.3)
    tar_train=tar_train.values
    #pred_train=pred_train.values
    print(pred_train.shape)
    print(tar_test.shape)
    score=[0]*pred_train.shape[0]         
    for train_index, test_index in loo.split(pred_train):
        X_train, X_test = pred_train[train_index], pred_train[test_index]
        y_train= tar_train[train_index]
        y_test = tar_train[test_index]
        etc.fit(X_train, y_train)
        y_test_pred=etc.predict(X_test)
        test_index=np.asscalar(test_index)
        score[test_index]=accuracy_score(y_test_pred,y_test)
    cv_acc_score=np.mean(score)*100
    print('Accuracy on Cross-validation set: ',cv_acc_score,'%')
    print(' ')
    print(' ')
    print('Predict on Test Set ...')
    print(' ')
    print(' ')
    predictions=etc.predict(pred_test)
    test_acc_score=accuracy_score(tar_test, predictions)*100
    print('Accuracy Score on Test Set',test_acc_score,'%')

model_build_etc(combnoout,combnoouty)
