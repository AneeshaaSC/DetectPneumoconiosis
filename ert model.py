import pandas as pd
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x:'%f'%x)
from sklearn import preprocessing
# read data from all different tabs in excel 
numoru=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightUpper')
numorm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightMiddle')
numorl=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightLower')

numoll=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftLower')
numolm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftMiddle')
numolu=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftUpper')

#check dimensions
print('zone1 shape:',numoru.shape) 
print('zone2 shape:',numorm.shape)
print('zone3 shape:',numorl.shape)
print('zone4 shape:',numolu.shape)
print('zone5 shape:',numolm.shape)
print('zone6 shape:',numoll.shape)


# Merge PatientNumMased and Labels of the 6 zones into one dataframe: ymain
ymain=numorm[['PatientNumMasked','Label']]
ymain.rename(columns={'Label':'y2'},inplace=True)
ymain=pd.merge(numoru[['Label','PatientNumMasked']], ymain, on='PatientNumMasked',how='outer')
ymain.rename(columns={'Label':'y1'},inplace=True)
ymain=pd.merge(numorl[['Label','PatientNumMasked']], ymain, on='PatientNumMasked',how='outer')
ymain.rename(columns={'Label':'y3'},inplace=True)
ymain=pd.merge(numolu[['Label','PatientNumMasked']], ymain, on='PatientNumMasked',how='outer')
ymain.rename(columns={'Label':'y4'},inplace=True)
ymain=pd.merge(numolm[['Label','PatientNumMasked']], ymain, on='PatientNumMasked',how='outer')
ymain.rename(columns={'Label':'y5'},inplace=True)
ymain=pd.merge(numoll[['Label','PatientNumMasked']], ymain, on='PatientNumMasked',how='outer')
ymain.rename(columns={'Label':'y6'},inplace=True)

# Replace Nans with 0 in ymain
ymain.fillna(0, inplace=True)

# If any one of the zones are labeled '1' or abnormal, Y will be 1
ymain['Y']=ymain[["y1", "y2","y3", "y4","y5", "y6"]].max(axis=1)

#separate predictors and target variables into different dataframes for all 6 zones
z1data=numoru.copy()
#check for duplicates, duplicate PatientNumMasked vlaues will be True
z1dups=z1data.duplicated(['PatientNumMasked'], keep=False)
# print how many duplicates were found
print('z1 duplicates counts: ',z1dups.groupby(z1dups).size())
# drop identifier and target variable
z1data.drop(['PatientNumMasked'],axis=1,inplace=True)
z1data.drop(['Label'],axis=1,inplace=True)
z1tar=numoru['Label']
# no duplicates in z1
z2data=numorm.copy()
z2dups=z2data.duplicated(['PatientNumMasked'], keep=False)
print('z2 duplicates counts: ',z2dups.groupby(z2dups).size())
z2data.drop(['PatientNumMasked'],axis=1,inplace=True)
z2data.drop(['Label'],axis=1,inplace=True)
z2tar=numorm['Label']
# no duplicates in z2
z3data=numorl.copy()
z3dups=z3data.duplicated(['PatientNumMasked'], keep=False)
print('z3 duplicates counts: ',z3dups.groupby(z3dups).size())
z3data.drop(['PatientNumMasked'],axis=1,inplace=True)
z3data.drop(['Label'],axis=1,inplace=True)
z3tar=numorl['Label']
# no duplicates in z3
z4data=numolu.copy()
z4dups=z4data.duplicated(['PatientNumMasked'], keep=False)
print('z4 duplicates counts: ',z4dups.groupby(z4dups).size())
z4data.drop(['PatientNumMasked'],axis=1,inplace=True)
z4data.drop(['Label'],axis=1,inplace=True)
z4tar=numolu['Label']
# no duplicates in z4
z5data=numolm.copy()
z5dups=z5data.duplicated(['PatientNumMasked'], keep=False)
print('z5 duplicates counts: ',z5dups.groupby(z5dups).size())
z5data.drop(['PatientNumMasked'],axis=1,inplace=True)
z5data.drop(['Label'],axis=1,inplace=True)
z5tar=numolm['Label']
# no duplicates in z5
z6data=numoll.copy()
z6dups=z6data.duplicated(['PatientNumMasked'], keep=False)
print('z6 duplicates counts: ',z6dups.groupby(z6dups).size())
z6data.drop(['PatientNumMasked'],axis=1,inplace=True)
z6data.drop(['Label'],axis=1,inplace=True)
z6tar=numoll['Label']
# no duplicates in z6


# standardize clustering variables to have mean=0 and sd=1
def centering(dfname):
    for i in dfname.columns:
        dfname[i]=preprocessing.scale(dfname[i].astype('float64'))
    return dfname

z1data=centering(z1data)    
z2data=centering(z2data)    
z3data=centering(z3data)    
z4data=centering(z4data) 
z5data=centering(z5data)    
z6data=centering(z6data) 
print(' ')
print('Variables are now centered')
print(' ')
# to perform leave one out cross validation
from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

# Perform feature selection   
from sklearn.ensemble import ExtraTreesClassifier    
from sklearn.feature_selection import SelectFromModel



def drop_useless_features(predictors,target):
    classifier=ExtraTreesClassifier()
    for train_index, test_index in loo.split(predictors):
        X_train, X_test = predictors.iloc[train_index], predictors.iloc[test_index]
        y_train, y_test = target[train_index], target[test_index]
        classifier=classifier.fit(X_train,y_train)
    model = SelectFromModel(classifier, prefit=True)
    predictors = model.transform(predictors)
    print('Number features selected: ',predictors.shape[1])
    return predictors

   
z2data=drop_useless_features(z2data,z2tar) 
print(z2data.shape)

z1data=drop_useless_features(z1data,z1tar) 
print(z1data.shape)

z3data=drop_useless_features(z3data,z3tar) 
print(z3data.shape)

z4data=drop_useless_features(z4data,z4tar) 
print(z4data.shape)

z5data=drop_useless_features(z5data,z5tar) 
print(z5data.shape)

z6data=drop_useless_features(z6data,z6tar) 
print(z6data.shape)

print(' ')
print('Feature Selection complete!')
print(' ')

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

def final_pred(predictors,target):
    classifier=ExtraTreesClassifier()
    tar_pred=cross_val_predict(classifier,predictors,target,cv=loo)
    """
    for train_index, test_index in loo.split(predictors):
        X_train, X_test = predictors.iloc[train_index], predictors.iloc[test_index]
        y_train= target[train_index]
        classifier=classifier.fit(X_train,y_train)
        tar_pred[test_index] = classifier.predict(X_test)
    """    
    score=accuracy_score(target,tar_pred)*100
    return tar_pred,score

print(' ')
print('Zone level prediction begins')
print(' ')
 
print(' ')
print('convert predictors to dataframe!')
print(' ')

z2data=pd.DataFrame(z2data)
z1data=pd.DataFrame(z1data)
z3data=pd.DataFrame(z3data)
z4data=pd.DataFrame(z4data)
z5data=pd.DataFrame(z5data)
z6data=pd.DataFrame(z6data)
print(' ')
print('call function!')
print(' ')

y2_pred_2,y2_acc_2=final_pred(z2data, z2tar)
y1_pred_2,y1_acc_2=final_pred(z1data, z1tar)
y3_pred_2,y3_acc_2=final_pred(z3data, z3tar)
y4_pred_2,y4_acc_2=final_pred(z4data, z4tar)
y5_pred_2,y5_acc_2=final_pred(z5data, z5tar)
y6_pred_2,y6_acc_2=final_pred(z6data, z6tar)

print(' ')
print('Label Prediction for the 6 zones done!')
print(' ')


print(' ')
print(' ')
print('Accuracy with Extremely Randomized Trees:')
print("Accuracy for y2: %.3f%% " % (y2_acc_2))
print("Accuracy for y3: %.3f%% " % (y3_acc_2))
print("Accuracy for y1: %.3f%% " % (y1_acc_2))
print("Accuracy for y4: %.3f%% " % (y4_acc_2))
print("Accuracy for y5: %.3f%% " % (y5_acc_2))
print("Accuracy for y6: %.3f%% " % (y6_acc_2))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(' ')
print(' ')


y2_pred_2=pd.DataFrame(y2_pred,columns = ["y2"])
y1_pred_2=pd.DataFrame(y1_pred,columns = ["y1"])
y3_pred_2=pd.DataFrame(y3_pred,columns = ["y3"])
y4_pred_2=pd.DataFrame(y4_pred,columns = ["y4"])
y5_pred_2=pd.DataFrame(y5_pred,columns = ["y5"])
y6_pred_2=pd.DataFrame(y6_pred,columns = ["y6"])

y2_pred.reset_index(inplace=True)
y1_pred.reset_index(inplace=True)
y3_pred.reset_index(inplace=True)
y4_pred.reset_index(inplace=True)
y5_pred.reset_index(inplace=True)
y6_pred.reset_index(inplace=True)

y2_pred_2.reset_index(inplace=True)
y1_pred_2.reset_index(inplace=True)
y3_pred_2.reset_index(inplace=True)
y4_pred_2.reset_index(inplace=True)
y5_pred_2.reset_index(inplace=True)
y6_pred_2.reset_index(inplace=True)

y2_patid=numorm['PatientNumMasked']
y2_patid=pd.DataFrame(y2_patid)
y1_patid=numoru['PatientNumMasked']
y1_patid=pd.DataFrame(y1_patid)
y3_patid=numorl['PatientNumMasked']
y3_patid=pd.DataFrame(y3_patid)
y4_patid=numolu['PatientNumMasked']
y4_patid=pd.DataFrame(y4_patid)
y5_patid=numolm['PatientNumMasked']
y5_patid=pd.DataFrame(y5_patid)
y6_patid=numoll['PatientNumMasked']
y6_patid=pd.DataFrame(y6_patid)



y2_patid.reset_index(inplace=True)
y1_patid.reset_index(inplace=True)
y3_patid.reset_index(inplace=True)
y4_patid.reset_index(inplace=True)
y5_patid.reset_index(inplace=True)
y6_patid.reset_index(inplace=True)

y1_pred=pd.merge(y1_pred, y1_patid, on='index')

y2_pred=pd.merge(y2_pred, y2_patid, on='index')

y3_pred=pd.merge(y3_pred, y3_patid, on='index')

y4_pred=pd.merge(y4_pred, y4_patid, on='index')

y5_pred=pd.merge(y5_pred, y5_patid, on='index')

y6_pred=pd.merge(y6_pred, y6_patid, on='index')

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

y1_pred_2=pd.merge(y1_pred_2, y1_patid, on='index')

y2_pred_2=pd.merge(y2_pred_2, y2_patid, on='index')

y3_pred_2=pd.merge(y3_pred_2, y3_patid, on='index')

y4_pred_2=pd.merge(y4_pred_2, y4_patid, on='index')

y5_pred_2=pd.merge(y5_pred_2, y5_patid, on='index')

y6_pred_2=pd.merge(y6_pred_2, y6_patid, on='index')



pred_y=pd.merge(y1_pred, y2_pred, on='PatientNumMasked',how='outer')
pred_y=pd.merge(pred_y, y3_pred, on='PatientNumMasked',how='outer')
pred_y=pd.merge(pred_y, y4_pred, on='PatientNumMasked',how='outer')
pred_y=pd.merge(pred_y, y5_pred, on='PatientNumMasked',how='outer')
pred_y=pd.merge(pred_y, y6_pred, on='PatientNumMasked',how='outer')
pred_y['Ypred']=pred_y[["y1", "y2","y3", "y4","y5", "y6"]].max(axis=1)

pred_y.fillna(0, inplace=True)

pred_y_2=pd.merge(y1_pred_2, y2_pred_2, on='PatientNumMasked',how='outer')
pred_y_2=pd.merge(pred_y_2, y3_pred_2, on='PatientNumMasked',how='outer')
pred_y_2=pd.merge(pred_y_2, y4_pred_2, on='PatientNumMasked',how='outer')
pred_y_2=pd.merge(pred_y_2, y5_pred_2, on='PatientNumMasked',how='outer')
pred_y_2=pd.merge(pred_y_2, y6_pred_2, on='PatientNumMasked',how='outer')
pred_y_2['Ypred']=pred_y_2[["y1", "y2","y3", "y4","y5", "y6"]].max(axis=1)

pred_y_2.fillna(0, inplace=True)

from sklearn.metrics import confusion_matrix
ymatrix = confusion_matrix(pred_y['Ypred'], ymain['Y'])
print('mat1')
print(ymatrix)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

ymatrix2 = confusion_matrix(pred_y_2['Ypred'], ymain['Y'])
print('mat2')
print(ymatrix2)

"""
print('y1 shape:',y1_pred.shape) 
print('y2 shape:',y2_pred.shape)
print('y3 shape:',y3_pred.shape)
print('y4 shape:',y4_pred.shape)
print('y5 shape:',y5_pred.shape)
print('y6 shape:',y6_pred.shape)
"""
print(' ')

temp2=pd.merge(ymain, pred_y_2, on='PatientNumMasked')
cp2=0
temp2['Ypred']=temp2['Ypred'].convert_objects(convert_numeric=True)

for i in range(len(temp2)):
    if temp2['Ypred'].iloc[i]==ymain['Y'].iloc[i]:
        cp2=cp2+1

acc2=(cp2/len(ymain))*100
print('Final Accuracy from ERT:', acc2)

from sklearn.metrics import precision_recall_fscore_support as score
precision2, recall2, fscore2, support2 = score(ymain['Y'], pred_y_2['Ypred'])
print('precision1: {}'.format(precision2))
print('recall1: {}'.format(recall2))


temp=pd.merge(ymain, pred_y, on='PatientNumMasked')
cp=0
temp['Ypred']=temp['Ypred'].convert_objects(convert_numeric=True)


for i in range(len(ymain)):
    if temp['Ypred'].iloc[i]==ymain['Y'].iloc[i]:
        cp=cp+1

acc=(cp/len(temp))*100
print('Final Accuracy FROM LR:', acc)


precision, recall, fscore, support = score(ymain['Y'], pred_y['Ypred'])
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))

