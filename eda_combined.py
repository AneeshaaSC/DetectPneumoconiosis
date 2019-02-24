import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sb

#Format the display a little bit
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format', lambda x:'%f'%x)

# read data from all different tabs in excel 
comb=pd.read_excel('combined.xlsx')
#plt.figure(figsize=(12, 9))
comb['Label'] = comb['Label'].convert_objects(convert_numeric=True)

numoru=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightUpper')
numorm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightMiddle')
numorl=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','RightLower')

numoll=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftLower')
numolm=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftMiddle')
numolu=pd.read_excel('CollatedPneumoconiosisData-GE Internal.xlsx','LeftUpper')

#All patients have same label in all zones


import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing

# standardize clustering variables to have mean=0 and sd=1
def centering(dfname):
    for i in dfname.columns:
            dfname[i]=preprocessing.scale(dfname[i].astype('float64'))
    return dfname

print(' ')
print('Centering Variables ...')
print(' ')

rucp=numoru.copy()
z1data=centering(numoru)
z1data.drop(['PatientNumMasked'],axis=1,inplace=True)
z1data.drop(['LabelRU'],axis=1,inplace=True)
z1data['Label']=rucp['LabelRU']    

rmcp=numorm.copy()
z2data=centering(numorm) 
z2data.drop(['PatientNumMasked'],axis=1,inplace=True)
z2data.drop(['LabelRM'],axis=1,inplace=True) 
z2data['Label']=rmcp['LabelRM']   
"""
rlcp=numorl.copy()
z3data=centering(numorl)
z3data.drop(['PatientNumMasked'],axis=1,inplace=True)
z3data.drop(['LabelRL'],axis=1,inplace=True) 
z3data['Label']=numorl['LabelRL'] 
"""   
z4data=centering(numolu) 
z4data['Label']=numolu['LabelLU'] 

z5data=centering(numolm) 
z5data['Label']=numolm['LabelLM'] 
   
z6data=centering(numoll) 
z6data['Label']=numoll['LabelLL'] 

#print(z1data['Label'].value_counts())

reg1=smf.logit(formula='Label ~ CoMatrix_Deg135_Correlation+CoMatrix_Deg90_Local_Homogeneity',data=comb).fit()
print(reg1.summary())
print(' ')
print(' ')
print('Odds Ratio')
print('-----------')
#print(np.exp(reg1.params)) 


params = reg1.params
conf = reg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))
"""
 
reg2=smf.logit(formula='Label ~ Hist_2_60_1_Skewness+Hist_2_60_2_Skewness',data=z2data).fit()
print(reg2.summary())
print(' ')
print(' ')
print('Odds Ratio')
print('-----------')
#print(np.exp(reg1.params)) 


params = reg2.params
conf = reg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))


# Create correlation matrix
corr_matrix = numolu.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
print(upper)
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print('correlated columns')
print(to_drop)
"""