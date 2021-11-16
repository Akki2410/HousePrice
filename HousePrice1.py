# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:22:27 2021

@author: aksha
Data House price
files: train and test
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.set_option("max_columns",None)
pd.set_option("max_colwidth", None)
#%matplotlib inline

#Load Files
df = pd.read_csv('C:/Users/aksha/Downloads/house price/train.csv')
test = pd.read_csv('C:/Users/aksha/Downloads/house price/test.csv')

#There is no SalePrice in test file
df.shape #1460 , 81
test.shape #1459, 80

##Drop ID columns--------------------------------------------------------------
df.drop(columns='Id',axis=1,inplace= True)
test.drop(columns='Id',axis=1,inplace= True)

#Check for Nulls---------------------------------------------------------------
fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,ax=ax)

obj = df.isnull().sum() 
for key,value in obj.iteritems():
    print(key,":",value)

obj = test.isnull().sum()
for key,value in obj.iteritems():
    print(key,":",value)


#replacing lotfrontage null values with mean
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())

#drop columns with sigularity for null values
df.drop(columns=['MiscFeature','PoolQC','Alley'],axis = 1 , inplace=True)
test.drop(columns=['MiscFeature','PoolQC','Alley'],axis = 1 , inplace=True)

#check for values in test file
test['MSZoning'].isnull().sum()
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
#--------------------------------
df_og = df.copy()
#--------------------------------
#replacing null values 
df['BsmtCond']=df['BsmtCond'].fillna('NoBsmtCond')
df['BsmtQual']=df['BsmtQual'].fillna('NoBsmt')
#test
test['BsmtCond']=test['BsmtCond'].fillna('NoBsmtCond')
test['BsmtQual']=df['BsmtQual'].fillna('NoBsmt')

#BsmtExposure
df['BsmtExposure'] = df['BsmtExposure'] .fillna('NoBsmtExpo')
test['BsmtExposure'] = test['BsmtExposure'] .fillna('NoBsmtExpo')

#BsmtFinType1
df['BsmtFinType1'] = df['BsmtFinType1'] .fillna('NoBsmtFinType1')
#test
test['BsmtFinType1'] = test['BsmtFinType1'] .fillna('NoBsmtFinType1')

#BsmtFinType2
df['BsmtFinType2'] = df['BsmtFinType2'] .fillna('NoBsmtFinType2')
#test
test['BsmtFinType2'] = test['BsmtFinType2'] .fillna('NoBsmtFinType2')

#Electrical
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

#FireplaceQu
df['FireplaceQu'] = df['FireplaceQu'] .fillna('NoFirePlace')
#test
test['FireplaceQu'] = test['FireplaceQu'] .fillna('NoFirePlace')


#GarageType #GarageFinish GarageQual GarageCond
df['GarageType'] = df['GarageType'] .fillna('NoGarageType')
df['GarageFinish'] = df['GarageFinish'] .fillna('NoGarageFinish')
df['GarageQual'] = df['GarageQual'] .fillna('NoGarageQual')
df['GarageCond'] = df['GarageCond'] .fillna('NoGarageCond')
#test
#GarageType #GarageFinish GarageQual GarageCond
test['GarageType'] = test['GarageType'] .fillna('NoGarageType')
test['GarageFinish'] = test['GarageFinish'] .fillna('NoGarageFinish')
test['GarageQual'] = test['GarageQual'] .fillna('NoGarageQual')
test['GarageCond'] = test['GarageCond'] .fillna('NoGarageCond')


#MasVnrArea #MasVnrType
df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
#test
test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])

df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
#test
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())

df['Fence'] = df['Fence'].fillna('NoFence')
test['Fence'] = test['Fence'].fillna('NoFence')

test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())


test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test['TotalBsmtSF'].isnull().sum()


test['BsmtFullBath'].mode()
test['BsmtHalfBath'].value_counts()
test['BsmtFullBath'].isnull().sum()
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])


test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])

test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mode()[0])
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())
#SaleType 
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])

#fill na values for GarageYrBlt with 2021 : then we will delete 2021-2021
df['GarageYrBlt'] =df['GarageYrBlt'].fillna(2021)
test['GarageYrBlt'] =test['GarageYrBlt'].fillna(2021)
df['GarageYrBlt'][df.GarageYrBlt==2021].count()


#-Splicols in fc:factor variable and nc : numerical variable======================================================

def splitcols(data):
  nc = data.select_dtypes(exclude = 'object').columns.values
  fc = data.select_dtypes(include = 'object').columns.values
  return(nc,fc)

nc,fc = splitcols(df)
nc_test,fc_test = splitcols(test)
#converting year to age
year_variable = [feature for feature in nc if 'Yr' in feature or 'Year' in feature]
print(year_variable)


from datetime import date
currentyear = date.today().year
currentyear
#df age
for feature in year_variable:
  df[feature] = currentyear - df[feature]
#test age  
for feature in year_variable:
    test[feature] = currentyear - test[feature]
#------------------------
df_year = df.copy()
test_year = test.copy()
#----------------------
#df = df_year.copy()
#test = test_year.copy()

df['GarageYrBlt'].isnull().sum()

df.isnull().sum()

print(nc)

#working with categorical variables in nc======================================
#check categorical cols : check unique values
#seperate categorical variable from numerical variables
categorical_features = []
for feature in nc:
 # print('Column Name: {} ,\n unique values\n{} '.format(feature,data_nc[feature].value_counts()))
  if len(df[feature].value_counts()) <= 15:
    categorical_features.append(feature)

categorical_features

categorical_features.remove('PoolArea')
categorical_features.remove('YrSold')
#as per obsevation remove below from categorical variable
#PoolArea #YrSold


for feature in categorical_features:
  print(feature,df[feature].value_counts())
  
#singularity categorical variable
#BsmtHalfBath
#KitchenAbvGr
df.drop(columns="BsmtHalfBath",axis=1,inplace=True)
df.drop(columns="KitchenAbvGr",axis=1,inplace=True)
#test
test.drop(columns="BsmtHalfBath",axis=1,inplace=True)
test.drop(columns="KitchenAbvGr",axis=1,inplace=True)

#removing entry of KitchenAbvGr and BsmtHalfBath categorical and nc
categorical_features.remove('BsmtHalfBath')
categorical_features.remove('KitchenAbvGr')
nc = np.delete(nc, np.where(nc == 'BsmtHalfBath'))
nc = np.delete(nc, np.where(nc == 'KitchenAbvGr'))


type(nc) #numpy.ndarray
type(categorical_features) #list

#remove categorical variable from nc
for c in categorical_features:
  nc = np.delete(nc, np.where(nc == c))
#----------------------------------------------------------------------
for feature in categorical_features:
  print(feature,df[feature].value_counts())

#in test MSSubClass had value 150 which is not present in train file so converting that to mode value
test['MSSubClass'][test['MSSubClass']==150]=test['MSSubClass'].mode()[0]
test['MSSubClass'].value_counts()

for feature in categorical_features:
  print(feature,test[feature].value_counts())

#As per obsevartion  below variables have extra values in test file
#FullBath  4   4
#Fireplaces 4  1
#GarageCars 5   1
#MoSold     2   81  --- convert this to 3

test['FullBath'] = test['FullBath'].replace(4,test['FullBath'].mode()[0])
test['FullBath'].value_counts()
test['Fireplaces'].value_counts()
test['Fireplaces'] = test['Fireplaces'].replace(4,test['Fireplaces'].mode()[0])
test['GarageCars'].value_counts()
test['GarageCars'] = test['GarageCars'].replace(5,test['GarageCars'].mode()[0])
test['MoSold'] = test['MoSold'].replace(2,3)
#convert values where count is very low to mode value-----------------------------------------------------------------------
#df--------------------------------------------------
df['GarageCars'][df['GarageCars']==4]=df['GarageCars'].mode()[0]
df['Fireplaces'][df['Fireplaces']==3]=df['Fireplaces'].mode()[0]

#TotRmsAbvGrd
df['TotRmsAbvGrd'][df['TotRmsAbvGrd']==3]=df['TotRmsAbvGrd'].mode()[0]
df['TotRmsAbvGrd'][df['TotRmsAbvGrd']==11]=df['TotRmsAbvGrd'].mode()[0]
df['TotRmsAbvGrd'][df['TotRmsAbvGrd']==12]=df['TotRmsAbvGrd'].mode()[0]
df['TotRmsAbvGrd'][df['TotRmsAbvGrd']==14]=df['TotRmsAbvGrd'].mode()[0]
df['TotRmsAbvGrd'][df['TotRmsAbvGrd']==2]=df['TotRmsAbvGrd'].mode()[0]

#BedroomAbvGr
df['BedroomAbvGr'][df['BedroomAbvGr']== 6] = df['BedroomAbvGr'].mode()[0]
df['BedroomAbvGr'][df['BedroomAbvGr']== 0] = df['BedroomAbvGr'].mode()[0]
df['BedroomAbvGr'][df['BedroomAbvGr']== 5] = df['BedroomAbvGr'].mode()[0]
df['BedroomAbvGr'][df['BedroomAbvGr']== 1] = df['BedroomAbvGr'].mode()[0]
df['BedroomAbvGr'][df['BedroomAbvGr']== 8] = df['BedroomAbvGr'].mode()[0]

df['HalfBath'][df['HalfBath']==2]=df['HalfBath'].mode()[0]

df['FullBath'][df['FullBath']==3]=df['FullBath'].mode()[0]
df['FullBath'][df['FullBath']==0]=df['FullBath'].mode()[0]

df['BsmtFullBath'][df['BsmtFullBath']==3]=df['BsmtFullBath'].mode()[0]
df['BsmtFullBath'][df['BsmtFullBath']==2]=df['BsmtFullBath'].mode()[0]

df['OverallCond'][df['OverallCond']==1]=df['OverallCond'].mode()[0]
df['OverallCond'][df['OverallCond']==2]=df['OverallCond'].mode()[0]
df['OverallCond'][df['OverallCond']==9]=df['OverallCond'].mode()[0]
df['OverallCond'][df['OverallCond']==3]=df['OverallCond'].mode()[0]

#OverallQual
df['OverallQual'][df['OverallCond']==3]=df['OverallQual'].mode()[0]
df['OverallQual'][df['OverallCond']==10]=df['OverallQual'].mode()[0]
df['OverallQual'][df['OverallCond']==2]=df['OverallQual'].mode()[0]
df['OverallQual'][df['OverallCond']==1]=df['OverallQual'].mode()[0]
#------------------------------------------------------------
#test
test['GarageCars'][test['GarageCars']==4]=test['GarageCars'].mode()[0]
test['Fireplaces'][test['Fireplaces']==3]=test['Fireplaces'].mode()[0]

#TotRmsAbvGrd
test['TotRmsAbvGrd'][test['TotRmsAbvGrd']==3]=test['TotRmsAbvGrd'].mode()[0]
test['TotRmsAbvGrd'][test['TotRmsAbvGrd']==11]=test['TotRmsAbvGrd'].mode()[0]
test['TotRmsAbvGrd'][test['TotRmsAbvGrd']==12]=test['TotRmsAbvGrd'].mode()[0]
test['TotRmsAbvGrd'][test['TotRmsAbvGrd']==14]=test['TotRmsAbvGrd'].mode()[0]
test['TotRmsAbvGrd'][test['TotRmsAbvGrd']==2]=test['TotRmsAbvGrd'].mode()[0]

#BedroomAbvGr
test['BedroomAbvGr'][test['BedroomAbvGr']== 6] = test['BedroomAbvGr'].mode()[0]
test['BedroomAbvGr'][test['BedroomAbvGr']== 0] = test['BedroomAbvGr'].mode()[0]
test['BedroomAbvGr'][test['BedroomAbvGr']== 5] = test['BedroomAbvGr'].mode()[0]
test['BedroomAbvGr'][test['BedroomAbvGr']== 1] = test['BedroomAbvGr'].mode()[0]
test['BedroomAbvGr'][test['BedroomAbvGr']== 8] = test['BedroomAbvGr'].mode()[0]

test['HalfBath'][test['HalfBath']==2]=test['HalfBath'].mode()[0]

test['FullBath'][test['FullBath']==3]=test['FullBath'].mode()[0]
test['FullBath'][test['FullBath']==0]=test['FullBath'].mode()[0]

test['BsmtFullBath'][test['BsmtFullBath']==3]=test['BsmtFullBath'].mode()[0]
test['BsmtFullBath'][test['BsmtFullBath']==2]=test['BsmtFullBath'].mode()[0]

test['OverallCond'][test['OverallCond']==1]=test['OverallCond'].mode()[0]
test['OverallCond'][test['OverallCond']==2]=test['OverallCond'].mode()[0]
test['OverallCond'][test['OverallCond']==9]=test['OverallCond'].mode()[0]
test['OverallCond'][test['OverallCond']==3]=test['OverallCond'].mode()[0]

#OverallQual
test['OverallQual'][test['OverallCond']==3]=test['OverallQual'].mode()[0]
test['OverallQual'][test['OverallCond']==10]=test['OverallQual'].mode()[0]
test['OverallQual'][test['OverallCond']==2]=test['OverallQual'].mode()[0]
test['OverallQual'][test['OverallCond']==1]=test['OverallQual'].mode()[0]
#-----------------------------
df_cat = df.copy()
test_cat = test.copy()
#--------------------------------

#==================Categorical features EDA done========================================================================
#fc(string) and nc are left====================================================
#Working with categorical variable fc==========================================
for feature in fc:
  print('Feature Name :{} ,\t\n {}'.format(feature,df[feature].value_counts()))
  #run this delete umwanted features and run again  


#below fields show Singularity 
  #Street #LandContour #Utilities #LandSlope #Condition2 #RoofMatl #BsmtCond #Heating #CentralAir #Electrical #Functional # GarageQual
  #GarageCond #PavedDrive
df.drop(columns=['Street','LandContour','Utilities','LandSlope','Condition2','RoofMatl','BsmtCond',
                 'Heating','CentralAir','Electrical','Functional','GarageQual',
                 'GarageCond','PavedDrive'],axis=1,inplace=True)

#remove from test
test.drop(columns=['Street','LandContour','Utilities','LandSlope','Condition2','RoofMatl','BsmtCond',
                 'Heating','CentralAir','Electrical','Functional','GarageQual',
                 'GarageCond','PavedDrive'],axis=1,inplace=True)

fc_drop = ['Street','LandContour','Utilities','LandSlope','Condition2','RoofMatl','BsmtCond',
                 'Heating','CentralAir','Electrical','Functional','GarageQual',
                 'GarageCond','PavedDrive']
for c in fc_drop:
  fc = np.delete(fc, np.where(fc == c))
  
df.shape
  

#############################################################################
groupedvalues  = df.groupby('Neighborhood')['SalePrice'].median().reset_index()
groupedvalues.head(120)
groupedvalues.sort_values(by=['SalePrice'])
#groupedvalues.Neighborhood
g = sns.barplot(x=groupedvalues.Neighborhood,y=groupedvalues.SalePrice,data=groupedvalues)
for index, row in groupedvalues.iterrows():
	g.text(row.name,row.SalePrice,round(row.SalePrice,2),color='black',ha='center')


#Blmngtn Crawfor CollgCr ClearCr
#Blueste Mitchel NAmes NPkVill Sawyer SWISU
#NridgHt NoRidge StoneBr
#Somerst Veenker Timber
#BrDale IDOTRR MeadowV
#BrkSide Edwards OldTown
#Gilbert SawyerW NWAmes

len(test['Neighborhood'].unique())
len(df['Neighborhood'].unique())

#decreasing values from 25 to 7 by combining valuesbased on SalePrice median
df['Neighborhood'] = df['Neighborhood'].replace(['Blmngtn','Crawfor','CollgCr','ClearCr'],'19k-20k')
df['Neighborhood'] = df['Neighborhood'].replace(['Blueste','Mitchel','NAmes','NPkVill','Sawyer','SWISU'],'13k-15k')
df['Neighborhood'] = df['Neighborhood'].replace(['NridgHt','NoRidge','StoneBr'],'27k-31k')
df['Neighborhood'] = df['Neighborhood'].replace(['Somerst','Veenker','Timber'],'21k-22k')
#df['Neighborhood'] = df['Neighborhood'].replace(['BrDale','IDOTRR','MeadowV'],'8k-10')
df['Neighborhood'] = df['Neighborhood'].replace(['BrDale','IDOTRR','MeadowV'],'8k-10k')
df['Neighborhood'] = df['Neighborhood'].replace(['BrkSide','Edwards','OldTown'],'11k-12k')
df['Neighborhood'] = df['Neighborhood'].replace(['Gilbert','SawyerW','NWAmes'],'17k-18k')
#----doing same for test
test['Neighborhood'] = test['Neighborhood'].replace(['Blmngtn','Crawfor','CollgCr','ClearCr'],'19k-20k')
test['Neighborhood'] = test['Neighborhood'].replace(['Blueste','Mitchel','NAmes','NPkVill','Sawyer','SWISU'],'13k-15k')
test['Neighborhood'] = test['Neighborhood'].replace(['NridgHt','NoRidge','StoneBr'],'27k-31k')
test['Neighborhood'] = test['Neighborhood'].replace(['Somerst','Veenker','Timber'],'21k-22k')
#test['Neighborhood'] = test['Neighborhood'].replace(['BrDale','IDOTRR','MeadowV'],'8k-10')
test['Neighborhood'] = test['Neighborhood'].replace(['BrDale','IDOTRR','MeadowV'],'8k-10k')
test['Neighborhood'] = test['Neighborhood'].replace(['BrkSide','Edwards','OldTown'],'11k-12k')
test['Neighborhood'] = test['Neighborhood'].replace(['Gilbert','SawyerW','NWAmes'],'17k-18k')

#Do the above for all categorical videos as per the barplot==============================
#currently I am not doing this for all variables and going ahead with dummy variable creation

#====================testing=============================================================
def plotBarSalePrice(dataset,ind_variable,target_variable):
    groupedvalues =dataset.groupby(ind_variable)[target_variable].median().reset_index()
    g=sns.barplot(x=groupedvalues.ind_variable,y=groupedvalues.target_variable,data=groupedvalues)
    for index, row in groupedvalues.iterrows():
        g.text(row.name,row.y,round(row.y,2),color='black',ha='center')
        
        
plotBarSalePrice(df,'Neighborhood','SalePrice')

#=================================================================================
df.shape
test.shape

nc
fc
categorical_features

df.MSSubClass.value_counts().sum()
df_dummy=df.copy()
test_dummy=test.copy()
#df= df_dummy.copy()
#test=test_dummy.copy()
#per Anova test to check relation with target ===============================================================
##################################################################
##################################################################
import statsmodels.api as sm
# ANOVA test
from statsmodels.formula.api import ols

#function
def anovatest(x,y,data):
    model = ols('x~y',data=data).fit()
    anova = sm.stats.anova_lm(model,typ=2)
    pval = anova['PR(>F)'][0]
    
    # return if the feature is significant or not
    if pval < 0.05:
        msg = '{} is significant'.format(x.name)
    else:
        msg = '{} is not significant'.format(x.name)
    
    return(msg)


#do label encoding on fc to do anova test keep the origninal intact
for features in fc:
    ordinal_label = {k:i for i , k in enumerate(df_dummy[features].unique(),0)}
    df_dummy[features] = df_dummy[features].map(ordinal_label)
    
for features in fc:    
    print(df_dummy[features].value_counts())



for features in fc:
    msg=anovatest(df_dummy[features], df_dummy[ 'SalePrice'], data=df_dummy)
    print(msg)
#Drop below features
#Condition1
#MasVnrType
#GarageFinish



for features in categorical_features:
    msg=anovatest(df[features], df[ 'SalePrice'], data=df)
    print(msg)
#MoSold

#===============================================================================
#creating dummy variables 
#combine  train and test dataset since there are extra value in test dataset
final_df=pd.concat([df,test],axis=0)
final_df.shape
'''
for c in fc:
    dummy = pd.get_dummies(final_df[c],drop_first = True , prefix = c)
    final_df = final_df.join(dummy)
'''
final_df.columns    
final_df.shape
#1399 , 211
df1.drop(columns=fc,inplace = True)
#================================================================
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True,prefix = fields)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


final_df=category_onehot_multcols(fc)

final_df =final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape
ndx=final_df['SalePrice'][final_df.SalePrice.isnull()].index
ndx

df_Train=final_df.iloc[:1460,:]
df_Test=final_df.iloc[1460:,:]

#=====working on nc variable =================================================
#plot boxplot and remove outliers-----------------------------------
#function for boxplot
def boxplot(data,c):
     sns.boxplot(x= data[c],data=data)
     plt.show()
     return(1)
#ploting for analysis--------------------------------------------------------------
for n in nc:
  df_nc =df.copy()
  if 0 in df_nc[n].unique():
    pass
  else:
    boxplot(df,n)


plt.figure(figsize=(15,7))
sns.scatterplot(x =df['YearBuilt'],y = df['SalePrice'])
plt.xlabel = 'YearBuilt'
plt.ylabel = 'SalePrice'


#------------------------------------------------------------------------------------------
nc_transform = ['LotArea','LotFrontage','GarageYrBlt','GrLivArea','YearBuilt','SalePrice']


for feature in nc_transform:
  df_Train[feature] = np.log(df_Train[feature])
  #test[feature]= np.log(test[feature])
nc_transform
#remove SalePrice from nc_transorm to log tranform other variable from test
nc_transform = np.delete(nc_transform, np.where(nc_transform == 'SalePrice'))
print(nc_transform)

for feature in nc_transform:
  df_Test[feature]= np.log(df_Test[feature])
  

df['YearBuilt']
#-------log transform done now check for mullticolinearity---------------------


#check for multicolinearity----------------------------------------------------
cormat = df_Train[nc].corr()
cormat = np.tril(cormat)

fig_dims = (15,15)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(cormat,vmin=-1, vmax=1,xticklabels = nc,yticklabels= nc,
            annot = True,square = False, linewidths=1, cmap = plt.cm.CMRmap_r,cbar=False,ax=ax)
plt.show()


#function to store values with mulitcolinearity in array
def correlation(dataset,threshold):
  col_corr = set()
  corr_matrix = dataset.corr()
  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if abs(corr_matrix.iloc[i,j]) > threshold:
        colname = corr_matrix.columns[i]
        col_corr.add(colname)
  return col_corr


corr_feautres = correlation(df_Train[nc],0.75)
corr_feautres
#removing corelated feature
df_Train.drop(columns=['1stFlrSF','GarageYrBlt'],axis=1,inplace=True)
df_Test.drop(columns=['1stFlrSF','GarageYrBlt'],axis=1,inplace=True)
nc = np.delete(nc, np.where(nc == '1stFlrSF'))
nc = np.delete(nc, np.where(nc == 'GarageYrBlt'))

#---------------------------------------------------------------------------------------------------
df_Train.shape
df_Test.shape

#======================================================================================================
#split into train and test

from sklearn.model_selection import train_test_split

def splitData(data,y,test_size1=0.3):
  Xtrain,Xtest,ytrain,ytest = train_test_split(data.drop(columns=y,axis=1),data[y],test_size=test_size1)
  return(Xtrain,Xtest,ytrain,ytest)

Xtrain,Xtest,ytrain,ytest = splitData(df_Train, 'SalePrice')

print(Xtrain.shape,Xtest.shape)

from sklearn.metrics import mean_squared_error
import statsmodels.api as smapi
from statsmodels.formula.api import ols

lm1 = smapi.OLS(ytrain,Xtrain).fit()
lm1.summary()

lm_p1= lm1.predict(Xtest)
lm_mse1 = np.round(mean_squared_error(ytest,lm_p1),2)
lm_rmse1 = np.round(np.sqrt(lm_mse1),2)

def prntMseRmse(model,mse,rmse):
    print( 'Model Name = {} MSE :{} RMSE:{}'.format(model,mse,rmse))
    
prntMseRmse('lm1', lm_mse1, lm_rmse1)

#------------------------Random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

RF1 = RandomForestRegressor().fit(Xtrain,ytrain)
RF1_p1 = RF1.predict(Xtest)
rf_mse1 = np.round(mean_squared_error(ytest,RF1_p1),2)
rf_rmse1 = np.round(np.sqrt(rf_mse1),2)

prntMseRmse('RandomeForest', rf_mse1, rf_rmse1)



RF_ptest =  RF1.predict(df_Test.drop(['SalePrice'],axis=1))
RF_ptest
Test_SalePrice = np.exp(RF_ptest)

import pickle
filename = 'finalized_model.pkl'
pickle.dump(RF1, open(filename, 'wb'))


pred=pd.DataFrame(Test_SalePrice)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)














