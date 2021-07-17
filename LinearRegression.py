#!/usr/bin/env python
# coding: utf-8

# # 1. Import Packages

# In[1]:


#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # 2. Load Data

# In[2]:


path='F:\Munesh Backup\Munesh\Imarticus\Projects\Linear Regression/train.csv'
data=pd.read_csv(path)


# # 3. Data Preparation

# In[3]:


# checking data types for variables in HousePrice dataframe
data.info()
data.shape


# # 3.1 Statistical Summary

# In[4]:


#dataframe with categorical object
data.describe(include='object')


# In[5]:


#dataframe with numrical features
data.describe(exclude='object')


# ## 3.2 Splitting Target Variables

# In[6]:


target=data['SalePrice']
target.head()


# In[7]:


#visualizing the distribution of Saleprice(dependent Variable) 
import seaborn as sns
sns.distplot(target,hist=True)


# In[8]:


#log Transformation
target_log=np.log(target)
sns.distplot(target_log,hist=True)


# In[9]:


# drop target variable from dataset
raw_data=data.copy()
data=data.drop('SalePrice',axis=1)
data.head()


# ## 3.3 Feature Engineering

# In[10]:


#MSSubClass=The building class
data['MSSubClass'] = data['MSSubClass'].apply(str)
#Changing OverallCond into a categorical variable
data['OverallCond'] = data['OverallCond'].astype(str)
#Year and month sold are transformed into categorical features.
data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)


# In[11]:


# Adding total sqfootage feature
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
# Removing TotalBsmtSF,1stFlrSF, 2ndFlrSF and Id
data = data.drop(["TotalBsmtSF"], axis=1)
data = data.drop(["1stFlrSF"], axis=1)
data = data.drop(["2ndFlrSF"], axis=1)
data = data.drop(["Id"], axis=1)
data.head()


# ## 3.4 Split Dataframe into numeric and categorical

# In[12]:


# save all categorical columns in list
categorical_columns = [col for col in data.columns.values if data[col].dtype == 'object']
# dataframe with categorical features
data_cat = data[categorical_columns]
# dataframe with numerical features
data_num = data.drop(categorical_columns, axis=1)
# Using describe function in numeric dataframe
data_num.describe()


# ## 3.5 Reduce Skewness for Numerical Features

# In[13]:


data_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


# In[14]:


from scipy.stats import skew
data_num_skew = data_num.apply(lambda x: skew(x.dropna()))
data_num_skew = data_num_skew[data_num_skew > .75]
# apply log + 1 transformation for all numeric features with skewnes over .75
data_num[data_num_skew.index] = np.log1p(data_num[data_num_skew.index])
data_num_skew


# In[15]:


data_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


# ## 3.6 Mean Normalization

# In[16]:


data_num = ((data_num - data_num.mean())/(data_num.max() - data_num.min()))
data_num.describe()


# In[17]:


data_num.hist(figsize=(16, 20),xlabelsize=8, ylabelsize=8);


# # 4 Missing Data analysis

# In[18]:


# first we'll visualize null count in overall dataframe
null_in_HousePrice = data.isnull().sum()
null_in_HousePrice = null_in_HousePrice[null_in_HousePrice > 0]
null_in_HousePrice.sort_values(inplace=True)
null_in_HousePrice.plot.bar()


# In[19]:


# Printing total Number of Missing Data
total=data.isnull().sum().sort_values(ascending=False)
percent=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.head(15)


# In[ ]:





# # 5. Missing Data treatment
# ## 5.1 Handling Missing Values in Numerical Count

# In[20]:


data_len = data_num.shape[0]
# check what is percentage of missing values in categorical dataframe
for col in data_num.columns.values:
    missing_values = data_num[col].isnull().sum()
    #print("{} - missing values: {} ({:0.2f}%)".format(col, missing_values, missing_values/data_len*100))
    # drop column if there is more than 50 missing values
    if missing_values > 260:
        #print("droping column: {}".format(col))
        data_num = data_num.drop(col, axis = 1)
        # if there is less than 260 missing values than fill in with median valu of column
    else:
        #print("filling missing values with median in column: {}".format(col))
        data_num = data_num.fillna(data_num[col].median())
        


# ## 5.2 Handling Missing values in categorical columns

# In[21]:


data_len = data_cat.shape[0]
# check what is percentage of missing values in categorical dataframe
for col in data_cat.columns.values:
    missing_values = data_cat[col].isnull().sum()
    print("{} - missing values: {} ({:0.2f}%)".format(col, missing_values, missing_values/data_len*100))
    # drop column if there is more than 50 missing values
    if missing_values > 50:
        print("droping column: {}".format(col))
        data_cat.drop(col, axis = 1)
    # if there is less than 50 missing values than fill in with mode value of column
    else:
        print("filling missing values with mode in column: {}".format(col))
        data_cat = data_cat.fillna(data_cat[col].mode())


# In[22]:


data_cat.describe()


# # 6. Dummy Coding for Categorical Varaible

# In[23]:


data_cat.columns


# In[24]:


# Using pandas.get_dummies function to Convert categorical variable into dummy/indicator variables
data_cat_dummies=pd.get_dummies(data_cat,drop_first=True)
# Viewing dimensionality of the DataFrame.
data_cat_dummies.head()


# In[25]:


#using concat function we merging two dataframe for furthere analysis
newdata = pd.concat([data_num, data_cat_dummies], axis=1)


# In[26]:


sns.factorplot("Fireplaces","SalePrice",data=raw_data,hue="FireplaceQu")


# In[27]:


# If fireplace is missing that means that house doesn't have a FireplaceQu
FireplaceQu = raw_data["FireplaceQu"].fillna('None')
pd.crosstab(raw_data.Fireplaces, raw_data.FireplaceQu)    


# In[28]:


sns.barplot(raw_data.OverallQual,raw_data.SalePrice)


# In[29]:


# MSZoning
labels = raw_data["MSZoning"].unique()
sizes = raw_data["MSZoning"].value_counts().values
explode=[0.1,0,0,0,0]
parcent = 100.*sizes/sizes.sum()
labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, parcent)]
colors = ['yellowgreen', 'gold', 'lightblue', 'lightcoral','blue']
patches, texts= plt.pie(sizes, colors=colors,explode=explode,
shadow=True,startangle=90)
plt.legend(patches, labels, loc="best")
plt.title("Zoning Classification")
plt.show()
sns.violinplot(raw_data.MSZoning,raw_data["SalePrice"])
plt.title("MSZoning wrt Sale Price")
plt.xlabel("MSZoning")
plt.ylabel("Sale Price");


# In[30]:


# SalePrice per Square Foot
SalePriceSF = raw_data['SalePrice']/raw_data['GrLivArea']
plt.hist(SalePriceSF, color="green")
plt.title("Sale Price per Square Foot")
plt.ylabel('Number of Sales')
plt.xlabel('Price per square feet');


# In[31]:


ConstructionAge = raw_data['YrSold'] - raw_data['YearBuilt']
plt.scatter(ConstructionAge, SalePriceSF)
plt.ylabel('Price per square foot (in dollars)')
plt.xlabel("Construction Age of house");


# In[32]:


# Heating and AC arrangements
sns.stripplot(x="HeatingQC", y="SalePrice",data=raw_data,hue='CentralAir',jitter=True,split=True)
plt.title("Sale Price vs Heating Quality");


# In[33]:


sns.boxplot(raw_data["FullBath"],raw_data["SalePrice"])
plt.title("Sale Price vs Full Bathrooms");


# In[34]:


# Kitchen Quality
sns.factorplot("KitchenAbvGr","SalePrice",data=raw_data,hue="KitchenQual")
plt.title("Sale Price vs Kitchen");


# ## 7.1 Correlation

# In[35]:


# Check Correlation
data_num.corr()


# ## 7.2 Correlation Plot

# In[36]:


# Plotting Correlation Plot
corr=data_num.corr()
plt.figure(figsize=(30,30))
sns.heatmap(corr[(corr>=0.5) | (corr<=-0.5)],cmap='YlGnBu',vmax=1,vmin=-1,linewidths=0.1,
           annot=True,annot_kws={"size":8},square=True)
plt.title('Correlation Between Figures')


# # 8 Linear Regression Modelling

# ## 8.1 Preparation of Dataset

# In[37]:


# Split the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(newdata,target_log,test_size=0.30,random_state=0)
print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)


# ## 8.2 Building a linear Regression Base Model

# In[38]:


#Building Linear Regression model using stats model
import statsmodels.api as sm

#Building linear Regression model using OLS
model1=sm.OLS(y_train,x_train).fit()

#Printing linear Regressio summary
model1.summary()


# In[39]:


def rmse(predictions,targets):
    differences=predictions-targets
    differences_squared = differences ** 2 
    mean_of_differences_squared = differences_squared.mean() 
    rmse_val = np.sqrt(mean_of_differences_squared) 
    return rmse_val


# In[40]:


cols = ['Model', 'R-Squared Value', 'Adj.R-Squared Value', 'RMSE']
models_report = pd.DataFrame(columns = cols)
# Predicting the model on test data
predictions1 = model1.predict(x_test)


# In[41]:


tmp1=pd.Series({'Model':"Base Linear Regression Model",
              'R-squared Value': model1.rsquared,
              'Adj. R-squared Value':model1.rsquared_adj,
              'RMSE':rmse(predictions1,y_test)})

model1_report=models_report.append(tmp1,ignore_index = True)
model1_report


# ## 8.3 Building Model with constant

# In[42]:


df_constant=sm.add_constant(newdata)
x_train1,x_test1, y_train1, y_test1 = train_test_split(df_constant, target_log, test_size = 0.30, random_state=0)

#building model with constant
model2 = sm.OLS(y_train1, x_train1).fit()
model2.summary2()


# In[43]:


# Predicting the model on test data
predictions2= model2.predict(x_test1)
tmp2=pd.Series({'Model':"Base Linear Regression Model",
              'R-squared Value': model2.rsquared,
              'Adj. R-squared Value':model2.rsquared_adj,
              'RMSE':rmse(predictions2,y_test)})

model2_report=models_report.append(tmp2,ignore_index = True)
model2_report


# ## 8.4 Calculating Variance Inflation factor

# In[44]:


print ("\nVariance Inflation Factor")
cnames = x_train1.columns
for i in np.arange(0,len(cnames)):
    xvars = list(cnames)
    yvar = xvars.pop(i)
    mod = sm.OLS(x_train1[yvar],(x_train1[xvars]))
    res = mod.fit()
    vif = 1/(1-res.rsquared)
    print (yvar,round(vif,3))


# In[45]:


vif_100 = ['MSSubClass_20','MSSubClass_60','RoofStyle_Gable','RoofStyle_Hip','RoofMatl_CompShg',
           'Exterior1st_MetalSd','Exterior1st_VinylSd','Exterior2nd_VinylSd','GarageQual_TA','GarageCond_TA']
# custom function to remove variables having higer VIF
to_keep = [x for x in x_train1 if x not in vif_100]
# print(to_keep)
x_train2 = x_train1[to_keep]
x_train2.head()


# ### 8.4.1 Building Model after removing VIF above 100

# In[46]:


model3=sm.OLS(y_train1,x_train2).fit()
model3.summary()


# In[47]:


vif_100 = ['MSSubClass_20','MSSubClass_60','RoofStyle_Gable','RoofStyle_Hip','RoofMatl_CompShg',
           'Exterior1st_MetalSd','Exterior1st_VinylSd','Exterior2nd_VinylSd','GarageQual_TA','GarageCond_TA']
# custom function to remove variables having higer VIF
to_keep = [x for x in x_test1 if x not in vif_100]
# print(to_keep)
x_test2 = x_test1[to_keep]
x_test2.head()


# In[48]:


predictions3 = model3.predict(x_test2)
tmp3 = pd.Series({'Model': " LRM after removing VIF above 100",'R-Squared Value' : model3.rsquared,
                  'Adj.R-Squared Value': model3.rsquared_adj,'RMSE': rmse(predictions3, y_test1)})
model3_report = models_report.append(tmp3, ignore_index = True)
model3_report


# In[49]:


# Removing variable having threshold value of above 10
print("\nVariance Inflation Factor")
cnames=x_train2.columns
for i in np.arange(0,len(cnames)):
    xvars=list(cnames)
    yvar=xvars.pop(i)
    mod=sm.OLS(x_train2[yvar],x_train2[xvars])
    res=mod.fit()
    vif=1/(1-res.rsquared)
    print(yvar,round(vif,3))


# In[50]:


VIF_10=VIF_10 = ['MSSubClass_20','MSSubClass_60','MSSubClass_90','YearBuilt','MasVnrArea','BsmtFinSF1',
                 'BsmtFinSF2','GrLivArea','GarageYrBlt','MiscVal','TotalSF','MSSubClass_190','MSSubClass_45'
                 ,'Neighborhood_Gilbert','Neighborhood_IDOTRR','MSSubClass_50','MSSubClass_80',
                 'MSZoning_FV','MSZoning_RL','MSZoning_RM','Neighborhood_BrkSide','Neighborhood_CollgCr',
                 'Neighborhood_Edwards', 'Neighborhood_NAmes','Neighborhood_OldTown','Neighborhood_Sawyer',
                 'Neighborhood_Somerst','Condition2_Norm','HouseStyle_1.5Unf','HouseStyle_2Story',
                 'HouseStyle_SLvl','Neighborhood_NWAmes', 'Condition2_Feedr','BldgType_2fmCon',
                 'Foundation_PConc','KitchenQual_TA','HouseStyle_SFoyer','MasVnrType_BrkFace','HouseStyle_1Story',
                 'Exterior1st_CemntBd','Exterior1st_HdBoard','Exterior1st_Plywood','Exterior1st_Wd Sdng',
                 'Exterior2nd_CmentBd','Exterior2nd_HdBoard','Exterior2nd_Plywood','Exterior2nd_Wd Sdng',
                 'MasVnrType_None','MasVnrType_Stone', 'ExterQual_Gd','ExterQual_TA','ExterCond_Fa','ExterCond_Gd',
                 'ExterCond_TA','BsmtQual_TA','BsmtFinType1_Unf','BsmtFinType2_Unf','Heating_GasA','Heating_GasW',
                 'Heating_Grav','GarageType_BuiltIn','SaleType_New','SaleCondition_Partial','GarageType_Attchd',
                 'GarageType_Detchd','MiscFeature_Shed','Functional_Typ']
to_keep=[x for x in x_train2 if x not in VIF_10]
x_train2=x_train2[to_keep]
x_train2.head()


# ### 8.4.2 Building model after removing VIF above 10

# In[51]:


model4=sm.OLS(y_train1,x_train2).fit()
model4.summary()


# In[52]:


x_test2=x_test2[to_keep]
x_test2.head()


# In[53]:


predictions4=model4.predict(x_test2)
tmp4=pd.Series({'Model':"LRM after removing VIF above 10",
               'R squared value':model4.rsquared,
               'Adj. R squared value':model4.rsquared_adj,
               'RMSE':rmse(predictions4,y_test1)})

model4_report=models_report.append(tmp4,ignore_index=True)
model4_report


# In[54]:


# Removing variable has threshold value of VIF above 5
print ("\nVariance Inflation Factor")
cnames = x_train2.columns
for i in np.arange(0,len(cnames)):
    xvars = list(cnames)
    yvar = xvars.pop(i)
    mod = sm.OLS(x_train2[yvar],(x_train2[xvars]))
    res = mod.fit()
    vif = 1/(1-res.rsquared)
    print (yvar,round(vif,3))


# In[55]:


VIF_5 = ['LotArea','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','PoolArea','MSSubClass_75',
         'RoofStyle_Shed','BsmtCond_TA','FireplaceQu_TA','PoolQC_Gd' ,'Condition1_Norm','MoSold_6','MoSold_7']
to_keep = [x for x in x_train2 if x not in VIF_5]
#print(to_keep)
x_train2 = x_train2[to_keep]
x_train2.head()


# ### 8.4.3 Building Model after removing VIF above 5

# In[56]:


model5 = sm.OLS(y_train1,x_train2).fit()
model5.summary()


# In[57]:


x_test2 = x_test2[to_keep]
x_test2.head()


# In[58]:


predictions5 = model5.predict(x_test2)
tmp5 = pd.Series({'Model': "LRM after removing VIF above 5",
                  'R-Squared Value' : model5.rsquared,
                  'Adj.R-Squared Value': model5.rsquared_adj,
                  'RMSE': rmse(predictions5, y_test1)})
model5_report = models_report.append(tmp5, ignore_index = True)
model5_report


# ## 8.5 Removing variables based on Insignificant Variables using P-value

# In[63]:


X=x_train2
Y=y_train1
X.info()


# In[74]:


initial_list=[]
threshold_in=0.05
threshold_out = 0.05
verbose=True
""" Perform a forward-backward feature selection
based on p-value from statsmodels.api.OLS
Arguments:
    X - pandas.DataFrame with candidate features
    y - list-like with the target
    initial_list - list of features to start with (column names of X)
    threshold_in - include a feature if its p-value < threshold_in
    threshold_out - exclude a feature if its p-value > threshold_out
    verbose - whether to print the sequence of inclusions and exclusions
Returns: list of selected features
Always set threshold_in < threshold_out to avoid infinite looping.
See https://en.wikipedia.org/wiki/Stepwise_regression for the details
"""
included = list(initial_list)
while True:
    changed=False
    # forward step
    excluded = list(set(X.columns)-set(included))
    new_pval = pd.Series(index=excluded)
    for new_column in excluded:
        model = sm.OLS(Y, sm.add_constant((X[included+[new_column]]))).fit()
        new_pval[new_column] = model.pvalues[new_column]
    best_pval = new_pval.min()
    if best_pval < threshold_in:
        best_feature = new_pval.argmin()
        included.append(best_feature)
        changed=True
        if verbose:
            print('Add {:30} with p-value {:.6}'.format(best_feature, best_pval))
            
    # backward step
    model = sm.OLS(Y, sm.add_constant(pd.DataFrame(X[included]))).fit()
    # use all coefs except intercept
    pvalues = model.pvalues.iloc[1:]
    worst_pval = pvalues.max() # null if pvalues is empty
    if worst_pval > threshold_out:
        changed=True
        worst_feature = pvalues.argmax()
        included.remove(worst_feature)
        if verbose:
            print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
    if not changed:
        break



# In[ ]:
included



# In[ ]:





# In[ ]:





# In[ ]:




