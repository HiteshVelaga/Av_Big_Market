# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:53:59 2020

@author: Hitesh
"""

import pandas as pd
train_data=pd.read_csv(r'G:\Hitesh\Av Big Market\train_v9rqX0R.csv')
test_data=pd.read_csv(r'G:\Hitesh\Av Big Market\test_AbJTz2l.csv')

train_data.isnull().sum(axis = 0)
# Item weight has 1463 nan values and outlet size has 2410 nan values
# Filling null with mean and mode
train_data_with_mean=train_data
train_data_with_mean['Item_Weight'].fillna((train_data['Item_Weight'].mean()),inplace=True)
train_data_with_mean['Outlet_Size'].fillna(('Medium'),inplace=True)
train_data_with_mean.isnull().sum(axis = 0)


# Preprocessing Item_Fat_Content
train_data_with_mean.Item_Fat_Content.unique()

train_data_with_mean['Item_Fat_Content']=train_data.Item_Fat_Content.apply(lambda x:1 if x=='reg' or x=='Regular' else 0)

train_data_with_mean.Item_Fat_Content.corr(train_data_with_mean.Item_Outlet_Sales)
#corr is 0.018 which is +ive so as fat content raises cost raises

train_data_with_mean.Item_Visibility.corr(train_data_with_mean.Item_Outlet_Sales)
#has -ve correlation

# checking for unique values and their counts
train_data_with_mean.Item_Type.unique()
train_data_with_mean.Item_Type.value_counts() 
train_data_with_mean.Outlet_Type.value_counts() 
train_data_with_mean.Outlet_Location_Type.value_counts() 
train_data_with_mean.Outlet_Identifier.value_counts() 

# Preprocessing outlet_Identifier,item_Type
from sklearn import preprocessing
item_Type=preprocessing.LabelEncoder()
outlet_Identifier=preprocessing.LabelEncoder()
train_data_with_mean.Item_Type=item_Type.fit_transform(train_data_with_mean.Item_Type)
train_data_with_mean.Outlet_Identifier=outlet_Identifier.fit_transform(train_data_with_mean.Outlet_Identifier)

# Preprocessing outlet_Size
train_data_with_mean['Outlet_Size']=train_data.Outlet_Size.apply(lambda x:0 if x=='Small' else (1 if x=='Medium' else 2))
train_data_with_mean.Outlet_Size.corr(train_data_with_mean.Item_Outlet_Sales)
#there is a +ve corr of 0.086 as size increases pirce increases

# Preprocessing Outlet_Location_Type
train_data_with_mean.Outlet_Location_Type=train_data_with_mean.Outlet_Location_Type.apply(lambda x:x.split()[1])

train_data_with_mean['Outlet_Location_Type']=train_data_with_mean['Outlet_Location_Type'].astype(int)
train_data_with_mean.Outlet_Location_Type.corr(train_data_with_mean.Item_Outlet_Sales)
#there is a +ve corr of 0.089 as loc changes pirce increases
# Preprocessing Establishment_Year
train_data_with_mean.Outlet_Establishment_Year=train_data_with_mean.Outlet_Establishment_Year.apply(lambda x: 2020-x)

train_data_with_mean.Outlet_Establishment_Year.corr(train_data_with_mean.Item_Outlet_Sales)

#Dropping unnecessary columns
train_data_with_mean.drop(['Item_Identifier','Outlet_Type'],axis=1,inplace=True)

#checking corr btw item mrp and outlet sales
train_data_with_mean.Item_MRP.corr(train_data_with_mean.Item_Outlet_Sales)
# they have highest corr of 0.57

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

X=train_data_with_mean[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type']]
Y=train_data_with_mean[['Item_Outlet_Sales']]
sc=StandardScaler()
sc_y=StandardScaler()
X=sc.fit_transform(X)
Y=sc_y.fit_transform(Y)
train_data_with_mean[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type']]=X
columnTransformer=ColumnTransformer(transformers=[('cat', OneHotEncoder(), [3, 5])],remainder='passthrough')
eg=train_data_with_mean

train_data_with_mean=columnTransformer.fit_transform(train_data_with_mean).toarray()
y=sc_y.fit_transform(train_data[['Item_Outlet_Sales']]).ravel()
train_data.drop(['Item_Outlet_Sales'],axis=1,inplace=True)
temp=train_data
train_data=columnTransformer.fit_transform(train_data).toarray()  


from sklearn.model_selection import train_test_split,GridSearchCV
X_train,X_test,Y_train,Y_test=train_test_split(train_data_with_mean[:,:-1],train_data_with_mean[:,-1],test_size=0.12)


from sklearn.ensemble import RandomForestRegressor
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(train_data_with_mean[:,:-1],train_data_with_mean[:,-1])

grid_search.best_params_


forest=RandomForestRegressor(bootstrap= True,
 max_depth= 90,
 max_features= 3,
 min_samples_leaf= 3,
 min_samples_split= 8,
 n_estimators= 300)

test_data.drop(['Item_Identifier','Outlet_Type'],axis=1,inplace=True)
test_data.isnull().sum(axis = 0)
test_data['Item_Weight'].fillna((test_data['Item_Weight'].mean()),inplace=True)
test_data['Outlet_Size'].fillna(('Medium'),inplace=True)



# Preprocessing Item_Fat_Content
test_data.Item_Fat_Content.unique()
test_data['Item_Fat_Content']=test_data.Item_Fat_Content.apply(lambda x:1 if x=='reg' or x=='Regular' else 0)

train_data_with_mean.Item_Fat_Content.corr(train_data_with_mean.Item_Outlet_Sales)
#corr is 0.018 which is +ive so as fat content raises cost raises

train_data_with_mean.Item_Visibility.corr(train_data_with_mean.Item_Outlet_Sales)
#has -ve correlation

# checking for unique values and their counts
train_data_with_mean.Item_Type.unique()
train_data_with_mean.Item_Type.value_counts() 
test_data.Outlet_Size.value_counts() 
train_data_with_mean.Outlet_Location_Type.value_counts() 
train_data_with_mean.Outlet_Identifier.value_counts() 

test_data.Item_Type=item_Type.transform(test_data.Item_Type)
test_data.Outlet_Identifier=outlet_Identifier.transform(test_data.Outlet_Identifier)

test_data['Outlet_Size']=test_data.Outlet_Size.apply(lambda x:0 if x=='Small' else (1 if x=='Medium' else 2))

test_data.Outlet_Location_Type=test_data.Outlet_Location_Type.apply(lambda x:int(x.split()[1]))

test_data.Outlet_Establishment_Year=test_data.Outlet_Establishment_Year.apply(lambda x: 2020-x)

X=test_data[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type']]
X=sc.transform(X)
test_data[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type']]=X
test_data=columnTransformer.transform(test_data).toarray()

forest.fit(train_data,y)
pred=forest.predict(test_data)
predictions=sc_y.inverse_transform(pred) 


sol=pd.read_csv(r'G:\Hitesh\Av Big Market\sample_submission_8RXa3c6.csv')
sol.Item_Outlet_Sales=predictions

sol.to_csv(r'G:\Hitesh\Av Big Market\sol.csv',index=False)
