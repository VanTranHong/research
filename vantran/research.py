import pandas as pd
import numpy as np
import math
import csv
import sklearn
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler


#reading files and removing rows or columns with too many missing places
datafile = r'/Users/vantran/Documents/vantran/coverted.csv'
def checkcategory(column):
    for i in column.values:
        if i!=0 and i!= 1 and i!= 'NaN':
            return False
    return True
        
data_consider = ['Hpyloriantigen','Hpyloriantibody','age','sex','maternaleducation','maternaloccupation','historyofvaccination','breastfeeding','dewormingmedicationinthelast6month','howmanypeoplelivinginyourhome','cigarettesmokersinthehouse','numberofsmokersinthehouse','typeofroof','wallsaremadeof','floortypeofthehouse','floorcoveredbyamaterial','wheredoyoucook','howoftenusecharcoalforcooking','howoftenusewoodforcooking','howoftenuseleavesforcooking','howoftenusedungforcooking','howoftenusenaftaorlambaforcooking','howoftenusegasforcooking','howoftenuseelectricityforcooking','mainsourceofwater','typeoftoilet','cat','dog','hen','coworox','sheep','horse','pig','goat','muleordonkey']   
min_max_scaler = preprocessing.MinMaxScaler()
data = pd.read_csv(datafile,skipinitialspace=True, header = 0)
print(data.shape)

data = data.loc[data['Hpyloriantigen'].isnull()==False]
data = data.loc[data['Hpyloriantibody'].isnull()==False]
data = data[data_consider]
data = data[data.columns[data.isnull().mean()<0.05]] #excluding columns with too many missing data
data = data.select_dtypes(exclude=['object']) #excluding columns with wrong data type

print(data.shape)




for column in data.columns:
    if checkcategory(data[column])==True:
        count1 = data[column].value_counts()[1]
        
        count0 = data[column].value_counts()[0]
       
        
        data[column].fillna(random.random()*(count1+count0)>count1, inplace = True)
    else:
        thesum = 0
        thecount = 0
        col = data[column]
        copy = col.copy()
        thecount = len(copy.values)-copy.isna().sum()
        copy.fillna(0, inplace = True)
        thesum = copy.sum() 
        ave = thesum/thecount  
        if col.isna().sum()>0:
            col.fillna(ave, inplace = True)
        col = (col-col.min())/(col.max()-col.min())
      

kf = KFold(n_splits=10)
print('antibody', data['Hpyloriantibody'].sum())

C = [10,1,.1,0.01]
for c in C:
    sumtestaccuracy =0
    sumtrainaccuracy = 0
    coef = 0
    
    
    for train, test in kf.split(data):
        train_data = data.iloc[train,:]
        test_data = data.iloc[test,:]
        x_train = train_data.drop(labels = ['Hpyloriantibody','Hpyloriantigen'], axis = 1)
        x_test = test_data.drop(labels = ['Hpyloriantibody','Hpyloriantigen'], axis = 1)
        y_train = train_data['Hpyloriantibody']
        y_test = test_data['Hpyloriantibody']
        
        
        sc = StandardScaler()

        # Fit the scaler to the training data and transform
        X_train_std = sc.fit_transform(x_train)

        # Apply the scaler to the test data
        X_test_std = sc.transform(x_test)
        clf =  LogisticRegression(penalty = 'l1', C=c, solver = 'liblinear')
        clf.fit(x_train,y_train)
        coef =  clf.coef_
        sumtrainaccuracy +=clf.score(X_train_std, y_train)
        sumtestaccuracy+= clf.score(X_test_std, y_test)
    print('C:', c)
    print('Coefficient of each feature:', coef)
    print('Training accuracy:', sumtrainaccuracy/10)
    print('Test accuracy:', sumtestaccuracy/10)
        
        
    
    














'''       

losreg.fit(X=x_train, y=y_train)
predictions = linreg.predict(X=x_test)
error = predictions-y_test
rmse = np.sqrt((np.sum(error**2)/len(x_test)))
coefs = linreg.coef_
features = x_train.columns
'''


'''
#regularization
alphas = np.linspace(0.0001, 1,100)
rmse_list = []
best_alpha = 0

for a in alphas:
    lasso = Lasso(fit_intercept = True, alpha = a, max_iter= 10000 )
    
    kf = KFold(n_splits=10)
    xval_err =0
    
    
    
    
    for train_index, validation_index in kf.split(x_train):
    
        lasso.fit(x_train.loc[train_index,:], y_train[train_index])
      
        p = lasso.predict(x_train.iloc[validation_index,:])
        err = p-y_train[validation_index]
        xval_err = xval_err+np.dot(err,err)
        rmse_10cv = np.sqrt(xval_err/len(x_train))
        rmse_list.append(rmse_10cv)
        best_alpha = alphas[rmse_list==min(rmse_list)]
      
        
#using the alpha calculated to calculate accuracy of the test
lasso = Lasso(fit_intercept = True, alpha = best_alpha)
lasso.fit(x_train, y_train)
predictionsOnTestdata_lasso = lasso.predict(x_test)
predictionErrorOnTestData_lasso = predictionErrorOnTestData_lasso-y_test
rmse_lasso = np.sqrt(np.dot(predictionErrorOnTestData_lasso,predictionErrorOnTestData_lasso)/len(predictionErrorOnTestData_lasso))
print(list(zip(x_train.columns, lasso.coef_)))
'''



        
        










    
