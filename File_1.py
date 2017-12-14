import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

#Importing the data and bifucating columns
data = pd.read_csv('HR_comma_sep.csv')

labencode = LabelEncoder()
data['sales'] = labencode.fit_transform(data['sales'])
data['salary'] = labencode.fit_transform(data['salary'])

X = data.iloc[:,:9].values   #X is a matrix here
y = data.iloc[:,9].values   #y is a vector here and not a matrix

#print(len(data))

#print(data['salary'].head())


#Feature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
#y not scaled as it is categorical

#Partioning the dataset into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0,random_state=0)

#Logistic Regression on Training set
classi = LogisticRegression(random_state=0)
classi.fit(X_train,y_train)

#Prediction on Test set
y_pred = classi.predict(X_train)

#Confusion Matrix
cm = confusion_matrix(y_train,y_pred)

print(cm)

correctpercen = (cm[0][0] + cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100
print(correctpercen)