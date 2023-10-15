# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Srihariharan S A
RegisterNumber:  212221040160
*/

import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
Initial Data Set: <br>
![initial](https://github.com/hariharan2383/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117346668/99e98ae9-d23c-4afa-b4db-0981a76aad6b) <br>

Data info : <br>
![info](https://github.com/hariharan2383/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117346668/8037d181-44c7-4653-ab14-3f1aa890a022) <br>

Optimization of null values : <br>
![optimize](https://github.com/hariharan2383/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117346668/4fc940f0-1869-436f-955e-eddefca177b1) <br>

Assignment of x value : <br>
![assignment](https://github.com/hariharan2383/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117346668/9204d7be-11dd-44c6-9052-078f5de62d20) <br>

Assignment of y value : <br>
![assignmentY](https://github.com/hariharan2383/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117346668/0109490f-8bac-4ea9-b734-26e466997ce5) <br>

Converting string literals to numerical values using label encoder : <br>
![Convert](https://github.com/hariharan2383/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117346668/42373079-1298-4e65-bb81-3dd785b20f94) <br>

Accuracy: <br>
![accuracy](https://github.com/hariharan2383/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117346668/5efb0e92-5c5a-4649-9b16-12b16d614a4d) <br>

Prediction: <br>
![predict](https://github.com/hariharan2383/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117346668/d43c43b8-1b22-4d7e-894c-70e8e628509f) <br>


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
