# Simple Linear Regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Importing the dataset
dataset = pd.read_csv(r'C:\Users\HP\Desktop\Study\Data Science\Database For Practice\insurance.csv')
#print(dataset['region'].value_counts())


# plotting the data
dataset.hist(figsize=(20,15))

print("correlation:\n",dataset.corr())

plt.figure(figsize = (15,10))
sns.heatmap(dataset.corr(), annot = True, cmap = 'YlGnBu')


# Encoding the data

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.preprocessing import LabelEncoder
gender = LabelEncoder()
X["gender"] = gender.fit_transform(X["gender"])

smoker=LabelEncoder()
X["smoker"]=smoker.fit_transform(X["smoker"])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[("Encoding",OneHotEncoder(),[5])],remainder="passthrough")
X=ct.fit_transform(X)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X=sc.fit_transform(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(X_train,y_train)


y_pred=regressor.predict(X_test)

# testing the accuracy of the model
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))