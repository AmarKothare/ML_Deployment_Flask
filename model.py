# importing all the necessary libraries

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("iris.csv")

#display first 5 rows of the dataset
print(df.head())

#Seperating the Independent(X) and dependent variables(y)
X=df[["Sepal_Length","Sepal_Width","Petal_Length","Petal_Width"]]
y=df[["Class"]]

# Splitting the dataset into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=50)

# Fearure Scaling
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_test)

# Intantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train,y_train)

# Make a Pickle file of our model"
pickle.dump(classifier,open("model.pkl","wb"))