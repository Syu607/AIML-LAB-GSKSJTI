import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split

data = pd.read_csv('tennis.csv')

X = data.iloc[:, :-1]

Y = data.iloc[:, -1]

obj1= LabelEncoder()
X.Outlook = obj1.fit_transform(X.Outlook)

obj2 = LabelEncoder()
X.Temperature = obj2.fit_transform(X.Temperature)

obj3 = LabelEncoder()
X.Humidity = obj3.fit_transform(X.Humidity)

obj4 = LabelEncoder()
X.Wind = obj4.fit_transform(X.Wind)

obj5 = LabelEncoder()
Y = obj5.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20)

from sklearn.naive_bayes import GaussianNB 
classifier = GaussianNB() 
classifier.fit(X_train, Y_train)
from sklearn.metrics import accuracy_score
print("Accuracy is:", accuracy_score(classifier.predict(X_test), Y_test))
