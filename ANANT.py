import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    'CGPA':[8.5,7.2,9.0,6.5,8.0],
    'Skills':[4,3,5,2,4],
    'Internship':[1,0,1,0,1],
    'Placed':[1,0,1,0,1]
}

df = pd.DataFrame(data)

X = df[['CGPA','Skills','Internship']]
y = df['Placed']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression()
model.fit(X_train,y_train)

print("Prediction:", model.predict([[8.2,4,1]]))