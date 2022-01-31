import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('salaries.csv')
df.head()

# split inputs and target
inputs = df.drop('salary_more_then_100k',axis='columns')
target = df['salary_more_then_100k']

# encode categorical cols with nums
le = LabelEncoder()
inputs_n = pd.DataFrame()
inputs_n['company'] = le.fit_transform(inputs['company'])
inputs_n['job'] = le.fit_transform(inputs['job'])
inputs_n['degree'] = le.fit_transform(inputs['degree'])

# create decision tree classifier
model = DecisionTreeClassifier()
model.fit(inputs_n, target)

# predict salary above 100k for google, sales exec, bachelor
model.predict([[2,2,1]])

