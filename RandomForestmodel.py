import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate


data = pd.read_csv("RicohPrinter-Preprocessed-Dataset.csv", encoding="utf8", dtype='unicode')
columns = ['Branded_Model_Name','SR_Number','SR_Cause_Code','SR_Symptom_Code','SR_Customer_Description','SR_CREATION_MONTH','res']
df = pd.DataFrame(data, columns)
df.astype('category')
#df = dataframe.drop(dataframe.index[0])

#df = pd.get_dummies(df)

print(df.head(5))

# Now we have a full prediction pipeline.
clf = RandomForestClassifier(criterion='gini', max_depth=20, min_samples_split=2, random_state=2)

y = df['res']
X = df.drop('res', axis=1)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7)
clf.fit(X_train, y_train)

print("model score: %.3f" % clf.score(X_test, y_test))
print("model score: " % cross_validate(clf, X_test, y_test))


from sklearn.metrics import confusion_matrix
y_prediction = clf.predict(X_test)
cm = confusion_matrix(y_test, y_prediction)
print(cm)


import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(cm, center=True)
plt.show()

