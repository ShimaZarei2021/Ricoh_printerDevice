import keras
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

data = pd.read_csv("data.csv", encoding="utf8", dtype='unicode')
columns = ['SR_Customer_Description', 'Branded_Model_Name', 'Activity_Id', 'Assigned_To', 'Activity_Status', 'Activity_Code', 'SR_Number','SR_Status', 'Activity_Sub_Status', 'SR_Cause_Code', 'SR_Symptom_Code',  'Activity__Type',]
dataframe = pd.DataFrame(data, columns=['SR_Customer_Description', 'Branded_Model_Name', 'Activity_Id', 'Assigned_To', 'Activity_Status', 'Activity_Code', 'SR_Number','SR_Status', 'Activity_Sub_Status', 'SR_Cause_Code', 'SR_Symptom_Code', 'Activity__Type']
)
df = dataframe.drop(dataframe.index[0])

df.fillna(value=0, inplace=True)

print(df.isnull().sum())

df = pd.DataFrame(df, columns=['SR_Customer_Description', 'Branded_Model_Name', 'Activity_Id', 'Assigned_To', 'Activity_Status', 'Activity_Code', 'SR_Number','SR_Status', 'Activity_Sub_Status', 'SR_Cause_Code', 'SR_Symptom_Code','Activity__Type' ])
if (df['Activity__Type'].count()>1):
   df.loc[(df.Activity__Type=='Inbound Call') | (df.Activity__Type=='Help Desk') | (df.Activity__Type=='Field Repair'), 'AT']='Field Repair'
   df.loc[(df.Activity__Type=='Inbound Call') | (df.Activity__Type=='Help Desk') , 'AT']='Help Desk'
   df.loc[(df.Activity__Type=='Inbound Call') | (df.Activity__Type=='Field Repair') , 'AT']='Field Repair'
   df.loc[(df.Activity__Type == 'Help Desk') | (df.Activity__Type == 'Field Repair') , 'AT']='Field Repair'

print(df.head(5))
print(df.shape)
df = df.astype('category')
df = pd.get_dummies(df)
X = df.iloc[:, 5:10].values
y = df.iloc[:, 12].values

'''
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
labelencoder_X_5 = LabelEncoder()
X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5])

print(X[:, 1])
#X = pd.DataFrame(X)
X = pd.get_dummies(X[:, 1])
'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#print(X_train, X_test)

#Initializing Neural Network
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 5))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 5))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid', input_dim = 5))
# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting our model
classifier.fit(X_train, y_train, batch_size = 2500, epochs = 3000)

# Predicting the Test set results
y_pred = classifier.predict_proba(X_test)
y_pred = (y_pred > 0)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(cm, center=True)
plt.show()
