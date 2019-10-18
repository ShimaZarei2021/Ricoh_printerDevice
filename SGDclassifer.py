import csv
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import matplotlib
from matplotlib import pyplot as plt


data = pd.read_csv("data.csv", encoding="utf8", dtype='unicode')
columns = ['SR_Customer_Description', 'Branded_Model_Name', 'Activity_Id', 'Assigned_To', 'Activity_Status', 'Activity_Code', 'SR_Number','SR_Status', 'Activity_Sub_Status', 'SR_Cause_Code', 'SR_Symptom_Code',  'Activity__Type',]
df = pd.DataFrame(data, columns=['SR_Customer_Description', 'Branded_Model_Name', 'Activity_Id', 'Assigned_To', 'Activity_Status', 'Activity_Code', 'SR_Number','SR_Status', 'Activity_Sub_Status', 'SR_Cause_Code', 'SR_Symptom_Code', 'Activity__Type']
)
df = df.drop(df.index[0])
#print(df.isnull().sum())
df.fillna(value=0, inplace=True)

if (df['Activity__Type'].count()>1):
   df.loc[(df.Activity__Type=='Inbound Call') | (df.Activity__Type=='Help Desk') | (df.Activity__Type=='Field Repair'), 'AT']='Field Repair'
   df.loc[(df.Activity__Type=='Inbound Call') | (df.Activity__Type=='Help Desk') , 'AT']='Help Desk'
   df.loc[(df.Activity__Type=='Inbound Call') | (df.Activity__Type=='Field Repair') , 'AT']='Field Repair'
   df.loc[(df.Activity__Type == 'Help Desk') | (df.Activity__Type == 'Field Repair') , 'AT']='Field Repair'

print(df.head(5))
df.drop(['Activity__Type'], axis=1)
df.astype('category')

FR = df[df['AT'] == 'FR'].shape[1]
HD = df[df['AT'] == 'HD'].shape[1]
In = df[df['AT'] =='In'].shape[1]

print(FR)
print(HD)
print(In)

plt.bar(14,FR,10, label="FR")
plt.bar(18,HD,10, label="HD")
plt.bar(20,In,10, label="In")


plt.legend()
plt.ylabel('Number of examples')
plt.title('Propoertion of examples')
plt.show()

text = df['SR_Customer_Description']
def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space,
    # which in effect deletes the punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)


df['SR_Customer_Description'] = df['SR_Customer_Description'].apply(remove_punctuation)
#print(dfn.head(10))

# extracting the stopwords from nltk library
sw = stopwords.words('Italian')
# displaying the stopwords
print(np.array(sw))
print("Number of stopwords: ", len(sw))

txt = df['SR_Customer_Description']
def stopwords(txt):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    txt = [word.lower() for word in txt.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(txt)

df['SR_Customer_Description'] = df['SR_Customer_Description'].apply(stopwords)
#print(dfn.head(10))

# create a count vectorizer object
count_vectorizer = CountVectorizer()
# fit the count vectorizer using the text data
count_vectorizer.fit(df['SR_Customer_Description'])
# collect the vocabulary items used in the vectorizer
dictionary = count_vectorizer.vocabulary_.items()

vocab = []
count = []
# iterate through each vocab and count append the value to designated lists
for key, value in dictionary:
    vocab.append(key)
    count.append(value)
# store the count in panadas dataframe with vocab as index
vocab_bef_stem = pd.Series(count, index=vocab)
# sort the dataframe
vocab_bef_stem = vocab_bef_stem.sort_values(ascending=False)

top_vacab = vocab_bef_stem.head(20)
top_vacab.plot(kind = 'barh', figsize=(5,10), xlim= (25230, 30260))
plt.show()

# create an object of stemming function
stemmer = SnowballStemmer("italian")
tx = df['SR_Customer_Description']
def stemming(tx):
    '''a function which stems each word in the given text'''
    tx = [stemmer.stem(word) for word in tx.split()]
    return " ".join(tx)

df['SR_Customer_Description']= df['SR_Customer_Description'].apply(stemming)
#print(dfn['SR_Customer_Description'])

# create the object of tfid vectorizer
tfid_vectorizer = TfidfVectorizer("italian")
# fit the vectorizer using the text data
tfid_vectorizer = tfid_vectorizer.fit(df['SR_Customer_Description'])

# collect the vocabulary items used in the vectorizer
dictionary =tfid_vectorizer.vocabulary_.items()


# lists to store the vocab and counts
vocab = []
count = []
# iterate through each vocab and count append the value to designated lists
for key, value in dictionary:
    vocab.append(key)
    count.append(value)
# store the count in panadas dataframe with vocab as index
vocab_after_stem = pd.Series(count, index=vocab)
# sort the dataframe
vocab_after_stem = vocab_after_stem.sort_values(ascending=False)
# plot of the top vocab
top_vacab = vocab_after_stem.head(5)
top_vacab.plot(kind = 'barh', figsize=(5,10), xlim= (15120, 15145))
plt.show()

text = df['SR_Customer_Description']
def length(text):
    return len(text)
df['length'] = df['SR_Customer_Description'].apply(length)
#dfn.head(10)


FR_data = df[df['AT'] == 'FR']
HD_data = df[df['AT'] == 'HD']
In_data = df[df['AT'] =='In']



#Histogram
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
bins = 100
plt.hist(FR_data['length'], alpha = 0.2, bins=bins, label='FR')
plt.hist(HD_data['length'], alpha = 1.0, bins=bins, label='HD')
plt.hist(In_data['length'], alpha = 0.8, bins=bins, label='In')

plt.xlabel('length')
plt.ylabel('numbers')
plt.legend(loc='upper right')
plt.xlim(0,4000)
plt.grid()
plt.show()


#TF-IDF Extraction
# extract the tfid representation matrix of the text data
tfid_matrix = tfid_vectorizer.transform(df['SR_Customer_Description'])
#print(tfid_matrix)
# collect the tfid matrix in numpy array
array = tfid_matrix.todense()
# store the tf-idf array into pandas dataframe
df = pd.DataFrame(array)
#print(df.head(10))

df['output'] = df['AT']
df['ST'] = df['SR_Status']
df.dropna(axis=0, how='all')
print(df.isnull())

#print(df.head(10))
features = df.columns.tolist()
output = df['output']
ST = df['ST']
print(features)
# removing the output and the id from features
features.remove('output')
features.remove('ST')
print(features)



alpha_list1 = np.linspace(0.006, 0.1, 20)
alpha_list1 = np.around(alpha_list1, decimals=4)
print(alpha_list1)

# parameter grid
parameter_grid = [{"alpha":alpha_list1}]
# classifier object
classifier1 = SGDClassifier()
# gridsearch object using 4 fold cross validation and neg_log_loss as scoring paramter
gridsearch1 = GridSearchCV(classifier1, parameter_grid, scoring='neg_log_loss', cv=2)
# fit the gridsearch
gridsearch1.fit(df[features], df[output])
results1 = pd.DataFrame()
# collect alpha list
results1['alpha'] = gridsearch1.cv_results_['param_alpha'].df
# collect test scores
results1['neglogloss'] = gridsearch1.cv_results_['mean_test_score'].df
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
plt.plot(results1['alpha'], -results1['neglogloss'])
plt.xlabel('alpha')
plt.ylabel('logloss')
plt.grid()
result = csv.writer(gridsearch1.best_params_)
print("Best parameter: ",gridsearch1.best_params_)
print("Best score: ",gridsearch1.best_score_)
