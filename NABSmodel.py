import nltk as nltk
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
#nltk.download()
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import csv

data = pd.read_csv("data.csv", encoding="utf8")
columns = ['Branded_Model_Name', 'Activity_Id', 'Assigned_To', 'Activity_Status', 'Activity_Type', 'Activity_Code', 'SR_Number','SR_Status', 'Activity_Sub_Status', 'SR_Cause_Code', 'SR_Symptom_Code', 'SR_Customer_Description']
dataframe = pd.DataFrame(data, columns=['Branded_Model_Name', 'Activity_Id', 'Assigned_To', 'Activity_Status', 'Activity_Type', 'Activity_Code', 'SR_Number','SR_Status', 'Activity_Sub_Status', 'SR_Cause_Code', 'SR_Symptom_Code', 'SR_Customer_Description']
)
df = dataframe.drop(dataframe.index[0])
#print(df.isnull().sum())
dfr = df.fillna("0")
#print(dfr)
print(dfr.isnull().sum())

dfr.to_csv("ndata.csv", index=True)
dfn = pd.DataFrame(dfr, columns=['Branded_Model_Name', 'Activity_Id', 'Assigned_To', 'Activity_Status', 'Activity_Type', 'Activity_Code', 'SR_Number','SR_Status', 'Activity_Sub_Status', 'SR_Cause_Code', 'SR_Symptom_Code', 'SR_Customer_Description'])
#print(dfn.head(5))
#print(dfn.shape)

FR = dfn[dfn['Activity_Type'] == 'FR'].shape[0]
HD = dfn[dfn['Activity_Type'] == 'HD'].shape[0]
SA = dfn[dfn['Activity_Type'] =='SA'].shape[0]
TD = dfn[dfn['Activity_Type'] =='TD'].shape[0]

plt.bar(14,FR,10, label="FR")
plt.bar(18,HD,10, label="HD")
plt.bar(20,SA,10, label="SA")
plt.bar(20,TD,10, label="TD")

plt.legend()
plt.ylabel('Number of examples')
plt.title('Propoertion of examples')
plt.show()

text = dfn['SR_Customer_Description']
def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space,
    # which in effect deletes the punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)


dfn['SR_Customer_Description'] = dfn['SR_Customer_Description'].apply(remove_punctuation)
#print(dfn.head(10))

# extracting the stopwords from nltk library
sw = stopwords.words('Italian')
# displaying the stopwords
print(np.array(sw))
print("Number of stopwords: ", len(sw))

txt = dfn['SR_Customer_Description']
def stopwords(txt):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    txt = [word.lower() for word in txt.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(txt)

dfn['SR_Customer_Description'] = dfn['SR_Customer_Description'].apply(stopwords)
#print(dfn.head(10))

# create a count vectorizer object
count_vectorizer = CountVectorizer()
# fit the count vectorizer using the text data
count_vectorizer.fit(dfn['SR_Customer_Description'])
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
tx = dfn['SR_Customer_Description']
def stemming(tx):
    '''a function which stems each word in the given text'''
    tx = [stemmer.stem(word) for word in tx.split()]
    return " ".join(tx)

dfn['SR_Customer_Description']= dfn['SR_Customer_Description'].apply(stemming)
#print(dfn['SR_Customer_Description'])

# create the object of tfid vectorizer
tfid_vectorizer = TfidfVectorizer("italian")
# fit the vectorizer using the text data
tfid_vectorizer = tfid_vectorizer.fit(dfn['SR_Customer_Description'])

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

text = dfn['SR_Customer_Description']
def length(text):
    '''a function which returns the length of text'''
    return len(text)
dfn['length'] = dfn['SR_Customer_Description'].apply(length)
#dfn.head(10)


FR_data = dfn[dfn['Activity_Type'] == 'FR']
HD_data = dfn[dfn['Activity_Type'] == 'HD']
SA_data = dfn[dfn['Activity_Type'] =='SA']
TD_data = dfn[dfn['Activity_Type'] =='TD']



#Histogram
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
bins = 500
plt.hist(FR_data['length'], alpha = 0.2, bins=bins, label='FR')
plt.hist(HD_data['length'], alpha = 1.0, bins=bins, label='HD')
plt.hist(SA_data['length'], alpha = 0.8, bins=bins, label='SA')
plt.hist(TD_data['length'], alpha = 0.8, bins=bins, label='TD')
plt.xlabel('length')
plt.ylabel('numbers')
plt.legend(loc='upper right')
plt.xlim(0,4000)
plt.grid()
plt.show()

#EDGAR ALLAN POE
# create the object of tfid vectorizer
#FR_tfid_vectorizer = TfidfVectorizer("italian")
# fit the vectorizer using the text data
#FR_tfid_vectorizer = FR_tfid_vectorizer.fit(FR_data['SR_Customer_Description'])
# collect the vocabulary items used in the vectorizer
#FR_dictionary = FR_tfid_vectorizer.vocabulary_.items()
#print(FR_dictionary)
# lists to store the vocab and counts
#vocab = []
#count = []
# iterate through each vocab and count append the value to designated lists
#for key, value in FR_dictionary:
 #   vocab.append(key)
  #  count.append(value)
# store the count in panadas dataframe with vocab as index
#FR_vocab = pd.Series(count, index=vocab)
# sort the dataframe
#FR_vocab = FR_vocab.sort_values(ascending=False)
# plot of the top vocab
#top_vacab = FR_vocab.head(20)
#top_vacab.plot(kind = 'barh', figsize=(5,10), xlim= (9700, 9740))
#plt.show()

#TF-IDF Extraction
# extract the tfid representation matrix of the text data
tfid_matrix = tfid_vectorizer.transform(dfn['SR_Customer_Description'])
# collect the tfid matrix in numpy array
array = tfid_matrix.todense()
# store the tf-idf array into pandas dataframe
df = pd.DataFrame(array)
#print(df.head(10))
df['output'] = dfn['Activity_Type']
df['ST'] = dfn['SR_Status']
print(df.head(10))
features = df.columns.tolist()
output = 'output'
# removing the output and the id from features
features.remove(output)
features.remove('ST')

alpha_list1 = np.linspace(0.006, 0.1, 20)
alpha_list1 = np.around(alpha_list1, decimals=4)
print(alpha_list1)

# parameter grid
parameter_grid = [{"alpha":alpha_list1}]
# classifier object
classifier1 = MultinomialNB()
# gridsearch object using 4 fold cross validation and neg_log_loss as scoring paramter
gridsearch1 = GridSearchCV(classifier1,parameter_grid, scoring = 'neg_log_loss', cv = 4)
# fit the gridsearch
gridsearch1.fit(df[features], df[output])
results1 = pd.DataFrame()
# collect alpha list
results1['alpha'] = gridsearch1.cv_results_['param_alpha'].dfn
# collect test scores
results1['neglogloss'] = gridsearch1.cv_results_['mean_test_score'].dfn
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
plt.plot(results1['alpha'], -results1['neglogloss'])
plt.xlabel('alpha')
plt.ylabel('logloss')
plt.grid()
result = csv.writer(gridsearch1.best_params_)
print("Best parameter: ",gridsearch1.best_params_)
print("Best score: ",gridsearch1.best_score_)

