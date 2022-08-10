# Importing Libraries and Installing Packages
import pandas as pd
import numpy as np
import re
import string
import pickle
from collections import Counter, OrderedDict
pd.options.mode.chained_assignment = None

#!pip install stanfordnlp
#stanfordnlp.download('hi')
import stanfordnlp
nlp = stanfordnlp.Pipeline(processors='tokenize,lemma',lang="hi")

# Dataset import
df = pd.read_csv('CommentsDataset.csv')
df

# Data Pre-Processing
# generating stopwords
def gen_stopword():
    st=pd.read_csv('hindi_stopwords.txt',sep='\n')
    stopwords=[]
    for i in range(len(st)):
        stopwords.append(st.loc[i, 'Stopwords'].strip())
    return stopwords

# lemmatization function
def hi_lemma(w):
    try:
        doc = nlp(w)
        tmp = [word.lemma for sent in doc.sentences for word in sent.words]
        return tmp[0]
    except:
        return w

def data_pre_processing(df_clean):
    # removing url links
    df_clean.Post = df_clean.Post.apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
    df_clean.Post = df_clean.Post.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))
    # removing @mention
    df_clean.Post = df_clean.Post.apply(lambda x: re.sub(r'@[\w]*', '', x))
    # removing all remaining characters that aren't hindi devanagari characters or white space
    df_clean.Post = df_clean.Post.apply(lambda x: re.sub(r"[^ऀ-ॿ\s]", '', x))
    # removing all special characters
    df_clean.Post = df_clean.Post.apply(lambda x: re.sub(r"[`'''`,~,!,@,#,$,%,^,&,*,(,),_,-,+,=,{,[,},},|,\,:,;,\",',<,,,>,.,?,/'''`\n।]", '', x))
    # removing emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    df_clean.Post = df_clean.Post.apply(lambda x: emoji_pattern.sub(r'', x))
    
    # removing stopwords
    stopwords = gen_stopword()
    df_clean.Post = df_clean.Post.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    
    # tokenization
    df_clean['token'] = df_clean.Post.apply(lambda x: x.split())
    
    # lemmatization
    df_clean['lemma_token'] = df_clean.token.apply(lambda x: [hi_lemma(y) for y in x])
    
    # remove repeated tokens
    df_clean['lemma_token'] = df_clean.lemma_token.apply(lambda x: list(OrderedDict.fromkeys(x)))
    
    df_clean['sentence'] = [' '.join(r) for r in df_clean['lemma_token'].values]
    
    return df_clean

df = data_pre_processing(df)
df

# Loading preprocessed dataset
df = pd.read_pickle("preprocessed_dataset.pkl")

X = df.sentence
y = df.drop(['sentence','token'],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TFIDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
def my_tokenizer(s):
    return s.split(' ')
tfidf = TfidfVectorizer(min_df=2,ngram_range=(1, 3),encoding='ISCII',tokenizer=my_tokenizer,stop_words=gen_stopword())
X_train = tfidf.fit_transform(df_train['sentence']).toarray()
y_train = df_train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
X_test = tfidf.transform(df_test['sentence'])
y_test = df_test[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]

# Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=2,ngram_range=(1,3),encoding='ISCII',tokenizer=my_tokenizer,stop_words=gen_stopword())
Xcv_train = vectorizer.fit_transform(df_train['sentence']).toarray()
Xcv_test = vectorizer.transform(df_test['sentence'])

# Saving TFIDF Vectorizer
pickle.dump(tfidf, open('tfidf.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
Rf = RandomForestClassifier(n_estimators=40,random_state=1,n_jobs=1)
categories = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
for l in categories:
    Rf.fit(X_train, y_train[l])
    prediction = Rf.predict(X_test)
    print(classification_report(y_test[l],prediction))

# RF Model Saving and Loading
pickle.dump(Rf,open('RandomForest.sav','wb'))

Rf = pickle.load(open("RandomForest.sav",'rb'))

# LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

def counter_word(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count
text = df.sentence
counter = counter_word(text)
num_words = len(counter)
max_length = 80

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X)

word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(X)
X_train = pad_sequences(train_sequences, maxlen=max_length, padding="pre", truncating="pre")

model = Sequential()
model.add(Embedding(num_words, 32, input_length=max_length))
model.add(LSTM(64))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
categories = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
for l in categories:
    history = model.fit(X_train, y_train[l], epochs=5, batch_size=32)
    scores = model.evaluate(X_valid, y_valid[l])
    print("Accuracy: %.2f%%" % (scores[1]*100),'\n')

# LSTM Model Saving and Loading
model.save('LSTM.h5')

from tensorflow import keras
model = keras.models.load_model('LSTM.h5')