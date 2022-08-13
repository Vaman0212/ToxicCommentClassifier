from flask import Flask, render_template, request, redirect, url_for
import re
from collections import OrderedDict
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import stanfordnlp
import warnings
warnings.filterwarnings('ignore')
app = Flask(__name__)

stopwords = [
    'के', 'का', 'एक', 'में', 'की', 'है', 'यह', 'और', 'से', 'हैं', 'को', 'पर', 'इस', 'होता', 'कि', 'जो',
    'मे', 'गया', 'करने', 'किया', 'लिये', 'अपने', 'ने', 'बनी', 'नहीं', 'तो', 'ही', 'या', 'एवं', 'दिया',
    'हो', 'इसका', 'था', 'द्वारा', 'हुआ', 'तक', 'साथ', 'करना', 'वाले', 'बाद', 'लिए', 'आप',
    'कुछ', 'सकते', 'किसी', 'ये', 'इसके', 'सबसे', 'इसमें', 'थे', 'दो', 'होने', 'वह', 'वे', 'करते',
    'बहुत', 'कहा', 'वर्ग', 'कई', 'करें', 'होती', 'अपनी', 'उनके', 'थी', 'यदि', 'हुई', 'जा', 'ना',
    'इसे', 'कहते', 'जब', 'होते', 'कोई', 'हुए', 'व', 'न', 'अभी', 'जैसे', 'सभी', 'करता', 'उनकी',
    'तरह', 'उस', 'आदि', 'कुल', 'एस', 'रहा', 'इसकी', 'सकता', 'रहे', 'उनका', 'इसी', 'रखें', 'अपना',
    'पे', 'उसके', 'अंदर', 'अत', 'अदि', 'अप', 'अपना', 'अपनि', 'अपनी', 'अपने', 'अभि', 'अभी', 'आदि',
    'आप', 'इंहिं', 'इंहें', 'इंहों', 'इतयादि', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका',
    'इसकि', 'इसकी', 'इसके', 'इसमें', 'इसि', 'इसी', 'इसे', 'उंहिं', 'उंहें', 'उंहों', 'उन', 'उनका', 'उनकि',
    'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसि', 'उसी', 'उसे', 'एक', 'एवं',
    'एस', 'एसे', 'ऐसे', 'ओर', 'और', 'कइ', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते',
    'कहा', 'का', 'काफि', 'काफ़ी', 'कि', 'किंहें', 'किंहों', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस',
    'किसि', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोइ', 'कोई', 'कोन', 'कोनसा', 'कौन',
    'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जहां', 'जा', 'जिंहें', 'जिंहों', 'जितना', 'जिधर', 'जिन', 'जिन्हें', 'जिन्हों',
    'जिस', 'जिसे', 'जीधर', 'जेसा', 'जेसे', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिंहें', 'तिंहों', 'तिन',
    'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थि', 'थी', 'थे', 'दबारा', 'दवारा', 'दिया', 'दुसरा', 'दुसरे', 'दूसरे',
    'दो', 'द्वारा', 'न', 'नहिं', 'नहीं', 'ना', 'निचे', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पुरा', 'पूरा', 'पे', 'फिर',
    'बनि', 'बनी', 'बहि', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भि', 'भितर', 'भी', 'भीतर', 'मगर', 'मानो',
    'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यहां', 'यहि', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रवासा', 'रहा', 'रहे', 'ऱ्वासा',
    'लिए', 'लिये', 'लेकिन', 'व', 'वगेरह', 'वरग', 'वर्ग', 'वह', 'वहाँ', 'वहां', 'वहिं', 'वहीं', 'वाले', 'वुह', 'वे',
    'वग़ैरह', 'संग', 'सकता', 'सकते', 'सबसे', 'सभि', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'हि',
    'ही', 'हुअ', 'हुआ', 'हुइ', 'हुई', 'हुए', 'हे', 'हें', 'है', 'हैं', 'हो', 'होता', 'होति', 'होती', 'होते', 'होना',
    'होने', 'अंदर', 'अत', 'अदि', 'अप', 'अपना', 'अपनि', 'अपनी', 'अपने', 'अभि', 'अभी', 'आदि', 'आप', 'इंहिं',
    'इंहें', 'इंहों', 'इतयादि', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकि', 'इसकी', 'इसके',
    'इसमें', 'इसि', 'इसी', 'इसे', 'उंहिं', 'उंहें', 'उंहों', 'उन', 'उनका', 'उनकि', 'उनकी', 'उनके', 'उनको', 'उन्हीं',
    'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसि', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'एसे', 'ऐसे', 'ओर', 'और', 'कइ',
    'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफि', 'काफ़ी', 'कि', 'किंहें',
    'किंहों', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसि', 'किसी', 'किसे', 'की', 'कुछ', 'कुल',
    'के', 'को', 'कोइ', 'कोई', 'कोन', 'कोनसा', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जहां', 'जा', 'जिंहें',
    'जिंहों', 'जितना', 'जिधर', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जेसा', 'जेसे', 'जैसा', 'जैसे', 'जो',
    'तक', 'तब', 'तरह', 'तिंहें', 'तिंहों', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थि', 'थी', 'थे', 'दबारा',
    'दवारा', 'दिया', 'दुसरा', 'दुसरे', 'दूसरे', 'दो', 'द्वारा', 'न', 'नहिं', 'नहीं', 'ना', 'निचे', 'निहायत', 'नीचे', 'ने',
    'पर', 'पहले', 'पुरा', 'पूरा', 'पे', 'फिर', 'बनि', 'बनी', 'बहि', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भि',
    'भितर', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यहां', 'यहि', 'यही', 'या', 'यिह',
    'ये', 'रखें', 'रवासा', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वगेरह', 'वरग', 'वर्ग', 'वह', 'वहाँ',
    'वहां', 'वहिं', 'वहीं', 'वाले', 'वुह', 'वे', 'वग़ैरह', 'संग', 'सकता', 'सकते', 'सबसे', 'सभि', 'सभी', 'साथ',
    'साबुत', 'साभ', 'सारा', 'से', 'सो', 'हि', 'ही', 'हुअ', 'हुआ', 'हुइ', 'हुई', 'हुए', 'हे', 'हें', 'है', 'हैं', 'हो',
    'होता', 'होति', 'होती', 'होते', 'होना', 'होने' ]

# tokenizer
def my_tokenizer(s):
    return s.split(' ')
# Lemmatizer
nlp = stanfordnlp.Pipeline(processors='tokenize,lemma',lang="hi")
# load vectorizer
tfidf = pickle.load(open("tfidf.pkl","rb"))
ctvec = pickle.load(open("vectorizer.pkl","rb"))
# Load RFC models
Rfc_t = pickle.load(open("RandomForestT.sav","rb"))
Rfc_c = pickle.load(open("RandomForestC.sav","rb"))
# Tokenizer
tokenizer = pickle.load(open('tokenizer.pkl','rb'))
# LSTM
model = load_model('LSTM.h5')

# lemmatization function
def hi_lemma(w):
    try:
        doc = nlp(w)
        tmp = [word.lemma for sent in doc.sentences for word in sent.words]
        return tmp[0]
    except:
        return w

# pre-processing the input comment
def preprocess(text):
    # removing url links
    func = lambda x: re.sub(r'http\S+', '', x)
    text = func(text)
    # removing new lines and tabs
    func = lambda x: re.sub(r"[\t\r]+", '', x)
    text = func(text)
    # removing @mention
    func = lambda x: re.sub(r'@[\w]*', '', x)
    text = func(text)
    # removing all special characters
    func = lambda x: re.sub(r"[`'''`,~,!,@,#,$,%,^,&,*,(,),_,-,+,=,{,[,},},|,\,:,;,\",',<,,,>,.,?,/'''`\n।]", '', x)
    text = func(text)
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
    func = lambda x: emoji_pattern.sub(r'', x)
    text = func(text)
    # removing all remaining characters that aren't hindi devanagari characters or white space
    func = lambda x: re.sub(r"[^ऀ-ॿ\s]", '', x)
    text = func(text)
    # removing stopwords
    func = lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])
    text = func(text)
    # tokenization
    func = lambda x: x.split(' ')
    text = func(text)
    # lemmatization
    func = lambda x: [hi_lemma(y) for y in x]
    text = func(text)
    # remove repeated tokens
    func = lambda x: list(OrderedDict.fromkeys(x))
    text = func(text)
    # generating clean sentence
    sentence = ' '.join(r for r in text)

    return sentence


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['GET','POST'])
def result():
    if request.method == "POST":
        comment = request.form["comment"]
        text = [preprocess(comment)]
        text1 = tfidf.transform(text)
        text2 = ctvec.transform(text)
        text3 = tokenizer.texts_to_sequences(text)
        text3 = pad_sequences(text3, maxlen=80, padding="pre", truncating="pre")
        score = []
        score_value = Rfc_t.predict_proba(text1)
        for i in range(3):
            score.append(round(score_value[i][0][1]*100,1))
        score_value = Rfc_c.predict_proba(text2)
        for i in range(3):
            score.append(round(score_value[i][0][1]*100,1))
        score_value = model.predict(text3)
        for i in range(3):
            score.append(round(score_value[0][i]*100,1))

        return render_template('results.html', text=comment, score=score)

@app.route('/reset')
def reset():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host="127.0.0.1", port="700", debug=True)