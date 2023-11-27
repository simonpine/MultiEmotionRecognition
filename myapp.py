import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')


tf_transformer = pickle.load(open('./transformModel_tf_transformer.pkl', 'rb'))
vcModel = pickle.load(open('./transformModel_vcModel.pkl', 'rb'))

stop_words = set(stopwords.words('english')) 
def removeStopwordsAndLower(text):
    
    words = text.lower().split() 
    filtered_words = [word for word in words if word not in stop_words] 
    return ' '.join(filtered_words)

def lemaAndStem(text):
    stemmer = SnowballStemmer("english")
    normalized_text = []
    for word in text.split():
        stemmed_word = stemmer.stem(word)
        normalized_text.append(stemmed_word)
    return ' '.join(normalized_text).replace(',', '')


def transformToPredict(textPredict):
    textPredict = removeStopwordsAndLower(textPredict)
    textPredict = lemaAndStem(textPredict)
    textPredict = vcModel.transform([textPredict])
    textPredict = tf_transformer.transform(textPredict)
    return textPredict

st.write('''
# Text Emotion Recognition
         
This app recognizes the emotion from a given text using Natural Language Processing techniques and Machine Learning algorithms.
         
''')

text = st.text_input('Write a text to identify the implicit emotion.') 

model = pickle.load(open('./trained_model_XGBoost.pkl', 'rb'))

st.button('Predict')

if(model.predict((transformToPredict(text)))[0] == 0):
    st.write('Anger 🤬')
elif (model.predict((transformToPredict(text)))[0] == 1):
    st.write('Fear 😨')
elif (model.predict((transformToPredict(text)))[0] == 2):
    st.write('happy 😀')
elif (model.predict((transformToPredict(text)))[0] == 3):
    st.write('Love 😍')
elif (model.predict((transformToPredict(text)))[0] == 4):
    st.write('Sadness 😭')
elif (model.predict((transformToPredict(text)))[0] == 5):
    st.write('Surprise 😱')