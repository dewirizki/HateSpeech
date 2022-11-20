from NLP_Models import TextMining as tm
from joblib import load
from NLP_Models import openewfile  as of
import pickle
import numpy as np

def loadmodel(filename):
    model = load(filename)
    return model

def preprocess(text, lemmit = True):
    if lemmit:
        text = tm.cleanText(text,fix=SlangS, pattern2 = True, lang = bahasa, lemma=lemmatizer, stops = stops, symbols_remove = True, numbers_remove = True, hashtag_remove=False, min_charLen = 2)
        text = tm.handlingnegation(text)
    else:
        text = tm.cleanText(text,fix=SlangS, pattern2 = True, lang = bahasa, lemma=lemmatizer, stops = stops, symbols_remove = True, numbers_remove = True, hashtag_remove=False, min_charLen = 2)
    return(text)

def loadtokenizer(filepath):
    with open(filepath, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return (tokenizer)

fSlang = of.openfile(path = './NLP_Models/slangword')
bahasa = 'id'
stops, lemmatizer = tm.LoadStopWords(bahasa, sentiment = False)
sw=open(fSlang,encoding='utf-8', errors ='ignore', mode='r');SlangS=sw.readlines();sw.close()
SlangS = {slang.strip().split(':')[0]:slang.strip().split(':')[1] for slang in SlangS}
tokenizersen = loadtokenizer(of.openfile(path = './NLP_Models/tokenizer_hatespeech'))
model = loadmodel(of.openfile(path = './NLP_Models/model_hatespeech'))

def hateSpeechPredict(text):
    text = str(text)
    text = [preprocess(text, lemmit = lemmatizer)]
    #text = tokenizersen.fit_transform(text)
    hate = model.predict_proba(text)
    hatelist = hate[0]
    hasil = {'hate': (hatelist[0]*100), 'nonhate': (hatelist[1]*100)}
    labels = ['hate', 'nonhate']
    conf_intv = float('%.3f'%(max(hasil.values())))
    result = labels[np.argmax(hatelist)]
    return {'model_pred': hasil, 'final_result': result, 'confidence': conf_intv}