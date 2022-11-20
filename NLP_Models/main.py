import pandas as pd
from NLP_Models import TextMining as tm
from NLP_Models import CleanText as ct
from NLP_Models import modelling as mdg


#HATESPEECH
dataPath = tm.crawlFiles('./NLP_Models/data/PORN/PORNDATA/', types = 'csv')
data = pd.concat([pd.read_csv(f) for f in dataPath])
data = data[['created_at', 'date', 'time', \
             'user_id', 'username', 'name',\
                 'tweet','mentions', 'urls', 'photos', 'replies_count', \
                     'retweets_count', 'likes_count', 'hashtags',\
                         'link', 'retweet', 'quote_url',\
                             'near', 'geo', 'source', 'user_rt_id', 'user_rt',\
                                 'retweet_id', 'reply_to', 'retweet_date']]
data['Label'] = 'hatespeech'

    
    

#HATESPEECH  
dataPathnhs = tm.crawlFiles('./NLP_Models/data/PORN/NONPORNDATA/', types = 'csv')
datanhs = pd.concat([pd.read_csv(f) for f in dataPathnhs])
datanhs = datanhs[['created_at', 'date', 'time', \
             'user_id', 'username', 'name',\
                 'tweet','mentions', 'urls', 'photos', 'replies_count', \
                     'retweets_count', 'likes_count', 'hashtags',\
                         'link', 'retweet', 'quote_url',\
                             'near', 'geo', 'source', 'user_rt_id', 'user_rt',\
                                 'retweet_id', 'reply_to', 'retweet_date']]
    
datanhs['Label'] = 'nonhatespeech'


dataFinal = pd.concat([data, datanhs])
dataFinal = dataFinal.reset_index(drop= True)    
dataFinal.to_json('./NLP_Models/data/dataFinal.json', orient='records')


#NEWLINE
#data = pd.read_json('./NLP_Models/data/dataFinal.json')
dataFinal.rename(columns={'tweet':'text'}, inplace=True)

dataFinal = ct.cleanningtext(data = dataFinal, both = True, onlyclean = False, sentiment = False)
dataFinal.to_json('./NLP_Models/data/dataFinalClean.json', orient='records')

#EDA




#Modelling
dataFinal = dataFinal[['text', 'cleaned_text' 'Label']]
modelSVC = mdg.modelling(data = dataFinal, modelname= '202106',\
                         crossval = False,  termfrequency = False, \
                             n_fold = 3, kernel = 'linear', n_jobs=1)






