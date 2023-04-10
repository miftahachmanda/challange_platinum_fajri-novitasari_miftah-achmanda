import pandas as pd
import re
import sqlite3

db = sqlite3.connect('database.db', check_same_thread=False)
mycurs = db.cursor()

#preprosesing data
#mengahapus  cleansing data

def remove_tweet(text):
    text = text.lower() #prosose merubah huruf menjadi kecil
    text = re.sub('\n',' ', text)
    text = re.sub('rt',' ', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub('user',' ', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
    text = re.sub(r'#', '', text)
    text = re.sub(r',','',text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub('  +',' ', text)
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = re.sub('  +',' ', text) 

    return text

# merubah tweet data csv
def upload_file(input_file):
    cleaning_data_file_dummy = input_file.iloc[:, 0]
    print(cleaning_data_file_dummy)

    for tweet in cleaning_data_file_dummy:
        tweet_bersih = remove_tweet(tweet)
        insert = "insert into tweet(tweet_dummy,tweet_bersih) values (?,?)"
        variabel = (tweet, tweet_bersih)
        mycurs.execute(insert, variabel)
        db.commit()
        print(tweet)

# Untuk Proses Text
def cleansing_text (input_text):
    output_text = remove_tweet(input_text)
    return output_text
  