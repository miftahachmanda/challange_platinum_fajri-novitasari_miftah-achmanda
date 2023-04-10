
from flask_swagger_ui import get_swaggerui_blueprint
from flask import Flask, request, jsonify, make_response, render_template
from datacleansing import upload_file, cleansing_text
import sqlite3
import pandas as pd

import pickle, re
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import joblib

# instantiate flask object
app = Flask(__name__)
# set app configs
app.config['JSON_SORT_KEYS'] = False 

# flask swagger configs
SWAGGER_URL = '/swagger'
API_URL = '/static/swag.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Data Analyst Sentiment!"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)

# Database
db = sqlite3.connect('database.db', check_same_thread=False) 
db.row_factory = sqlite3.Row
mycursor = db.cursor()

### =================================== HOME PAGE =================================== ###

@app.route("/", methods=['GET','POST'])
def home():
	return render_template("index.html", content="Guys")


### ================================= ANALYST SENTIMENT ================================= ###

#CNN
file = open("model/cnn/x_pad_sequences.pickle", 'rb')
feature_file_from_cnn = pickle.load(file)
file.close()

tokenizer = joblib.load('model/cnn/tokenizer.pickle')
model_file_from_cnn = load_model("model/cnn/modelcnn.h5")

def predict_sentiment_cnn(text):
    sentiment = ['negative', 'neutral', 'positive']
    text = [cleansing_text(text)]
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_cnn.shape[1])
    prediction = model_file_from_cnn.predict(feature)
    get_sentiment = np.argmax(prediction[0])
    return sentiment[get_sentiment]

def sentiment_cnn_csv(input_file):
    column = input_file.iloc[:, 0]
    print(column)

    for data_file in column: # Define and execute query for insert cleaned text and sentiment to sqlite database
        data_clean = cleansing_text(data_file)
        sent = predict_sentiment_cnn(data_clean)
        query = "insert into sentiment_cnn (original_text, clean_text, analyst_sentiment) values (?,?,?)"
        val = (data_file,data_clean,sent)
        db.execute(query, val)
        db.commit()
        print(data_file)

### INPUT TEXT ###
@app.route("/cnn", methods=['POST'])
def cnn():
    original_text = str(request.form["text"]) #get text from user
    text = cleansing_text(original_text) #cleaning text
    text_sentiment = predict_sentiment_cnn(text)
    
    query = "insert into sentiment_cnn (original_text, clean_text, analyst_sentiment) values (?,?,?)"
    variable = (original_text, text, str(text_sentiment))
    mycursor.execute(query, variable)
    db.commit()
    
    # Define API response
    json_response = {
        'description': "Analysis Sentiment Success!",
        'original_text' : original_text,
        'text' : text,
        'sentiment' : text_sentiment
    }
    response_data = jsonify(json_response)
    return response_data

@app.route("/cnn", methods = ["GET"])
def get_cnn():    
    data_query = "select * from sentiment_cnn"
    #execute data_query
    select_text_from_data_query = mycursor.execute(data_query)
    text_sentiment = [dict(cnn_id=row[0], original_text=row[1], clean_text=row[2], analyst_sentiment=row[3])
                      for row in select_text_from_data_query.fetchall()]
    return jsonify(text_sentiment)

### UPLOAD CSV FILE ###
@app.route("/cnn/csv", methods=['POST'])
def cnn_csv():

    # Get file
    file = request.files['file']
    try:
            datacsv = pd.read_csv(file, encoding='iso-8859-1')
    except:
            datacsv = pd.read_csv(file, encoding='utf-8')

    # Cleaning file
    sentiment_cnn_csv(datacsv)

    # Define API response
    select_data = db.execute("SELECT * FROM sentiment_cnn")
    db.commit
    data = [
        dict(cnn_id=row[0], original_text=row[1], clean_text=row[2], analyst_sentiment=row[3])
    for row in select_data.fetchall()
    ]
    
    return jsonify(data)


#-----------------------LSTM---------------------------#
file = open("model/lstm/x_pad_sequences.pickle", 'rb')
feature_file_from_lstm = pickle.load(file)
file.close()

tokenizer = joblib.load('model/lstm/tokenizer.pickle')
model_file_from_lstm = load_model("model/lstm/modellstm.h5")

def predict_sentiment_lstm(text):
    sentiment = ['negative', 'neutral', 'positive']
    text = [cleansing_text(text)]
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = np.argmax(prediction[0])
    return sentiment[get_sentiment]


def sentiment_lstm_csv(input_file):
    column = input_file.iloc[:, 0]
    print(column)

    for data_file in column: # Define and execute query for insert cleaned text and sentiment to sqlite database
        data_clean = cleansing_text(data_file)
        sent = predict_sentiment_lstm(data_clean)
        query = "insert into sentiment_lstm (original_text, clean_text, analyst_sentiment) values (?,?,?)"
        val = (data_file,data_clean,sent)
        db.execute(query, val)
        db.commit()
        print(data_file)

@app.route("/lstm", methods=['POST'])
def lstm():
    original_text = str(request.form["text"]) #get text from user
    text = cleansing_text(original_text) #cleaning text
    text_sentiment = predict_sentiment_lstm(text)
    
    query = "insert into sentiment_lstm (original_text, clean_text, analyst_sentiment) values (?,?,?)"
    variable = (original_text, text, str(text_sentiment))
    mycursor.execute(query, variable)
    db.commit()

    # Define API response
    json_response = {
        'description': "Analysis Sentiment Success!",
        'original_text' : original_text,
        'text' : text,
        'sentiment' : text_sentiment
    }
    response_data = jsonify(json_response)
    return response_data

@app.route("/lstm", methods = ["GET"])
def get_lstm():    
    data_query = "select * from sentiment_lstm"
    #execute data_query
    select_text_from_data_query = mycursor.execute(data_query)
    text_sentiment = [dict(lstm_id=row[0], original_text=row[1], clean_text=row[2], analyst_sentiment=row[3])
                      for row in select_text_from_data_query.fetchall()]
    return jsonify(text_sentiment)

### UPLOAD CSV FILE ###
@app.route("/lstm/csv", methods=['POST'])
def lstm_csv():

    # Get file
    file = request.files['file']
    try:
            datacsv = pd.read_csv(file, encoding='iso-8859-1')
    except:
            datacsv = pd.read_csv(file, encoding='utf-8')

    # Cleaning file
    sentiment_lstm_csv(datacsv)

    # Define API response
    select_data = db.execute("SELECT * FROM sentiment_lstm")
    db.commit
    data = [
        dict(lstm_id=row[0], original_text=row[1], clean_text=row[2], analyst_sentiment=row[3])
    for row in select_data.fetchall()
    ]
    
    return jsonify(data)

### ================================= ERROR HANDLING ================================= ###

@app.errorhandler(400)
def handle_400_error(_error):
    "Return a http 400 error to client"
    return make_response(jsonify({'error': 'Misunderstood'}), 400)


@app.errorhandler(401)
def handle_401_error(_error):
    "Return a http 401 error to client"
    return make_response(jsonify({'error': 'Unauthorised'}), 401)


@app.errorhandler(404)
def handle_404_error(_error):
    "Return a http 404 error to client"
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.errorhandler(500)
def handle_500_error(_error):
    "Return a http 500 error to client"
    return make_response(jsonify({'error': 'Server error'}), 500)



#Run Server
if __name__ == '__main__':
    app.run(debug=True)
