from distutils.log import debug
from flask import Flask, jsonify, redirect, render_template, send_from_directory, session, request, url_for
import sqlite3
import os
#import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import time

import matplotlib
import sklearn
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

# import for svm
import pickle

import librosa

np.set_printoptions(suppress=True)
model = keras.models.load_model('model/NN(AudioSplit).h5')
# with open('model/SVM.pkl', 'rb') as modelFile:
    # svm = pickle.load(modelFile)

# database name
db = 'sd3.db'

# functions
def createUserSession(name, age, gender):
    print(name, age, gender)
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("SELECT * FROM user WHERE username ='"+name+"';")
    if(len(cur.fetchall())==0):
        cur.execute("INSERT INTO user (username) VALUES ('"+name+"');")
        con.commit()
    cur.close()
    con.close()

# connections/requests

app = Flask(__name__, template_folder='templates')
SESSION_TYPE = "redis"
PERMANENT_SESSION_LIFETIME = 1800
app.config.update(SECRET_KEY=os.urandom(24))
app.static_folder = 'static'

@app.route('/')
def index():
    return render_template("index.html", Name=session.get('user'), Age=session.get('age'), Gender=session.get('gender'))

@app.route('/record', methods=['POST'])
def record():
    userData = request.form
    session['user'] = userData["name"]
    session['age'] = userData["age"]
    session['gender'] = userData["genderPOST"]
    # print(request.form)
    createUserSession(userData["name"], userData["age"], userData["genderPOST"])
    return render_template("record.html")


@app.route('/record', methods=['GET'])
def recordGet():
    if (session.get('user') == None):
        return redirect(url_for('index'))
    
    return render_template("record.html")
    

@app.route('/confirmRecord', methods=['POST'])
def confirmRecord():
    if (session.get('user') == None):
        return "session user not found", 500

    filepath = 'audio/'+str(session['user'])+'.wav'

    # savefile
    if(os.path.exists(filepath)):
        os.remove(filepath)

    with open(filepath, mode='bx') as f:
        f.write(request.data)
    f.close()

    # add path to db
    name = session['user']
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("SELECT * FROM user WHERE username ='"+name+"';")
    user_id = cur.fetchone()[0]
    
    cur.execute("SELECT * FROM audio WHERE user_id ='"+str(user_id)+"';")
    if(len(cur.fetchall())==0):
        cur.execute("INSERT INTO audio (path, user_id) VALUES ('"+filepath+"', '"+str(user_id)+"');")
        con.commit()
    cur.close()
    con.close()

    response = {"status": "success"}
    return response, 200

@app.route('/viewData')
def viewData():
    if (session.get('user') == None):
        return redirect(url_for('index'))
    audioExists = False
    # get audio data path check if exists
    name = session['user']
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("SELECT * FROM user WHERE username ='"+name+"';")
    user_id = cur.fetchone()[0]
    
    cur.execute("SELECT * FROM audio WHERE user_id ='"+str(user_id)+"';")
    if(len(cur.fetchall())>0):
        audioExists = True
    cur.close()
    con.close()

    return render_template("data.html", audioData = audioExists)

@app.route('/getImage')
def getImage():

    filepath = 'image/'+str(session['user'])+'.jpg'

    # savefile
    if(os.path.exists(filepath)):
        os.remove(filepath)

    #  get audio path
    name = session['user']
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("SELECT * FROM user WHERE username ='"+name+"';")
    user_id = cur.fetchone()[0]
    cur.execute("SELECT * FROM audio WHERE user_id ='"+str(user_id)+"';")
    audioPath = cur.fetchone()[2]

    # create image
    sample_rate, samples = wavfile.read(audioPath)
    samples = samples[:,0]
    
    # normalize
    # samples = samples/np.max(np.abs(samples), axis=0)
    samples = sklearn.preprocessing.minmax_scale(samples, feature_range=(-1,1))
    
    timeLength = np.linspace(0, len(samples)/sample_rate, num = len(samples))
    
    plt.figure(figsize=(12,6))
    plt.title("Sound Wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.plot(timeLength, samples)
    plt.savefig(filepath)
    # plt.show()
    while(not os.path.exists(filepath)):
        time.sleep(1)

    # add path to db
    cur.execute("SELECT * FROM image WHERE user_id ='"+str(user_id)+"';")
    if(len(cur.fetchall())==0):
        cur.execute("INSERT INTO image (path, user_id) VALUES ('"+filepath+"', '"+str(user_id)+"');")
        con.commit()
    cur.close()
    con.close()

    return send_from_directory("image", name+".jpg")

# @app.route('/results')
# def results():
#     if (session.get('user') == None):
#         return redirect(url_for('index'))
#     audioExists = False
#     # get audio data path check if exists
#     name = session['user']
#     con = sqlite3.connect(db)
#     cur = con.cursor()
#     cur.execute("SELECT * FROM user WHERE username ='"+name+"';")
#     user_id = cur.fetchone()[0]
    
#     cur.execute("SELECT * FROM audio WHERE user_id ='"+str(user_id)+"';")
#     if(len(cur.fetchall())>0):
#         audioExists = True
#     cur.close()
#     con.close()

#     return render_template("results.html", audioData = audioExists)

@app.route('/getResults', methods=["POST"])
def getResults():

    #  get audio path
    name = session['user']
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("SELECT * FROM user WHERE username ='"+name+"';")
    user_id = cur.fetchone()[0]
    cur.execute("SELECT * FROM audio WHERE user_id ='"+str(user_id)+"';")
    audioPath = cur.fetchone()[2]

    # get audio
    sample_rate, samples = wavfile.read(audioPath)
    samples = samples[:,0]
    # normalize
    samples = sklearn.preprocessing.minmax_scale(samples, feature_range=(-1,1))
    # trim audio
    samples, index = librosa.effects.trim(samples, top_db=10, frame_length=256, hop_length=64)
    # get features
    mfccs = librosa.feature.mfcc(samples, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    mfccs = tf.reshape(mfccs, [-1,40])
    # process audio with ML
    
    prediction = model.predict(mfccs)
    max_index = np.argmax(prediction)
    probability = str(np.amax(prediction))

    # prediction = svm.predict(mfccs)

    MLResult = None
    # prediction for svm
    # if(prediction[0] == 'healthy'):
    #     MLResult = "healthy"
    # elif(prediction[0] == 'positive'):
    #     MLResult = "positive"


    # prediction for NN
    if(max_index==0):
        MLResult = "healthy"
    elif(max_index==1):
        MLResult = "positive"


    # add results to db
    cur.execute("SELECT * FROM results WHERE user_id ='"+str(user_id)+"';")
    # add to db for NN
    if(len(cur.fetchall())==0):
        cur.execute("INSERT INTO results (results, probability, user_id) VALUES ('"+MLResult+"', '"+probability+"', '"+str(user_id)+"');")
        con.commit()
    else:
        cur.execute("UPDATE results SET results = '"+MLResult+"', probability = '"+probability+"' WHERE user_id = '"+str(user_id)+"';")
        con.commit()
    
    # add to db for svm
    # if(len(cur.fetchall())==0):
    #     cur.execute("INSERT INTO results (results, user_id) VALUES ('"+MLResult+"', '"+str(user_id)+"');")
    #     con.commit()
    # else:
    #     cur.execute("UPDATE results SET results = '"+MLResult+"' WHERE user_id = '"+str(user_id)+"';")
    #     con.commit()

    cur.close()
    con.close()

    returnDict = {
        "Result": MLResult,
        "Probability": probability
    }
    return jsonify(returnDict)

if __name__ == '__main__':
    # uncomment to enable other devices to connect on local network
    app.run(debug=True, host="0.0.0.0", ssl_context=('ssl/server.crt', 'ssl/server.key'))
    # app.run(debug=True)