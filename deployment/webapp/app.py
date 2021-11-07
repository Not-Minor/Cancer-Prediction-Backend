import cv2
from flask import Flask,jsonify,request,send_from_directory,render_template
from keras_preprocessing import image
# import tensorflow as tf
from keras.models import load_model
import numpy as np
import os

app=Flask(__name__)

STATIC_FOLDER='G:/Code/py prog/machine learning-deep learning/my projects/breast cancer histology/deployment/webapp/static'

UPLOAD_FOLDER=STATIC_FOLDER+'/uploads/' #path to the folder where we will store the upload before prediction
MODEL_FOLDER=STATIC_FOLDER+'/models' #path to the folders where we'll store the models

def predict(fullpath):
    data=image.load_img(fullpath,target_size=(75,75,3))
    data=np.expand_dims(data,axis=0)
    data = data.astype('float') / 255

    model=load_model(MODEL_FOLDER+'/BCModel.h5')
    result=model.predict(data)

    pred_prob=model.predict_proba(data)

    return result,pred_prob

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload',methods=['GET','POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file=request.files['image']
        fullname=os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(fullname)

        result,pred_prob=predict(fullname)
        pred=np.argmax(result,axis=1)
        prob=np.max(pred_prob,axis=1)
        prob_str=' '.join(map(str, np.round_(prob*100,decimals=2)))

        if pred==0:
            label='IDC POSITIVE'
            accuracy=prob_str
            return render_template('predict.html',image_file=file.filename,prediction_text='There is a {}% chance that these are {} lungs.'.format(accuracy,label))

        elif pred==1:
            label='IDC NEGATIVE'
            accuracy=prob_str
            return render_template('predict.html',image_file=file.filename,prediction_text='There is a {}% chance that these lungs have {}.'.format(accuracy,label))

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER,filename)

if __name__=='__main__':
    app.run(debug=True)
        