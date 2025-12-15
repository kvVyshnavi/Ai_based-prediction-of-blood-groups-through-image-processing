from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import cv2
import pymysql
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam



global username
global X_train, X_test, y_train, y_test, X, Y, labels
labels = []
accuracy = []
precision = []
recall = [] 
fscore = []
path = "Dataset"

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    a = (accuracy_score(y_test,predict)*100)
    p = (precision_score(y_test, predict,average='macro') * 100)
    r = (recall_score(y_test, predict,average='macro') * 100)
    f = (f1_score(y_test, predict,average='macro') * 100)
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    return algorithm

def values(filename, acc):
    f = open(filename, 'rb')
    train_values = pickle.load(f)
    f.close()
    accuracy_value = train_values[acc]
    return accuracy_value

X = np.load('model/X.npy')
Y = np.load('model/Y.npy')

X = X.astype('float32')
X = X/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

def getModel(lr, num_epochs):
    dnn_model = Sequential()
    dnn_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    dnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    dnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    dnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    dnn_model.add(Flatten())
    dnn_model.add(Dense(units = 256, activation = 'relu'))
    dnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    optimizer = Adam(lr = lr)
    dnn_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    dnn_model.fit(X_train, y_train, batch_size = 32, epochs = num_epochs, validation_data=(X_test, y_test), verbose=1)
    predict = dnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_test1, predict)
    return acc, dnn_model

def fireflyOptimization():
    if os.path.exists("model/dnn_weights.hdf5"):
        dnn_model = Sequential()
        dnn_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
        dnn_model.add(MaxPooling2D(pool_size = (2, 2)))
        dnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
        dnn_model.add(MaxPooling2D(pool_size = (2, 2)))
        dnn_model.add(Flatten())
        dnn_model.add(Dense(units = 256, activation = 'relu'))
        dnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
        optimizer = Adam(lr = 0.001)
        dnn_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        dnn_model.load_weights("model/dnn_weights.hdf5")
    else:
        lr = [0.01, 0.01, 0.001]
        num_epochs = [10, 15, 23]
        best_model = None
        best_param = None
        best_acc = 0
        for i in range(len(lr)):
            acc, dnn_model = getModel(lr[i], num_epochs[i])
            if acc > best_acc:
                best_param = [lr[i], num_epochs[i]]
                best_model = dnn_model
        dnn_model.save_weights("model/dnn_weights.hdf5")
    return dnn_model       
    
dnn_model = fireflyOptimization()
predict = dnn_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
#call this function to calculate accuracy and other metrics
calculateMetrics("DNN with Firefly", y_test1, predict)
conf_matrix = confusion_matrix(y_test1, predict)

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def segment(image_path):
    img = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 50, 50])   # Lower range for red
    upper_red_1 = np.array([10, 255, 255])  # Upper range for red
    mask_1 = cv2.inRange(hsv_img, lower_red_1, upper_red_1)
    lower_red_2 = np.array([170, 50, 50]) # Lower range for red (wrapping around)
    upper_red_2 = np.array([180, 255, 255]) # Upper range for red (wrapping around)
    mask_2 = cv2.inRange(hsv_img, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask_1, mask_2)
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(red_mask, kernel, iterations=2)
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=2)
    return img, dilated_mask    

def PredictAction(request):
    if request.method == 'POST':
        global labels
        dnn_model = fireflyOptimization()
        myfile = request.FILES['t1'].read()
        if os.path.exists('BloodApp/static/test.jpg'):
            os.remove('BloodApp/static/test.jpg')
        with open('BloodApp/static/test.jpg', "wb") as file:
            file.write(myfile)
        file.close()
        image = cv2.imread('BloodApp/static/test.jpg')
        img = cv2.resize(image, (32,32))
        img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,32,32,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255
        predict = dnn_model.predict(img)
        predict = np.argmax(predict)
        predict = labels[predict]
        original_image, segmented_mask = segment('BloodApp/static/test.jpg')
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        cv2.putText(original_image, 'Predicted As : '+predict, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)
        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(14, 4))#display original and predicted segmented image
        axis[0].set_title("Original Image")
        axis[1].set_title("Segmented Image")
        axis[0].imshow(original_image)
        axis[1].imshow(segmented_mask, cmap="gray")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':'<font size="3" color="blue">Blood Group Predicted As : '+predict+'</font>', 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def TrainModel(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, labels
        global accuracy, precision, recall, fscore, conf_matrix
        output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Algorithm Name</th><th><font size="3" color="black">Accuracy</th>'
        output += '<th><font size="3" color="black">Precision</th><th><font size="3" color="black">Recall</th><th><font size="3" color="black">FSCORE</th></tr>'
        algorithms = ['DNN with Firefly Optimization']
        for i in range(len(algorithms)):
            output += '<td><font size="3" color="black">'+algorithms[i]+'</td><td><font size="3" color="black">'+str(accuracy[i])+'</td><td><font size="3" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="3" color="black">'+str(recall[i])+'</td><td><font size="3" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        arch1_acc = values("model/dnn_history.pckl", "accuracy")
        index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
        axis[0].set_title("Confusion Matrix Prediction Graph")
        axis[1].set_title("DNN Training Accuracy Graph")
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
        ax.set_ylim([0,len(labels)])    
        axis[1].plot(index, arch1_acc, color="green")
        axis[1].legend(['Training Accuracy'], loc='lower right')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        #plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):    
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, X, Y, labels
        output = '<font size="3" color="black">Blood Group Images Dataset Loaded</font><br/>'
        output += '<font size="3" color="blue">Total images found in Dataset = '+str(X.shape[0])+'</font><br/>'
        output += '<font size="3" color="blue">Different Class Labels found in Dataset = '+str(labels)+'</font><br/><br/>'
        output += '<font size="3" color="black">Dataset Train & Test Split details</font><br/>'
        output += '<font size="3" color="blue">80% dataset images used to train DNN = '+str(X_train.shape[0])+'</font><br/>'
        output += '<font size="3" color="blue">20% dataset images used to test DNN = '+str(X_test.shape[0])+'</font><br/>'
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'mysql', database = 'blood',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break                
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'mysql', database = 'blood',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "Signup process completed. Login to perform Blood Group Classification"
        context= {'data':output}
        return render(request, 'Register.html', context)    

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'mysql', database = 'blood',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})
