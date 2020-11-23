from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix  , classification_report ,accuracy_score
import numpy as np
import pandas as pd
import json

data = pd.read_csv("student-mat.csv" , sep=";")
input_data =data.iloc[: ,-3:].values
output_data = np.zeros(shape = (len(input_data)  ,1) , dtype = int)

for iter in range(len(input_data)):
    if np.mean(input_data[iter]) >=9:
        output_data[iter][0] =1



x_train,  x_test , y_train , y_test = train_test_split(input_data , output_data ,
                                                       random_state=786 , test_size=.2)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

with open("model_parameters.json" ,"r") as jsonp:
    hyperparameters = json.load(jsonp)

def create_model(layers ,activation):
    model = Sequential()
    for layers_index in range(len(layers)):
        model.add(Dense(layers[layers_index ] , activation= activation))

    model.add(Dense(1, activation = "sigmoid"))
    model.compile(optimizer= "adam" , metrics =["accuracy"] , loss ="binary_crossentropy")
    return model

model = create_model(hyperparameters["layers"] , hyperparameters["activation_func"])

model.fit(x= x_train, y=y_train , epochs = hyperparameters["epochs"])

score = model.predict_classes(x_test)

loss , acc = model.evaluate(x_test, y_test)

with open("main_model_confusion_matrix.txt" , "w") as cmp:
    cmp.write("confusion matrix for main model  \n\n" + str(confusion_matrix(y_test ,score))+\
            "\n\naccuracy : %s"% (acc) +\
            "\n\nloss : %s"%(loss))

with open("main_model_classification_report.txt" , "w") as crp:
    crp.write("classification report for main model :\n\n"  +str(classification_report(y_test ,score)))

