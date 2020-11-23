import tensorflow as tf
from tensorflow.keras.models import Sequential ,save_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split , GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from data_hadnling  import  input_data , output_data
from grid_parameters import activation , layers
import json

x_train,  x_test , y_train , y_test = train_test_split(input_data , output_data ,
                                                       random_state=786 , test_size=.2)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



def model_creation(layers , activation_func):
    model=  Sequential()
    for layer_nodes in layers:
        model.add(Dense(layer_nodes , activation= activation_func))
    model.add(Dense(1 , activation = "sigmoid"))
    model.compile(optimizer= "adam", metrics= ["accuracy"] , loss = "binary_crossentropy")
    return model

model = KerasClassifier(build_fn= model_creation ,verbose=1)

param_grid = dict(layers  =  layers, activation_func = activation , epochs =[200])
grid = GridSearchCV(estimator = model  , param_grid =  param_grid)

grid_result = grid.fit(x_train , y_train )
score = grid.predict(x_test)

with open("grid_result.txt"  ,"w") as gridp:
    gridp.write("best accuracy : \n\n"  + str(accuracy_score(y_test , score))+"\n\n")
    gridp.write("best paramteres : \n\n" + str(grid_result.best_params_))

with open("model_parameters.json", "w") as jsonp:
    json.dump(grid_result.best_params_,jsonp )



with open("confusion_matrix.txt" , "w") as cmp:
    cmp.write("confiusion matrix : \n\n " +str(confusion_matrix(y_test ,score)))

with open("classification_report.txt" ,"w") as crp:
    crp.write("classification report : \n\n" + str(classification_report(y_test , score)))

