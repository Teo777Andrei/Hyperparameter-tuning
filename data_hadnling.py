import pandas as pd
import numpy as np

data = pd.read_csv("student-mat.csv" , sep=";")
input_data =data.iloc[: ,-3:].values
output_data = np.zeros(shape = (len(input_data)  ,1) , dtype = int)

for iter in range(len(input_data)):
    if np.mean(input_data[iter]) >=9:
        output_data[iter][0] =1


