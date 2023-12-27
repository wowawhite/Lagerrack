#load StandardScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler

#create fake y_train dataframe
y_train = pd.DataFrame ([1,2,3,4], columns =['y_train'])
scalery = StandardScaler().fit(y_train)

#transform the y_test data
y_test = pd.DataFrame ([1,2,3,4], columns =['y_test'])
y_test = scalery.transform(y_train)

# print transformed y_test
print("this is the scaled array:",y_test)

#inverse the y_test data back to 1,2,3,4
y_new = pd.DataFrame (y_test, columns =['y_new'])
y_new_inverse = scalery.inverse_transform(y_new)

# print inversed to original y_test
print("this is the inversed array:",y_new_inverse)