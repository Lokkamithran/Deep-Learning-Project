import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# pre processing
employee_dataset = pd.read_csv('supervised_dataset.csv')
employee_dataset["ip_type"].replace({'default' : 1,'datacenter' : 0},inplace=True)
employee_dataset["source"].replace({'E' : 1,'F' : 0},inplace=True)
employee_dataset["classification"].replace({'outlier' : 1,'normal' : 0},inplace=True)

employee_dataset.dropna(subset=['inter_api_access_duration(sec)'], inplace=True)
employee_dataset.dropna(subset=['api_access_uniqueness'], inplace=True)

# print(employee_dataset.isnull().any())

# exit()

# print(employee_dataset.classification.value_counts())
train, test = train_test_split(employee_dataset, test_size=0.3)
hidden_units = 100
learning_rate = 0.005
no_epochs=100

# scaler=StandardScaler()
model = Sequential()
model.add(Dense(hidden_units, input_dim=9, activation='tanh'))
model.add(Dense(hidden_units, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

sgd=optimizers.legacy.SGD(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['acc'])

train_x=train.iloc[:,2:11]
train_y=train.iloc[:,11]

model.fit(train_x, train_y, epochs=no_epochs, batch_size=1,  verbose=2)

test_x=test.iloc[:,2:11]
predictions = model.predict(test_x)

rounded = [int(round(x[0])) for x in predictions]

og_values = test.iloc[:,11]
ind = 0
eq = 0
for rec in og_values:
    if rec == rounded[ind]:
        eq += 1
    ind += 1

print("\n\nAccuracy: ", eq/len(rounded)*100, "%\n")