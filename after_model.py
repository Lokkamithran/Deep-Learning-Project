import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# pre processing
employee_dataset = pd.read_csv('supervised_dataset.csv')
employee_dataset = employee_dataset.iloc[:,2:]
employee_dataset["ip_type"].replace({'default' : 1,'datacenter' : 2},inplace=True)
employee_dataset["source"].replace({'E' : 1,'F' : 2},inplace=True)
employee_dataset["classification"].replace({'outlier' : 1,'normal' : 0},inplace=True)

employee_dataset.dropna(subset=['inter_api_access_duration(sec)'], inplace=True)
employee_dataset.dropna(subset=['api_access_uniqueness'], inplace=True)

# print(employee_dataset.dtypes)

scaler = MinMaxScaler()
employee_dataset[employee_dataset.columns] = scaler.fit_transform(employee_dataset[employee_dataset.columns])
# print(employee_dataset.head())

train, test = train_test_split(employee_dataset, test_size=0.3)
hidden_units = 100
learning_rate = 0.0001
no_epochs = 100

# scaler=StandardScaler()
model = Sequential()
model.add(Dense(hidden_units, input_dim=9, activation='tanh'))
model.add(Dense(hidden_units, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

sgd=optimizers.legacy.SGD(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['acc'])

train_x=train.iloc[:,:9]
train_y=train.iloc[:,9]

smote = SMOTE()
train_x_oversampled, train_y_oversampled = smote.fit_resample(train_x, train_y)
X_train = pd.DataFrame(train_x_oversampled)
Y_train = pd.DataFrame(train_y_oversampled, columns=train_y.to_frame().columns)
# print(Y_train['~classification'].value_counts()[0.0], Y_train['classification'].value_counts()[1.0], Y_train['classification'].value_counts())

model.fit(train_x_oversampled, train_y_oversampled, epochs=no_epochs, batch_size=1,  verbose=2)

test_x=test.iloc[:,:9]
predictions = model.predict(test_x)

rounded = [int(round(x[0])) for x in predictions]

og_values = test.iloc[:,9]
ind = 0
eq = 0
for rec in og_values:
    if rec == rounded[ind]:
        eq += 1
    ind += 1

print("\n\nAccuracy: ", eq/len(rounded)*100, "%\n")