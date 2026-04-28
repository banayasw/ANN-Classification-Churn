import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle

# Load dataset
data = pd.read_csv('Churn_Modelling.csv')
# print(data.head())

# Preprocess the data
# drop irrelevant columns
data = data.drop(['RowNumber','CustomerId','Surname'],axis=1)
# print(data.head())

# Encode categorical variables
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
# print(data)

# One hot encoding 'Geography'
from sklearn.preprocessing import OneHotEncoder
onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']])
# print(geo_encoder)

onehot_encoder_geo.get_feature_names_out(['Geography'])

geo_encoded_df = pd.DataFrame(geo_encoder.toarray(),columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
# print(geo_encoded_df)

# Comine one hot encoder columns with the  original data
data = pd.concat([data.drop('Geography',axis=1),geo_encoded_df],axis=1)
# print(data.head())


# Save the encoder and scaler
# with open('label_encoder_gender.pkl','wb') as file:
#     pickle.dump(label_encoder_gender,file)

# with open('onehot_encoder_geo.pkl','wb') as file:
#     pickle.dump(onehot_encoder_geo,file)


# Divide the dataset into independent and dependent features
X = data.drop('Exited',axis=1)
y = data['Exited']

# Split the data in training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Scale these features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# with open('scaler.pkl','wb') as file:
#     pickle.dump(scaler,file)


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers  import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

## Build our ANN model

model = Sequential([Dense(64,activation='relu',input_shape=(X_train.shape[1],)), ##HL1 connected with input layer
                    Dense(32,activation='relu'), ## HL2
                    Dense(1,activation='sigmoid') ## output layer

])

# model.summary()

opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
loss=tensorflow.keras.losses.BinaryCrossentropy()

# Compile the model
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])

# Setup the tensorboard
import datetime
log_dir = "logs/fit"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)


# Set up early stopping
early_stopping_back = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

## Train the model

history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,
            callbacks=[tensorflow_callback,early_stopping_back])

model.save('model.h5')

# Load TensorBoard Extension
# %load_ext tensorboard

# %tensorboard --logdir logs/fit20260427-194710

from tensorflow.keras.models import load_model

## Load the scaler.pkl, one hot and trained model
model = load_model('model.h5')

## load the encoders and scaler

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


# Example input_data

input_data = {
    'CreditScore':600,
    'Geography':'France',
    'Gender':'Male',
    'Age':40,
    'Tenure':3,
    'Balance':60000,
    'NumOfProducts':2,
    'HasCrCard':1,
    'IsActiveMember':1,
    'EstimatedSalary':50000
}

geo_encoded = onehot_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_df = pd.DataFrame([input_data])

# Encode categorical variables
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

## Concatination with one hot encoded
input_df = pd.concat([input_df.drop("Geography",axis=1),geo_encoded_df],axis=1)

# Scaling the input_data
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

prediction_probability = prediction[0][0]
print(prediction_probability)

