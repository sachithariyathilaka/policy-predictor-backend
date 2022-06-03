from flask import Flask
from flask import jsonify
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.keras.models import model_from_json

app = Flask(__name__)


@app.route('/train')
def train():
    # load the dataset
    dataset = loadtxt('dataset.csv', delimiter=',')

    # split into input (X) and output (y) variables
    X = dataset[:, 0:6]
    y = dataset[:, 6:11]

    # define the keras model
    model = Sequential()
    model.add(Dense(6, input_dim=6, activation='relu'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(5, activation='softmax'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(X, y, epochs=500, batch_size=100)

    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model.h5")

    return jsonify(accuracy)


@app.route('/predict')
def predict():
    # load the dataset
    dataset = loadtxt('dataset.csv', delimiter=',')

    # split into input (X) and output (y) variables
    X = dataset[:, 0:6]
    y = dataset[:, 6:11]

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model.h5")

    # Make class Predictions with the model
    score = loaded_model.predict_proba([X])
    return jsonify(score[0].tolist())

if __name__ == '__main__':
    app.run()
