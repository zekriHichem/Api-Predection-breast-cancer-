from flask import Flask,jsonify,request
import json
from flask_cors import CORS



import pickle
from statistics import mode
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier




app = Flask(__name__)

CORS(app)

@app.route('/',methods=['POST'])
def hello_world():
    print(request.data)
    dataDict = json.loads(request.data)
    print(dataDict)
    radius_mean = dataDict['radius_mean']
    texture_mean = dataDict['texture_mean']
    perimeter_mean = dataDict['perimeter_mean']
    area_mean = dataDict['area_mean']
    smoothness_mean = dataDict['smoothness_mean']
    compactness_mean = dataDict['compactness_mean']
    concavity_mean = dataDict['concavity_mean']
    concave_points_mean = dataDict['concave_points_mean']
    symmetry_mean = dataDict['symmetry_mean']
    fractal_dimension_mean = dataDict['fractal_dimension_mean']
    data = [[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
             concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean]]
    print("good")
    print(data)

    m1 = pickle.load(open('model_knn.sav', 'rb'))
    m2 = pickle.load(open('model_random_forest.sav', 'rb'))
    m3 = pickle.load(open('model_extra_trees.sav', 'rb'))

    predictions = []
    predictions.append(*m1.predict(data))
    predictions.append(*m2.predict(data))
    predictions.append(*m3.predict(data))

    prediction = mode(predictions)
    print(prediction)
    if prediction:
        result = {'diagnostic': 'M'}
    else:
        result = {'diagnostic': 'B'}

    return jsonify(result)


if __name__ == '__main__':
    app.run('0.0.0.0',8081)
