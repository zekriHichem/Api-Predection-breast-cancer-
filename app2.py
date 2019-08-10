from flask import Flask,jsonify,request


import json
from flask_cors import CORS
from flask_uploads import UploadSet, configure_uploads, IMAGES

from PIL import Image


import pickle
from statistics import mode
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

import torch
from torchvision import transforms
import torchvision.models as models
from torch import nn


app = Flask(__name__)



photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)
CORS(app)

class_names = {
    0: "CALC",
    1: "CIRC",
    2: "SPIC",
    3: "MISC",
    4: "ARCH",
    5: "ASYM",
    6: "NORM"
}


def predict_anomaly_class(img_path, model):
    transformations = transforms.Compose((
        transforms.Resize(224),
        transforms.ToTensor()
    ));

    img = Image.open(img_path)
    #img = Image.new("RGB", img.size)
    x = transformations(img)
    # add batch_size of 1
    x = x.unsqueeze(0)
    out = model(x)
    _, class_idx = out.topk(1)
    class_idx = int(class_idx)
    return class_names[class_idx]


class_names_s = {
    0: "B",
    1: "M"
}


def predict_severity_class(img_path, model):
    transformations = transforms.Compose((
        transforms.Resize(224),
        transforms.ToTensor()
    ));

    img = Image.open(img_path)
    #img = Image.new("RGB", img.size)
    x = transformations(img)
    # add batch_size of 1
    x = x.unsqueeze(0)
    out = model(x)
    _, class_idx = out.topk(1)
    class_idx = int(class_idx)
    return class_names_s[class_idx]


@app.route('/',methods=['POST'])
def hello_world():
    print("hhhhhhh")
    print((request.files["image"] is None))

    print(request.files["image"])
    if request.method == 'POST' and 'image' in request.files:
        filename = photos.save(request.files['image'])
    filepath = "static/img/" + filename
    print(filename)
    result =''
    model_class = getClassModel()
    resultclass=predict_anomaly_class(filepath,model_class)
    if resultclass != "NORM" :
        model_severity_class = getDiagModel()
        resultServityClass = predict_severity_class(filepath,model_severity_class)
        result = {
            "classe":resultclass,
            "diag":resultServityClass
        }
    else:
        result={
            "classe":resultclass,
            "diag":"null"
        }

    print(result)
    return jsonify(result)


def getDiagModel():
    model_severity_class = models.resnet18()
    num_ftrs = model_severity_class.fc.in_features
    model_severity_class.fc = nn.Linear(num_ftrs, 2)
    state_dict = torch.load('model-severity-class-prediction.pth', map_location='cpu')
    model_severity_class.load_state_dict(state_dict)
    model_severity_class.eval()
    return model_severity_class


def getClassModel():
    model_class = models.resnet18()
    num_ftrs = model_class.fc.in_features
    model_class.fc = nn.Linear(num_ftrs, 7)
    state_dict = torch.load('model-class-prediction.pth', map_location='cpu')
    model_class.load_state_dict(state_dict)
    model_class.eval()
    return model_class


if __name__ == '__main__':
    app.run('0.0.0.0',8082)
