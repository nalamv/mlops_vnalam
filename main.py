from flask import Flask, request, jsonify
from joblib.parallel import method
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
import joblib
from skopt import BayesSearchCV

app = Flask(__name__) # Initialize Flask app

# with open("./models/iris_classifier.pkl", "rb") as fileObj:
#     iris_Model = pickle.load(fileObj)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Venkata's Flask App!"

@app.route("/getSquare",methods=["POST"])
def getSquare():
    data=request.get_json()
    number=data.get("number")
    return jsonify({"square":number**2})

@app.route("/predict", methods=["POST"]) #<-- this is the controller
def iris_prediction(): # <-- this is view function
    data = request.get_json()
    sepal_lenght = data.get("sl")
    petal_lenght = data.get("pl")
    sepal_width = data.get("sw")
    petal_width = data.get("pw")
    print(f'{sepal_lenght},{sepal_width},{petal_lenght},{petal_width}')
    with open("./models/iris_classifier.pkl", "rb") as fileObj:
        iris_model_new = pickle.load(fileObj)
    flower_type = iris_model_new.predict([[sepal_lenght, sepal_width, petal_lenght, petal_width]])
    return jsonify({"predcited_flower_type": flower_type[0]})


def train_model(max_depth, min_samples_leaf,criterion):
    print(f'{max_depth},{min_samples_leaf},{criterion}')
    df = pd.read_csv("Iris.csv")
    X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    Y = df["Species"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with open('models/iris_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save model
    #joblib.dump(model, "models/iris_classifier.pkl")
    return  accuracy


@app.route("/model-training", methods=["POST"])
def train():
    try:
        # Get JSON payload
        data = request.get_json()
        depth = data.get('depth')
        min_leaf = data.get('min_leaf')
        criterion = data.get('criterion')

        # Train model
        model_new_accuracy=train_model(depth,min_leaf,criterion)

        # Return response
        return jsonify({
            'status': 'success',
            'message': f'Model trained and saved',
            'new_accuracy':model_new_accuracy
        })
    except Exception as e:
     return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == "__main__":
    app.run()