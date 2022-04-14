from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def predict():

    if request.method == 'GET':
        return render_template("linreg.html")
    elif request.method == 'POST':
        print(request.form)
        feats=list(dict(request.form).values())
        print(feats)
        print(type(feats[1]))
        feats=np.array(feats).reshape(1,-1)
        print(feats)
        model = joblib.load("model-development/model.pkl")
        result = model.predict(feats)
        result=result.flatten()
        result=result[0]
        print(result)
        return render_template('linreg.html', result=result)
    else:
        return "Unsupported Request Method"

if __name__ == '__main__':
    app.run(port=5000, debug=True)
