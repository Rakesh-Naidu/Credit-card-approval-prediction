import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__)
meanstd = pd.read_csv("meanstd.csv",index_col="index")
model = pickle.load(open('finalized_model_knn.sav','rb'))

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return render_template('index.html')

def scale(lis):
    scaled = []
    ans = (lis[0]-meanstd.loc['PriorDefault_t']['mean'])/meanstd.loc['PriorDefault_t']['std']
    scaled.append(ans)
    ans = (lis[1]-meanstd.loc['YearsEmployed']['mean'])/meanstd.loc['YearsEmployed']['std']
    scaled.append(ans)
    ans = (lis[2]-meanstd.loc['CreditScore']['mean'])/meanstd.loc['CreditScore']['std']
    scaled.append(ans)
    ans = (lis[3]-meanstd.loc['Income']['mean'])/meanstd.loc['Income']['std']
    scaled.append(ans)
    # print(scaled)
    return scaled

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            features = [float(x) for x in request.form.values()]
            scaled_feat = scale(features)
            final_feat = [np.array(scaled_feat)]
            prediction = model.predict(final_feat)
            # print(prediction)
            if prediction==1:
                return render_template('index.html', prediction_text = "Credit Card Approved 🥳")
            else:
                return render_template('index.html', prediction_text = "Credit Card Not Approved 😔")
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
