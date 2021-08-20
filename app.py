import pandas as pd
import os
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('logit.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	input_features = { k: [float(v)] for k, v in request.form.items()}
	features = pd.DataFrame.from_dict(input_features)
	prediction = model.predict(features)
	
	if prediction[0] == 1:
		survival = 'survived'
	else:
		survival = 'did not survive'
	
	return render_template('index.html', prediction_text='Passenger {}'.format(survival))

if __name__ == '__main__':
	port = os.environ.get("PORT", 5000)
	app.run(debug=False, host="0.0.0.0", port=port)
