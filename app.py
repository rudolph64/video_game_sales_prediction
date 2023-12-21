from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['plt']
    data2 = request.form['yor']
    data3 = request.form['gen']
    data4 = request.form['pub']
    data5 = request.form['Critic_Score']
    data6 = request.form['Critic_Count']
    data7 = request.form['User_Score']
    data8 = request.form['Rating']
    data9 = request.form['NA_Sales']
    data10 = request.form['EU_Sales']
    data11 = request.form['Other_Sales']
    data12 = request.form['JP_Sales']
    arr = np.array([[float(data1), float(data2), float(data3), float(data4), float(data5),
                 float(data6), float(data7), float(data8), float(data9), float(data10),
                 float(data11), float(data12)]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
