
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle 

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)