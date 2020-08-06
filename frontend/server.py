import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import base64
import json
import pickle

import numpy as np
import requests
from flask import Flask, request, jsonify
from flask import render_template
from flask_restful import reqparse, abort, Api, Resource
from backend.core.nlu.engine.probabilistic_engine import ProbabilisticNLUEngine

app = Flask(__name__)
api = Api(app)
engines = {'nlu' : ProbabilisticNLUEngine()}
for engine in engines.keys():
    engines[engine].fit()
    engines[engine].eval()

# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


@app.route('/nlu/parse_intent/', methods=['GET', 'POST'])
def intent_parser():
    if request.method == 'POST':
        # Decoding and pre-processing base64 image
        inp = request.form['message']
        pred = engines['nlu'].parse(inp)
        # Returning JSON response to the frontend
        print(pred)
        intent = pred['intent']
        tags = f"{intent} intent | "
        for tag_name in pred['tags'].keys():
            tags += f"{tag_name}@{pred['tags'][tag_name]['value'][0]}"
        return tags
    else:
        return render_template('main.html')