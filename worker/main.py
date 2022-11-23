from builtins import type
import modman
import works
import json
import numpy as np
import torch
from time import sleep
from flask import Flask, Response, jsonify, request, render_template
from time import sleep
import matplotlib.pyplot as plt
import csv
import datetime
from os import getpid
import os

# Model Name / ALIAS (in Client)
ALIAS = 'experiment_01'

ALL_CLIENTS = {}
AGGREGATION_SCORE = {}
AGGREGATION_LOCK = False
numberof_appearance = {}

round = 0
clients_verify_stats = {}
clients_round = {}
global_reward = [[],[]]
COMPLETE = False
CURRENT_UPDATE_COUNT = None
app = Flask(__name__)

@app.route("/")
def hello():
    return "Param Server"

def aggregate(all_params_wscore, global_params):
    global round
    round +=1
    average_params=works.Federated_average(all_val_params)
    return average_params

@app.route('/api/model/aggregate_lock', methods=['POST'])
def aggregate_lock():
    global AGGREGATION_LOCK
    print(f'Got aggregation lock request from server')
    datas = request.get_json()
    update_count = datas['update_count']
    print(f'Aggregation lock: {AGGREGATION_LOCK}')
    if AGGREGATION_LOCK:
        print(f" force to lock is true, cur:{CURRENT_UPDATE_COUNT}/ upd:{update_count}")
        return jsonify({'lock': True})
    else:
        return jsonify({'lock': False})


@app.route('/api/model/aggregate_params', methods=['POST'])
def aggregate_params():
    global AGGREGATION_LOCK
    global CURRENT_UPDATE_COUNT
    AGGREGATION_LOCK = True
    print(f'Got aggregation request from server')
    datas = request.get_json()
    all_params = datas['all_params']
    global_params = datas['global_params']
    CURRENT_UPDATE_COUNT = datas['update_count']
    aggregated_params, round, message = aggregate(all_params_wscore=all_params, global_params=global_params)
    print(f'Aggregation complete.\n Sending aggregated params for round: {round}')
    CURRENT_UPDATE_COUNT +=1
    payload = {
        'params': modman.convert_tensor_to_list(aggregated_params),
        'round': round,
        'n_clients': len(all_params),
        'message': message,
        'update_count': CURRENT_UPDATE_COUNT
    }
    AGGREGATION_LOCK = False
    print(f'Aggregation lock set: {AGGREGATION_LOCK}')
    return jsonify(payload)


if __name__ == "__main__":
    #app.run(debug=True, port=5500)

    # for listening to any network
    app.run(host="0.0.0.0", debug=False, port=5600)
