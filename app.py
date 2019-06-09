#!/usr/bin/env python

import pdb
import os
from flask import Flask, request, redirect, url_for, flash, render_template
# from werkzeug import secure_filename
# from azure.storage.blob import BlockBlobService, AppendBlobService
# import string
# import random
import requests

import json
import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np

app = Flask(__name__, instance_relative_config=True)
app.config['MAX_CONTENT_LENGTH'] = 0.5 * 1024 * 1024

# Azure Storage account access key
TRAIN_SECRET_KEY = str(os.getenv("TRAIN_SECRET_KEY"))
AZURE_BLOB_ACCOUNT = str(os.getenv("AZURE_BLOB_ACCOUNT"))
IS_LOCAL = bool(os.getenv("IS_LOCAL"))
container = "testcontainer"  # TODO: switch to prod container once ready (using environ vars)

# blob_service = BlockBlobService(account_name=account, account_key=key)
# blob_service = AppendBlobService(
#     account_name=AZURE_BLOB_ACCOUNT, account_key=TRAIN_SECRET_KEY)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    html = '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
    if request.method == 'POST':
        file = request.files['file']
        
        # filename = secure_filename(file.filename)
        # fileextension = filename.rsplit('.', 1)[1]
        # pdb.set_trace()
        # with open(file.read(), 'r') as f:
        #     pdb.set_trace()
        #     first_line = f.readline()
        #     if not is_number(first_line):
        #         next(f)
        #     for idx, line in enumerate(f):
        #         if not is_number(line):
        #             flash(f'Line {idx} is not a number')
        #             return html
        res = requests.post('http://52.224.186.42/score',
                            json=json.loads(file.read()))

        feature = 'Line'
        plot_res = create_plot(feature, json.loads(res.json()))
        return render_template('index.html', plot=plot_res)




        # return '''
        # <!doctype html>
        # <title>Prediction</title>
        # <p>''' + res.json() + '''</p>



        # <script>
        # var _table_ = document.createElement('table'),
        # _tr_ = document.createElement('tr'),
        # _th_ = document.createElement('th'),
        # _td_ = document.createElement('td');

        # // Builds the HTML Table out of myList json data from Ivy restful service.
        # function buildHtmlTable(arr) {
        # var table = _table_.cloneNode(false),
        #     columns = addAllColumnHeaders(arr, table);
        # for (var i = 0, maxi = arr.length; i < maxi; ++i) {
        #     var tr = _tr_.cloneNode(false);
        #     for (var j = 0, maxj = columns.length; j < maxj; ++j) {
        #     var td = _td_.cloneNode(false);
        #     cellValue = arr[i][columns[j]];
        #     td.appendChild(document.createTextNode(arr[i][columns[j]] || ''));
        #     tr.appendChild(td);
        #     }
        #     table.appendChild(tr);
        # }
        # return table;
        # }

        # // Adds a header row to the table and returns the set of columns.
        # // Need to do union of keys from all records as some records may not contain
        # // all records
        # function addAllColumnHeaders(arr, table) {
        # var columnSet = [],
        #     tr = _tr_.cloneNode(false);
        # for (var i = 0, l = arr.length; i < l; i++) {
        #     for (var key in arr[i]) {
        #     if (arr[i].hasOwnProperty(key) && columnSet.indexOf(key) === -1) {
        #         columnSet.push(key);
        #         var th = _th_.cloneNode(false);
        #         th.appendChild(document.createTextNode(key));
        #         tr.appendChild(th);
        #     }
        #     }
        # }
        # table.appendChild(tr);
        # return columnSet;
        # }

        # document.body.appendChild(buildHtmlTable([''' + res.json()  + ''']));
        # </script>

        # '''
                    
        # Randomfilename = id_generator()
        # filename = Randomfilename + '.' + fileextension
        # try:
        #     pass
        #     # blob_service.create_blob(container, "UM77VCHMQK16HOSDFTA0THU3LX66JX0T.txt", if_none_match="*")
        #     # blob_service.create_blob_from_stream(container, "UM77VCHMQK16HOSDFTA0THU3LX66JX0T.txt", file)
        # except Exception:
        #     print('create_blob Exception=' + Exception)
        # try:
        #     blob_service.append_blob_from_stream(container, "UM77VCHMQK16HOSDFTA0THU3LX66JX0T.txt", file)
        # except:
        #     print('append_blob_from_stream Exception=' + Exception)
        # ref = 'http://' + AZURE_BLOB_ACCOUNT + '.blob.core.windows.net/' + container + '/' + 'UM77VCHMQK16HOSDFTA0THU3LX66JX0T.txt' #filename
        # return '''
        # <!doctype html>
        # <title>File Link</title>
        # <h1>Uploaded File Link</h1>
        # <p>''' + ref + '''</p>
        # <img src="''' + ref + '''">
        # '''


    return html

def create_plot(feature, pred_json):
    if feature == 'Line':
        count = 10 #TODO: make count dynamic
        xScale = np.linspace(0, 1, count)
        acoustic_data_scale = np.random.randn(count)
        time_to_failure_scale = pred_json['result'].split(',')

        # Create traces
        acoustic_data = go.Scatter(
            x=xScale,
            y=acoustic_data_scale
        )
        time_to_failure = go.Scatter(
            x=xScale,
            y=time_to_failure_scale
        )
        data = [acoustic_data, time_to_failure]
        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        return graphJSON

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# def id_generator(size=32, chars=string.ascii_uppercase + string.digits):
#     return ''.join(random.choice(chars) for _ in range(size))


if __name__ == '__main__':
    app.run(debug=True)
