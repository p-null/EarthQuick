#!/usr/bin/env python

import batch as util
import config

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

from tqdm import tqdm_notebook as tqdm
from joblib import Parallel, delayed
import scipy as sp
import itertools
import gc

from tsfresh.feature_extraction import feature_calculators
import librosa
import pywt

import azure.storage.blob as azureblob
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batch_auth
import azure.batch.models as batchmodels

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

app = Flask(__name__, instance_relative_config=True)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


blob_client = azureblob.BlockBlobService(
    account_name=config._STORAGE_ACCOUNT_NAME,
    account_key=config._STORAGE_ACCOUNT_KEY)

# append_blob_service = AppendBlobService(
#     account_name=AZURE_BLOB_ACCOUNT, account_key=TRAIN_SECRET_KEY)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    html = '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type="file" name="file[]" multiple="">
         <input type="submit" value="Upload">
    </form>
    '''
    if request.method == 'POST':
        # file = request.files['file']
        input_container_name = 'temp'
        blob_client.create_container(input_container_name, fail_on_exist=False)

        file_streams = request.files.getlist("file[]")

        # test = sample_test_gen(uploaded_files)

        # Upload the data files.
        input_files = [
            util.upload_file_to_container(
                blob_client, input_container_name, file_stream)
            for file_stream in file_streams]
        # try:
        #     for file_stream in file_streams:
        #         util.upload_file_to_container(
        #             blob_client, input_container_name, file_stream)
        # except Exception:
        #     print('create_blob_from_stream Exception=' + Exception)

        # Create a Batch service client. We'll now be interacting with the Batch
        # service in addition to Storage
        credentials = batch_auth.SharedKeyCredentials(config._BATCH_ACCOUNT_NAME,
                                                    config._BATCH_ACCOUNT_KEY)
        batch_client = batch.BatchServiceClient(
            credentials,
            base_url=config._BATCH_ACCOUNT_URL)

        try:
            # Create the pool that will contain the compute nodes that will execute the
            # tasks.
            util.create_pool(batch_client, config._POOL_ID)

            # Create the job that will run the tasks.
            util.create_job(batch_client, config._JOB_ID, config._POOL_ID)

            # Add the tasks to the job.
            # TODO
            util.add_tasks(batch_client, config._JOB_ID, input_files)

            # Pause execution until tasks reach Completed state.
            # TODO
            util.wait_for_tasks_to_complete(batch_client,
                                    config._JOB_ID,
                                    datetime.timedelta(minutes=30))
        except batchmodels.BatchErrorException as err:
            util.print_batch_exception(err)
            raise

        raw_data = [f.read().decode("utf-8") for f in file_streams]
        acoustic_data = [raw.split('\n') for raw in raw_data]
        acoustic_data = [j for i in acoustic_data for j in i]
        data = [parse_sample_test(StringIO(raw)) for raw in raw_data]
        data = np.vstack(data)
        features = ['var_num_peaks_2_denoise_simple',
                    'var_percentile_roll50_std_20', 'var_mfcc_mean4',  'var_mfcc_mean18']
        test = pd.DataFrame(data, columns=features)
        test_X = test[features].values

        test_json = []
        for row in test_X:
            test_json.append({
                features[0]: row[0],
                features[1]: row[1],
                features[2]: row[2],
                features[3]: row[3],
            })
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
        #TODO: have test_json by this stage after it has gone through batch processing 
        pdb.set_trace()
        res = requests.post('http://52.224.188.74/score',
                            json=test_json)
        # res = requests.post('http://52.224.186.42/score',
        #                     json=json.loads(test.to_json()))

        feature = 'Line'
        acoustic_data = [int(a) for a in acoustic_data if a.isdigit()]
        time_to_failure = [float(t) for t in json.loads(res.json())['result'].split(',')]
        plot_res = create_plot(feature, acoustic_data, time_to_failure)
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

def create_plot(feature, acoustic_data, time_to_failure):
    if feature == 'Line':
        xScale = np.linspace(0, len(acoustic_data), len(acoustic_data))
        xScale2 = np.linspace(150000//2, len(acoustic_data)+(150000//2), len(time_to_failure))
        # Create traces
        acoustic_data = go.Scatter(
            x=xScale,
            y=acoustic_data,
            name='acoustic data'
        )
        time_to_failure = go.Scatter(
            x=xScale2,
            y=[t*10 for t in time_to_failure],
            name='time to failure',
        )
        layout = go.Layout(
            title='Earthquick',
            yaxis=dict(
                title='acoustic data'
            ),
            yaxis2=dict(
                title='time to failure',
                overlaying='y',
                side='right'
            )
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
