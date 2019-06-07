import os
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
from azure.storage.blob import BlockBlobService, AppendBlobService
import string
import random
import requests

app = Flask(__name__, instance_relative_config=True)

# Azure Storage account access key
TRAIN_SECRET_KEY = str(os.getenv("TRAIN_SECRET_KEY"))
AZURE_BLOB_ACCOUNT = str(os.getenv("AZURE_BLOB_ACCOUNT"))
IS_LOCAL = bool(os.getenv("IS_LOCAL"))
container = "testcontainer"  # TODO: switch to prod container once ready (using environ vars)

# blob_service = BlockBlobService(account_name=account, account_key=key)
blob_service = AppendBlobService(
    account_name=AZURE_BLOB_ACCOUNT, account_key=TRAIN_SECRET_KEY)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        # filename = secure_filename(file.filename)
        # fileextension = filename.rsplit('.', 1)[1]
        # Randomfilename = id_generator()
        # filename = Randomfilename + '.' + fileextension
        try:
            pass
            # blob_service.create_blob(container, "UM77VCHMQK16HOSDFTA0THU3LX66JX0T.txt", if_none_match="*")
            # blob_service.create_blob_from_stream(container, "UM77VCHMQK16HOSDFTA0THU3LX66JX0T.txt", file)
        except Exception:
            print('create_blob Exception=' + Exception)
        try:
            blob_service.append_blob_from_stream(container, "UM77VCHMQK16HOSDFTA0THU3LX66JX0T.txt", file)
        except:
            print('append_blob_from_stream Exception=' + Exception)
        ref = 'http://' + AZURE_BLOB_ACCOUNT + '.blob.core.windows.net/' + container + '/' + 'UM77VCHMQK16HOSDFTA0THU3LX66JX0T.txt' #filename
        return '''
        <!doctype html>
        <title>File Link</title>
        <h1>Uploaded File Link</h1>
        <p>''' + ref + '''</p>
        <img src="''' + ref + '''">
        '''
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


def id_generator(size=32, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


if __name__ == '__main__':
    app.run(debug=True)
