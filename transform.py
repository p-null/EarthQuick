import pdb
import os

import numpy as np
import pandas as pd

from tqdm import tqdm_notebook as tqdm
from joblib import Parallel, delayed
import scipy as sp
import itertools
import gc

from tsfresh.feature_extraction import feature_calculators
import librosa
import pywt

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

noise = np.random.normal(0, 0.5, 150000)

def denoise_signal_simple(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    #univeral threshold
    uthresh = 10
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard')
                 for i in coeff[1:])
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')


def feature_gen(z):
    X = pd.DataFrame(index=[0], dtype=np.float64)

    z = z + noise
    z = z - np.median(z)

    den_sample_simple = denoise_signal_simple(z)
    mfcc = librosa.feature.mfcc(z)
    mfcc_mean = mfcc.mean(axis=1)
    percentile_roll50_std_20 = np.percentile(
        pd.Series(z).rolling(50).std().dropna().values, 20)

    X['var_num_peaks_2_denoise_simple'] = feature_calculators.number_peaks(
        den_sample_simple, 2)
    X['var_percentile_roll50_std_20'] = percentile_roll50_std_20
    X['var_mfcc_mean18'] = mfcc_mean[18]
    X['var_mfcc_mean4'] = mfcc_mean[4]

    return X


def sample_test_gen(uploaded_files):
    X = pd.DataFrame()
    submission = pd.read_csv(uploaded_files, sep='\n', index_col='seg_id')
    result = Parallel(n_jobs=4, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(
        delayed(parse_sample_test)(seg_id, uploaded_files) for seg_id in tqdm(submission.index))
    data = [r.values for r in result]
    data = np.vstack(data)
    X = pd.DataFrame(data, columns=result[0].columns)
    return X

def parse_sample_test(seg_id):
    sample = pd.read_csv('../input/test/' + seg_id + '.csv',
                         dtype={'acoustic_data': np.int32})
    delta = feature_gen(sample['acoustic_data'].values)
    delta['seg_id'] = seg_id
    return delta
##

def batch_process(file_path=None):
    # X = pd.DataFrame()
    sample = pd.read_csv(file_path, sep='\n', dtype={
                             'acoustic_data': np.int32})
    result = feature_gen(sample['acoustic_data'].values)
    # pdb.set_trace()
    # data = [r.values for r in result]
    data = np.vstack(result.values)
    # X = pd.DataFrame(data, columns=result[0].columns)
    features = ['var_num_peaks_2_denoise_simple',
                'var_percentile_roll50_std_20', 'var_mfcc_mean4',  'var_mfcc_mean18']
    test = pd.DataFrame(data, columns=features)
    test_X = test[features].values
    return test


    # raw_data = [f.read().decode("utf-8") for f in file_streams]
    # acoustic_data = [raw.split('\n') for raw in raw_data]
    # acoustic_data = [j for i in acoustic_data for j in i]
    # data = [parse_sample_test(StringIO(raw)) for raw in raw_data]
    # data = np.vstack(data)
    # features = ['var_num_peaks_2_denoise_simple',
    #             'var_percentile_roll50_std_20', 'var_mfcc_mean4',  'var_mfcc_mean18']
    # test = pd.DataFrame(data, columns=features)
    # test_X = test[features].values


if __name__ == '__main__':
    # Map command line arguments to function arguments.
    batch_process(*sys.argv[1:])
