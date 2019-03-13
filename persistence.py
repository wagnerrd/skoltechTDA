import pandas as pd
import numpy as np
import pickle
from ripser import ripser, plot_dgms
from multiprocessing import Pool
import argparse

'''
Usage:
python persistence.py /home/robert/forex/data/currency/forex.pkl /home/robert/forex/data/tda/ --predict_delay 15 --dim 3 --Tau 5 --dT 2 --n_processes 4

path_to_tda_data is the directory where you want the output pickle file. It will be saved as:
persistences_15-3-5-2.pkl

15: prediction delay
3: dimension of sliding window
5: in-line jump (see getSlidingWindow function)
2: timesteps to skip between lines in sliding window (also see getSlidingWindow function)
'''

# Fixes missing values. I'll probably do mean interpolation eventually, but not yet
def fixna(ts):
    return ts[~np.isnan(ts)]


def getSlidingWindow(x, dim, Tau, dT):
    '''
    This function takes time series x (without time-part)
    and returns a massive X, which has sliding windows as columns.
    dim=3, Tau=5, dT=2 on range() object yields:
    [0, 5, 10]
    [2, 7, 12]
    [4, 11, 14] etc.
    '''
    N = len(x)
    NWindows = int(np.floor((N-dim*Tau)/dT)) # The number of windows
    if NWindows <= 0:
        print("Error: Tau too large for signal extent")
        return np.zeros((3, dim))
    X = np.zeros((NWindows, dim)) # Create a 2D array which will store all windows
    idx = np.arange(N)
    for i in range(NWindows):
        # Figure out the indices of the samples in this window
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))+2
        if end > len(x):
            X = X[0:i, :]
            break
        # Do spline interpolation to fill in this window, and place
        # it in the resulting array
        X[i, :] = x[idxx] #interp.spline(idx[start:end+1], x[start:end+1], idxx)
    return X


# This function takes in a large time series along with index of window start, length of window
# returns +1 for increase one plength after end of series, -1 for decrease or level values, along with window
# return None for end of December
def classify(timeseries, index, sectionlength=1440):
#     plength = 5
#     dim = 3
#     Tau = 3
#     dT = 2
    if len(timeseries) < index + sectionlength + plength:
        return 0, [[]]
    section = timeseries.iloc[index:index+sectionlength]
    sectionendval = section.iloc[-1]
    futureval = timeseries.iloc[index+sectionlength+plength]
    increase = futureval - sectionendval
    slidingwindow = getSlidingWindow(section.values, dim, Tau, dT)
    if increase > 0:
        return 1, slidingwindow
    else:
        return 0, slidingwindow


def make_filtration(ts, sectionlength=1440):
    dim0cls0 = []
    dim0cls1 = []
    dim1cls0 = []
    dim1cls1 = []
    cleants = fixna(ts)
    for i in range(0, len(cleants)-2*sectionlength, sectionlength):
        cls, window = classify(cleants, i)
        filtration = ripser(window)['dgms']
        if cls == 0:
            dim0cls0.append(filtration[0])
            dim1cls0.append(filtration[1])
        elif cls == 1:
            dim0cls1.append(filtration[0])
            dim1cls1.append(filtration[1])
    return (dim0cls0, dim0cls1, dim1cls0, dim1cls1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate persistence diagrams')
    parser.add_argument('path_to_forex_data')
    parser.add_argument('path_to_tda_data')
    parser.add_argument('--predict_delay')
    parser.add_argument('--dim')
    parser.add_argument('--Tau')
    parser.add_argument('--dT')
    parser.add_argument('--n_processes', required=False)
    
    args = vars(parser.parse_args())
    indata = args['path_to_forex_data']
    outdata = args['path_to_tda_data']
    plength = int(args['predict_delay'])
    dim = int(args['dim'])
    Tau = int(args['Tau'])
    dT = int(args['dT'])
    if args['n_processes'] is None:
        n_processes = 4
    else:
        n_processes = int(args['n_processes'])
    
    #path = '/home/robert/forex'
    #filename = path + '/data/currency/forex.pkl'
    pkl_file = open(indata, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()

    with Pool(n_processes) as pool:

        # we need a sequence of columns to pass pool.map
        seq = [data[col_name] for col_name in data.columns]

        # pool.map returns results as a list
        results_list = pool.map(make_filtration, seq)
        pool.close()
        pool.join()


    dim0cls0 = []
    dim0cls1 = []
    dim1cls0 = []
    dim1cls1 = []
    for item in results_list:
        dim0cls0 = dim0cls0 + item[0]
        dim0cls1 = dim0cls1 + item[1]
        dim1cls0 = dim1cls0 + item[2]
        dim1cls1 = dim1cls1 + item[3]
    filtrations = [dim0cls0, dim0cls1, dim1cls0, dim1cls1]
    
    outdata = outdata + 'persistences_' + str(plength) + '-' + str(dim) + '-' + str(Tau) + '-' + str(dT) + '.pkl'
    output = open(outdata, 'wb')
    pickle.dump(filtrations, output)
    output.close()
