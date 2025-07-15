import sys
import numpy as np
import torch
from scipy.signal.filter_design import cheby1
from scipy.signal import dlti
from scipy.signal.signaltools import filtfilt, lfilter

     
def decimate(x, rate, ripple=0.05, n=None, ftype='iir', axis=-1, zero_phase=True):
    x = np.asarray(x)
    
    """   ftype == 'iir'   """
    if n is None: 
        n = 8
    # system = dlti(*cheby1(n, ripple, 0.8 / q))
    system = dlti(*cheby1(n, ripple, rate))
    b, a = system.num, system.den

    result_type = x.dtype
    if result_type.kind in 'bui':
        result_type = np.float64
    b = np.asarray(b, dtype=result_type)
    a = np.asarray(a, dtype=result_type)

    sl = [slice(None)] * x.ndim
    a = np.asarray(a)

    if zero_phase:
        y = filtfilt(b, a, x, axis=axis)
    else:
        y = lfilter(b, a, x, axis=axis)
    sl[axis] = slice(None, None, 2)

    # return y[tuple(sl)]
    return y 

if __name__ == '__main__':
    print('filter.py')
    input = torch.zeros(112000,)
    x = input.numpy()
    y = decimate(x, 0.5) # cut off frequency = 0.5
    print(y.shape)
