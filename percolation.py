#!/usr/bin/env python

from utils import *
import sys

def main():
    try:
        _, init_config_file = sys.argv
    except ValueError as err:
        print(err)

    # comments inside the file should start with #
    L, T, p0, pk, dp = np.loadtxt(init_config_file)
    print(
        'Loaded parameters:',
        f'{L = }',
        f'{T = }',
        f'{p0 = }',
        f'{pk = }',
        f'{dp = }',
        sep='\n'
    )
    for pi in tqdm(np.arange(p0, pk, dp), desc='Progress '):
        run_model(pi, int(T), int(L), savepath='./data')

def initialize_numba_funcs():
    # run function numba decoraded functions 
    # with some simple arguments
    print('Initializing simulation.')
    hkalg(np.array([[1,1],[1,0]]))
    burning_method(np.array([[1,1],[1,0]]))

if __name__=='__main__':
    initialize_numba_funcs()
    main()
