import numpy as np
from tqdm import tqdm
from algorithms import *
from multiprocessing import Pool
from pathlib import Path

def analyze_lattice(lattice):
    """
    Analyze given lattice for clusters sizes distribution
    and if there exisits a spanning (wrapping) cluster
    with use of the burning method and Hoshen-Kopelman
    algorith (both implemented in algorithms.py file)

    Args:
        lattice: (2d np.ndarray)
            input 2-dimensional square lattice to analyze

    Returns:
        masses: np.ndarray
            array of clusters masses (aka sizes)
        wrapped: boolean
            True if the biggest cluster streches from top
            to the bottom of the lattice else False
    """
    _, wrapped = burning_method(lattice.copy())
    
    # cluster sizes
    masses, _, _ = hkalg(lattice.copy())
    masses = masses[masses>0]
    
    return masses, wrapped

def analyze_results(results, T):
    """
    _summary_

    Args:
        results (_type_): _description_
        T (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_of_wrapped = 0
    # num of columns to allocate
    to_allocate = max(results, key=lambda x: x[0].size)[0].size
    clusters_sizes = np.zeros([T, to_allocate] )
    # separete mass arrays from boolean values
    for i, res in enumerate(results):
        clusters_sizes[i,:res[0].size] = res[0]
        if res[1]: # if wrapped
            num_of_wrapped += 1
    return clusters_sizes.astype(np.int16), num_of_wrapped

def save_distribiution(clusters_sizes, savepath, params):
    """
    _summary_

    Args:
        clusters_sizes (_type_): _description_
        savepath (_type_): _description_
    """
    sizes, counts = np.unique(clusters_sizes[clusters_sizes != 0].astype(np.int64), return_counts=True)
    savepath = Path(savepath)
    p, L, T = params
    np.savetxt(
        savepath/f'Dist_p{p:.3f}L{L}T{T}.txt',
        np.array([sizes, counts]).T,
        delimiter='\t', fmt=['%d', '%d']
    )

def run_model(
            p, T, L, parallel=True, n_workers=None,
            save_results_to_txt=True, savepath='.'
        ):
    '''
    for given probability p, # of trials T and lattice size L
    calculates spanning cluster probrability,
    cluster distribution
    and average size of the biggest cluster
    '''
    # set the numer of workers if in parallel mode
    # if n_workers is not specified then os.cpu_count() 
    # is being used instead
    n_workers = n_workers if parallel else 1
    # generate generator of random lattices
    rng = np.random.default_rng()
    lattices = (rng.choice([0,1], size=(L,L), p=[1-p, p]) for _ in range(T))

    # initialize the pool of workers
    with Pool(n_workers) as pool:
        # returns list of tuples (masses: np.ndarray, wrapped: boolean)
        results = list(tqdm(
            pool.imap(analyze_lattice, lattices, chunksize=100),
            total=T, leave=False, desc=f'{p = :.3f}'
        ))
    
    ######### results processing #########
    clusters_sizes, num_of_wrapped = analyze_results(results, T)

    spanning_cluster_prob = num_of_wrapped / T
    avg_max_size = clusters_sizes.max(axis=1).mean()

    ######### saving clusters distribiution #########
    if save_results_to_txt:
        save_distribiution(clusters_sizes, savepath, (p, L, T))

    ######### saving others #########
    filename = f'Ave_L{L}T{T}.txt'
    with open(filename, 'a') as f:
        f.write(f'{p:.3f}\t{spanning_cluster_prob}\t{avg_max_size}\n')
        
    return None if save_results_to_txt else spanning_cluster_prob, avg_max_size, clusters_sizes