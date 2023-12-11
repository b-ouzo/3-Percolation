import numpy as np
from numba import njit

@njit
def nn(point):
    '''
    auxilary function for the burning method function
    '''
    i, j = point
    return ((i-1,j), (i+1,j), (i,j-1), (i,j+1))

@njit
def burning_method(lattice):
    lattice[0, lattice[0,:]!=0] = 2
    set_on_fire = np.argwhere(lattice[:1,:]==2)
    t = 3
    L = len(lattice)

    wrapped = False
    while True:
        if len(set_on_fire)==0 or wrapped:
            break
        for neighbours in set(map(nn, set_on_fire)):
            for point in neighbours:
                i, j = point
                if i>-1 and i<L and j>-1 and j<L and lattice[i,j]==1:
                    if i == L-1:
                        wrapped = True
                    lattice[i,j] = t
        set_on_fire = np.argwhere(lattice==t)
        t += 1
        
    return lattice, wrapped

@njit
def find_label(label, links):
    while label != links[label]:
        label = links[label]
    return label

@njit
def hkalg(N):
    L, _ = N.shape
    k = 2
    M = [-1, -1]
    links = {2:2, 0:0}
    for i in range(L):
        for j in range(L):
            if N[i,j]==1:
                # 0 if non-existing neighbour
                left = 0 if j-1<0 else links[N[i,j-1]]
                above = 0 if i-1<0 else links[N[i-1,j]]
                # neigbouring non-exist or unoccupied
                if (left==0 and above==0):
                    N[i, j] = k
                    k += 1
                    links[k] = k
                    M.append(1)
                # one of the neighbours is occupied
                elif above>1 and left==0: # the one above the site
                    N[i,j] = above
                    M[N[i,j]] += 1
                elif above==0 and left>1: # the one left to the site
                    N[i,j] = left
                    M[N[i,j]] += 1
                # if both neighbours belong to the same cluster
                elif left == above:
                    N[i,j] = above
                    M[N[i,j]] += 1
                else:
                    smaller, larger = sorted([above, left])
                    N[i,j] = smaller
                    M[smaller] += M[larger] + 1
                    links[larger] = links[smaller]
                    M[larger] *= -1

    for i in range(L):
        for j in range(L):
            N[i,j] = find_label(N[i,j], links)
                    
    return np.array(M), N, links