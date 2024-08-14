
import numpy as np


def hp_range(grid_file='<ABSOLUTE_PATH_TO_CODE_HOMEFOLDER>/examples/obd/conf/hyper_params.txt'):
    hp_file = open(grid_file, "w")
    #Ns = np.concatenate((np.logspace(2,5,6, dtype=np.int64),np.logspace(2,5,6, dtype=np.int64)*2,np.logspace(2,5,6, dtype=np.int64)*5))
    Ns = np.logspace(2,6,10, dtype=np.int64)
    Ns.sort()
    print(Ns)
    N_iter = 100
    N_actions = [25, 50, 75, 100]
    for N in Ns:
        for iter in range(N_iter):
            #for n_action in N_actions:
            text = '--iteration ' + str(iter) + ' ' + ' ' + '--N ' + str(N) + '\n'
            hp_file.write(text)
    hp_file.close() 

if __name__ == '__main__':
    hp_range()