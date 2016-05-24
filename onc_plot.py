"""
Plots ONC vs neighborhood size

Usage: python onc_plot.py

Author(s): Wei Chen (wchen459@umd.edu)
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import glob
from itertools import cycle
import ConfigParser


def get_neighbors(X, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    return indices

def num_intersections(a, b):
    return len(set(a).intersection(b))-1

def num_topo_ordered(a, b):
    # Return the number of items in list b that has the right topological order as in list a
    a = list(a)
    b = list(b)
    count = 0
    pre = -1
    for item in a:
        if item in b:
            if b.index(item)>pre:
                count += 1
            pre = b.index(item)
    return count-1

def get_metric(X_in, X_out, n_neighbors, metric='UNC'):
    # Compute the unordered neighborhoods coincidence (UNC) index
    indices_in = get_neighbors(X_in, n_neighbors)
    indices_out = get_neighbors(X_out, n_neighbors)
    m = X_in.shape[0]
    u = 0
    for i in range(m):
        if metric == 'UNC':
            u += num_intersections(indices_in[i], indices_out[i])
        elif metric == 'ONC':
            u += num_topo_ordered(indices_in[i], indices_out[i])
        else:
            print 'Wrong metric!'
    return float(u)/m/n_neighbors

def save_fig(metric):
    
    range_k = range(2, 116, 2)
    
    plt.figure()
    linewidth = 2
    plt.rc("font", size=fontsize)
    
    sources = {'sf_linear': 'Superformula (linear)',
               'sf_s_nonlinear': 'Superformula (slightly nonlinear)',
               'sf_v_nonlinear': 'Superformula (very nonlinear)',
               'glass': 'Glass'}
    plt.title(sources[source])
    
    plt.xticks(np.arange(min(range_k)-1, max(range_k)+2, 20, dtype=np.int))
    plt.xlabel('Neighborhood size')
    plt.ylabel(metric)
    colors = ['b', 'g', 'r', 'y', 'c', 'm', 'y', 'k']
    lss = ['-', '--', '-.', ':']
    colorcycler = cycle(colors)
    lscycler = cycle(lss)
    lines = []
    txtfile = open(save_dir2+"mean_onc.txt", "w")
    names = ['PCA', 'Kernel PCA', 'Autoencoder']

    for name in names:
        print 'Getting ' + metric + ': ' + name

        fname = save_dir2 + name + '.csv'
        X_dr = np.genfromtxt(fname, delimiter=',')
        X_dr = np.reshape(X_dr, (X_dr.shape[0], -1))
        
        l = []
        for n_neighbors in range_k:
            c = get_metric(X, X_dr, n_neighbors, metric=metric)
            l.append(c)

        onc_mean = np.mean(l)
        print onc_mean
        
        txtfile.write('%.3f\n' % onc_mean)
        line, = plt.plot(range_k, l, color=next(colorcycler), ls=next(lscycler), linewidth=linewidth, label=name)
        lines.append(line)
            
    plt.legend(loc=4, handles=lines, fontsize=16)
    plt.savefig(save_dir2+metric+'.png', dpi=300)

if __name__ == "__main__":
    
    fontsize = 18
    
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
       
    RESULTS_DIR = config.get('Global', 'RESULTS_DIR')
    
    source = config.get('Global', 'source')
    example_dir = RESULTS_DIR + source + '/'
    
    n_control_points = config.getint('Global', 'n_control_points') 
    save_dir1 = example_dir + 'n_control_points = ' + str(n_control_points) + '/'
    X = np.genfromtxt(save_dir1+'parametric.csv', delimiter=',')
    
    n_features = config.getint('Global', 'n_features')
    save_dir2 = save_dir1+'semantic_dim = '+str(n_features)+'/'
    fnames = glob.glob(save_dir2+'*.csv')

    metric = 'ONC'
    save_fig(metric)
    print metric + '.png saved!'

