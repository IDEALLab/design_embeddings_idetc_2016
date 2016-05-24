'''
Plots reconstruction error vs semantic space dimensionality

Usage: python metric_err.py

Author(s): Wei Chen (wchen459@umd.edu)
'''

import matplotlib.pyplot as plt
import numpy as np

plt.rc("font", size=22)

examples = ['glass', 'sf_linear', 'sf_s_nonlinear', 'sf_v_nonlinear']
titles = {'glass':              'Glass',
          'sf_linear':          'Superformula (linear)',
          'sf_s_nonlinear':     'Superformula (slightly nonlinear)',
          'sf_v_nonlinear':     'Superformula (very nonlinear)'}

n = len(examples)
x = range(1, 6)

for i in range(n):

    plt.figure()
    plt.xticks(np.arange(min(x), max(x)+1, dtype=np.int))
    plt.xlabel('Semantic space dimensionality')
    plt.ylabel('Reconstruction error')
    plt.xlim(0.5, 5.5)
    
    errs = np.zeros((3,5))
    for j in x:
        # Read reconstruction errors in rec_err.txt
        txtfile1 = open('./results/1-13/'+examples[i]+'/n_control_points = 20/semantic_dim = '
                       +str(j)+'/rec_err.txt', 'r')
        txtfile2 = open('./results/116-128/'+examples[i]+'/n_control_points = 20/semantic_dim = '
                       +str(j)+'/rec_err.txt', 'r')
        k = 0
        for (line1,line2) in zip(txtfile1,txtfile2):
            errs[k, j-1] = (float(line1)+float(line2))/2
            k += 1

#    plt.ylim(0., 1.1*np.max(errs))
    line_pca, = plt.plot(x, errs[0], '-ob', label='PCA', ms=10)
    line_kpca, = plt.plot(x, errs[1], '-vg', label='Kernel PCA', ms=10)
    line_ae, = plt.plot(x, errs[2], '-sr', label='Autoencoder', ms=10)
    plt.legend(handles=[line_pca, line_kpca, line_ae], fontsize=21)
    plt.title(titles[examples[i]])
    plt.tight_layout()
    fig_name = 'n_features_vs_rec_error_'+examples[i]+'.png'
    plt.savefig('./results/'+fig_name, dpi=300)
    print fig_name+' saved!'
