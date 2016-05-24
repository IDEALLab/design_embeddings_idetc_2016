"""
Parameterizes sample shapes and trains models using PCA, kernel PCA and the autoencoder.

data : geometric representation of sample shapes (xy coordinates of the B-spline control points)
features : representation of samples in the semantic space

Usage: python training.py

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
"""

import os
from functools import partial
import ConfigParser

import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
from scipy.spatial import distance, ConvexHull

import dim_reduction
from deep_network import stacked_ae
import shape_plot
from parametric_space import get_parametric

def create_dir(path):
    if os.path.isdir(path): 
        pass 
    else: 
        os.mkdir(path)
    
def get_reconstruction_error(data, data_rec, x_plots, n_control_points, splines, parameterizations, n_init_points = 1000):
    n_data_points = data.shape[0]
    u0 = splines[0].uniform_parameterisation(n_init_points)

    reconstruction_error = 0
            #ax[k].set_aspect('equal')
    for index in range(n_data_points):
        m = splines[index].M(parameterizations[index], x_plots[index]).tolist()
        mx = max([y for (x, y) in m])
        mn = min([y for (x, y) in m])

        m2 = splines[0].M(u0, data_rec[index].reshape((-1,2))).tolist()
        mx2 = max([y for (x, y) in m2])
        mn2 = min([y for (x, y) in m2])

        add = sum([distance.euclidean(np.multiply(1/(mx - mn),a),np.multiply(1/(mx2 - mn2),b))
            for (a, b) in zip(splines[index].M(parameterizations[index], x_plots[index]).tolist(),
                splines[0].M(u0, data_rec[index].reshape((-1,2))).tolist())])
        reconstruction_error += add

    #reconstruction_error /= (n_control_points * n_data_points)
    reconstruction_error /= n_data_points * len(x_plots)
    return reconstruction_error

def model_analysis(fs, source_dir, train, test, n_features, n_control_points):
    ''' Build model instances using optimized hyperparameters and evaluate using testing data '''

    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    
    n_samples = config.getint('Global', 'n_samples')
    source = config.get('Global', 'source')
    
    # Specify number of training and testing samples
    n_samples_train = len(train)
    n_samples_test = len(test)
    n_samples = n_samples_train + n_samples_test
        
    print('Source: '+source+' | B-spline control points: '+str(n_control_points)+' | Semantic features: '+str(n_features)+
          ' | Training samples: '+str(n_samples_train)+' | Testing samples: '+str(n_samples_test))
        
    data = np.zeros((n_samples, 2*n_control_points))
    
    x_plots_train, splines_train, parameterizations_train, data[train], save_dir1 = get_parametric(source_dir, train, source)
    x_plots_test, splines_test, parameterizations_test, data[test], save_dir1 = get_parametric(source_dir, test, source)
    
    save_dir2 = save_dir1 + 'semantic_dim = '+str(n_features)+'/'
    create_dir(save_dir2)

    # Normalize each sample
    s = preprocessing.MaxAbsScaler()
    data_norm = s.fit_transform(data.T).T # Min-Max normalization
    np.savetxt(save_dir1+"parametric.csv", data_norm[train], delimiter=",")

    txtfile = open(save_dir2+"rec_err.txt", "w")

    for f in fs:

        # Get semantic features
        features, name, inv_transform = f(data_norm, n_features, train, test)

        print('Algorithm: ' + name + ' completed.')
        create_dir(save_dir2 + name)
        
        # Min-Max normalization
        features_norm = preprocessing.MinMaxScaler().fit_transform(features)
        np.savetxt(save_dir2+name+".csv", features_norm[train], delimiter=",")

        # Get reconstructed data (B-spline control points)
        data_rec = inv_transform(features)
        data_rec = s.inverse_transform(data_rec.T).T

        # Get reconstruction error
        rec_err_train = get_reconstruction_error(data[train], data_rec[train], x_plots_train,
                                                 n_control_points, splines_train, parameterizations_train)
        rec_err_test = get_reconstruction_error(data[test], data_rec[test], x_plots_test,
                                                n_control_points, splines_test, parameterizations_test)

        print 'Training error: ', rec_err_train
        print 'Testing error: ', rec_err_test
        
        txtfile.write('%.3f\n' % rec_err_test)
        
        # Convex hull of training samples in the semantic space
        if n_features > 1:
            hull = ConvexHull(features_norm[train])
            boundary = hull.equations
        
        # Get semantic space sparsity
        #sparsity, kde = metric_sparsity.get_sparsity(features_norm[train], boundary)
        #print 'Sparsity: ', sparsity
        kde = KernelDensity(kernel='epanechnikov', bandwidth=0.15).fit(features_norm[train])
        
        print('Saving 2D plots...')
        if source == 'glass':
            shape_plot.scatter_plot(features_norm, data_rec, train, test, parameterizations_train, parameterizations_test,
                                    splines_train, splines_test, x_plots_train, x_plots_test, save_dir2, name)
        else:
            shape_plot.scatter_plot(features_norm, data_rec, train, test, parameterizations_train, parameterizations_test,
                                    splines_train, splines_test, x_plots_train, x_plots_test, save_dir2, name, mirror=False)

        if n_features==2 or n_features==3:
            min_maxes = zip(map(min,zip(*features[train])),map(max,zip(*features[train])))
            
            if source == 'glass':
                shape_plot.plot_semantic_space_grid(8, n_features, min_maxes, inv_transform, save_dir2, 
                                                    name, splines_train[0], boundary, kde)
            else:
                shape_plot.plot_semantic_space_grid(8, n_features, min_maxes, inv_transform, save_dir2, 
                                                    name, splines_train[0], boundary, kde, mirror=False)
                                                    
    txtfile.close()


if __name__ == "__main__":
    
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    
    n_control_points = config.getint('Global', 'n_control_points')
    n_features = config.getint('Global', 'n_features')
    n_samples = config.getint('Global', 'n_samples')
    
    source = config.get('Global', 'source')
    IMAGE_DIR = config.get('Global', 'IMAGE_DIR')
    source_dir = IMAGE_DIR+source+'/'
    
    alls = range(n_samples)
    test_start = config.getint('Global', 'test_start')
    test_end = config.getint('Global', 'test_end')
    test = range(test_start-1, test_end)
    train = [item for item in alls if item not in test]

    # Get optimized hyperparameters
    hp = ConfigParser.ConfigParser()
    hp.read('./hp-opt/hp_'+source+'_'+str(n_samples)+'_'+str(n_features)+'_'+str(test_start)+'-'+str(test_end)+'.ini')
    
    kernel = hp.get('kpca', 'kernel')
    gamma = hp.getfloat('kpca', 'gamma')
    alpha = hp.getfloat('kpca', 'alpha')

    hidden_size_l1 = hp.getint('ae4l', 'hidden_size_l1')
    hidden_size_l2 = hp.getint('ae4l', 'hidden_size_l2')
    hidden_size_l3 = hp.getint('ae4l', 'hidden_size_l3')
    learning_rate = hp.getfloat('ae4l', 'learning_rate')
    lr_decay = hp.getfloat('ae4l', 'lr_decay')
    regularizer = hp.get('ae4l', 'regularizer')
    weight_decay = hp.getfloat('ae4l', 'weight_decay')
    momentum = hp.getfloat('ae4l', 'momentum')

    fs = [dim_reduction.pca,
          partial(dim_reduction.kpca, kernel=kernel, gamma=gamma, alpha=alpha),
          partial(stacked_ae, hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2, hidden_size_l3=hidden_size_l3, 
                  learning_rate=learning_rate, lr_decay=lr_decay, regularizer=regularizer, l=weight_decay, momentum=momentum)
          ]
    

    model_analysis(fs, source_dir, train, test, n_features, n_control_points)
        
    print('All completed :)')
