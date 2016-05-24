"""
Plots samples or new shapes in the semantic space.

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
"""

from matplotlib import pyplot as plt
from sklearn import preprocessing
import numpy as np
import itertools

def scatter_plot(features, data_rec, train, test, parameterizations_train, parameterizations_test, 
                 splines_train, splines_test, x_plots_train, x_plots_test, save_path, name_algorithm, mirror=True):
    
    ''' Create 3D scatter plot and corresponding 2D projections
        of at most the first 3 dimensions of data'''
    
    plt.rc("font", size=font_size)
    n_samples_train = len(train)
    n_samples_test = len(test)
    n_dim = features.shape[1]
    
    if n_dim == 3:
        # Create a 3D scatter plot
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection = '3d')
        ax3d.scatter(features[:,0], features[:,1], features[:,2])
        ax3d.set_title(name_algorithm)
        plt.savefig(save_path+name_algorithm+'/3d.png', dpi=300)
        plt.close()
    
    features_train = features[train]
    features_test = features[test]
    
    # Project 3D plot to 2D plots and label each point
    figs = []
    ax = []
    k = 0
    for i in range(0, n_dim-1):
        for j in range(i+1, n_dim):
            figs.append(plt.figure())
            ax.append(figs[k].add_subplot(111))
            
            # Plot training data
            for index in range(n_samples_train):
                ax[k].scatter(features_train[index,i], features_train[index,j], s = 7)
                m = splines_train[index].M(parameterizations_train[index], x_plots_train[index]).tolist()
                mx = max([y for (x, y) in m])
                mn = min([y for (x, y) in m])
                xscl = .08 / (mx - mn)
                yscl = .08 / (mx - mn)
                ax[k].plot( *zip(*[(x * xscl + features_train[index, i], -y * yscl + features_train[index, j])
                                   for (x, y) in m]), color='red')
                if mirror:
                    ax[k].plot( *zip(*[(-x * xscl + features_train[index, i], -y * yscl + features_train[index, j]) 
                                   for (x, y) in m]), color='red')
                '''
                # Draw reconstructed samples for training data
                m2 = splines_train[index].M(parameterizations_train[index], data_rec_train[index].reshape((-1,2))).tolist()
                mx2 = max([y for (x, y) in m2])
                mn2 = min([y for (x, y) in m2])
                xscl2 = .08 / (mx2 - mn2)
                yscl2 = .08 / (mx2 - mn2)
                ax[k].plot( *zip(*[(x * xscl2 + features_train[index, i], -y * yscl2 + features_train[index, j])
                                   for (x, y) in m2]), color='green', alpha=0.5)
                if mirror:
                    ax[k].plot( *zip(*[(-x * xscl2 + features_train[index, i], -y * yscl2 + features_train[index, j])
                                   for (x, y) in m2]), color='green', alpha=0.5)
                '''
            #Plot testing data
            for index in range(n_samples_test):
                ax[k].scatter(features_test[index, i], features_test[index, j], s = 7)
                m = splines_test[index].M(parameterizations_test[index], x_plots_test[index]).tolist()
                mx = max([y for (x, y) in m])
                mn = min([y for (x, y) in m])
                xscl = .08 / (mx - mn)
                yscl = .08 / (mx - mn)
                ax[k].plot( *zip(*[(x * xscl + features_test[index, i], -y * yscl + features_test[index, j])
                                   for (x, y) in m]), color='blue')
                if mirror:
                    ax[k].plot( *zip(*[(-x * xscl + features_test[index, i], -y * yscl + features_test[index, j]) 
                                   for (x, y) in m]), color='blue')
                '''
                # Draw reconstructed samples for testing data
                m2 = splines_test[index].M(parameterizations_test[index], data_rec_test[index].reshape((-1,2))).tolist()
                mx2 = max([y for (x, y) in m2])
                mn2 = min([y for (x, y) in m2])
                xscl2 = .08 / (mx2 - mn2)
                yscl2 = .08 / (mx2 - mn2)
                ax[k].plot( *zip(*[(x * xscl2 + features_test[index, i], -y * yscl2 + features_test[index, j])
                                   for (x, y) in m2]), color='cyan', alpha=0.7)
                if mirror:
                    ax[k].plot( *zip(*[(-x * xscl2 + features_test[index, i], -y * yscl2 + features_test[index, j])
                                   for (x, y) in m2]), color='cyan', alpha=0.7)                    
                '''
            ax[k].set_title(name_algorithm)
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)
            plt.xlabel('Dimension-'+str(i+1))
            plt.ylabel('Dimension-'+str(j+1))
            
            #ax[k].text(-0.1, -0.1, 'training error = '+str(err_train)+' / testing error = '+str(err_test))
            
            k += 1
            plt.savefig(save_path+name_algorithm+'/'+str(i+1)+'-'+str(j+1)+'.png', dpi=300)
            plt.close()

def plot_semantic_space_grid(points_per_axis, n_dim, min_maxes, inverse_transform, save_path, name_algorithm, 
                             spline, boundary=None, kde=None, n_init_points=1000, mirror=True):
    
    ''' Plot reconstructed glass contours in the semantic space.
        If the semantic space is 3D (i.e., n_dim=3), plot one slice of the 3D space at each time. '''
    
    plt.rc("font", size=font_size)
    linewidth = 3
    u0 = spline.uniform_parameterisation(n_init_points)
    lincoords = []
    
    for i in range(0,n_dim):
        lincoords.append(np.linspace(min_maxes[i][0],min_maxes[i][1],points_per_axis))
    coords = list(itertools.product(*lincoords)) # Create a list of coordinates in the semantic space
    coords_norm = preprocessing.MinMaxScaler().fit_transform(coords) # Min-Max normalization
    if kde is not None:
        # Density evaluation for coords_norm
        kde_scores = np.exp(kde.score_samples(coords_norm))
    coords_norm = coords_norm.tolist()
    data_rec = inverse_transform(np.array(coords)) # Reconstruct B-spline control points
    
    # Determine if the i-th item of coords_norm is in the convex hull
    
    indices = []
    for i in range(len(coords)):
        c = tuple(coords_norm[i]) + (1,)
        if boundary is not None:
            e = np.dot(boundary, np.expand_dims(c, axis=1))
        if boundary is None or np.all(e <= 0):
            #if kde is None or kde_scores[i] > 0.25:
                indices.append(i)
    
    if n_dim < 3:
        # Create a 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
            ax.scatter(coords_norm[i][0], coords_norm[i][1], s = 7)
            m = spline.M(u0, data_rec[i].reshape((-1,2))).tolist()
            mx = max([y for (x, y) in m])
            mn = min([y for (x, y) in m])
            xscl = .7 / (mx - mn) / points_per_axis
            yscl = .7 / (mx - mn) / points_per_axis
            
            alpha = kde_scores[i] + .3
            if alpha > 1:
                alpha = 1

            ax.plot( *zip(*[(x * xscl + coords_norm[i][0], -y * yscl + coords_norm[i][1]) for (x, y) in m]), linewidth=linewidth, color='blue', alpha=alpha)
            if mirror:
                ax.plot( *zip(*[(-x * xscl + coords_norm[i][0], -y * yscl + coords_norm[i][1]) for (x, y) in m]), linewidth=linewidth, color='blue', alpha=alpha)
        
        ax.set_title(name_algorithm, fontsize=20)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Dimension-1')
        plt.ylabel('Dimension-2')
        plt.savefig(save_path+name_algorithm+'/' + 'semantic_space.png', dpi=300)
        
        if kde is not None:
            for i in indices:
                # Compute and annotate sparsity for coords_norm[i]
                #kde_score = np.exp(kde.score_samples(np.reshape(coords_norm[i], (1, -1))))[0]
                ax.annotate('{:.2f}'.format(kde_scores[i]), (coords_norm[i][0], coords_norm[i][1]), fontsize=12)
            plt.savefig(save_path+name_algorithm+'/'+'semantic_space_sparsity.png', dpi=300)
            plt.close()
        
    else:
        # Create slices of 2D plots for n_dim = 3
        coords_norm = np.array(coords_norm)[indices,:]
        data_rec = data_rec[indices,:]
        # Sort coords_norm and data_rec simultanously by the 3rd column in coords_norm (z coordinates)
        cc = np.concatenate((coords_norm, data_rec), axis=1)
        cc = cc[np.argsort(cc[:,2])]
        coords_norm = cc[:,:3]
        data_rec = cc[:,3:]
        k = 0
        figs = []
        ax = []
        figs.append(plt.figure())
        ax.append(figs[k].add_subplot(111))
        z = coords_norm[0,2]
        for i in range(len(coords_norm)):
            if coords_norm[i, 2] == z:
                ax[k].scatter(coords_norm[i, 0], coords_norm[i, 1], s = 7)
                m = spline.M(u0, data_rec[i].reshape((-1,2))).tolist()
                mx = max([y for (x, y) in m])
                mn = min([y for (x, y) in m])
                xscl = .7 / (mx - mn) / points_per_axis
                yscl = .7 / (mx - mn) / points_per_axis
                ax[k].plot( *zip(*[(x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color='blue')
                if mirror:
                    ax[k].plot( *zip(*[(-x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color='blue')
                
            else:
                ax[k].set_title(name_algorithm+' (z = '+str(z)+')')
                plt.xlim(-0.1, 1.1)
                plt.ylim(-0.1, 1.1)
                plt.xlabel('Dimension-1')
                plt.ylabel('Dimension-2')
                plt.savefig(save_path+name_algorithm+'/semantic_space_z='+str(z)+'.png', dpi=300)
                plt.close()
                
                k += 1
                z = coords_norm[i, 2]
                figs.append(plt.figure())
                ax.append(figs[k].add_subplot(111))
                ax[k].scatter(coords_norm[i, 0], coords_norm[i, 1], s = 7)
                m = spline.M(u0, data_rec[i].reshape((-1,2))).tolist()
                mx = max([y for (x, y) in m])
                mn = min([y for (x, y) in m])
                xscl = .7 / (mx - mn) / points_per_axis
                yscl = .7 / (mx - mn) / points_per_axis
                ax[k].plot( *zip(*[(x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color='blue')
                if mirror:
                    ax[k].plot( *zip(*[(-x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color='blue')

#            if kde is not None:
#                for i in indices:
#                    # Compute and annotate sparsity for coords_norm[i]
#                    #kde_score = np.exp(kde.score_samples(np.reshape(coords_norm[i], (1, -1))))[0]
#                    ax[k].annotate('{:.2f}'.format(kde_scores[i]), (coords_norm[i][0], coords_norm[i][1]), fontsize=16)
#                plt.savefig(save_path+name_algorithm+'/'+'semantic_space_sparsity.png', dpi=300)
#                plt.close()
        
        ax[k].set_title(name_algorithm+' (z = '+str(z)+')')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Dimension-1')
        plt.ylabel('Dimension-2')
        plt.savefig(save_path+name_algorithm+'/semantic_space_z='+str(z)+'.png', dpi=300)
        plt.close()

def plot_original_space_grid(points_per_axis, n_dim, min_maxes, inverse_transform, save_path, name_algorithm, 
                             spline, n_init_points=1000, mirror=True):
    
    ''' Plot reconstructed glass contours in the semantic space.
        If the semantic space is 3D (i.e., n_dim=3), plot one slice of the 3D space at each time. '''
    
    print "plotting original space"

    plt.rc("font", size=font_size)
    u0 = spline.uniform_parameterisation(n_init_points)
    lincoords = []
    
    for i in range(0,n_dim):
        lincoords.append(np.linspace(min_maxes[i][0],min_maxes[i][1],points_per_axis))
    coords = list(itertools.product(*lincoords)) # Create a list of coordinates in the semantic space
    coords_norm = preprocessing.MinMaxScaler().fit_transform(coords) # Min-Max normalization
    data_rec = inverse_transform(np.array(coords)) # Reconstruct B-spline control points
    indices = range(len(coords))

    if n_dim < 3:
        # Create a 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
            ax.scatter(coords_norm[i, 0], coords_norm[i, 1], s = 7)
            m = spline.M(u0, data_rec[i].reshape((-1,2))).tolist()
            mx = max([y for (x, y) in m])
            mn = min([y for (x, y) in m])
            xscl = .7 / (mx - mn) / points_per_axis
            yscl = .7 / (mx - mn) / points_per_axis

            color = 'blue'

            ax.plot( *zip(*[(x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color=color)
            if mirror:
                ax.plot( *zip(*[(-x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color=color)

        ax.set_title(name_algorithm, fontsize=20)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('a')
        plt.ylabel('b')
        plt.savefig(save_path+'semantic_space.png', dpi=300)
        
        plt.close()
        
    else:
        # Create slices of 2D plots for n_dim = 3
        coords_norm = np.array(coords_norm)[indices,:]
        data_rec = data_rec[indices,:]
        # Sort coords_norm and data_rec simultanously by the 3rd column in coords_norm (z coordinates)
        cc = np.concatenate((coords_norm, data_rec), axis=1)
        cc = cc[np.argsort(cc[:,2])]
        coords_norm = cc[:,:3]
        data_rec = cc[:,3:]
        k = 0
        figs = []
        ax = []
        figs.append(plt.figure())
        ax.append(figs[k].add_subplot(111))
        z = coords_norm[0,2]
        for i in range(len(coords_norm)):
            if coords_norm[i, 2] == z:
                ax[k].scatter(coords_norm[i, 0], coords_norm[i, 1], s = 7)
                m = spline.M(u0, data_rec[i].reshape((-1,2))).tolist()
                mx = max([y for (x, y) in m])
                mn = min([y for (x, y) in m])
                xscl = .7 / (mx - mn) / points_per_axis
                yscl = .7 / (mx - mn) / points_per_axis
                ax[k].plot( *zip(*[(x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color=color)
                if mirror:
                    ax[k].plot( *zip(*[(-x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color=color)
                
            else:
                ax[k].set_title(name_algorithm+' (Semantic feature 3 = '+str(z)+')')
                plt.xlim(-0.1, 1.1)
                plt.ylim(-0.1, 1.1)
                plt.xlabel('Semantic feature 1')
                plt.ylabel('Semantic feature 2')
                plt.savefig(save_path+name_algorithm+'/semantic_space_z='+str(z)+'.png', dpi=300)
                plt.close()
                
                k += 1
                z = coords_norm[i, 2]
                figs.append(plt.figure())
                ax.append(figs[k].add_subplot(111))
                ax[k].scatter(coords_norm[i, 0], coords_norm[i, 1], s = 7)
                m = spline.M(u0, data_rec[i].reshape((-1,2))).tolist()
                mx = max([y for (x, y) in m])
                mn = min([y for (x, y) in m])
                xscl = .7 / (mx - mn) / points_per_axis
                yscl = .7 / (mx - mn) / points_per_axis
                ax[k].plot( *zip(*[(x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color=color)
                if mirror:
                    ax[k].plot( *zip(*[(-x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color=color)

        ax[k].set_title(name_algorithm+' (Semantic feature 3 = '+str(z)+')')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Semantic feature 1')
        plt.ylabel('Semantic feature 2')
        plt.savefig(save_path+name_algorithm+'/semantic_space_z='+str(z)+'.png', dpi=300)
        plt.close()

def plot_original_space_examples(points_per_axis, n_dim, min_maxes, inverse_transform, save_path, name_algorithm, 
                             spline, samples, n_init_points=1000, mirror=True):
    
    ''' Plot reconstructed glass contours in the semantic space.
        If the semantic space is 3D (i.e., n_dim=3), plot one slice of the 3D space at each time. '''
    
    print "plotting original space"

    plt.rc("font", size=font_size)
    u0 = spline.uniform_parameterisation(n_init_points)
    
    coords = samples
    coords_norm = preprocessing.MinMaxScaler().fit_transform(coords) # Min-Max normalization
    data_rec = inverse_transform(np.array(coords)) # Reconstruct B-spline control points
    indices = range(len(coords))

    if n_dim < 3:
        # Create a 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
            ax.scatter(coords_norm[i, 0], coords_norm[i, 1], s = 7)
            m = spline.M(u0, data_rec[i].reshape((-1,2))).tolist()
            mx = max([y for (x, y) in m])
            mn = min([y for (x, y) in m])
            xscl = .7 / (mx - mn) / points_per_axis
            yscl = .7 / (mx - mn) / points_per_axis

            color = 'red'

            ax.plot( *zip(*[(x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color=color)
            if mirror:
                ax.plot( *zip(*[(-x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color=color)

        ax.set_title(name_algorithm, fontsize=20)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('a')
        plt.ylabel('b')
        plt.savefig(save_path+'samples.png', dpi=300)
        
        plt.close()
        
    else:
        # Create slices of 2D plots for n_dim = 3
        coords_norm = np.array(coords_norm)[indices,:]
        data_rec = data_rec[indices,:]
        # Sort coords_norm and data_rec simultanously by the 3rd column in coords_norm (z coordinates)
        cc = np.concatenate((coords_norm, data_rec), axis=1)
        cc = cc[np.argsort(cc[:,2])]
        coords_norm = cc[:,:3]
        data_rec = cc[:,3:]
        k = 0
        figs = []
        ax = []
        figs.append(plt.figure())
        ax.append(figs[k].add_subplot(111))
        z = coords_norm[0,2]
        for i in range(len(coords_norm)):
            if coords_norm[i, 2] == z:
                ax[k].scatter(coords_norm[i, 0], coords_norm[i, 1], s = 7)
                m = spline.M(u0, data_rec[i].reshape((-1,2))).tolist()
                mx = max([y for (x, y) in m])
                mn = min([y for (x, y) in m])
                xscl = .7 / (mx - mn) / points_per_axis
                yscl = .7 / (mx - mn) / points_per_axis
                ax[k].plot( *zip(*[(x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color=color)
                if mirror:
                    ax[k].plot( *zip(*[(-x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color=color)
                
            else:
                ax[k].set_title(name_algorithm+' (Semantic feature 3 = '+str(z)+')')
                plt.xlim(-0.1, 1.1)
                plt.ylim(-0.1, 1.1)
                plt.xlabel('Semantic feature 1')
                plt.ylabel('Semantic feature 2')
                plt.savefig(save_path+name_algorithm+'/semantic_space_z='+str(z)+'.png', dpi=300)
                plt.close()
                
                k += 1
                z = coords_norm[i, 2]
                figs.append(plt.figure())
                ax.append(figs[k].add_subplot(111))
                ax[k].scatter(coords_norm[i, 0], coords_norm[i, 1], s = 7)
                m = spline.M(u0, data_rec[i].reshape((-1,2))).tolist()
                mx = max([y for (x, y) in m])
                mn = min([y for (x, y) in m])
                xscl = .7 / (mx - mn) / points_per_axis
                yscl = .7 / (mx - mn) / points_per_axis
                ax[k].plot( *zip(*[(x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color=color)
                if mirror:
                    ax[k].plot( *zip(*[(-x * xscl + coords_norm[i, 0], -y * yscl + coords_norm[i, 1]) for (x, y) in m]), color=color)
        
        ax[k].set_title(name_algorithm+' (Semantic feature 3 = '+str(z)+')')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Semantic feature 1')
        plt.ylabel('Semantic feature 2')
        plt.savefig(save_path+name_algorithm+'/semantic_space_z='+str(z)+'.png', dpi=300)
        plt.close()

font_size = 18
