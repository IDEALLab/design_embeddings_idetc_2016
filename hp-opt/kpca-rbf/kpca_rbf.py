'''
Objective function for hyperparameter optimization of kernel PCA

Author(s): Wei Chen (wchen459@umd.edu)
'''

import pickle
import os
import sys
import ConfigParser
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import KFold
sys.path.append('../..')
from dim_reduction import kpca

def mapping(gamma, alpha, samples):    
    
    config = ConfigParser.ConfigParser()
    config.read('../../config.ini')
    
    n_components = config.getint('Global', 'n_features')
    n_folds = config.getint('Global', 'n_folds')
    
    source = config.get('Global', 'source')
    data = np.load('../../raw_parametric_'+source+'.npy')[samples]
    n_images = data.shape[0]
    print n_images
    
    # Normalize each sample
    s = preprocessing.MaxAbsScaler()
    data_norm = s.fit_transform(data.T).T # Min-Max normalization

    #np.savetxt("parametric.csv", data_norm, delimiter=",")

    # K-fold cross-validation
    kf = KFold(n_images, n_folds=n_folds, shuffle=True)
    rec_err_test = 0
    i = 1
    
    for train, test in kf:
        
        print 'cross validation: %d' % i
        i += 1
        
        # Get loss
        cost = kpca(data_norm, n_components, train, test, kernel='rbf', gamma=gamma, alpha=alpha, evaluation=True)
        rec_err_test += cost

    rec_err_test /= n_folds
    #print 'reconstruction error = %f' % rec_err_test

    return rec_err_test

def main(job_id, params):
        
    print params
    
    config = ConfigParser.ConfigParser()
    config.read('../../config.ini')
    n_features = config.getint('Global', 'n_features')
    n_samples = config.getint('Global', 'n_samples')
        
    source = config.get('Global', 'source')
	
    # Specify number of training and testing samples
    alls = range(n_samples)
    test_start = config.getint('Global', 'test_start')
    test_end = config.getint('Global', 'test_end')
    test = range(test_start-1, test_end)
    train = [item for item in alls if item not in test]
	
    rec_err_cv = mapping(params['gamma'][0], params['alpha'][0], train)
	
    errname = '../temp/err_kpca'
    if os.path.isfile(errname):
        with open(errname, 'rb') as f:
            l_err = pickle.load(f) # l_err = [rec_err_cv, count]
    else:
        l_err = [1., 0]

    cfgname = '../hp_'+source+'_'+str(n_samples)+'_'+str(n_features)+'_'+str(test_start)+'-'+str(test_end)+'.ini'
    hp = ConfigParser.ConfigParser()
    hp.read(cfgname)
        
    if not hp.has_section('kpca'):
        # Create the section if it does not exist.
        hp.add_section('kpca')
        hp.set('kpca','kernel','rbf')
        hp.write(open(cfgname,'w'))
        hp.read(cfgname)
        
    if rec_err_cv < l_err[0]:
        # Modify the config file if new reconstruction error is smaller
        hp.set('kpca','gamma',params['gamma'][0])
        hp.set('kpca','alpha',params['alpha'][0])
        l_err = [rec_err_cv, 0]
        hp.write(open(cfgname,'w'))
    else:
        l_err[1] += 1
            
    with open(errname, 'wb') as f:
        pickle.dump(l_err, f)
            
    print 'optimal: ', l_err[0]
    print 'count: ', l_err[1]
    
    return rec_err_cv
    
