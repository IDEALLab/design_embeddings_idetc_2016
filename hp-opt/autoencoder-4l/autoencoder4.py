'''
Objective function for hyperparameter optimization of an autoencoder with 3 hidden layers

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
from deep_network import stacked_ae

def mapping(hidden_size_l1, hidden_size_l2, hidden_size_l3, learning_rate, lr_decay, regularizer, weight_decay, momentum, samples):    
    
    config = ConfigParser.ConfigParser()
    config.read('../../config.ini')
    
    n_features = config.getint('Global', 'n_features')
    n_folds = config.getint('Global', 'n_folds')
    regularizers = ['l1', 'l2']
    reg_fn = regularizers[regularizer]
    
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
        cost = stacked_ae(data_norm, n_features, train, test, hidden_size_l1, hidden_size_l2, hidden_size_l3, 0, 
                          learning_rate, lr_decay, reg_fn, weight_decay, momentum, evaluation=True)
        rec_err_test += cost

    rec_err_test /= n_folds
    #print 'reconstruction error = %f' % rec_err_test
    
    con1 = int(hidden_size_l1-hidden_size_l2-1)   # hidden_size_l1 > hidden_size_l2
    con2 = int(hidden_size_l2-hidden_size_l3-1)   # hidden_size_l2 > hidden_size_l3

    return {
        "mapping"           : rec_err_test,
        "l2_less_l1"        : con1, 
        "l3_less_l2"        : con2
    }

    #return rec_err_test

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
		
    errname = '../temp/err_ae4l'
    if os.path.isfile(errname):
        with open(errname, 'rb') as f:
            l_err = pickle.load(f) # l_err = [rec_err_cv, count]
    else:
        l_err = [1., 0]

    cfgname = '../hp_'+source+'_'+str(n_samples)+'_'+str(n_features)+'_'+str(test_start)+'-'+str(test_end)+'.ini'
    hp = ConfigParser.ConfigParser()
    hp.read(cfgname)
    
    results = mapping(params['size_l1'][0], params['size_l2'][0], params['size_l3'][0], params['learning_rate'][0], 
                      params['lr_decay'][0], params['regularizer'][0], params['weight_decay'][0], params['momentum'][0], train)
    rec_err_cv = results["mapping"]
    
    if not hp.has_section('ae4l'):
        # Create the section if it does not exist.
        hp.add_section('ae4l')
        hp.write(open(cfgname,'w'))
        hp.read(cfgname)
    
    if rec_err_cv < l_err[0]:
        # Modify the config file if new reconstruction error is smaller
        hp.set('ae4l','hidden_size_l1',params['size_l1'][0])
        hp.set('ae4l','hidden_size_l2',params['size_l2'][0])
        hp.set('ae4l','hidden_size_l3',params['size_l3'][0])
        hp.set('ae4l','learning_rate',params['learning_rate'][0])
        hp.set('ae4l','lr_decay',params['lr_decay'][0])
        regularizers = ['l1', 'l2']
        reg_fn = regularizers[params['regularizer'][0]]
        hp.set('ae4l','regularizer',reg_fn)
        hp.set('ae4l','weight_decay',params['weight_decay'][0])
        hp.set('ae4l','momentum',params['momentum'][0])
        l_err = [rec_err_cv, 0]
        hp.write(open(cfgname,'w'))		
    else:
        l_err[1] += 1
            
    with open(errname, 'wb') as f:
        pickle.dump(l_err, f)
            
    print 'optimal: ', l_err[0]
    print 'count: ', l_err[1]
    
    return results
    
