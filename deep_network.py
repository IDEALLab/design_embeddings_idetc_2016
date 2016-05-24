"""
Builds and trains autoencoders.

Author(s): Wei Chen (wchen459@umd.edu)
"""

from functools import partial
from keras.models import Sequential
from keras.layers.core import Dense, AutoEncoder
from keras.layers import containers
from keras.optimizers import SGD
from keras.regularizers import l1, l2
from sklearn.metrics import mean_squared_error


def single_layer_autoencoder(outer_dim, hidden_dim, W_regularizer, learning_rate, lr_decay, momentum):
    
    autoencoder = Sequential()
    encoder = containers.Sequential([Dense(hidden_dim, input_dim=outer_dim, activation=activation, W_regularizer=W_regularizer)])
    decoder = containers.Sequential([Dense(outer_dim, input_dim=hidden_dim, activation=activation, W_regularizer=W_regularizer)])
    autoencoder.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
    sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=momentum, nesterov=True)
    autoencoder.compile(loss=obj_fcn, optimizer=sgd)
    
    return autoencoder

def ae(data, feature_dim, train, test, learning_rate, lr_decay, reg_fn, l, momentum, evaluation):
    ''' Autoencoder '''
    
    batch_size=len(train)
    data_dim = data.shape[1]
    
    model = single_layer_autoencoder(data_dim, feature_dim, reg_fn(l), learning_rate, lr_decay, momentum)
    model.fit(data[train], data[train], batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
    
    output = model.predict(data)
    
    # Reconstruction
    model_rec = Sequential()
    model_rec.add(Dense(data_dim, input_dim=feature_dim, activation=activation, weights=model.layers[0].decoder.get_weights()[0:2]))
    model_rec.layers[0].get_input(False) # Get input from testing data
    model_rec.compile(loss='mse', optimizer='sgd')
    
    if evaluation:
        data_rec = model_rec.predict(output[test])
        loss = mean_squared_error(data[test], data_rec)
        return loss
    
    name = 'Autoencoder'
    
    return output, name, model_rec.predict

def ae2l(data, feature_dim, train, test, hidden_size, learning_rate, lr_decay, reg_fn, l, momentum, evaluation):
    ''' 2-layer stacked autoencoder '''
    
    batch_size=len(train)
    data_dim = data.shape[1]
    
    if pre_training:
        # Pre-training (greedy layer-wise training)
        if verbose:
            print('Pre-training for 1st layer...')
        ae1 = single_layer_autoencoder(data_dim, hidden_size, reg_fn(l), learning_rate, lr_decay, momentum)
        ae1.fit(data[train], data[train], batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        a1 = ae1.predict(data[train]) # 1st hidden layer's activation
        w1_en = ae1.layers[0].encoder.get_weights()
        w1_de = ae1.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
        
        if verbose:
            print('Pre-training for 2nd layer...')
        ae2 = single_layer_autoencoder(hidden_size, feature_dim, reg_fn(l), learning_rate, lr_decay, momentum)
        ae2.fit(a1, a1, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        w2_en = ae2.layers[0].encoder.get_weights()
        w2_de = ae2.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
    
    else:
        w1_en = None
        w1_de = None
        w2_en = None
        w2_de = None
    
    # Fine-tuning
    if verbose:
        print('Fine-tuning...')
    model = Sequential()
    encoder = containers.Sequential([Dense(hidden_size, input_dim=data_dim, activation=activation, weights=w1_en, 
                                           W_regularizer=reg_fn(l)),
                                     Dense(feature_dim, activation=activation, weights=w2_en, W_regularizer=reg_fn(l))])
    decoder = containers.Sequential([Dense(hidden_size, input_dim=feature_dim, activation=activation, weights=w2_de, 
                                           W_regularizer=reg_fn(l)),
                                     Dense(data_dim, activation=activation, weights=w1_de, W_regularizer=reg_fn(l))])
    model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
    sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss=obj_fcn, optimizer=sgd)
    model.fit(data[train], data[train], nb_epoch=nb_epoch, verbose=verbose)
    if verbose:
        print('--------------------------------------------------------------')
        
    output = model.predict(data)
    
    # Reconstruction
    model_rec = Sequential()
    model_rec.add(Dense(hidden_size, input_dim=feature_dim, activation=activation, weights=model.layers[0].decoder.get_weights()[0:2]))
    model_rec.add(Dense(data_dim, activation=activation, weights=model.layers[0].decoder.get_weights()[2:4]))
    model_rec.layers[0].get_input(False) # Get input from testing data
    model_rec.compile(loss='mse', optimizer='sgd')
    
    if evaluation:
        data_rec = model_rec.predict(output[test])
        loss = mean_squared_error(data[test], data_rec)
        return loss
    
    name = '2-layer stacked autoencoder'
    
    return output, name, model_rec.predict

def ae3l(data, feature_dim, train, test, hidden_size_l1, hidden_size_l2, learning_rate, lr_decay, reg_fn, l, momentum, evaluation):
    ''' 3-layer stacked autoencoder '''
    
    batch_size=len(train)
    data_dim = data.shape[1]
    
    if pre_training:
        # Pre-training (greedy layer-wise training)
        if verbose:
            print('Pre-training for 1st layer...')
        ae1 = single_layer_autoencoder(data_dim, hidden_size_l1, reg_fn(l), learning_rate, lr_decay, momentum)
        ae1.fit(data[train], data[train], batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        a1 = ae1.predict(data[train]) # 1st hidden layer's activation
        w1_en = ae1.layers[0].encoder.get_weights()
        w1_de = ae1.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
        
        if verbose:
            print('Pre-training for 2nd layer...')
        ae2 = single_layer_autoencoder(hidden_size_l1, hidden_size_l2, reg_fn(l), learning_rate, lr_decay, momentum)
        ae2.fit(a1, a1, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        a2 = ae2.predict(a1) # 1st hidden layer's activation
        w2_en = ae2.layers[0].encoder.get_weights()
        w2_de = ae2.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
        
        if verbose:
            print('Pre-training for 3rd layer...')
        ae3 = single_layer_autoencoder(hidden_size_l2, feature_dim, reg_fn(l), learning_rate, lr_decay, momentum)
        ae3.fit(a2, a2, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        w3_en = ae3.layers[0].encoder.get_weights()
        w3_de = ae3.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
    
    else:
        w1_en = None
        w1_de = None
        w2_en = None
        w2_de = None
        w3_en = None
        w3_de = None
    
    # Fine-tuning
    if verbose:
        print('Fine-tuning...')
    model = Sequential()
    encoder = containers.Sequential([Dense(hidden_size_l1, input_dim=data_dim, activation=activation, weights=w1_en, 
                                           W_regularizer=reg_fn(l)),
                                     Dense(hidden_size_l2, activation=activation, weights=w2_en, W_regularizer=reg_fn(l)),
                                     Dense(feature_dim, activation=activation, weights=w3_en, W_regularizer=reg_fn(l))])
    decoder = containers.Sequential([Dense(hidden_size_l2, input_dim=feature_dim, activation=activation, weights=w3_de, 
                                           W_regularizer=reg_fn(l)),
                                     Dense(hidden_size_l1, activation=activation, weights=w2_de, W_regularizer=reg_fn(l)),
                                     Dense(data_dim, activation=activation, weights=w1_de, W_regularizer=reg_fn(l))])
    model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
    sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss=obj_fcn, optimizer=sgd)
    model.fit(data[train], data[train], nb_epoch=nb_epoch, verbose=verbose)
    if verbose:
        print('--------------------------------------------------------------')

    output = model.predict(data)
    
    # Reconstruction
    model_rec = Sequential()
    model_rec.add(Dense(hidden_size_l2, input_dim=feature_dim, activation=activation, weights=model.layers[0].decoder.get_weights()[0:2]))
    model_rec.add(Dense(hidden_size_l1, activation=activation, weights=model.layers[0].decoder.get_weights()[2:4]))
    model_rec.add(Dense(data_dim, activation=activation, weights=model.layers[0].decoder.get_weights()[4:6]))
    model_rec.layers[0].get_input(False) # Get input from testing data
    model_rec.compile(loss='mse', optimizer='sgd')
    
    if evaluation:
        data_rec = model_rec.predict(output[test])
        loss = mean_squared_error(data[test], data_rec)
        return loss
    
    name = '3-layer stacked autoencoder'
    
    return output, name, model_rec.predict

def ae4l(data, feature_dim, train, test, hidden_size_l1, hidden_size_l2, hidden_size_l3, learning_rate, lr_decay, reg_fn, l, 
         momentum, evaluation):
    ''' 4-layer stacked autoencoder '''
    
    batch_size=len(train)
    data_dim = data.shape[1]
    
    if pre_training:
        # Pre-training (greedy layer-wise training)
        if verbose:
            print('Pre-training for 1st layer...')
        ae1 = single_layer_autoencoder(data_dim, hidden_size_l1, reg_fn(l), learning_rate, lr_decay, momentum)
        ae1.fit(data[train], data[train], batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        a1 = ae1.predict(data[train]) # 1st hidden layer's activation
        w1_en = ae1.layers[0].encoder.get_weights()
        w1_de = ae1.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
        
        if verbose:
            print('Pre-training for 2nd layer...')
        ae2 = single_layer_autoencoder(hidden_size_l1, hidden_size_l2, reg_fn(l), learning_rate, lr_decay, momentum)
        ae2.fit(a1, a1, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        a2 = ae2.predict(a1) # 1st hidden layer's activation
        w2_en = ae2.layers[0].encoder.get_weights()
        w2_de = ae2.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
        
        if verbose:
            print('Pre-training for 3rd layer...')
        ae3 = single_layer_autoencoder(hidden_size_l2, hidden_size_l3, reg_fn(l), learning_rate, lr_decay, momentum)
        ae3.fit(a2, a2, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        a3 = ae3.predict(a2) # 2nd hidden layer's activation
        w3_en = ae3.layers[0].encoder.get_weights()
        w3_de = ae3.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
            
        if verbose:
            print('Pre-training for 4th layer...')
        ae4 = single_layer_autoencoder(hidden_size_l3, feature_dim, reg_fn(l), learning_rate, lr_decay, momentum)
        ae4.fit(a3, a3, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        w4_en = ae4.layers[0].encoder.get_weights()
        w4_de = ae4.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
    
    else:
        w1_en = None
        w1_de = None
        w2_en = None
        w2_de = None
        w3_en = None
        w3_de = None
        w4_en = None
        w4_de = None
    
    # Fine-tuning
    if verbose:
        print('Fine-tuning...')
    model = Sequential()
    encoder = containers.Sequential([Dense(hidden_size_l1, input_dim=data_dim, activation=activation, weights=w1_en, 
                                           W_regularizer=reg_fn(l)),
                                     Dense(hidden_size_l2, activation=activation, weights=w2_en, W_regularizer=reg_fn(l)),
                                     Dense(hidden_size_l3, activation=activation, weights=w3_en, W_regularizer=reg_fn(l)),
                                     Dense(feature_dim, activation=activation, weights=w4_en, W_regularizer=reg_fn(l))])
    decoder = containers.Sequential([Dense(hidden_size_l3, input_dim=feature_dim, activation=activation, weights=w4_de, 
                                           W_regularizer=reg_fn(l)),
                                     Dense(hidden_size_l2, activation=activation, weights=w3_de, W_regularizer=reg_fn(l)),
                                     Dense(hidden_size_l1, activation=activation, weights=w2_de, W_regularizer=reg_fn(l)),
                                     Dense(data_dim, activation=activation, weights=w1_de, W_regularizer=reg_fn(l))])
    model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
    sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss=obj_fcn, optimizer=sgd)
    model.fit(data[train], data[train], nb_epoch=nb_epoch, verbose=verbose)
    if verbose:
        print('--------------------------------------------------------------')

    output = model.predict(data)
    
    # Reconstruction
    model_rec = Sequential()
    model_rec.add(Dense(hidden_size_l3, input_dim=feature_dim, activation=activation, weights=model.layers[0].decoder.get_weights()[0:2]))
    model_rec.add(Dense(hidden_size_l2, activation=activation, weights=model.layers[0].decoder.get_weights()[2:4]))
    model_rec.add(Dense(hidden_size_l1, activation=activation, weights=model.layers[0].decoder.get_weights()[4:6]))
    model_rec.add(Dense(data_dim, activation=activation, weights=model.layers[0].decoder.get_weights()[6:8]))
    model_rec.layers[0].get_input(False) # Get input from testing data
    model_rec.compile(loss='mse', optimizer='sgd')
    
    if evaluation:
        data_rec = model_rec.predict(output[test])
        loss = mean_squared_error(data[test], data_rec)
        return loss
    
    #name = '4-layer stacked autoencoder'
    name = 'Autoencoder'
    
    return output, name, model_rec.predict

def ae5l(data, feature_dim, train, test, hidden_size_l1, hidden_size_l2, hidden_size_l3, hidden_size_l4, learning_rate, lr_decay,
         reg_fn, l, momentum, evaluation):
    ''' 5-layer stacked autoencoder '''
    
    batch_size=len(train)
    data_dim = data.shape[1]
    
    if pre_training:
        # Pre-training (greedy layer-wise training)
        if verbose:
            print('Pre-training for 1st layer...')
        ae1 = single_layer_autoencoder(data_dim, hidden_size_l1, reg_fn(l), learning_rate, lr_decay, momentum)
        ae1.fit(data[train], data[train], batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        a1 = ae1.predict(data[train]) # 1st hidden layer's activation
        w1_en = ae1.layers[0].encoder.get_weights()
        w1_de = ae1.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
        
        if verbose:
            print('Pre-training for 2nd layer...')
        ae2 = single_layer_autoencoder(hidden_size_l1, hidden_size_l2, reg_fn(l), learning_rate, lr_decay, momentum)
        ae2.fit(a1, a1, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        a2 = ae2.predict(a1) # 1st hidden layer's activation
        w2_en = ae2.layers[0].encoder.get_weights()
        w2_de = ae2.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
        
        if verbose:
            print('Pre-training for 3rd layer...')
        ae3 = single_layer_autoencoder(hidden_size_l2, hidden_size_l3, reg_fn(l), learning_rate, lr_decay, momentum)
        ae3.fit(a2, a2, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        a3 = ae3.predict(a2) # 2nd hidden layer's activation
        w3_en = ae3.layers[0].encoder.get_weights()
        w3_de = ae3.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
            
        if verbose:
            print('Pre-training for 4th layer...')
        ae4 = single_layer_autoencoder(hidden_size_l3, hidden_size_l4, reg_fn(l), learning_rate, lr_decay, momentum)
        ae4.fit(a3, a3, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        a4 = ae4.predict(a3) # 2st hidden layer's activation
        w4_en = ae4.layers[0].encoder.get_weights()
        w4_de = ae4.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')

        if verbose:
            print('Pre-training for 5th layer...')
        ae5 = single_layer_autoencoder(hidden_size_l4, feature_dim, reg_fn(l), learning_rate, lr_decay, momentum)
        ae5.fit(a4, a4, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
        w5_en = ae5.layers[0].encoder.get_weights()
        w5_de = ae5.layers[0].decoder.get_weights()
        if verbose:
            print('--------------------------------------------------------------')
    
    else:
        w1_en = None
        w1_de = None
        w2_en = None
        w2_de = None
        w3_en = None
        w3_de = None
        w4_en = None
        w4_de = None
        w5_en = None
        w5_de = None
    
    # Fine-tuning
    if verbose:
        print('Fine-tuning...')
    model = Sequential()
    encoder = containers.Sequential([Dense(hidden_size_l1, input_dim=data_dim, activation=activation, weights=w1_en, 
                                           W_regularizer=reg_fn(l)),
                                     Dense(hidden_size_l2, activation=activation, weights=w2_en, W_regularizer=reg_fn(l)),
                                     Dense(hidden_size_l3, activation=activation, weights=w3_en, W_regularizer=reg_fn(l)),
                                     Dense(hidden_size_l4, activation=activation, weights=w4_en, W_regularizer=reg_fn(l)),
                                     Dense(feature_dim, activation=activation, weights=w5_en, W_regularizer=reg_fn(l))])
    decoder = containers.Sequential([Dense(hidden_size_l4, input_dim=feature_dim, activation=activation, weights=w5_de, 
                                           W_regularizer=reg_fn(l)),
                                     Dense(hidden_size_l3, activation=activation, weights=w4_de, W_regularizer=reg_fn(l)),
                                     Dense(hidden_size_l2, activation=activation, weights=w3_de, W_regularizer=reg_fn(l)),
                                     Dense(hidden_size_l1, activation=activation, weights=w2_de, W_regularizer=reg_fn(l)),
                                     Dense(data_dim, activation=activation, weights=w1_de, W_regularizer=reg_fn(l))])
    model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
    sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss=obj_fcn, optimizer=sgd)
    model.fit(data[train], data[train], nb_epoch=nb_epoch, verbose=verbose)
    if verbose:
        print('--------------------------------------------------------------')

    output = model.predict(data)
    
    # Reconstruction
    model_rec = Sequential()
    model_rec.add(Dense(hidden_size_l4, input_dim=feature_dim, activation=activation, weights=model.layers[0].decoder.get_weights()[0:2]))
    model_rec.add(Dense(hidden_size_l3, activation=activation, weights=model.layers[0].decoder.get_weights()[2:4]))
    model_rec.add(Dense(hidden_size_l2, activation=activation, weights=model.layers[0].decoder.get_weights()[4:6]))
    model_rec.add(Dense(hidden_size_l1, activation=activation, weights=model.layers[0].decoder.get_weights()[6:8]))
    model_rec.add(Dense(data_dim, activation=activation, weights=model.layers[0].decoder.get_weights()[8:10]))
    model_rec.layers[0].get_input(False) # Get input from testing data
    model_rec.compile(loss='mse', optimizer='sgd')
    
    if evaluation:
        data_rec = model_rec.predict(output[test])
        loss = mean_squared_error(data[test], data_rec)
        return loss
    
    name = '5-layer stacked autoencoder'
    
    return output, name, model_rec.predict
    
def stacked_ae(data, feature_dim, train, test, hidden_size_l1=0, hidden_size_l2=0, hidden_size_l3=0, hidden_size_l4=0, 
               learning_rate=.1, lr_decay=1e-8, regularizer='l2', l=0., momentum=0.9, evaluation=False):
    ''' Select number of layers for autoencoder based on arguments 
        hidden_size_l1, hidden_size_l2, hidden_size_l3 and hidden_size_l4 '''
    
    reg_fn = regularizers[regularizer]
    
    if hidden_size_l1 == 0:
        f = ae
    elif hidden_size_l2 == 0:
        f = partial(ae2l, hidden_size=hidden_size_l1)
    elif hidden_size_l3 == 0:
        f = partial(ae3l, hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2)
    elif hidden_size_l4 == 0:
        f = partial(ae4l, hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2, hidden_size_l3=hidden_size_l3)
    else:
        f = partial(ae5l, hidden_size_l1=hidden_size_l1, hidden_size_l2=hidden_size_l2, hidden_size_l3=hidden_size_l3,
                    hidden_size_l4=hidden_size_l4)
    
    return f(data, feature_dim, train, test, learning_rate=learning_rate, lr_decay=lr_decay, reg_fn=reg_fn, l=l, momentum=momentum,
             evaluation=evaluation)


verbose = 0
pre_training = True
obj_fcn = 'mse'
activation = 'tanh'
nb_epoch = 5000
regularizers = {'l2': l2, 'l1': l1}
