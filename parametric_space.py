"""
The code does three things:
1) Takes in the source of samples (e.g., superformular variables or glassware images)
2) Uses B-spline curves to fit their contours
3) Gets the xy coordinates of their B-spline control points

data : geometric representation of sample shapes (xy coordinates of the B-spline control points)

Usage: python parametric_space.py

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
"""

import glob
import os
import sys
import ConfigParser
import numpy as np
from glass import process_image
import superformula
import random
import math


def create_dir(path):
    if os.path.isdir(path): 
        pass 
    else: 
        os.mkdir(path)
        
def fit_samples(source_dir, indices, image_save_dir, n_control_points):
    ''' Fit all samples with B-spline curves '''
    x_plots = []
    splines = []
    parameterizations = []
    image_paths = glob.glob(source_dir+"*.*")
    for index in indices:
        image_name = os.path.splitext(os.path.basename(image_paths[index]))[0]+'.png'
        image_save_path = image_save_dir + image_name
        try:
            x_plot, parameterization, spline = process_image(image_paths[index], image_save_path, n_control_points)
            print('Processing: ' + os.path.basename(image_paths[index]))
            x_plots.append(x_plot)
            splines.append(spline)
            parameterizations.append(parameterization)
        except:
            print "For "+image_paths[index]
            print "Unexpected error:", sys.exc_info()[0]
        
    return x_plots, splines, parameterizations

def get_all_superformula_splines(source_dir, indices, image_save_dir, num_control_points, num_func_points=1000):
    ''' Fit all superformula samples with B-spline curves '''
    x_plots = []
    splines = []
    parameterizations = []
    image_paths = glob.glob(source_dir+"*.*")
    inputs = np.load(image_paths[0])
    for index in indices:
        image_save_path = image_save_dir+str(index+1)+'.png'
        x_plot, parameterization, spline, converged = superformula.superformula_spline(*inputs[index], im_save_path=image_save_path,
                                                                            num_control_points=num_control_points,
                                                                            num_func_points=num_func_points)
        print('' + str(index+1) + ' - Processing: ' + str(inputs[index]))
        if converged:
            x_plots.append(x_plot)
            splines.append(spline)
            parameterizations.append(parameterization)
        else:
            print "Failed to converge. Trying another."
    
    return x_plots, splines, parameterizations

def source_glass(n_samples):
    data_source = fit_samples

    return data_source

def source_superformula_linear(n_samples):
    data_source = get_all_superformula_splines
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    
    source_dir = config.get('Global', 'IMAGE_DIR')
    fname = source_dir+'sf_linear/inputs.npy'
    
    inputs = []
    if not os.path.isfile(fname): # If input file .npy not exist in source directory
        for i in range(2 * n_samples):      # Give a bunch of extra inputs so that we can skip ones that fail to converge
            a = random.uniform(1,6)
            b = random.uniform(1,6)
            inputs.append([a,b,3,7,18,18])      # linear
        np.save(fname, inputs)
        
    else:
        print 'Using the existing input file.'
        '''image_paths = glob.glob(source_dir+"sf_linear/*.*")
        inputs = np.load(image_paths[0])
    
    n_control_points = config.getint('Global', 'n_control_points')
    results_dir = config.get('Global', 'RESULTS_DIR') + "sf_linear/"

    def inv_transform(data):
        return [superformula.superformula_spline(d[0], d[1], 3, 7, 18, 18,
            im_save_path=None, num_control_points=n_control_points, num_func_points=1000)[0] for d in data]

    spline = uniform_bspline.UniformBSpline(2, n_control_points, 2, is_closed=True)

    create_dir(results_dir + "linear_original_space/")

    orig_space_min = config.getfloat('Global', 'orig_space_min')
    orig_space_max = config.getfloat('Global', 'orig_space_max')

    shape_plot.plot_original_space_grid(6, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], inv_transform, results_dir, 
                                "Superformula (linear)", spline, mirror=mirror)
    samples = [l[0:2] for l in inputs]
    shape_plot.plot_original_space_examples(6, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], inv_transform, results_dir, 
                                "Superformula (linear)", spline, samples, mirror=mirror)
    print "Done generating original space graph"
    '''
    return data_source

def source_superformula_s_nonlinear(n_samples):
    data_source = get_all_superformula_splines
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    
    source_dir = config.get('Global', 'IMAGE_DIR')
    fname = source_dir+'sf_s_nonlinear/inputs.npy'
    
    inputs = []
    if not os.path.isfile(fname): # If input file .npy not exist in source directory
        for i in range(2 * n_samples):      # Give a bunch of extra inputs so that we can skip ones that fail to converge
            a = random.uniform(1,6)
            b = random.uniform(1,6)
            inputs.append([a,b,3,7 - a + b,12 + a,12 + b])  # slightly nonlinear
        np.save(fname, inputs)
        
    else:
        print 'Using the existing input file.'
        '''image_paths = glob.glob(source_dir+"sf_s_nonlinear/*.*")
        inputs = np.load(image_paths[0])
    
    n_control_points = config.getint('Global', 'n_control_points')
    results_dir = config.get('Global', 'RESULTS_DIR') + "sf_s_nonlinear/"
    
    def inv_transform(data):

        return [superformula.superformula_spline(d[0], d[1], 3, 7 - d[0] + d[1], 12 + d[0], 12 + d[1],
            im_save_path=None, num_control_points=n_control_points, num_func_points=1000)[0] for d in data]

    spline = uniform_bspline.UniformBSpline(2, n_control_points, 2, is_closed=True)

    create_dir(results_dir + "s_nonlinear_original_space/")

    orig_space_min = config.getfloat('Global', 'orig_space_min')
    orig_space_max = config.getfloat('Global', 'orig_space_max')

    shape_plot.plot_original_space_grid(6, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], inv_transform, results_dir, 
                                "Superformula (slightly nonlinear)", spline, mirror=mirror)
    samples = [l[0:2] for l in inputs]
    shape_plot.plot_original_space_examples(6, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], inv_transform, results_dir, 
                                "Superformula (slightly nonlinear)", spline, samples, mirror=mirror)
    print "Done generating original space graph"
    '''
    return data_source

def source_superformula_v_nonlinear(n_samples):
    data_source = get_all_superformula_splines
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    
    source_dir = config.get('Global', 'IMAGE_DIR')
    fname = source_dir+'sf_v_nonlinear/inputs.npy'
    
    inputs = []
    if not os.path.isfile(fname): # If input file .npy not exist in source directory
        for i in range(2 * n_samples):      # Give a bunch of extra inputs so that we can skip ones that fail to converge
            a = random.uniform(1,6)
            b = random.uniform(1,6)
            inputs.append([a,b,math.floor(3 + (a + b) % 4),7,12 + a,12 + b])  # very nonlinear
                # require m to be an integer so that the period doesn't increase
        np.save(fname, inputs)
            
    else:
        print 'Using the existing input file.'
        '''image_paths = glob.glob(source_dir+"sf_v_nonlinear/*.*")
        inputs = np.load(image_paths[0])
    
    n_control_points = config.getint('Global', 'n_control_points')
    results_dir = config.get('Global', 'RESULTS_DIR') + "sf_v_nonlinear/"

    def inv_transform(data):
        return [superformula.superformula_spline(d[0], d[1], 3 + (d[0] + d[1]) % 4, 7, 12 + d[0], 12 + d[1],
            im_save_path=None, num_control_points=n_control_points, num_func_points=1000)[0] for d in data]

    spline = uniform_bspline.UniformBSpline(2, n_control_points, 2, is_closed=True)

    create_dir(results_dir + "v_nonlinear_original_space/")

    orig_space_min = config.getfloat('Global', 'orig_space_min')
    orig_space_max = config.getfloat('Global', 'orig_space_max')

    shape_plot.plot_original_space_grid(6, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], inv_transform, results_dir, 
                                "Superformula (very nonlinear)", spline, mirror=mirror)
    samples = [l[0:2] for l in inputs]
    shape_plot.plot_original_space_examples(6, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], inv_transform, results_dir, 
                                "Superformula (very nonlinear)", spline, samples, mirror=mirror)
    print "Done generating original space graph"
    '''
    return data_source

def get_parametric(source_dir, indices, source):
    
    create_dir(source_dir)
    
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    
    n_control_points = config.getint('Global', 'n_control_points')
    
    test_start = config.getint('Global', 'test_start')
    test_end = config.getint('Global', 'test_end')
    RESULTS_DIR = config.get('Global', 'RESULTS_DIR')+str(test_start)+'-'+str(test_end)+'/'
    create_dir(RESULTS_DIR)
    
    create_dir(RESULTS_DIR)
    
    example_dir = RESULTS_DIR + source + '/'
    create_dir(example_dir)
            
    save_dir1 = example_dir + 'n_control_points = ' + str(n_control_points) + '/'
    create_dir(save_dir1)

    image_save_dir = save_dir1+'im/'
    create_dir(image_save_dir)
    
    n_samples = len(indices)
    
    sources = {'sf_linear'      : source_superformula_linear,
               'sf_s_nonlinear' : source_superformula_s_nonlinear,
               'sf_v_nonlinear' : source_superformula_v_nonlinear,
               'glass'          : source_glass}
               
    data_source = sources[source](n_samples)

    # Only want the original space diagrams
    # exit()

    # Fit images with B-spline curves
    x_plots, splines, parameterizations = data_source(source_dir, indices, image_save_dir, n_control_points)

    data = np.zeros((n_samples, 2*n_control_points))
    for index in range(n_samples):
        data[index,:] = np.reshape(x_plots[index], 2*n_control_points)
        
    return x_plots, splines, parameterizations, data, save_dir1


if __name__ == "__main__":
    
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    
    IMAGE_DIR = config.get('Global', 'IMAGE_DIR')
    source = config.get('Global', 'source')
    source_dir = IMAGE_DIR+source+'/'
    n_samples = config.getint('Global', 'n_samples')
    n_control_points = config.getint('Global', 'n_control_points')
    
    print('Fitting B-spline curves to contours with No. of control points: '+str(n_control_points))
    x_plots, splines, parameterizations, data, save_dir1 = get_parametric(source_dir, range(n_samples), source)

    np.save('raw_parametric_%s.npy' % source, data)
    print 'Parametric data saved in raw_parametric_%s.npy.' % source
    
    
