'''
For the superformula example, plots the origianl samples and shapes in the a-b space

Usage: python sf_original_plot.py

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
'''

import glob
import os
import ConfigParser
import numpy as np
import superformula
import uniform_bspline
import shape_plot

def create_dir(path):
    if os.path.isdir(path): 
        pass 
    else: 
        os.mkdir(path)
        
def sf_linear(inputs):
    
    def inv_transform(data):
        return [superformula.superformula_spline(d[0], d[1], 3, 7, 18, 18,
            im_save_path=None, num_control_points=n_control_points, num_func_points=1000)[0] for d in data]

    spline = uniform_bspline.UniformBSpline(2, n_control_points, 2, is_closed=True)

    create_dir(results_dir + "linear_original_space/")

    orig_space_min = config.getfloat('Global', 'orig_space_min')
    orig_space_max = config.getfloat('Global', 'orig_space_max')

    shape_plot.plot_original_space_grid(6, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], inv_transform, results_dir, 
                                "Superformula (linear)", spline, mirror=False)
    samples = [l[0:2] for l in inputs]
    shape_plot.plot_original_space_examples(6, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], inv_transform, results_dir, 
                                "Superformula (linear)", spline, samples, mirror=False)
    print "Done generating original space graph"

def sf_s_nonlinear(inputs):
    
    def inv_transform(data):

        return [superformula.superformula_spline(d[0], d[1], 3, 7 - d[0] + d[1], 12 + d[0], 12 + d[1],
            im_save_path=None, num_control_points=n_control_points, num_func_points=1000)[0] for d in data]

    spline = uniform_bspline.UniformBSpline(2, n_control_points, 2, is_closed=True)

    create_dir(results_dir + "s_nonlinear_original_space/")

    orig_space_min = config.getfloat('Global', 'orig_space_min')
    orig_space_max = config.getfloat('Global', 'orig_space_max')

    shape_plot.plot_original_space_grid(6, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], inv_transform, results_dir, 
                                "Superformula (slightly nonlinear)", spline, mirror=False)
    samples = [l[0:2] for l in inputs]
    shape_plot.plot_original_space_examples(6, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], inv_transform, results_dir, 
                                "Superformula (slightly nonlinear)", spline, samples, mirror=False)
    print "Done generating original space graph"

def sf_v_nonlinear(inputs):

    def inv_transform(data):
        return [superformula.superformula_spline(d[0], d[1], 3 + (d[0] + d[1]) % 4, 7, 12 + d[0], 12 + d[1],
            im_save_path=None, num_control_points=n_control_points, num_func_points=1000)[0] for d in data]

    spline = uniform_bspline.UniformBSpline(2, n_control_points, 2, is_closed=True)

    create_dir(results_dir + "v_nonlinear_original_space/")

    orig_space_min = config.getfloat('Global', 'orig_space_min')
    orig_space_max = config.getfloat('Global', 'orig_space_max')

    shape_plot.plot_original_space_grid(6, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], inv_transform, results_dir, 
                                "Superformula (very nonlinear)", spline, mirror=False)
    samples = [l[0:2] for l in inputs]
    shape_plot.plot_original_space_examples(6, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], inv_transform, results_dir, 
                                "Superformula (very nonlinear)", spline, samples, mirror=False)
    print "Done generating original space graph"

    
if __name__ == "__main__":

    font_size = 18
    
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    
    source = config.get('Global', 'source')
    results_dir = config.get('Global', 'RESULTS_DIR') + source + '/'
    n_control_points = config.getint('Global', 'n_control_points')

    IMAGE_DIR = config.get('Global', 'IMAGE_DIR')
    source_dir = IMAGE_DIR+source+'/'
    image_paths = glob.glob(source_dir+"*.*")
    inputs = np.load(image_paths[0])

    sources = {'sf_linear'      : sf_linear,
               'sf_s_nonlinear' : sf_s_nonlinear,
               'sf_v_nonlinear' : sf_v_nonlinear}

    print source
    sources[source](inputs)
    
