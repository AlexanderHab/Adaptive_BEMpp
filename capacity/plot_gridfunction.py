#########################################################
#########################################################
#File is written in context of the  Capacity paper
#
#
#written by Alexander Haberl
#########################################################
#########################################################
# import packages

import bempp.api
import numpy as np
from IPython.core.debugger import Tracer


fname_grid = 'test_cube_zz_est_theta0.4step_step_8.msh'
fname_coeffs  = 'coeffs_test_cube_zz_est_theta0.4step_step_8.npy' 


#import initial grid
grid = bempp.api.import_grid(fname_grid)
space = bempp.api.function_space(grid, 'DP', 0)

coeffs = np.load(fname_coeffs)
fun = bempp.api.GridFunction(space, coefficients=coeffs)

fun.plot()


