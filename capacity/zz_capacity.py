#########################################################
#########################################################
#File is written in context of the  Capacity paper
# 
#Adaptive algorithm to compute the capacity of a given polyhedra. 
#Algorithm uses a ZZ-type a posteriori error estimator and Dörfler-markign
#
#Code needs a functioning BEM++ library with NVB
# grid_view.py - Badhack is not necessarily requiered
#
#
#written by Alexander Haberl
#########################################################
#########################################################
# import packages

import bempp.api
import numpy as np
import timeit
from IPython.core.debugger import Tracer
from scipy.sparse.linalg import gmres

##########################################################
# Set some options for the Adaptive Algorithm

#number of adaptive steps
steps =30
#polynomial degree
p=0 #currently the code only works for p=0

#marking parameter
theta = 0.4


#import initial grid
grid = bempp.api.import_grid('cube_12.msh')

#file name for result file
save_name = 'cube_zz_est_theta' + str(theta) + 'step_'  


# BEM++ options
bempp.api.global_parameters.quadrature.double_singular = 8
bempp.api.global_parameters.quadrature.near.double_order = 8 #
bempp.api.global_parameters.quadrature.near.single_order = 8 #
bempp.api.global_parameters.quadrature.medium.double_order =8
bempp.api.global_parameters.quadrature.medium.single_order =8
bempp.api.global_parameters.quadrature.far.double_order =8
bempp.api.global_parameters.quadrature.far.single_order =8

#bempp.api.global_parameters.assembly.boundary_operator_assembly_type = 'dense'
bempp.api.global_parameters.hmat.max_rank = 256
bempp.api.global_parameters.hmat.max_block_size = 2048 # default 2048
bempp.api.global_parameters.hmat.eps = 1E-10
bempp.api.global_parameters.hmat.admissibility = 'strong' #default = weak


##########################################################
# RHS
def one_fun(x, n, domain_index, res):
    res[0] = -1


#########################################################
# initialize storage arrays

#initialize array to store error
plot_error =  np.zeros(steps,dtype='float64')
# initialize array to store the number of elements in each adaptive step
number_of_elements = np.zeros(steps,dtype = 'int')
# initialize array to store the capacity
capacity =  np.zeros(steps,dtype='float64')
#initialize array to store the solving time
solve_time =  np.zeros(steps,dtype='float64')
#initialize array to store the assembly_time time
assembly_time =  np.zeros(steps,dtype='float64')
#initialize array to store the time to compute the estimator
est_time =  np.zeros(steps,dtype='float64')
# initialize array to store the number of iterations in each adaptive step
iter_number = np.zeros(steps,dtype = 'int')


#########################################################
#Adaptive loop
#########################################################
for step_counter in range(steps):
   
    print("#########################################################################")
    print("Adaptive Algorithm for the capacity (ZZ-type esimator)")
    print("#########################################################################")
    print("adaptive step", step_counter+1)


    print("Building up BEM-Operators")
    time_1 = timeit.time.time()	


    # initialize function spaces and operators
    const_space = bempp.api.function_space(grid, "DP", p)
    lin_space = bempp.api.function_space(grid, "P",p+1)
    bary_space = bempp.api.function_space(grid.barycentric_grid(), 'DP', p)
    bary_lin_space = bempp.api.function_space(grid.barycentric_grid(), 'DP', p+1) 
    
    base_slp = bempp.api.operators.boundary.laplace.single_layer(bary_space, bary_space, bary_space)
    slp, hyp = bempp.api.operators.boundary.laplace.single_layer_and_hypersingular_pair(grid, spaces='dual', base_slp=base_slp,stabilization_factor=1)  

    rank_one_op = bempp.api.RankOneBoundaryOperator(hyp.domain, hyp.range, hyp.dual_to_range)
    hyp_regularized = hyp + rank_one_op
    
    
    #initialize right hand side
    rhs_fun = bempp.api.GridFunction(slp.range, fun=one_fun)

    #initialize preconditioned BEM-operator 
    lhs = slp * hyp_regularized
    discrete_op = lhs.strong_form()


    time_2 = timeit.time.time()
    print("assembly time in s =", time_2 - time_1)
    assembly_time[step_counter]=time_2 -time_1


    ############################################################
    # solve via GMRES
    ############################################################

    print("solve via gmres")
    time_1 = timeit.time.time()

    number_of_iterations = 0	
    def callback(x):
        global number_of_iterations
        number_of_iterations += 1
    

    sol_vec, info = gmres(discrete_op, rhs_fun.coefficients,tol=1e-10, callback=callback)
    sol_fun = hyp_regularized * bempp.api.GridFunction(hyp.domain, coefficients=sol_vec)

    print("Number of iterations: {0}".format(number_of_iterations))


    iter_number[step_counter] = number_of_iterations

    time_2 = timeit.time.time()
    print("Time for solving in s =", time_2 - time_1)
    solve_time[step_counter]=time_2 -time_1


    ############################################################
    # compute the capacity
    ############################################################
    capacity[step_counter]=-sol_fun.integrate()[0,0]
    print("The capacity is {0}.".format(capacity[step_counter]))


    ############################################################
    # compute the error estimator

    # compute the ZZ error estimator 
    # eta = ||h^{1/2} (phi_\ell - I_\ell \phi_\ell)  ||_{L^2(T)) 	
    # interpret the value of \phi_\ell \in P^0(\TT^\dual) 
    # as node value of $I_\ell \phi_\ell \in S^1(\TT_\ell)

    # for more details we refer to the capacity paper  
    ############################################################

    print("compute the errror estimator")
    time_1 = timeit.time.time()

	
    bary_grid = grid.barycentric_grid()
    bary_map = grid.barycentric_descendents_map()
    bary_index_set = bary_grid.leaf_view.index_set()

    ############################################################
    ############################################################


    index_set = grid.leaf_view.index_set()
    elements = grid.leaf_view.entity_count(0) * [None]
    for element in grid.leaf_view.entity_iterator(0):
        elements[index_set.entity_index(element)] = element
        #grid.mark(element) # mark for uniform refinement 	# uniform refinement


    local_zz_est_squared_bary = np.zeros(bary_grid.leaf_view.entity_count(0), dtype='float64')
    local_est = np.zeros(grid.leaf_view.entity_count(0), dtype='float64')
    #The "new_space" is a space S^0(\TT) but internally based on the barycentric refinement
    #if normal space is used, the map_lin_to_bary_lin does not work
    new_space = bempp.api.function_space(grid,"B-P",1)	    

    map_dual_to_bary = bempp.api.operators.boundary.sparse.identity(sol_fun.space,bary_lin_space,bary_lin_space)
    map_lin_to_bary_lin =  bempp.api.operators.boundary.sparse.identity(new_space,bary_lin_space,bary_lin_space)
		 

    Iphi_coefficients = sol_fun.coefficients
    Iphi = bempp.api.GridFunction(new_space, coefficients=Iphi_coefficients)

    # lift both functions to P^1(\TT_bary)
    phi = map_dual_to_bary * sol_fun
    Iphi_lin_bary = map_lin_to_bary_lin * Iphi

    diff = phi - Iphi_lin_bary
	

    for element in bary_grid.leaf_view.entity_iterator(0):
        index = bary_index_set.entity_index(element)
        norm = diff.l2_norm(element)**2 
        local_zz_est_squared_bary[index] = norm
  

    for m in range(bary_map.shape[0]):
        element = grid.leaf_view.element_from_index(m)
        mesh_size = element.geometry.volume
        for n in range(bary_map.shape[1]):
            local_est[m] += local_zz_est_squared_bary[bary_map[m, n]]
        local_est[m] = local_est[m]*mesh_size**(1./2)
	
    total_zz_est = np.sum(local_est)
    
    print("Squared ZZ-Estimator: {0}".format(total_zz_est))
    time_2 = timeit.time.time()
    print("Time estimatring in s =", time_2 - time_1)
    est_time[step_counter]=time_2 -time_1 


    ############################################################
    #implement Doerfler-Marking
    ############################################################
    print("do the marking")
    time_1 = timeit.time.time()    

    marking_parameter = theta
    error_indicator_mesh =  total_zz_est

    # get the indices of the sortet array (descending)
    idx = np.argsort((-1)*local_est)

    ind_on_marked =0 
    counter = 0 

    while (ind_on_marked < marking_parameter * error_indicator_mesh):
        ind_on_marked += local_est[idx[counter]]
        grid.mark(elements[idx[counter]])
        counter += 1
   
  
    print("number of total elements",grid.leaf_view.entity_count(0))
    print("number of marked elemens",counter)
    print("error estimator",error_indicator_mesh)


    #collect data for the plot   
    number_of_elements[step_counter] = grid.leaf_view.entity_count(0)
    plot_error[step_counter] = error_indicator_mesh 


    time_2 = timeit.time.time()
    print("Time for marking in s =", time_2 - time_1)


    ############################################################
    #plot the error estimator 
    ############################################################

    # Resort the error contributions
    sorted_local_est = np.zeros_like(local_est)
    index_set = grid.leaf_view.index_set()
    for element in grid.leaf_view.entity_iterator(0):
        index = index_set.entity_index(element)
        global_dof_index = const_space.get_global_dofs(element, dof_weights=False)[0]
        sorted_local_est[global_dof_index] = local_est[index]

    final_res = bempp.api.GridFunction(const_space, coefficients=sorted_local_est)


    ############################################################
    # save grid
    ############################################################



    
    fname = save_name + 'step_' + str(step_counter) + '.msh'
    fname_coeffs = 'coeffs_' + save_name + 'step_' + str(step_counter) + '.npy'

    bempp.api.export(grid=grid, file_name=fname)
    np.save(fname_coeffs ,final_res.coefficients)

    ############################################################
    # refine
    ############################################################

    print("refining step")
    if step_counter != steps-1:
        ftos,grid = grid.refine()
        print("grid has been refined")
        print("New number of elements {0}".format(grid.leaf_view.entity_count(0)))

    else:
        print("grid has NOT been refined (last step)")
        print("Number of elements {0}".format(grid.leaf_view.entity_count(0)))


    ############################################################
    # save_data
    ############################################################

    fname = save_name + str(step_counter) + '.txt'
 
    with open(fname,'wb') as outfile:

        outfile.write(bytes('###################################### \n','UTF-8'))
        outfile.write(bytes('# Capacity for residual error estimator \n','UTF-8'))
        outfile.write(bytes('###################################### \n','UTF-8'))

        outfile.write(bytes('###################################### \n','UTF-8'))
        outfile.write(bytes('#Polynomial degree and adaptive steps \n','UTF-8'))
        np.savetxt(outfile,[p,steps], delimiter=" ", fmt="%f")

        outfile.write(bytes('###################################### \n','UTF-8'))
        outfile.write(bytes('#Capacity \n','UTF-8'))
        np.savetxt(outfile,capacity)

        outfile.write(bytes('###################################### \n','UTF-8'))
        outfile.write(bytes('#Error estimator \n','UTF-8'))
        np.savetxt(outfile,plot_error)
	
        
        outfile.write(bytes('###################################### \n','UTF-8'))
        outfile.write(bytes('#Number of elements \n','UTF-8'))
        np.savetxt(outfile,number_of_elements)


        outfile.write(bytes('###################################### \n','UTF-8'))
        outfile.write(bytes('#store the assembly_time \n','UTF-8'))
        np.savetxt(outfile,assembly_time)

        outfile.write(bytes('###################################### \n','UTF-8'))
        outfile.write(bytes('#store the solving time \n','UTF-8'))
        np.savetxt(outfile,solve_time)

        outfile.write(bytes('###################################### \n','UTF-8'))
        outfile.write(bytes('#store the est time \n','UTF-8'))
        np.savetxt(outfile,est_time)


        outfile.write(bytes('###################################### \n','UTF-8'))
        outfile.write(bytes('#number of iterations \n','UTF-8'))
        np.savetxt(outfile,iter_number)
