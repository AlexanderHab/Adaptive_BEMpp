###############################################################################
# SOLVE V \phi = 1 adaptively
# for plynomial degree p <= pmax

# compare results with uniform refinement

# using residual error estimator


###############################################################################


import bempp.api
import numpy as np
import timeit

###############################################################################
# BEm++ settings


bempp.api.global_parameters.quadrature.double_singular += 6
bempp.api.global_parameters.quadrature.near.double_order +=6
bempp.api.global_parameters.quadrature.medium.double_order += 6
bempp.api.global_parameters.quadrature.far.double_order += 6
bempp.api.global_parameters.quadrature.far.single_order += 6


#bempp.api.global_parameters.assembly.boundary_operator_assembly_type = 'dense'
bempp.api.global_parameters.hmat.max_rank = 256
bempp.api.global_parameters.hmat.max_block_size = 2048 # default 2048
bempp.api.global_parameters.hmat.eps = 1E-6
bempp.api.global_parameters.hmat.admissibility = 'strong' #default = weak
bempp.api.global_parameters.hmat.coarsening = 'false' # default = true









###############################################################################
# define scatterer

#grid = bempp.api.shapes.sphere(h=0.1)



###############################################################################
#numerical settings
ada_steps=30
unif_steps = 13
nEmax = 110000

p=0

# define wavewnumber
#kvec=[1];
kvec=[1,2,4,8,16];


save_name = 'diss_helmholtz_Lshape_pp'


###############################################################################
#1) uniform refinement
###############################################################################



for k in kvec:

	unif_indicator =  np.zeros([unif_steps],dtype='float64')
	unif_estimator =  np.zeros([unif_steps],dtype='float64')
	unif_oscillation =  np.zeros([unif_steps],dtype='float64')


	unif_number_of_elements = np.zeros([unif_steps],dtype = 'int')
	unif_iteration_steps = np.zeros([unif_steps],dtype = 'int')



	##################################################################
	# define wave direction
	a_x = 1./np.sqrt(2)
	a_y = 1./np.sqrt(2)
	a_z = 0 

	#################################
	# plain wave
	#def dirichlet_data(x,n,domain_index,result):
	#	result[0] = np.exp(1j*(k_x*x[0] + k_y*x[1] + k_z*x[2]))

	def dirichlet_data(x,n,domain_index,result):
		result[0] = np.exp(1j*k*(a_x*x[0] + a_y*x[1] + a_z*x[2]))  
    
	#def neumann_data(x,n,domain_index,result):
	#	result[0] = 1j*(k_x*n[0] + k_y*n[1] + k_z*n[2]) * np.exp(1j* (k_x*x[0] + k_y*x[1] + k_z*x[2]))


	#################################	
	#point source
#	ps=[0.001,0.001,0]
#	def dirichlet_data(x,n,domain_index,result):
#		result[0] = np.exp(1j*k*np.sqrt((x[0]-ps[0])**2 +(x[1]-ps[1])**2 +(x[2]-ps[2])**2 ))*
#			( (x[0]-ps[0])**2 +(x[1]-ps[1])**2 +(x[2]-ps[2])**2 )**(-1./2)
    


	
	##################################################################
	#import grid

	#grid = bempp.api.import_grid('cube_final.msh')
	grid = bempp.api.import_grid('Lshape_helmholtz.msh')
	#grid = bempp.api.import_grid('/home/ahaberl/Desktop/work_in_progress/capacity_paper/meshes/Lshape_rot_sym.msh')


	##################################################################
	# start uniform loop

	for step_counter in range(unif_steps):
    
   
		print("#########################################################################")
		print("kvec=",kvec)
		print("#########################################################################")
		print("k=",k," in uniform step", step_counter+1)
		print("#########################################################################")
		

		#p1_space = bempp.api.function_space(grid, "P", p+1)
		p0_space = bempp.api.function_space(grid, "DP", p)
		space = bempp.api.function_space(grid, "DP", p+1)
		#osc_space = bempp.api.function_space(grid, "P", p+5)
    
		dirichlet_fun = bempp.api.GridFunction(p0_space, fun=dirichlet_data)

		###############################################################################
		# build operators
	
		print("build up operators and compute the right hand side")

		time_1 = timeit.time.time()

		slp = bempp.api.operators.boundary.helmholtz.single_layer(
			p0_space, p0_space, p0_space,k)


		rhs = - dirichlet_fun

		time_2 = timeit.time.time()
		print("computing the right hand side time in s =", time_2 - time_1) 

		###############################################################################
		# solve the system

		print("solve the system")
		time_1 = timeit.time.time()

		############### use cg 
		#number_of_iterations = 0
		#def callback(x):
		#	global number_of_iterations
		#	number_of_iterations += 1
		#	if (number_of_iterations % 100 ==0):
		#		print("-----cg iteration:",number_of_iterations) 

		#phi,info =  bempp.api.linalg.cg(slp, rhs, tol=1E-4, callback = callback)
		#print("Number of cg - iterations: {0}".format(number_of_iterations))


		############### use lu
#		phi = bempp.api.linalg.lu(slp, rhs)

		############### use gmres 
		from scipy.sparse.linalg import gmres

		number_of_iterations = 0
		def callback(x):
			global number_of_iterations
			number_of_iterations += 1
			if (number_of_iterations % 1000 ==0):
				print("-----gmres iteration:",number_of_iterations) 
	
	
		discrete_op = slp.strong_form()
	
		sol_vec, info = gmres(discrete_op, rhs.coefficients,tol=1e-5,callback=callback)
		print("Number of gmres - iterations: {0}".format(number_of_iterations))

		phi = bempp.api.GridFunction(p0_space, coefficients=sol_vec)	#31.1


		############### 

		time_2 = timeit.time.time()
		print("solving time in s =", time_2 - time_1) 
   





	 	###############################################################################
		# build operators  for the residual
		print("compute the residual")
		time_1 = timeit.time.time()	

		my_slo_op = bempp.api.operators.boundary.helmholtz.single_layer(
			p0_space, space, space,k)

		my_id_op = bempp.api.operators.boundary.sparse.identity(
			space, space, space)

		dirichlet_fun_res = bempp.api.GridFunction(space, fun=dirichlet_data)

	
		residual = (-1)*my_id_op*dirichlet_fun_res - my_slo_op*phi
	
		time_2 = timeit.time.time()
		print("computing the residual, time in s =", time_2 - time_1)

		############################################################################### 
		# compute the error estimator and do the marking
	
		print("compute the error estimator")
		time_1 = timeit.time.time()	
 
		error_estimator = np.zeros(grid.leaf_view.entity_count(0),dtype='float64')
		
		elements = []
		#compute the estimator on each element
		for i,element in enumerate(grid.leaf_view.entity_iterator(0)):
			elements.append(element)
			grid.mark(element)
    
			# compute error indicators
			error_estimator[i] = element.geometry.volume**(1./2)*residual.surface_grad_norm(element)**2

       
    
		error_estimator_mesh = sum(error_estimator)

		time_2 = timeit.time.time()
		print("computing the estimator, time in s =", time_2 - time_1)
	

		############################################################################### 
		# compute the data oscillation


		############################################################################### 
		# compute the combined error_indicator = estimator + oscillation

		error_indicator = np.zeros(grid.leaf_view.entity_count(0),dtype='float64')
	
		error_indicator = error_estimator 
	
		error_indicator_mesh = sum(error_indicator)

    
		print("number of elements",i+1)
		print("error estimator",error_indicator_mesh)
                    

		###############################################################################    
		# collect data for the plot   
		unif_number_of_elements[step_counter] = grid.leaf_view.entity_count(0)
		unif_indicator[step_counter] = np.sqrt(error_indicator_mesh)
		unif_estimator[step_counter] = np.sqrt(error_estimator_mesh)
		unif_iteration_steps[step_counter] = number_of_iterations
    

		###############################################################################    
		# refine grid 



		print("refine grid")
		if step_counter != unif_steps-1:
			grid.refine()                                          
		print("grid has been refined")
		print("New number of elements {0}".format(grid.leaf_view.entity_count(0)))


		if(grid.leaf_view.entity_count(0) >=nEmax):
			break




	###############################################################################
	# 2) adaptive refinement for different k
	###############################################################################
	#initialize array to store error
	ada_estimator =  np.zeros([ada_steps],dtype='float64')
	ada_indicator =  np.zeros([ada_steps],dtype='float64')
	ada_oscillation =  np.zeros([ada_steps],dtype='float64')
	
	ada_number_of_elements = np.zeros([ada_steps],dtype = 'int')
	ada_iteration_steps = np.zeros([ada_steps],dtype = 'int')



	##################################################################
	# define wave direction
	a_x = 1./np.sqrt(2)
	a_y = 1./np.sqrt(2)
	a_z = 0 

	#################################
	# plain wave
	#def dirichlet_data(x,n,domain_index,result):
	#	result[0] = np.exp(1j*(k_x*x[0] + k_y*x[1] + k_z*x[2]))

	def dirichlet_data(x,n,domain_index,result):
		result[0] = np.exp(1j*k*(a_x*x[0] + a_y*x[1] + a_z*x[2]))  
    
	#def neumann_data(x,n,domain_index,result):
	#	result[0] = 1j*(k_x*n[0] + k_y*n[1] + k_z*n[2]) * np.exp(1j* (k_x*x[0] + k_y*x[1] + k_z*x[2]))
    
	def neumann_data(x,n,domain_index,result):
		result[0] = 1j*(k_x*n[0] + k_y*n[1] + k_z*n[2]) * np.exp(1j* (k_x*x[0] + k_y*x[1] + k_z*x[2]))


	#################################	
	#point source
#	ps=[0.001,0.001,0]
#	def dirichlet_data(x,n,domain_index,result):
#		result[0] = np.exp(1j*k*np.sqrt((x[0]-ps[0])**2 +(x[1]-ps[1])**2 +(x[2]-ps[2])**2 ))*
#			( (x[0]-ps[0])**2 +(x[1]-ps[1])**2 +(x[2]-ps[2])**2 )**(-1./2)


	#grid = bempp.api.import_grid('cube_final.msh')
	grid = bempp.api.import_grid('Lshape_helmholtz.msh')
	#grid = bempp.api.import_grid('/home/ahaberl/Desktop/work_in_progress/capacity_paper/meshes/cube_12_new_2.msh')
	#grid = bempp.api.import_grid('/home/ahaberl/Desktop/work_in_progress/capacity_paper/meshes/Lshape_rot_sym.msh')

	for step_counter in range(ada_steps):
    

		print("#########################################################################")
		print("kvec=",kvec)
		print("#########################################################################")
		print("k=",k," in adaptive step", step_counter+1)
		print("#########################################################################")


		#p1_space = bempp.api.function_space(grid, "P", p+1)
		p0_space = bempp.api.function_space(grid, "DP", p)
		space = bempp.api.function_space(grid, "DP", p+1)
		#osc_space = bempp.api.function_space(grid, "P", p+5)
    
		dirichlet_fun = bempp.api.GridFunction(p0_space, fun=dirichlet_data)

		###############################################################################
		# build operators

		print("build up operators and compute the right hand side")

		time_1 = timeit.time.time()


		slp = bempp.api.operators.boundary.helmholtz.single_layer(
			p0_space, p0_space, p0_space,k)

	
		rhs = (-1)* dirichlet_fun


		time_2 = timeit.time.time()
		print("computing the right hand side time in s =", time_2 - time_1) 

		###############################################################################
		# solve the system

		print("solve the system")
		time_1 = timeit.time.time()

		############### use cg 
		#number_of_iterations = 0
		#def callback(x):
		#	global number_of_iterations
		#	number_of_iterations += 1
		#	if (number_of_iterations % 1000 ==0):
		#		print("-----cg iteration:",number_of_iterations) 

		#phi,info =  bempp.api.linalg.cg(slp, rhs, tol=1E-8, callback = callback)
		#print("Number of cg - iterations: {0}".format(number_of_iterations))


		############### use lu
#		phi = bempp.api.linalg.lu(slp, rhs)

		############### use gmres 
		from scipy.sparse.linalg import gmres
#
		number_of_iterations = 0
		def callback(x):
			global number_of_iterations
			number_of_iterations += 1
			if (number_of_iterations % 1000 ==0):
				print("-----gmres iteration:",number_of_iterations) 
#
#
		discrete_op = slp.strong_form()
#
		sol_vec, info = gmres(discrete_op, rhs.coefficients,tol=1e-5,callback=callback)
		print("Number of gmres - iterations: {0}".format(number_of_iterations))
#
		phi = bempp.api.GridFunction(p0_space, coefficients=sol_vec)	#31.1


		############################################################

		time_2 = timeit.time.time()
		print("solving time in s =", time_2 - time_1) 
   
	

		###############################################################################
		# build operators  for the residual
		print("compute the residual")
		time_1 = timeit.time.time()	
	
		my_slo_op = bempp.api.operators.boundary.helmholtz.single_layer(
			p0_space, space, space,k)


		my_id_op = bempp.api.operators.boundary.sparse.identity(
			space, space, space)


		dirichlet_fun_res = bempp.api.GridFunction(space, fun=dirichlet_data)

	
		residual = (-1)*my_id_op*dirichlet_fun_res - my_slo_op*phi

		time_2 = timeit.time.time()
		print("computing the residual, time in s =", time_2 - time_1)

		############################################################################### 
		# compute the error estimator and do the marking

		print("compute the error estimator")
		time_1 = timeit.time.time()	
 
		error_estimator = np.zeros(grid.leaf_view.entity_count(0),dtype='float64')
		local_capacity = np.zeros(grid.leaf_view.entity_count(0),dtype='float64')

  
		elements = []
		#compute the estimator on each element
		for i,element in enumerate(grid.leaf_view.entity_iterator(0)):
			elements.append(element)
    
			# compute error indicators
			error_estimator[i] = element.geometry.volume**(1./2)*residual.surface_grad_norm(element)**2

       
    
		error_estimator_mesh = sum(error_estimator)

		time_2 = timeit.time.time()
		print("computing the estimator, time in s =", time_2 - time_1)
	

		############################################################################### 
		# compute the data oscillation
		

		############################################################################### 
		# compute the combined error_indicator = estimator + oscillation

		error_indicator = np.zeros(grid.leaf_view.entity_count(0),dtype='float64')

		error_indicator = error_estimator 

		error_indicator_mesh = sum(error_indicator)



		###############################################################################
		# implement Doerfler-Marking

		marking_parameter = 0.5
    
		# get the indices of the sortet array (descending)
		idx = np.argsort((-1)*error_indicator)

		ind_on_marked =0 
		counter = 0 

		# do the marking
		while (ind_on_marked < marking_parameter * error_indicator_mesh):
			ind_on_marked += error_indicator[idx[counter]]
			grid.mark(elements[idx[counter]])
			counter += 1
    
    
		print("number of elements",i+1)
		print("number of marked elemens",counter)
		print("error estimator",error_indicator_mesh)

		###############################################################################                           
		# collect data for the plot   
    
		ada_number_of_elements[step_counter] = grid.leaf_view.entity_count(0)
		ada_indicator[step_counter] = np.sqrt(error_indicator_mesh)
		ada_estimator[step_counter] = np.sqrt(error_estimator_mesh)

		ada_iteration_steps[step_counter] = number_of_iterations	

		###############################################################################
		# print the estimator

		save_name_grid_res = 'res_' + save_name + '_k'+str(k)+'_step' + str(step_counter)+'.msh'

		sorted_local_errors = np.zeros_like(error_estimator)
		index_set = grid.leaf_view.index_set()
		for element in grid.leaf_view.entity_iterator(0):
			index = index_set.entity_index(element)
			global_dof_index = p0_space.get_global_dofs(element, dof_weights=False)[0]
			sorted_local_errors[global_dof_index] = error_estimator[index]

		final_res = bempp.api.GridFunction(p0_space, coefficients=sorted_local_errors)


		bempp.api.export(grid_function=final_res ,file_name=save_name_grid_res)





		print("refining step")
		if step_counter != ada_steps-1:
			grid.refine()                                          
			print("grid has been refined")
			print("New number of elements {0}".format(grid.leaf_view.entity_count(0)))

		else:
			print("grid has NOT been refined (last step)") 
			print("Number of elements {0}".format(grid.leaf_view.entity_count(0)))
		


	 


		if(grid.leaf_view.entity_count(0) >=nEmax):
			break



	###############################################################################
 	# write data into a file
	###############################################################################
	#save grid


	save_name_txt = save_name + '_k'+str(k)+'.txt'
	save_name_grid = save_name + '_k'+str(k)+'.msh'

	print("save grid")
	bempp.api.export(grid=grid, file_name=save_name_grid)



	with open(save_name_txt,'wb') as outfile:
 
		outfile.write(bytes('###################################### \n','UTF-8'))
		outfile.write(bytes('# direct Bem with residual error estimator and oscillation\n','UTF-8'))
		outfile.write(bytes('###################################### \n','UTF-8'))
  
		outfile.write(bytes('#Polynomial degree, uniform steps and adaptive steps \n','UTF-8'))
		np.savetxt(outfile,[k,unif_steps,ada_steps])
  
  
#		outfile.write(bytes('###################################### \n','UTF-8'))
#		outfile.write(bytes('#Error indicator (esimtator + oscillation) adaptive\n','UTF-8'))
#		np.savetxt(outfile,ada_indicator)

		outfile.write(bytes('###################################### \n','UTF-8'))
		outfile.write(bytes('#Error estimator adaptive\n','UTF-8'))
		np.savetxt(outfile,ada_estimator)



		outfile.write(bytes('###################################### \n','UTF-8'))	
		outfile.write(bytes('#Error indicator (esimtator + oscillation) uniform\n','UTF-8'))
		np.savetxt(outfile,unif_indicator)


		outfile.write(bytes('###################################### \n','UTF-8'))
		outfile.write(bytes('#Error estimator uniform\n','UTF-8'))
		np.savetxt(outfile,unif_estimator)


		outfile.write(bytes('###################################### \n','UTF-8'))
		outfile.write(bytes('#Number of uniform elements \n','UTF-8'))
		np.savetxt(outfile,unif_number_of_elements)
 

		outfile.write(bytes('###################################### \n','UTF-8'))
		outfile.write(bytes('#Number of adaptive elements \n','UTF-8'))
		np.savetxt(outfile,ada_number_of_elements)


		outfile.write(bytes('###################################### \n','UTF-8'))
		outfile.write(bytes('#Number of uniform iterations  \n','UTF-8'))
		np.savetxt(outfile,unif_iteration_steps)


		outfile.write(bytes('###################################### \n','UTF-8'))
		outfile.write(bytes('#Number of adaptive iterations \n','UTF-8'))
		np.savetxt(outfile,ada_iteration_steps)




#		save_name_csv = save_name + '_k'+str(k)+'.csv'
#
#		import csv
#		with open(save_name_csv, 'w', newline='') as csvfile:
#			spamwriter = csv.writer(csvfile, delimiter=' ')
#  		 	
#			spamwriter.writerow(ada_number_of_elements)
#			spamwriter.writerow(ada_estimator)
#
#			spamwriter.writerow(unif_number_of_elements)
#			spamwriter.writerow(unif_estimator)



      
 






        
        
        

