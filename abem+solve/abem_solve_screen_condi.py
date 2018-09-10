###############################################################################
#Code for the ABEM+solve paper in 3D

#using direct approach
#using residual error estimator

#code  needs a new version of BEM++
#+ modifikation of the mesh refinement


###############################################################################


import bempp.api
import numpy as np
import timeit
import math
from IPython.core.debugger import Tracer
from scipy.sparse.linalg import gmres

###############################################################################
# define polynomial degree
p=0;


###############################################################################
#set quadrature and h-matrix options

bempp.api.global_parameters.quadrature.double_singular =8
bempp.api.global_parameters.quadrature.near.double_order = 8  # 4
bempp.api.global_parameters.quadrature.near.single_order = 8 # 4
bempp.api.global_parameters.quadrature.medium.double_order = 8 # 3
bempp.api.global_parameters.quadrature.medium.single_order = 8 # 3
bempp.api.global_parameters.quadrature.far.double_order = 8 #  2
bempp.api.global_parameters.quadrature.far.single_order = 8 #  2

bempp.api.global_parameters.assembly.boundary_operator_assembly_type = 'dense'
#bempp.api.global_parameters.hmat.max_rank = 4096
#bempp.api.global_parameters.hmat.max_block_size = 32768  # default 2048
#bempp.api.global_parameters.hmat.eps = 1E-8
#bempp.api.global_parameters.hmat.admissibility = 'strong' #default = weak


###############################################################################
# define right hand side for direct BEM

def dirichlet_data(x, n, ind, res):
	#res[0] = x[0]+x[1]+x[2]
	#res[0] = 2*x[0]**2 - 2*x[1]**2 #- 4*x[2]**2
	#res[0] = 8*x[0]**5 - 40*x[0]**3*x[1]**2 + 15*x[0]*x[1]**4 -40*x[0]**3*x[2]**2+30*x[0]*x[1]**2*x[2]**2 + 15*x[0]*x[2]**4
	res[0] = 1
	#res[0] = x[2]*np.real((x[0]+1.j*x[1])**(2./3))
	#res[0] = np.real((x[0]+1.j*x[1])**(2))
	#res[0] = 1./(4*math.pi)*1./((x[0]+0.001)**2 + (x[1]+0.001)**2 + (x[2]+0.001)**2)**(1./2)


###############################################################################
# define right hand side for indirect BEM

#def dirichlet_data(x, n, ind, res):
#	res[0] = 1




###############################################################################


#number of adaptive steps
ada_steps = 200

#Max elements
max_el = 5000

# doerfler parameter
theta_vec = [0.5]

#pcg parameter
lambda_cg_vec = [0.001]






###############################################################################
###############################################################################
# define the precontitioning function (Matrix-Vektor product)

def precMl(x,H_list,I_list,D_list,AzeroInv,steps):
	
	x_temp = [0]*steps
	for i in range(steps-1,-1,-1):
		#print("precMl... first loop i:", i)
		# note that H_list[i] is already transposed
		x_temp[i] = (H_list[i].transpose() * D_list[i]*H_list[i]).dot(x) #wrong in general
		#x_temp[i] = np.dot((H_list[i].transpose()) , np.dot( D_list[i], H_list[i] )).dot(x)
		x = (I_list[i].transpose()).dot(x)
	
	x = AzeroInv.dot(x)

	for i in range(steps): 
		#print("precMl... first loop i:", i)
		x = I_list[i].dot(x)
		x = x + x_temp[i]
	return x

def prec_id(x,H_list,I_list,D_list,AzeroInv,steps):
	return x

def precDiag(x,slp_mat):
	for i in range(slp_mat.shape[0]):
		x[i] = x[i]*(slp_mat[i,i]**(-1))
	return x



###############################################################################
###############################################################################
# define power iteration 



def power_iteration(A,Pop,max_it):
	n, d = A.shape
	it_number = 0 

	ev = Pop(np.random.rand(d)) 
	ev = ev / np.sqrt(ev.dot(ev))
	
	#Tracer()()
	while True:

		Av = Pop(A.dot(ev))
		ev= Av / np.sqrt(Av.dot(Av))
		v = ev.dot(Pop(A).dot(ev))

		if it_number >= max_it:
			break

		it_number = it_number +1



	return ev,v


###############################################################################
###############################################################################
# define inverse power iteration 

	

def inv_power_iteration(A,Pop,max_it):
	n, d = A.shape
	it_number = 0 
	

	ev = Pop(np.random.rand(d)) 
	ev = ev / np.sqrt(ev.dot(ev))

	while True:
		
		wv, info = gmres(Pop(A),ev,x0 = ev, tol=1e-10,maxiter=50)	
		#Tracer()()
		#print(info)
		ev= wv/ np.sqrt(wv.dot(wv))
		v = ev.dot(Pop(A).dot(ev))

		if it_number >= max_it:
			break

		it_number = it_number +1

	return ev,v





###############################################################################
# define right hand side for direct BEM

#def dirichlet_data(x, n, ind, res):
	#res[0] = x[0]+x[1]+x[2]
	#res[0] = 2*x[0]**2 - 2*x[1]**2 #- 4*x[2]**2
	#res[0] = 8*x[0]**5 - 40*x[0]**3*x[1]**2 + 15*x[0]*x[1]**4 -40*x[0]**3*x[2]**2+30*x[0]*x[1]**2*x[2]**2 + 15*x[0]*x[2]**4
	#res[0] = 1
	#res[0] = x[2]*np.real((x[0]+1.j*x[1])**(2./3))
	#res[0] = np.real((x[0]+1.j*x[1])**(2))
	#res[0] = 1./(4*math.pi)*1./((x[0]+0.001)**2 + (x[1]+0.001)**2 + (x[2]+0.001)**2)**(1./2)


###############################################################################
# define right hand side for indirect BEM
def dirichlet_data(x, n, ind, res):
	res[0] = 1





###############################################################################
###############################################################################


###############################################################################
# 2) adaptive loop
###############################################################################

for theta in theta_vec:
	for lambda_cg in lambda_cg_vec:
		#save name
		save_name = 'screen_cond_test'+'theta' + str(theta) + 'lambda' + str(lambda_cg)


		#import grid
		grid = bempp.api.import_grid('Lshape_rot_sym_screen.msh')
		#grid = bempp.api.import_grid('cube_12.msh')

		#initialize array to store error
		ada_estimator =  np.zeros(ada_steps,dtype='float64')
		ada_indicator =  np.zeros(ada_steps,dtype='float64')
		ada_oscillation =  np.zeros(ada_steps,dtype='float64')
		ada_pcg_steps =  np.zeros(ada_steps,dtype='int')
		ada_cond_number =  np.zeros(ada_steps,dtype='float64')
		ada_cond_number_precon =  np.zeros(ada_steps,dtype='float64')
		ada_cond_number_diag =  np.zeros(ada_steps,dtype='float64')

		# initialize array to store the number of elements in each adaptive step
		ada_number_of_elements = np.zeros(ada_steps,dtype = 'int')

		# lists to store the precon-matrices
		H_list = []
		I_list = []
		D_list = []
		step_counter =-1		 


	
		while (grid.leaf_view.entity_count(0) <= max_el):
				step_counter = step_counter +1

		#for step_counter in range(ada_steps):
			    
			   
				print("#########################################################################")
				print("adaptive step", step_counter,"theta = ", theta, "and lambda = ",lambda_cg)
			
				#p1_space = bempp.api.function_space(grid, "P", p+1)
				p0_space = bempp.api.function_space(grid, "DP", p)
				space = bempp.api.function_space(grid, "DP", p+1)
				#osc_space = bempp.api.function_space(grid, "P", p+2)
				 

				###############################################################################
				# build operators

				print("build up operators and compute the right hand side")

				time_1 = timeit.time.time()

				#identity = bempp.api.operators.boundary.sparse.identity(
				#	p1_space, p1_space, p0_space)
				#dlp = bempp.api.operators.boundary.laplace.double_layer(
				#	p1_space, p1_space, p0_space)
				slp = bempp.api.operators.boundary.laplace.single_layer(
					p0_space, p0_space, p0_space)

				slp_mat = bempp.api.as_matrix(slp.weak_form())

				##################
				# for direct method	
			
				#dirichlet_fun = bempp.api.GridFunction(p1_space, fun=dirichlet_data)
				#rhs = (.5*identity+dlp)*dirichlet_fun
				
			

				##################
				#for indirect method
				dirichlet_fun = bempp.api.GridFunction(p0_space, fun=dirichlet_data)
				rhs = dirichlet_fun


				time_2 = timeit.time.time()
				print("computing the right hand side time in s =", time_2 - time_1) 


				
				###############################################################################
				# compute the preconditioner

				print("compute preconditioner")

				time_1 = timeit.time.time()


				#get A_0^(-1) 
				if step_counter ==0:
					AzeroInv = np.linalg.inv(slp_mat)
					phi_old_coeffs = np.zeros(grid.leaf_view.entity_count(0),dtype=float)
					phi_old = bempp.api.GridFunction(p0_space, coefficients = phi_old_coeffs)


				else:

					


					##########################################################################
					# build Haar matrix H
					#first in old grid
				
					nEl = grid.leaf_view.entity_count(0)
					nEd =  grid.leaf_view.entity_count(1)
					insertion_to_real = np.zeros(grid.leaf_view.entity_count(0),dtype='int')

					for i,element in enumerate(grid.leaf_view.entity_iterator(0)):	#loop goes over the insertion indices!!!!!!!!!
						#print("index i",i)
						el = grid.leaf_view.element_from_index(i)
						el_insertion_index = grid.element_insertion_index(el)	# get the insertion index of el
						insertion_to_real[el_insertion_index] = i



					H_trans = grid.leaf_view.edge_to_element_matrix.copy() #this is the transposed Haar-Matrix without scaling
					H_trans.dtype = float	


					#can be made faster if we just go over the no-zero entries and scale each one properly
					[row,col] = np.nonzero(H_trans)
					counter = 0 						#counter for the sign of the scaling factor
				
					for i,j in enumerate(row):
						edge_index = j
						el_index = col[i]					# get right element index
						edge = grid.leaf_view.edge_from_index(edge_index)	# get the corresponding edge
						vol_edge = edge.geometry.volume
						el = grid.leaf_view.element_from_index(el_index)	# get the corresponding element
						#el = grid.element_from_insertion_index(el_index)  #DEBUGG
						vol_el = el.geometry.volume

						#get the correct sign
						#get corners
						corners = el.geometry.corners
						corners_edge = edge.geometry.corners
						fun_sign = -1
						#check if edge has the same orientation as the triangle-> then sign = +
						if (corners[:,[0,1]] == corners_edge).all():
							fun_sign = 1
						if (corners[:,[1,2]] == corners_edge).all():
							fun_sign = 1
						if (corners[:,[2,0]] == corners_edge).all():
							fun_sign = 1
						H_trans[edge_index,el_index] = fun_sign *vol_edge/vol_el 	
					
					##########################################################################
					#build diagonal matrices


					H = grid.leaf_view.edge_to_element_matrix.copy() 
					H = H.transpose()			# now transposed

					 
					#use f2s array
					# if f2s-row contains more the one non -1 entry -> the element has been refined
					ref_elements = np.zeros(nEl,dtype=int)
					ref_insertion_index = []
					ref_insertion_index_father = np.nonzero(ftos[:,3]>-2)[0]  # get the insertion indices of father elements
					for i in ref_insertion_index_father:
						for j in ftos[i,:]:
							if j>-1:
								ref_insertion_index.append(j)

					ref_elements = insertion_to_real[ref_insertion_index]	# get the real indices
					#ref_elements = ref_insertion_index.copy() # test # DEBUGG

					#For screen Problem, consider only the edges in the inside		

					# find the indices of all edges of the refined elements
					ref_edges = np.nonzero(H[ref_elements,:])[1]
					ref_edges = np.unique(ref_edges) 		# clear double entries			


					#build the diagonal matrix
					diag_vec = np.zeros(nEd,dtype='float64')
					#loop over all refined edges
					for i in ref_edges:
						#get the cooresponding elements
						elements_for_edge =  np.nonzero(grid.leaf_view.edge_to_element_matrix[i,:])[1]
						
						#if edge is not on the boundary create corresponding entry
						# otherwise do nothing for boundary edge on the screen
						if (elements_for_edge.shape[0]==2):
							#get elements
							el_plus = grid.leaf_view.element_from_index(elements_for_edge[0])
							el_minus = grid.leaf_view.element_from_index(elements_for_edge[1])
							vol_plus = el_plus.geometry.volume
							vol_minus = el_minus.geometry.volume	
						
							#get edge
							edge =  grid.leaf_view.edge_from_index(i)
							vol_edge = edge.geometry.volume
						
							diag_vec[i] = ( (vol_edge/vol_plus)**2 * slp_mat[elements_for_edge[0],elements_for_edge[0]]  - 2*(vol_edge/vol_plus)*(vol_edge/vol_minus) * slp_mat[elements_for_edge[1],elements_for_edge[0]] + (vol_edge/vol_minus)**2 * slp_mat[elements_for_edge[1],elements_for_edge[1]] )**(-1)
				
					from scipy.sparse import dia_matrix
					diag = dia_matrix((diag_vec,0),shape=(nEd,nEd),dtype='float64')


					##########################################################################
					#build I matrices

					nEl_old = ftos.shape[0]

					I = np.zeros((nEl,nEl_old),dtype='float64')

					for i in range(nEl_old):		# go through father_to_son_array
						for j in ftos[i,:]:
							if (j>-1):		# if ftos[i,j]>-1 than the j-th element of T_[\ell+1] is the son of the i-th element in T_[\ell] 
								I[insertion_to_real[j],i] = 1			

					from scipy.sparse import csc_matrix	
					I = csc_matrix(I)

					##########################################################################
					#store matrices in list
					# note that index is shifted -> for ada_counter 1 -> index in list is 0 
					H_list.append(H_trans)
					I_list.append(I)
					D_list.append(diag)
				
				
					##########################################################################
					# prolongate old solution phi to the new mesh
					
					#get coefficients
					phi_old_coeffs = I.dot( phi_coeffs)
					phi_old = bempp.api.GridFunction(p0_space, coefficients= phi_old_coeffs)

				time_2 = timeit.time.time()
				print("computing the preconditioner time in s =", time_2 - time_1) 


				###############################################################################
				# build operators  for the residual
				print("building operators for the estimator")

				time_1 = timeit.time.time()


				my_slo_op = bempp.api.operators.boundary.laplace.single_layer(
					p0_space, space, space)

				
				#my_id_op = bempp.api.operators.boundary.sparse.identity(
				#	p1_space, space, space)

				#my_id_p1_p0 = bempp.api.operators.boundary.sparse.identity(
				#	p1_space, p0_space, p0_space)

				
				#my_dlp_op = bempp.api.operators.boundary.laplace.double_layer(
				#	p1_space, space, space)


				##############
				# for direct method
				#dirichlet_fun_res = bempp.api.GridFunction(p1_space, fun=dirichlet_data)
				#rhs_res = (.5*my_id_op + my_dlp_op)*dirichlet_fun_res




				##############
				# for indirect method
				dirichlet_fun_res = bempp.api.GridFunction(space, fun=dirichlet_data)
				rhs_res = dirichlet_fun_res


				time_2 = timeit.time.time()
				print("computing operators for the residual time in s =", time_2 - time_1) 

				###############################################################################
				# solve the system
				#implement the cg solver


				print("sovle + estimate")

				time_1 = timeit.time.time()


				error_estimator_mesh = 0  	
				pcg_error = np.inf

				rhs_fun_cg = rhs

				#DEBUG
				debug_id = bempp.api.operators.boundary.sparse.identity(
					p0_space, p0_space, p0_space)
				
				mass_mat = bempp.api.as_matrix(debug_id.weak_form()).copy()
				
				#DEBUG ############### use solve
#				phi_exact_coeff = np.linalg.solve(slp_mat, mass_mat.dot(rhs.coefficients))
#				phi_exact = bempp.api.GridFunction(p0_space, coefficients = (phi_exact_coeff))
				

				

				res_cg = mass_mat.dot(rhs_fun_cg.coefficients) - slp_mat.dot( phi_old.coefficients)
				h_cg = precMl(res_cg.copy(),H_list,I_list,D_list,AzeroInv,step_counter)
				#h_cg = precDiag(res_cg.copy(),slp_mat) #diagonal precon
				d_cg = h_cg.copy()
				pcg_step_counter = 0

				while(pcg_error >= lambda_cg*error_estimator_mesh):

					pcg_step_counter +=1
				
					print("##########################")	
					print("adaptive step:", step_counter, "pcg-step:", pcg_step_counter) 
					z = slp_mat.dot(d_cg)
					
					alpha_cg = (res_cg.dot(h_cg)) / (d_cg.dot(z))
					phi_coeffs =phi_old_coeffs + alpha_cg * d_cg 
					#phi = bempp.api.GridFunction(p0_space, coefficients=phi_coeffs)
					
					res_cg_new = res_cg - alpha_cg * z;
					h_cg_new = precMl(res_cg_new.copy(),H_list,I_list,D_list,AzeroInv,step_counter)
					#h_cg_new = precDiag(res_cg_new.copy(),slp_mat) #digonal precon
					beta_cg = res_cg_new.dot(h_cg_new) / (res_cg.dot(h_cg))
					
					d_cg = h_cg_new + beta_cg * d_cg

					#DEBUG 
					norm_res_cg = res_cg_new.dot(slp_mat.dot(res_cg_new)) #DEBUG
					print("||res_cg||_en :", norm_res_cg) #DEBUG

					res_cg = res_cg_new
					h_cg = h_cg_new	
					
				
					############################################################################### 
					# compute the error estimator
					
					phi = bempp.api.GridFunction(p0_space, coefficients= phi_coeffs)
					residual = rhs_res  - my_slo_op*phi
					#print("l2 norm residual:", residual.l2_norm())
					#residual = (.5*my_id_op + my_dlp_op)*dirichlet_fun_res - my_slo_op*phi


					#print("compute the error estimator")
					time_1 = timeit.time.time()	
			 
					error_estimator = np.zeros(grid.leaf_view.entity_count(0),dtype='float64')
			  
					elements = []
					#compute the estimator on each element
					for i,element in enumerate(grid.leaf_view.entity_iterator(0)):
						elements.append(element)
			    
						# compute error indicators
						error_estimator[i] = element.geometry.volume**(1./2)*residual.surface_grad_norm(element)**2
						#grid.mark(element)		#uniform refinemenet (store with theta = 0.99)
			       
			    
					error_estimator_mesh = sum(error_estimator)
					error_estimator_mesh = np.sqrt(error_estimator_mesh)

					time_2 = timeit.time.time()
					#print("computing the estimator, time in s =", time_2 - time_1)
					print("error estimator", error_estimator_mesh)
					
					
					phi_diff = phi_coeffs - phi_old_coeffs
					pcg_error = slp_mat.dot(phi_diff) 
					pcg_error = np.sqrt(phi_diff.dot(pcg_error))

					phi_old_coeffs = phi_coeffs

#					diff_to_exact = phi_exact.coefficients - phi_coeffs
#					error_to_real = slp_mat.dot(diff_to_exact)
#					error_to_real =  np.sqrt(diff_to_exact.dot(error_to_real))
#					print("error to real pcg solution:",error_to_real ) #DEBUG
					print("pcg error",pcg_error)


					
				############################################################

				time_2 = timeit.time.time()
				print("solving time in s =", time_2 - time_1) 
			   
				



				

				############################################################################### 
#not needed of indirect bem	# compute the data oscillation
				# note that, a grid function is the L2-orthogonal projection onto the given space
#				print("compute the oscillations")
#				time_1 = timeit.time.time()	

				oscillation = np.zeros(grid.leaf_view.entity_count(0),dtype='float64')

#				dirichlet_fun_osc = bempp.api.GridFunction(osc_space, fun=dirichlet_data)
#
#				id_osc_osc = 	my_id_op = bempp.api.operators.boundary.sparse.identity(
#					osc_space, osc_space, osc_space).weak_form().sparse_operator
#
#				id_p1_osc = bempp.api.operators.boundary.sparse.identity(
#					p1_space, osc_space, osc_space).weak_form().sparse_operator
#
#				# embed the data oscillation in osc_space
#				from scipy.sparse.linalg import spsolve
#
#				dirichlet_fun_p1_in_osc_coeffs = spsolve(id_osc_osc, id_p1_osc*dirichlet_fun.coefficients)
#				dirichlet_fun_p1_in_osc = bempp.api.GridFunction(osc_space, coefficients = dirichlet_fun_p1_in_osc_coeffs)
#				
#				oscillation_fun = dirichlet_fun_osc - dirichlet_fun_p1_in_osc
#
#				
#				for i,element in enumerate(grid.leaf_view.entity_iterator(0)):
#					oscillation[i] = element.geometry.volume**(1./2)*oscillation_fun.surface_grad_norm(element)**2
#
#
				oscillation_mesh = sum(oscillation)
#
#				time_2 = timeit.time.time()
#				print("computing the oscillations, time in s =", time_2 - time_1)

				############################################################################### 
				# compute the combined error_indicator = estimator + oscillation

				error_indicator = np.zeros(grid.leaf_view.entity_count(0),dtype='float64')

				error_indicator = error_estimator + oscillation

						
				error_indicator_mesh = sum(error_indicator)



				###############################################################################
				# implement Doerfler-Marking

				marking_parameter = theta
			    
				# get the indices of the sortet array (descending)
				idx = np.argsort((-1)*error_indicator)

				ind_on_marked =0 
				counter = 0 

				# do the marking
				while (np.sqrt(ind_on_marked) < marking_parameter * np.sqrt(error_indicator_mesh)):
					ind_on_marked += error_indicator[idx[counter]]
					grid.mark(elements[idx[counter]])
					counter += 1
			    
			    
				print("number of elements",i+1)
				print("number of marked elemens",counter)
				print("error indicator (est + osc):",error_indicator_mesh)

				###############################################################################
				 #comupte the condition number
				###############################################################################
				
				print("compute the condition numbers")


				# Construct a linear operator that computes P^-1 * x.
				def PrecMLop(x):
					y = precMl(x.copy(),H_list,I_list,D_list,AzeroInv,step_counter) 	
					return y
				
				def test_diag_op(x):	
					y = precDiag(x.copy(),slp_mat)
					return y

				def id_op_cond(x):
					y=x.copy()
					return y 

				#Tracer()()
				dim = slp_mat.shape[0]
				import scipy.sparse.linalg as spla
				Pinv = spla.LinearOperator((dim, dim), matvec =PrecMLop )
				Id_op_cond = spla.LinearOperator((dim, dim), matvec =id_op_cond )
				Diag_op_cond = spla.LinearOperator((dim,dim), matvec = test_diag_op)

				#Tracer()()
				print("comp. condition number non precondtioned ") 
				#compute condition number for non precondditioned 
				vec,max_ev_non_prec = power_iteration(slp_mat,Id_op_cond,20)
				vec,min_ev_non_prec = inv_power_iteration(slp_mat,Id_op_cond,20)
				
				cond_number = max_ev_non_prec / min_ev_non_prec
				
				print("comp. condition number precondtioned ") 
				#compute condition number for precondditioned matrix
				vec,max_ev_prec = power_iteration(slp_mat,Pinv,20)
				vec,min_ev_prec = inv_power_iteration(slp_mat,Pinv,20)
				
				cond_number_precon = max_ev_prec / min_ev_prec

				#Tracer()()
				exact_non_cond_number = np.linalg.cond(slp_mat,'fro')	



				#compute condition number for diagonal preconditioner
				vec,max_ev_diagprec = power_iteration(slp_mat,Diag_op_cond,20)
				vec,min_ev_diagprec = inv_power_iteration(slp_mat,Diag_op_cond,20)
				
				cond_number_diag = max_ev_diagprec / min_ev_diagprec

				print("->->->->->->->->test_condition_number", exact_non_cond_number/ cond_number )  
				print("->->->->->->->->max eigenvalues, non-prec,ML-prec,diag-prec:", max_ev_non_prec,max_ev_prec,max_ev_diagprec )  
				print("->->->->->->->->min eigenvalues, non-prec,ML-prec,diag-prec:", min_ev_non_prec,min_ev_prec,min_ev_diagprec )  
				############################################################

				#Tracer()()
				ada_cond_number[step_counter] = cond_number
				ada_cond_number_precon[step_counter] = cond_number_precon
				ada_cond_number_diag[step_counter] = cond_number_diag

				
				
				print("condition number non preconditioned =", cond_number) 
				print("condition number preconditioned =", cond_number_precon) 
				print("condition number diagonal preconditioned =", cond_number_diag) 
				###############################################################################                           
				# collect data for the plot   
			    
				ada_number_of_elements[step_counter] = grid.leaf_view.entity_count(0)
				ada_indicator[step_counter] = np.sqrt(error_indicator_mesh)
				ada_oscillation[step_counter] = np.sqrt(oscillation_mesh)
				ada_estimator[step_counter] = error_estimator_mesh # is already ^(1/2)
				ada_pcg_steps[step_counter] = pcg_step_counter


				###############################################################################
				# save grid 
				###############################################################################
				
				save_name_grid = save_name + 'step' +str(step_counter) +'.msh'
				bempp.api.export(grid=grid, file_name=save_name_grid)
				#grid.plot()

				###############################################################################
				print("refining step")
				if step_counter != ada_steps-1:
					#old_grid = grid
					ftos,grid = grid.refine()                                          
					print("grid has been refined")
					print("New number of elements {0}".format(grid.leaf_view.entity_count(0)))

				else:
					print("grid has NOT been refined (last step)") 
					print("Number of elements {0}".format(grid.leaf_view.entity_count(0)))

				


				###############################################################################
				 # write data into a file
				###############################################################################
				save_name_txt = save_name + 'step' + str(step_counter) +'.txt'

				
				with open(save_name_txt,'wb') as outfile:
			 
					outfile.write(bytes('###################################### \n','UTF-8'))
					outfile.write(bytes('# ABEM + solve example direct Bem \n','UTF-8'))
					outfile.write(bytes('###################################### \n','UTF-8'))
			  
					outfile.write(bytes('#Polynomial degree, adaptive steps, theta, lambda_cg \n','UTF-8'))
					np.savetxt(outfile,[p,ada_steps,theta,lambda_cg])
			  
					outfile.write(bytes('###################################### \n','UTF-8'))
					outfile.write(bytes('#Number of adaptive elements \n','UTF-8'))
					np.savetxt(outfile,ada_number_of_elements)

					outfile.write(bytes('###################################### \n','UTF-8'))
					outfile.write(bytes('#Condition number \n','UTF-8'))
					np.savetxt(outfile,ada_cond_number)

					outfile.write(bytes('###################################### \n','UTF-8'))
					outfile.write(bytes('#Condition number preconditioned matrix \n','UTF-8'))
					np.savetxt(outfile,ada_cond_number_precon)
			  
					outfile.write(bytes('###################################### \n','UTF-8'))
					outfile.write(bytes('#Condition number digonal matrix \n','UTF-8'))
					np.savetxt(outfile,ada_cond_number_diag)
			  
					outfile.write(bytes('#Error indicator (esimtator + oscillation) adaptive\n','UTF-8'))
					np.savetxt(outfile,ada_indicator)

					outfile.write(bytes('###################################### \n','UTF-8'))
					outfile.write(bytes('#Number PCG-steps \n','UTF-8'))
					np.savetxt(outfile,ada_pcg_steps)


			###############################################################################
			# end of adaptive loop
			###############################################################################


###############################################################################
# plot the actual residual

# Resort the error contributions
#sorted_local_errors = np.zeros_like(error_indicator)
#index_set = grid.leaf_view.index_set()
#for element in grid.leaf_view.entity_iterator(0):
#    index = index_set.entity_index(element)
#    global_dof_index = p0_space.get_global_dofs(element, dof_weights=False)[0]
#    sorted_local_errors[global_dof_index] = error_indicator[index]


#bempp.api.GridFunction(p0_space, coefficients=sorted_local_errors).plot()



###############################################################################
# plot other stuff

#plot phi
#phi.plot()

# plot the right hand side
#residual.plot()

#plot the dirichlet data
#dirichlet_fun.plot()



###############################################################################
# save grid and plot grid



#grid.plot()



###############################################################################
# plot the error estomator

#import matplotlib.pyplot as plt


#referenz_gerade =  ada_number_of_elements**(-1./3)
#referenz_gerade_2 =  ada_number_of_elements**(-4./3)*3


#plt.loglog(unif_number_of_elements,unif_indicator,'o-',linewidth=4, color='blue',label='unif')
#plt.loglog(ada_number_of_elements,ada_indicator,'o-',linewidth=4,color='red',label='adaptive')


#plt.loglog(ada_number_of_elements,referenz_gerade,'o--',linewidth=3,color='green',label='1/3')
#plt.loglog(ada_number_of_elements,referenz_gerade_2,'o--',linewidth=3,color='cyan',label='2/3')

#plt.legend()
#plt.show()





        
        

