import cplex
from cplex.exceptions import CplexError
import numpy as np
from matplotlib import pyplot as plt
import cProfile
import time
import concurrent.futures
####################################################################################################################################################################
def check_coverage(C,M):
    
    problem = cplex.Cplex()
    problem.set_results_stream(None)
    problem.objective.set_sense(problem.objective.sense.minimize)
    d=M.shape[1]
    variable_names=[f"x{i}" for i in range(d)]
    problem.variables.add(obj=[0.0]*d, names=variable_names,lb=[-cplex.infinity]*d,ub=[cplex.infinity]*d)
    for i in range(C.shape[0]):
        problem.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=variable_names, val=list(C[i]))
            ],
            senses=["G"],  # 'G' for >=
            rhs=[0.0]
        )
    for j in range(M.shape[0]):
        problem.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=variable_names, val=list(M[j]))
            ],
            senses=["L"],  # 'L' for <=
            rhs=[-1]
        )
    
    problem.solve()
    status = problem.solution.get_status()
    status_string = problem.solution.get_status_string(status)

    if status == problem.solution.status.optimal:
        #print("Primal problem is feasible and solved optimally.")
        #print("Optimal solution:")
        #print(problem.solution.get_values())
        return problem.solution.get_values()
    elif status == problem.solution.status.infeasible:
        return True
        
    else:
        print("An unexpected solution status was encountered.")
#########################################################################################################################################################################
def check_rank(W,x):
    c=0
    for w in W:
        if w@x>=0:
            c+=1
    return c
###########################################################################################################################################################
def calc_A(C,M):

    # Initialize the CPLEX problem
    problem = cplex.Cplex()
    #problem.set_log_stream(None)
    #problem.set_error_stream(None)
    #problem.set_warning_stream(None)
    problem.set_results_stream(None)
    problem.parameters.preprocessing.presolve.set(0)
    # Define the LP problem as a minimization problem
    problem.objective.set_sense(problem.objective.sense.minimize)
    d=M.shape[1]
    variable_names=[f"x{i}" for i in range(d)]
    # Define the coefficients of the variables in the objective function
    problem.variables.add(obj=[0.0]*d, names=variable_names,lb=[-cplex.infinity]*d,ub=[cplex.infinity]*d)
    # Define the constraints
    for i in range(C.shape[0]):
        problem.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=variable_names, val=list(C[i]))
            ],
            senses=["G"],  # 'G' for >=
            rhs=[0.0]
        )
    for j in range(M.shape[0]):
        problem.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=variable_names, val=list(-M[j]))
            ],
            senses=["G"],  # 'L' for <=
            rhs=[1.0]
        )
    # Solve the problem
    problem.solve()
    
    # Check the status of the solution
    status = problem.solution.get_status()
    status_string = problem.solution.get_status_string(status)

    if status == problem.solution.status.optimal:
        # print("Primal problem is feasible and solved optimally.")
        # print("Optimal solution:")
        # print(problem.solution.get_values())
        # print("Optimal objective value:")
        # print(problem.solution.get_objective_value())
        return False
    elif status == problem.solution.status.infeasible:
        # For infeasible problems, we can get the dual Farkas proof of infeasibility

        dual_farkas = problem.solution.advanced.dual_farkas()
        #print(f"ray dual farkas: {dual_farkas}")
        A = [i - C.shape[0] for i, val in enumerate(dual_farkas[0]) if val > 0 and i >= C.shape[0]]
        #print(f" constraints num: {problem.linear_constraints.get_num()}, constraint rhs: {problem.linear_constraints.get_rhs()}, A: {A}")
        return A
    else:
        print("An unexpected solution status was encountered.")

#######################################################################################################################################################################
def findCell(C,M):
    if C.shape[0]==M.shape[1]-1:
        cover=check_coverage(C,M)
        if cover==True:
            return False
        return cover
   # if C.shape[0]!=0:
   #     M=M[[np.linalg.matrix_rank(np.vstack((C,i)))>r for i in M]]
    # random vectors are linearly independent with prob 1
    A=calc_A(C,M)
    if A==False:
        return check_coverage(C,M)
        #return True
    for i in A:
        if C.shape[0]==0:
            rec=findCell(np.array([M[i]]),np.vstack((M[:i],M[i+1:])))
            if rec!=False:
                return rec
        else:
            rec=findCell(np.vstack((C,M[i])),np.vstack((M[:i],M[i+1:])))
            if rec!=False:
                return rec
    return False

def unpack_and_findCell(args):
    return findCell(*args)
def findCell_parallel(C,M):
    A=calc_A(C,M)
    if A==False:
        return check_coverage(C,M)
        #return True
    calls=[(np.array([M[i]]),np.vstack((M[:i],M[i+1:]))) for i in A]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(unpack_and_findCell, calls)
        for res in results:
            if res!=False:
                return res
    return False

def unpack_and_findCell_constrained(args):
    return findCell_constrained(*args)
def findCell_parallel_constrained(C,M,additional):
    A=calc_A(C,M)
    if A==False:
        return check_coverage(C,M)
    calls=[(np.vstack((C,np.array([M[i]]))),np.vstack((M[:i],M[i+1:])),additional) for i in A]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(unpack_and_findCell_constrained, calls)
        for res in results:
            if res!=False:
                return res
    return False
def findCell_constrained(C,M,additional):
    if C.shape[0]==M.shape[1] +additional:
        return False
   # if C.shape[0]!=0:
   #     M=M[[np.linalg.matrix_rank(n p.vstack((C,i)))>r for i in M]]
    # random vectors are linearly independent with prob 1
    A=calc_A(C,M)
    if A==False:
        return check_coverage(C,M)
        #return True
    for i in A:
        if C.shape[0]==0:
            rec=findCell_constrained(np.vstack((C,M[i])),np.vstack((M[:i],M[i+1:])),additional)
            if rec!=False:
                return rec
        else:
            rec=findCell_constrained(np.vstack((C,M[i])),np.vstack((M[:i],M[i+1:])),additional)
            if rec!=False:
                return rec
    return False
#########################################################################################################################################################################################
def stop_criterion(C):
        
    problem = cplex.Cplex()
    problem.set_results_stream(None) 
    problem.objective.set_sense(problem.objective.sense.minimize)
    d=C.shape[1]
    variable_names=[f"x{i}" for i in range(d)]
    problem.variables.add(obj=[0.0]*d, names=variable_names,lb=[-cplex.infinity]*d,ub=[cplex.infinity]*d)
    for i in range(C.shape[0]):
        problem.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=variable_names, val=list(C[i]))
            ],
            senses=["G"],  # 'G' for >=
            rhs=[1.0]
        )
    
    
    problem.solve()
    status = problem.solution.get_status()

    if status == problem.solution.status.optimal:
        return False
    elif status == problem.solution.status.infeasible:
        return True
        
    else:
        print("An unexpected solution status was encountered.")
######################################################################################

def cmap(W):
    R=np.linspace(-20,20,300)
    X,Y=np.meshgrid(R,R)
    res=np.zeros((X.shape[0],Y.shape[0]))
    for w in W:
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                if w@ np.array([X[i][j],Y[i][j]])>=0:
                    res[i][j]+=1
    fig, ax = plt.subplots()

    ctf=ax.contourf(X, Y, res,levels=np.arange(0, W.shape[0] + 2) - 0.5,colors=["blue","red","orange","yellow"]+["green"]*(W.shape[0]-3))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.colorbar(ctf, ticks=np.arange(0, W.shape[0] + 1))
    plt.show() 
    return res
def restricted_cmap(C,W):
    R=np.linspace(-20,20,300)
    X,Y=np.meshgrid(R,R)
    res=np.zeros((X.shape[0],Y.shape[0]))
    for w in W:
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                if w@ np.array([X[i][j],Y[i][j]])>=0:
                    res[i][j]+=1
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            for c in C:
                if c@ np.array([X[i][j],Y[i][j]])<0:
                    res[i][j]=-1
    fig, ax = plt.subplots()
    #masked_res = np.ma.masked_where(res == -1, res)
    ctf=ax.contourf(X, Y, res,levels=np.arange(-1, W.shape[0] + 2) - 0.5,colors=["black"]+["blue","orange","yellow"]+["green"]*(W.shape[0]-2),extend="neither")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.colorbar(ctf, ticks=np.arange(0, W.shape[0] + 1))
    plt.show() 
    return res
    ##################################################################################################################
def monteCarlo(W,c):
    m,d=W.shape
    for i in range(d**c):
        x=np.random.normal(size=d)
        output=W@x
        count=np.sum(output>0)
        if count<d:
            return list(x)
    return False
def van_der_corput(n, base):
    vdc, denom = 0, 1
    while n > 0:
        n, remainder = divmod(n, base)
        denom *= base
        vdc += remainder / denom
    return vdc

def halton_sequence(n_points, dim):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if dim > len(primes):
        raise ValueError("Number of dimensions exceeds available primes.")
    
    halton_points = np.zeros((n_points, dim))
    for d in range(dim):
        base = primes[d]
        for i in range(n_points):
            halton_points[i, d] = van_der_corput(i, base)-0.5
    return halton_points
def halton_sequence_positive(n_points, dim):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if dim > len(primes):
        raise ValueError("Number of dimensions exceeds available primes.")
    
    halton_points = np.zeros((n_points, dim))
    for d in range(dim):
        base = primes[d]
        for i in range(n_points):
            halton_points[i, d] = van_der_corput(i, base)
    return halton_points
    
def HaltonCarlo(W,c):
    m,d=W.shape
    halton_points=halton_sequence(d**c,d)
    for x in halton_points:
        active=W@x
        active=np.sum(active>0)
        if active<d:
            return list(x)
    return False
def HaltonCarlo_positive(W,c):
    m,d=W.shape
    halton_points=halton_sequence_positive(d**c+1,d)
    for x in range(1,len(halton_points)):
        active=W@halton_points[x]
        active=np.sum(active>0)
        if active<d:
            return list(halton_points[x])
    return False
def monteCarlo_positive(W,c):
    m,d=W.shape
    for i in range(d**c):
        x=np.abs(np.random.normal(size=d))
        output=W@x
        count=np.sum(output>0)
        if count<d:
            return list(x)
    return False
##############################################################################################################################
def initW_structured(m,d):
    W=np.random.normal(size=(m,d))
    for i in range(m):
        if i%(2*d)<d:
            W[i][i%d]=abs(W[i][i%d])
        else:
            W[i][i%d]=-abs(W[i][i%d])

import numpy as np

def orthogonal_init(shape):

    rows, cols = shape
    flat_shape = (max(rows, cols), max(rows, cols))
    random_matrix = np.random.randn(*flat_shape)
    q, r = np.linalg.qr(random_matrix)
    q = q * np.sign(np.diag(r))
    print(q)
    orthogonal_matrix = q[:rows, :cols]
    return orthogonal_matrix



###############################################################################################
def main():
    
    res=[]
    d=2
    res_carlo=[]
    res_halton=[]
    for d in range(2,3): 
        print(f"start for d={d}")
        C=np.identity(d)
        d_list=[]
        d_list_Carlo=[]
        d_list_Halton=[]
        count_carlo=[0,0,0,0]
        for m in range(2,15):
            print(m,d_list,d_list_Carlo,d_list_Halton)
            count=0
            count_carlo=[0,0,0,0]
            count_halton=[0,0,0,0]
            for i in range(1000):
                R=np.random.normal(size=(m,d))

                if  findCell_constrained(C,R,d):
                    count+=1
                for c in range(4,8):
                    if monteCarlo_positive(R,c):
                        count_carlo[c-4]+=1
                    if HaltonCarlo_positive(R,c):
                        count_halton[c-4]+=1
            d_list.append(count)
            d_list_Carlo.append(count_carlo)
            d_list_Halton.append(count_halton)
        res.append(d_list)
        res_carlo.append(d_list_Carlo)
        res_halton.append(d_list_Halton)

    return res
if __name__ == '__main__':
    main()
    #cProfile.run('main()')
