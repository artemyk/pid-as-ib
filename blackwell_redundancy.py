# -*- coding: utf-8 -*-
r"""
Compute the proposed Blackwell redundancy measure, $I_\cap^\prec$, from
 A Kolchinsky, A Novel Approach to the Partial Information Decomposition, Entropy, 2022.

Given a joint distribution p_{Y,X_1,X_2}, this redundancy measure is 
the solution to the following optimization problem:

R = min_{s_{Q|Y}} I_s(Y;Q) 
             s.t. s_{Q|Y} ⪯ p_{X_i|Y} ∀i

This can in turn be re-written as:
 R = min_{s_{Q|Y},s_{Q|X_1}, .., s_{Q|X_n}} I_s(Y;Q)
     s.t. ∀i,q,y : Σ_{x_i} s(q|x_i) p(x_i,y) = s(q|y)p(y)

Note that this is optimization problem involes a maximization of 
a convex function subject to a system of linear constraints.  This 
system of linear constraints defines a convex polytope, and the maximum 
of the function will lie at one of the vertices of the polytope.  We 
proceed by finding these vertices, iterating over them, and finding 
the vertex with the largest value of I(Y;Q).
"""

import numpy as np



def get_blackwell_redundancy(pY, conditionals, nQ=None):
    nY = len(pY)
    nS = len(conditionals)
    
    if not (nQ is None or (isinstance(nQ, int) and nQ >= 1)):
        raise Exception('Parameter nQ should be None or positive integer')

    if nY <= 1 or nQ == 1:
        # Trivial case where target or redundancy has a single outcome, redundancy has to be 0
        return 0, {}
    
    # variablesQgiven holds ppl variables that represent conditional probability
    # values of s(Q=q|X_i=x_i) for different sources i, as well as s(Q=q|Y=y)
    # for the target Y (recall that Q is our redundancy random variable)
    variablesQgiven = {} 

    var_ix        = 0  # counter for tracking how many variables we've created
    

    if nQ is None:
        # Calculate the maximum number of outcomes we will require for Q, per Theorem A1
        nQ = sum([condp.shape[0]-1 for condp in conditionals]) + 1
                             
    # Iterate over all the random variables (R.V.s): i.e., all the sources + the target 
    for rvndx in range(nS+1):
        variablesQgiven[rvndx] = {}

        # get marginals
        mP = conditionals[rvndx]@pY if rvndx < nS else pY
            
        if any(mP == 0):
            raise Exception('All marginals should have full support ' +
                            '(to proceed, drop outcomes with 0 probability)')

        # Iterate over outcomes of current R.V.
        for v in range(len(mP)):
            sum_to_one = 0 
            for q in range(nQ):
                # represents s(Q=q|X_rvndx=v) if rvndx from 0..nS-1
                #        and s(Q=q|Y=v)       otherwise
                variablesQgiven[rvndx][(q, v)] = var_ix 
                var_ix += 1
    
    num_vars = var_ix 

    A_eq  , b_eq   = [], []  # linear constraints Ax =b
    A_ineq, b_ineq = [], []  # linear constraints Ax<=b


    for rvndx in range(nS+1):
        # get marginal for random variable rvndx
        mP = conditionals[rvndx]@pY if rvndx < nS else pY
            
        for v in range(len(mP)):  # iterate over outcomes of this random variable
            sum_to_one = np.zeros(num_vars)
            for q in range(nQ):
                var_ix = variablesQgiven[rvndx][(q, v)]

                # Non-negative constraint on each variable
                z = np.zeros(num_vars)
                z[var_ix] = -1
                A_ineq.append(z) 
                b_ineq.append(0)

                sum_to_one[var_ix] = 1

            if rvndx < nS:
                # Linear constraint that enforces Σ_q s(Q=q|X_rvndx=v) = 1
                A_eq.append(sum_to_one)
                b_eq.append(1)

    # Now we add the constraint:
    #    ∀i,q,y : Σ_{x_i} s(q|x_i) p(x_i,y) = s(q|y)p(y)
    for rvndx, condp in enumerate(conditionals):
        # Compute joint marginal of target Y and source X_rvndx
        joint_dist = condp*pY[None,:]
        for q in range(nQ):
            for y in range(nY):
                z = np.zeros(num_vars)
                cur_mult   = 0. 
                for x in range(joint_dist.shape[0]):
                    pXY         = joint_dist[x,y]
                    cur_mult   += pXY
                    z[variablesQgiven[rvndx][(q, x)]] = pXY 
                z[variablesQgiven[nS][(q, y)]] = -cur_mult
                A_eq.append(z)
                b_eq.append(0)

    # Define a matrix sol_mx that allows for fast mapping from solution vector 
    # returned by ppl (i.e., a particular extreme point of our polytope) to a 
    # joint distribution over Q and Y
    mul_mx = np.zeros((num_vars, nQ*nY))
    for (q,y), k in variablesQgiven[nS].items():
        mul_mx[k, q*nY + y] += pY[y]

    def entr(x):
        x = x + 1e-18
        return -x*np.log2(x)

    H_Y = entr(pY).sum()

    def objective(x):
        # Map solution vector x to joint distribution over Q and Y
        pQY     = x.dot(mul_mx).reshape((nQ,nY))
        if np.any(pQY<-1e-6):
            raise Exception("Invalid probability values")
        pQY[pQY<0] = 0
        probs_q = pQY.sum(axis=1) + 1e-18
        H_YgQ   = entr(pQY/probs_q[:,None]).sum(axis=1).dot(probs_q)
        v       =  H_Y - H_YgQ
        if 0>v>-1e-6: 
            v   = 0  # round to zero if it is negative due to numerical issues
        return v

    # The following uses ppl to turn our system of linear inequalities into a 
    # set of extreme points of the corresponding polytope. It then calls 
    # get_solution_val on each extreme point
    x_opt, v_opt = maximize_convex_function(
        f=objective,
        A_eq=np.array(A_eq), 
        b_eq=np.array(b_eq), 
        A_ineq=np.array(A_ineq), 
        b_ineq=np.array(b_ineq))

    sol = {}
    sol['p(Q,Y)'] = x_opt.dot(mul_mx).reshape((nQ,nY))

    # Compute conditional distributions of Q given each source X_i
    for rvndx, condp in enumerate(conditionals):
        pX = condp@pY
        cK = 'p(Q|X%d)'%rvndx
        sol[cK] = np.zeros( (nQ, len(pX) ) )
        for (q,v_ix), k in variablesQgiven[rvndx].items():
            sol[cK][q,v_ix] = x_opt[k]

    # Return mutual information I(Q;Y) and solution information
    return v_opt, sol





# ***********************************************************************
# ***********************************************************************
"""
Code for maximizing a convex function over a polytope, as defined
by a set of linear equalities and inequalities.

This uses the fact that the maximum of a convex function over a
polytope will be achieved at one of the extreme points of the polytope.

Thus, the maximization is done by taking a system of linear inequalities,
using the pypoman library to create a list of extreme
points, and then evaluating the objective function on each point.

We use various techniques to first prune away redundant constraints.
"""

import numpy as np
import scipy

try:
    import pypoman
except ImportError:
    print("Error: pypoman package not found. Please install pypoman in order to compute Blackwell redundancy")

import scipy.spatial.distance as sd

def remove_duplicates(A,b):
    # Removes duplicate rows from system (in)equalities given by A and b
    while True:
        N = len(A)
        mx = np.hstack([b[:,None], A])
        dists = sd.squareform(sd.pdist(mx))
        duplicates_found = False
        for ndx1 in range(N):
            A1, b1 = A[ndx1,:], b[ndx1]
            keep_rows = np.ones(N, bool)
            keep_rows[ndx1+1:] = dists[ndx1,ndx1+1:]>1e-08
                    
            if not np.all(keep_rows):
                duplicates_found = True
                A = A[keep_rows,:]
                b = b[keep_rows]
                break

        if not duplicates_found:
            break 

    return A, b


def eliminate_redundant_constraints(A,b):
    # Eliminate redundant constraints from the inequality system Ax <= b
    
    init_num_cons = A.shape[0] # initial number of constraints
    
    N = A.shape[1]             # number of variables
    bounds = [(None,None),]*N
    
    keep_rows         = list(range(A.shape[0]))
    nonredundant_rows = set([])
    
    while True:
        eliminated = False
        for i in keep_rows:  # test whether row i is redundant
            
            if i in nonredundant_rows:  # already tested this row
                continue
                
            # Current row specifies the constraint b >= a^T x
            # Let A' and b' indicate the constraints in all the other rows 
            # If b >= max_x a^T x  such that b' >= A'x, then this constraint is redundant and can be eliminated

            other_rows = [j for j in keep_rows if j != i]
            
            A_other, b_other = A[other_rows], b[other_rows]
            
            c = scipy.optimize.linprog(-A[i], A_ub=A_other, b_ub=b_other, bounds=bounds)


            if c.success and c.status == 0 and -c.fun <= b[i] + 1e-15:  # solver succeeded and this row is redundant
                keep_rows = other_rows
                eliminated = True
                break
            
            else:
                # row is not redundant
                nonredundant_rows.add(i)

        if not eliminated:
            break
            
    A, b = A[keep_rows], b[keep_rows]
    return A, b

def maximize_convex_function(f, A_ineq, b_ineq, A_eq=None, b_eq=None):
    """
    Maximize a convex function over a polytope.

    Parameters
    ----------
    f : function
        Objective function to maximize
    A_ineq : matrix
        Specifies inequalities matrix, should be num_inequalities x num_variables
    b_ineq : array
        Specifies inequalities vector, should be num_inequalities long
    A_eq : matrix
        Specifies equalities matrix, should be num_equalities x num_variables
    b_eq : array
        Specifies equalities vector, should be num_equalities long

    Returns tuple optimal_extreme_point, maximum_function_value

    """

    best_x, best_val = None, -np.inf
    
    A_ineq = A_ineq.astype('float')
    b_ineq = b_ineq.astype('float')
    A_ineq, b_ineq = remove_duplicates(A_ineq, b_ineq)

    if A_eq is not None:
        # pypoman doesn't support equality constraints. We remove equality 
        # constraints by doing a coordinate transformation.

        A_eq = A_eq.astype('float')
        b_eq = b_eq.astype('float')
        A_eq, b_eq = remove_duplicates(A_eq, b_eq)

        # Get one solution that satisfies A x0 = b
        x0 = scipy.linalg.lstsq(A_eq, b_eq)[0]
        assert(np.abs(A_eq.dot(x0) - b_eq).max() < 1e-5)

        # Get projector onto null space of A, it satisfies AZ=0 and Z^T Z=I
        Z = scipy.linalg.null_space(A_eq)
        # Now every solution can be written as x = x0 + Zq, since A x = A x0 = b 

        # Inequalities get transformed as
        #   A'x <= b'  to   A'(x0 + Zq) <= b  to  (A'Z)q <= b - A'x0

        b_ineq = b_ineq - A_ineq.dot(x0)
        A_ineq = A_ineq.dot(Z)

        A_ineq, b_ineq = remove_duplicates(A_ineq, b_ineq)
        
        transform = lambda q: Z.dot(q) + x0

    else:
        transform = lambda x: x

    A_ineq, b_ineq = eliminate_redundant_constraints(A_ineq, b_ineq)
    
    extreme_points = pypoman.compute_polytope_vertices(A_ineq, b_ineq)
    
    for v in extreme_points:
        x = transform(v)
        val = f(x)
        if val > best_val:
            best_x, best_val = x, val

    if best_x is None:
        raise Exception('No extreme points found!')

    return best_x, best_val



