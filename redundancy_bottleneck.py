# -*- coding: utf-8 -*-

"""
Code to implement Redundancy Bottleneck algorithm, from
  * A Kolchinsky, "Partial information decomposition as information bottleneck", 2024.
"""

import numpy as np
import warnings
import collections
Result = collections.namedtuple('Result', [
                'beta',
                'objective',
                'prediction', 
                'compression', 
                'src_prediction',
                'src_compression',
                'zs_ids',
                'rQgZS',
                'pY_ZS',
                'pZ_SY',
                'pS_ZY',
                ])

IS_CLOSE_EPS = 1e-15

def check_p(x):      # check to make sure x is a valid distribution (nonnegative and sums to 1)
    assert(-IS_CLOSE_EPS < np.sum(x)-1 < IS_CLOSE_EPS)
    assert(np.all(x>=0))


def check_condp(x):  # check to make sure x is a valid conditional distribution
    assert(x.ndim == 2)
    xsums = np.sum(x,axis=0)
    assert(np.all(xsums - 1 <  IS_CLOSE_EPS))
    assert(np.all(xsums - 1 > -IS_CLOSE_EPS))
    assert(np.all(x>=0))
    

def entropy(p):        # Shannon entropy of distribution p, in bits
    return sum([ -v*np.log2(v) for v in p.flat if v > IS_CLOSE_EPS] )


def mi(p):             # Mutual information for joint distribution p[x,y], in bits
    return entropy(p.sum(axis=0)) + entropy(p.sum(axis=1)) - entropy(p)

def cond_entropy(p):   # Conditional entropy p[x|y], in bits
    return entropy(p) - entropy(p.sum(axis=0))

def kl(p,q):    #  Compute KL divergence of distribution p, in bits
    return sum([ v*np.log2(v/q.flat[ix]) for ix, v in enumerate(p.flat) if v > IS_CLOSE_EPS] )



def conditionalize(joint): 
    """ Convert joint distribution p[x,y] into the conditional distribution p(x|y)=p[x,y]/p[y].
        For y where p[y] = 0, we arbitrarily set p(x|y)=1 for x=1, so as to have a 
        well-defined conditional distribution
    """
    marginal  = joint.sum(axis=0)
    ixs       = marginal != 0
    p         = joint.copy()
    p[:,ixs]  = p[:,ixs]/marginal[ixs]
    p[0,~ixs] = 1
    return p




def get_rb_value(
          beta,                    # Parameter that controls trade-off between prediction and compression            
          pY,                      # 1D np.array, distribution over outcomes of target Y
          src_cond_dists,          # list of 2D np.array, conditional distribution X_i|Y for each source
          pS=None,                 # 1D np.array specifying distribution p(s) over sources. If none, using uniform
          nQ=None,                 # Cardinality of bottleneck random variable.
                                   #   If None, we will use nQ = nZS + 1 (as described in the article)
          num_retries    = 1,      # Number of times to rerun optimization (best outcome will be picked)
          baseline_rQgZSs = None,  # List of baseline r(Q|Z,S) to try as initializations
          max_iters      = 1000,   # Maximum iterations for convergence
          tol            = 1e-6,   # Stopping criterion for convergence
          verbose        = False,  # Print debugging information
          squared_langragian=False,
          pp=2,
          ):
    """
    Solve the redundancy bottleneck (RB) Lagrangian problem,
        max I(Q;Y) - beta * I(Q;S|Y)

    We use the alternating projections algorithm described in the paper. 

    Returns a namedtuple, containing information about best Q found:
      prediction  I(Q;Y|S) in bits,
      compression I(Q;S|Y) in bits,
      optimal map Pr(q|s,z),
      prediction values for each source,
      etc.
    """


    assert(tol  > 0)
    assert(beta > 0)


    check_p(pY)
    nY    = len(pY)
    H_Y   = entropy(pY)
    ylist = np.flatnonzero(pY)

    for p in src_cond_dists:
        assert( p.shape[1] == nY)
        check_condp(p)
    nS = len(src_cond_dists)

    cur_pS = np.ones(nS)/nS
    if pS is not None:
        cur_pS = np.array(pS)
        check_p(cur_pS)
        assert(cur_pS.ndim == 1 and len(cur_pS)==nS)
        assert(np.all(cur_pS > 0))  # Source distribution should have full support


    # Create some handy indexing dictionaries/functions
    zs_ids = []
    for src, p in enumerate(src_cond_dists):
        for z in np.flatnonzero( p@pY):  # Add non-zero probability outcomes to list of ids
            zs_ids.append( (z,src) )
    nZS    = len(zs_ids)
    nZ     = max([p.shape[0] for p in src_cond_dists])

    if nQ is None:
        nQ = nZS+1
    nQ = int(nQ)
    nYQ = nY*nQ 
    
    pZ_SY = np.zeros( (nZ, nY*nS) )
    pY_ZS = np.zeros( (nY, nZS  ) )
    pS_ZY = np.zeros( (nS, nY*nZ) )
    for y in ylist:
        for zs, (z, src) in enumerate(zs_ids):
            sy = src*nY + y
            zy = z  *nY + y
            pY_ZS[y,zs] = pZ_SY[z,sy] = pS_ZY[src,zy] = cur_pS[src]*pY[y]*src_cond_dists[src][z,y]
            
            
    pYgSZ  = conditionalize(pY_ZS)
    pZgSY  = conditionalize(pZ_SY)

    check_condp(pYgSZ)
    check_condp(pZgSY)
    

    # *******************************
    # We define some useful functions    
    # *******************************
    def get_dists(rQgZS):
        pQ_Y   = np.zeros((nQ, nY))
        pQ_S   = np.zeros((nQ, nS))
        pY_QS  = np.zeros((nY, nS*nQ))
        pQ_SY  = np.zeros((nQ, nS*nY))
        pZ_SYQ = np.zeros((nZ, nS*nYQ))
        for zs, (z, src) in enumerate(zs_ids):
            for q in range(nQ):
                qs = q*nS+src
                for y in ylist:
                    sy  = src*nY + y
                    syq = src*nYQ+y*nQ+q

                    j = rQgZS[q,zs] * pY_ZS[y,zs]
                    pQ_Y[q,y]     += j
                    pQ_S[q,src]   += j
                    pY_QS[y,qs]   += j
                    pQ_SY[q,sy]   += j
                    pZ_SYQ[z,syq] += j

        miQ_YgS = H_Y - cond_entropy(pY_QS)
        miQ_SgY = cond_entropy(pQ_Y) - cond_entropy(pQ_SY)

        if squared_langragian:
            if pp==-1:
                obj = miQ_YgS-beta*np.exp(miQ_SgY)
            else:
                obj = miQ_YgS-beta*(miQ_SgY+1)**pp
        else:
            obj = miQ_YgS-beta*miQ_SgY

        return obj, miQ_YgS, miQ_SgY, pQ_Y, pQ_S, pY_QS, pQ_SY, pZ_SYQ

    def do_run(rQgZS):
        check_condp(rQgZS)
        
        cur_obj, prev_obj = -np.inf, -np.inf
        iter_num = 0

        while True:
            cur_obj, miQ_YgS, miQ_SgY, pQ_Y, pQ_S, pY_QS, pQ_SY, pZ_SYQ = get_dists(rQgZS)
            
            if verbose:
                print("iter/obj/miQ_YgS/miQ_SgY:", iter_num, cur_obj, miQ_YgS, miQ_SgY)
        
            # Do checks
            if np.isinf(cur_obj) or np.isnan(cur_obj):
                print("WARNING! Invalid objective found, stopping this run")
                break

            if prev_obj >= cur_obj + 1e-2: # objective shouldn't decrease too much
                print("WARNING! Objective decreased (", prev_obj, " ->", cur_obj, "), stopping this run")
                break

            if -tol < cur_obj-prev_obj < tol:
                if verbose: print("Converged on iteration", iter_num)
                break

            if iter_num >= max_iters:
                if verbose: print("Warning: did not converge after " + str(max_iters) + " iterations")
                break

            wQ_Y     = pQ_Y
            wYgQS    = conditionalize(pY_QS)
            wZgSYQ   = conditionalize(pZ_SYQ)
    
            ln_rQgZS  = np.zeros((nQ, nZS))    # stores log r(q|z,s)
            for zs, (z, src) in enumerate(zs_ids):
                for q in range(nQ):
                    a, b = 0, 0
                    qs = q*nS+src
                    for y in ylist:
                        sy  = src*nY + y
                        syq = src*nYQ+y*nQ+q

                        p  = pYgSZ[y,zs]
                        if p < IS_CLOSE_EPS:  # too close to zero
                            continue
        
                        v1 = wYgQS[y,qs]
                        v2 = wQ_Y[q,y]*wZgSYQ[z,syq]

                        a += p*np.log(v1) if v1 > 0 else -np.inf 
                        b += p*np.log(pY[y]*pZgSY[z,sy])
                        b -= p*np.log(v2) if v2 > 0 else -np.inf
                        
                    if squared_langragian:
                        if pp == -1:
                            ln_rQgZS[q,zs] = a/(np.exp(miQ_SgY)*beta) - b
                        else:
                            ln_rQgZS[q,zs] = a/(pp*(miQ_SgY+1)**(pp-1)*beta) - b
                    else:
                        ln_rQgZS[q,zs] = a/beta - b

                # Rescale to avoid numerical overflows in np.exp. This doesn't affect the 
                # results since its cancelled by conditionalize
                ln_rQgZS[:,zs] -= np.max(ln_rQgZS[:,zs])

            rQgZS = conditionalize(np.exp(ln_rQgZS))

            prev_obj  = cur_obj
            iter_num += 1


        return cur_obj, rQgZS


    # ******************************



    best_obj, best_rQgZS = -np.inf, None
    if baseline_rQgZSs is not None:
        for i, init_rQgZS in enumerate(baseline_rQgZSs):
            if verbose: print("* Trying with baseline_rQgZS initialization #", i)
            obj, rQgZS = do_run(init_rQgZS)
            if best_obj < obj:
                best_obj, best_rQgZS = obj, rQgZS

    if nQ >= nZS:
        if verbose: print("* Trying with identity rQgZS initialization")
        init_rQgZS = np.zeros( (nQ, nZS) )
        pertubedEye = np.eye(nZS) + np.random.random((nZS,nZS))*1e-2
        init_rQgZS[:nZS,:nZS] = conditionalize(pertubedEye)
        obj, rQgZS = do_run(init_rQgZS)
        if best_obj < obj:
            best_obj, best_rQgZS = obj, rQgZS

    for try_ix in range(num_retries):
        if verbose: print("* Doing random initialization, try", try_ix)
        init_rQgZS = np.random.random((nQ,nZS))
        init_rQgZS = conditionalize(init_rQgZS)
        obj, rQgZS = do_run(init_rQgZS)
        if best_obj < obj:
            best_obj, best_rQgZS = obj, rQgZS
        



    # Compute some additional quantities of interest for the best run
    cur_obj, miQ_YgS, miQ_SgY, pQ_Y, pQ_S, pY_QS, pQ_SY, pZ_SYQ = get_dists(best_rQgZS)
    pQgSY = conditionalize(pQ_SY)
    pQgY  = conditionalize(pQ_Y)
    pQgS  = conditionalize(pQ_S)

    # Compute contributions to prediction from the different sources
    src_pred = np.zeros(nS)  # Specific cond. mutual information terms, I(Q;Y|S=s)
    src_comp = np.zeros(nS)  # Specific cond. mutual information terms, I(Q;S=s|Y)
    for src in range(nS):
        for y in ylist:
            sy = src*nY + y
            for q in range(nQ):
                condp          = pQ_SY[q,sy]/cur_pS[src]
                if condp       < IS_CLOSE_EPS: continue
                src_pred[src] += condp*np.log2(pQgSY[q,sy]/pQgS[q,src])
                src_comp[src] += condp*np.log2(pQgSY[q,sy]/pQgY[q,y])


    # Return information about optimal map found
    return Result(beta            = beta,               # beta value of the RB Lagrangian
                  objective       = cur_obj,            # Objective value of the RB Lagrangian
                  prediction      = miQ_YgS,            # Overall prediction value
                  compression     = miQ_SgY,            # Overall compression value
                  src_prediction  = src_pred,           # Prediction broken down by source
                  src_compression = src_comp,           # Compression broken down by source
                  rQgZS           = best_rQgZS,         # Optimal map Pr(Q|Z,S)
                  zs_ids          = zs_ids,             # List of indexes of (z,s)
                  pY_ZS           = pY_ZS,              # Joint distribution P(y,(z,s))
                  pZ_SY           = pZ_SY,              # Joint distribution P(z,(s,y))
                  pS_ZY           = pS_ZY,              # Joint distribution P(s,(z,y))
                  )




# try:
#     from numba import jit_module
#     jit_module(nopython=True, error_model="numpy")

# except ModuleNotFoundError:
#     warnings.warn("Please install numba package for faster performance!")



