import numpy as np
import redundancy_bottleneck as rb


pY1 = np.array([0.6, 0.4])
src_conditionals1 = [np.array([[0.95, 0.25],
                              [0.05, 0.75],]),
                    np.array([[0.8 , 0.4],
                              [0.2 , 0.6]])]
r1 = rb.get_rb_value(beta=10, pY=pY1, src_cond_dists=src_conditionals1, num_retries=10)

# Zero weight target outcome
pY2 = np.array([0.6, 0.4,0.])
src_conditionals2 = [np.array([[0.95, 0.25,0],
                              [0.05, 0.75,1],]),
                    np.array([[0.8 , 0.4,0],
                              [0.2 , 0.6 ,1]])]
r2 = rb.get_rb_value(beta=10, pY=pY2, src_cond_dists=src_conditionals2, num_retries=10)

assert(np.abs(r1.prediction  - r2.prediction)<1e-2)
assert(np.abs(r1.compression - r2.compression)<1e-2)

pY3 = np.array([0.0, 1.0])
r3 = rb.get_rb_value(beta=10, pY=pY3, src_cond_dists=src_conditionals1)
assert(np.abs(r3.prediction)<1e-2)
assert(np.abs(r3.compression)<1e-2)



# AND gate
pY4 = np.array([0.75, 0.25,])
srcC= np.array([[2/3, 0], [1/3, 1]])
r4 = rb.get_rb_value(beta=100, pY=pY4, src_cond_dists=[srcC,srcC], num_retries=10)
assert(np.abs(r4.prediction-0.311)<1e-2)