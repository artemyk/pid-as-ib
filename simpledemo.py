import numpy as np
import redundancy_bottleneck as rb

# Compute redundancy bottleneck for simple two-source system
pY = np.array([0.6, 0.4])    # Distribution of target p(y)
src_conditionals = (np.array([[0.95, 0.25],
                              [0.05, 0.75]]),
                    np.array([[0.8 , 0.4],
                              [0.2 , 0.6]]))

beta = 100
r = rb.get_rb_value(beta=beta, pY=pY, src_cond_dists=src_conditionals, num_retries=10)
print("Redundancy bottleneck at beta=%g: prediction=%9.6f, compression=%9.6f" % (beta, r.prediction, r.compression))

# Compare to Blackwell redundancy, if necessary packages are installed
try:
    import pypoman  # computational geometric package
    import blackwell_redundancy
    r2 = blackwell_redundancy.get_blackwell_redundancy(pY, src_conditionals)
    print("Blackwell redundancy             : redundancy=%9.6f" % r2[0])
except ModuleNotFoundError:
    print("If you wish to compute Blackwell redundancy, please install pypoman package")
