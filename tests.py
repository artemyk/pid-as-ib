import unittest
import numpy as np
import redundancy_bottleneck as rb

class TestRB(unittest.TestCase):

    src_conditionals1 = (np.array([[0.95, 0.25],
                                  [0.05, 0.75],]),
                        np.array([[0.8 , 0.4],
                                  [0.2 , 0.6]]))
    src_conditionals2 = (np.array([[1, 1],
                              [0, 0],]),
                    np.array([[1 , 0],
                              [0 , 1]]))
    src_conditionals3 = (np.array([[1, 0],
                              [1, 0],]),
                    np.array([[1 , .5],
                              [0 , .5]]))

    def test_zeroprob_y(self):
        pY1 = np.array([0.6, 0.4])
        r1 = rb.get_rb_value(beta=10, pY=pY1, src_cond_dists=self.src_conditionals1, num_retries=10)

        # Zero weight target outcome
        pY2 = np.array([0.6, 0.4,0.])
        src_conditionals2 = (np.array([[0.95, 0.25,0],
                                      [0.05, 0.75,1],]),
                            np.array([[0.8 , 0.4,0],
                                      [0.2 , 0.6 ,1]]))
        r2 = rb.get_rb_value(beta=10, pY=pY2, src_cond_dists=src_conditionals2, num_retries=10)

        self.assertAlmostEqual(r1.prediction , r2.prediction , places=2)
        self.assertAlmostEqual(r1.compression, r2.compression, places=2)

    def test_zero_y(self):
        pY = np.array([0.0, 1.0])
        r = rb.get_rb_value(beta=10, pY=pY, src_cond_dists=self.src_conditionals1)
        self.assertAlmostEqual(r.prediction ,0,2)
        self.assertAlmostEqual(r.compression,0,2)

    def test_AND_gate(self):
        pY = np.array([0.75, 0.25])
        srcC= np.array([[2/3, 0], [1/3, 1]])
        r = rb.get_rb_value(beta=100, pY=pY, src_cond_dists=(srcC,srcC), num_retries=10)
        self.assertAlmostEqual(r.prediction,0.311,2)

    def test_int_y_prob(self):
        pY = np.array([0, 1])
        r = rb.get_rb_value(beta=10, pY=pY, src_cond_dists=self.src_conditionals1)
        self.assertAlmostEqual(r.prediction ,0,2)
        self.assertAlmostEqual(r.compression,0,2)

    def test_int_src(self):
        pY = np.array([0, 1])
        r = rb.get_rb_value(beta=10, pY=np.array([0.75, 0.25]), src_cond_dists=self.src_conditionals2)
        r = rb.get_rb_value(beta=10, pY=np.array([0, 1]), src_cond_dists=self.src_conditionals2)

if __name__ == '__main__': 
    unittest.main() 
