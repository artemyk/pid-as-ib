# Partial information decomposition as information bottleneck

Python code to implement Redundancy Bottleneck (RB) algorithm, described in:
  * A Kolchinsky, "Partial information decomposition as information bottleneck", 2024.


In order to speed up computation of RB, it is strongly recommended to install [`numba`](https://numba.pydata.org/), e.g., via
```
pip3 install numba
```
However, the code should work without it also.

Includes:
* `redundancy_bottleneck.py`: contains the main function for computing RB, `get_rb_value`
* `simpledemo.py`: simple demonstration of how to call the RB function on a simple two-source system
* `PaperFigures.ipynb`: Jupyter notebook that generate figures from the manuscript
* `blackwell_redundancy.py`: code to compute our previous measure of "Blackwell redundancy" [link](https://www.mdpi.com/1099-4300/24/3/403). Please note this code requires [`pypoman`](https://github.com/stephane-caron/pypoman) library to be installed.
 
