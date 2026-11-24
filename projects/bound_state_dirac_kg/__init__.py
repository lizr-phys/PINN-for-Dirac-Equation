"""Independent stationary vector-coupled Dirac and Klein--Gordon experiment."""
import os

# Set before importing NumPy/SciPy in the command-line entry points.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
