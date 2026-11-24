"""Independent stationary Dirac/KG scattering experiments."""
import os

# Set before importing numpy/scipy; campaigns run serially for comparable timings.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

