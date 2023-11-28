import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh

# ---- FINAL VERSION OF REQUIRED BASE FUNCTIONS (for simulations from the paper) ----------------#


#---------------------------------------------------------------------------------------------------#
#--                          Generate data for simulations                                        --#
#---------------------------------------------------------------------------------------------------#

def genBSplineBasis(D, n , degree = 4):
    num_control_points = degree + 1   # Define the number of control points (degree + 1)
    knots = np.linspace(-4, 4, D - num_control_points)
    basis_functions = []  # to store the basis funcs
    values_at_points = []  # Initialize an empty list to store the values of each basis function at 100 points
    evaluation_points = np.linspace(-4, 4, 100)        # Evaluate the B-spline basis functions at 100 points between -4 and 4
    for i in range(D):
        control_points = np.zeros(D)
        control_points[i] = 1.0
        basis_function = BSpline(knots, control_points, degree)
        values = [basis_function(x) for x in evaluation_points]                 # Evaluate the basis function at 100 points
        basis_functions.append(basis_function)
        values_at_points.append(values)
    spline_vals = np.array(values_at_points)        # spline values x - axis locations
    return  spline_vals   # evaluation_points, needed only for plotting basis functions


def matrixN_i(spline_vals, D, n):
    #np.random.seed(seed)
    row_indices = np.arange(1, D + 1)       
    variances = 1 / row_indices**2  #+ 0.25
    N_i = np.random.normal(0, variances[:, np.newaxis], size=(D, n))   # random normal
    clipped_N_i = np.clip(N_i, -4, 4)
    epsilon_i = np.dot(clipped_N_i.T, spline_vals)
    return epsilon_i

def mat_Psi(n, kappa):
    row_indices = np.arange(1, n + 1)   
    sigmas =   1/row_indices #+2
    sigma_mat = np.outer(sigmas, sigmas)
    #np.random.seed(seed)
    Psi = np.random.normal(0, sigma_mat)     # random normal
    Psi = Psi / np.sqrt(np.max(eigh(Psi @ Psi.T)[0]))  #extracts the maximum eigenvalue
    theta = kappa*Psi
    return theta

def error_process(epsilon_i, theta):
    epsilon_im1 = epsilon_i[:-1, :]
    init_row = np.zeros((1, epsilon_im1.shape[1]))
    epsilon_im1 = np.vstack((init_row, epsilon_im1))
    eta_i = epsilon_i + np.matmul(theta, epsilon_im1)
    return eta_i

def fMA1(n, D =20, kappa =0.9):
    spline_values = genBSplineBasis(D, n)
    epsiln_i = matrixN_i(spline_values, D, n)
    theta = mat_Psi(n, kappa)
    eta_i = error_process(epsiln_i, theta) 
    return eta_i





#####################---- end of fMA1 process ------------#############
