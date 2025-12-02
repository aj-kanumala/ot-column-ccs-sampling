# Impoer necessary libraries
import numpy as np
import pandas as pd
import time
import ot


# Compute Wasserstein distance between two point clouds
def compute_wasserstein_distance(pc1, pc2, reg=1):
    """
    Takes in two pointclouds pc1 and pc2, i.e., pc1 = [x1,y1,w1;...;xp,yp,wp], etc. and computes the Wasserstein-2 distance
    between pc1 and pc2.
    """
    # Reshaping and stripping out (x,y) locations (Upts, Vpts) and weights (Uwts, Vwts)
    pc1pts = np.ascontiguousarray(pc1[:,0:2])
    pc2pts = np.ascontiguousarray(pc2[:,0:2])
    pc1wts = np.ascontiguousarray(pc1[:,2])
    pc2wts = np.ascontiguousarray(pc2[:,2])
    pc1wts = pc1wts/np.sum(pc1wts) # normalize to make a probability
    pc2wts = pc2wts/np.sum(pc2wts) # normalize to make a probability  
    M = ot.dist(pc1pts, pc2pts, metric = 'sqeuclidean')  # Compute squared euclidean distance on the pointcloud points
    W = ot.sinkhorn2(pc1wts,pc2wts, M, reg=reg) # Compute exact squared Wasserstein-2 distance between U and V  
    return W

def Wasserstein_Matrix(image_list, reg=1, squared=True):
    """
    The function compute the (squared if squared=True) Wasserstein Distance Matrix between N images
    image_list: python list of pointcloud representations
    """
    N = len(image_list) #number of images
    distance = np.zeros((N,N))  # initialize the distance matrix
    tic = time.perf_counter()
    for i in range(N):
        for j in range(i+1,N):
            if squared==True:
                distance[i,j] = compute_wasserstein_distance(image_list[i], image_list[j], reg=reg)
                # print(f"Row:{i} | Column:{j} | Distance:{distance[i,j]:.2f}")  #------Debug 
            else:
                distance[i,j] = compute_wasserstein_distance(image_list[i], image_list[j], reg=reg)**.5
    distance += distance.T  
    toc = time.perf_counter()
    total_time = (toc - tic)/60
    print(f"Computed {distance.shape}-distance matrix in {total_time:.2f} mins")
    return distance

# ---------- Helper function ---------- #
def set_default_params_CCS(params):
    """Set default parameters for CCS if missing."""
    defaults = {'p': 0.2, 'delta': 0.3}
    for k, v in defaults.items():
        params.setdefault(k, v)
    return params


# ---------- Existing CCS function ---------- #
def CCS(X, params_CCS, rng=None):
    """Cross-Concentrated Sampling on pre-computed full distance matrix (generic version)."""
    params_CCS = set_default_params_CCS(params_CCS)
    rng = np.random.default_rng(rng)
    p = params_CCS['p']
    delta = params_CCS['delta']
    m, n = X.shape
    num_c = round(n * delta)
    J_ccs = rng.choice(n, num_c, replace=False)
    C = X[:, J_ccs]
    ubc = min(num_c * m, int(np.ceil(p * num_c * m)))
    C_obs_ind = rng.choice(num_c * m, ubc, replace=False)
    C_Obs = np.zeros((m, num_c), dtype=X.dtype)
    C_Obs.flat[C_obs_ind] = C.flat[C_obs_ind]
    X_Omega_ccs = np.zeros((m, n), dtype=X.dtype)
    X_Omega_ccs[:, J_ccs] = C_Obs
    selected_indices = np.argwhere(X_Omega_ccs != 0)
    selected_indices = np.unique(selected_indices, axis=0)
    return X_Omega_ccs, J_ccs, selected_indices, C_obs_ind



# --------- (New) Column-CCS based partial distance computation --------------- #
def Wass_Matrix_CCS_Col(image_list, params_CCS, squared=True, rng=None, reg=1):
    """
    Compute a sparse Wasserstein Distance Matrix using CCS sampling, only from selected columns.   
    Parameters:
    - image_list: List of point clouds ([x, y, w] arrays).
    - p: Sampling probability within selected columns (default: 0.3).
    - delta: Fraction of columns to sample (default: 0.2).
    - squared: If True, compute squared Wasserstein distance; else, take square root.
    Returns:
    - distance: Sparse NxN NumPy array with Wasserstein distances.
    - J_ccs: Sampled column indices.
    - sampled_positions: Nx3 NumPy array of [row_idx, col_idx, distance] for sampled positions.
    """
    params_CCS = set_default_params_CCS(params_CCS)
    rng = np.random.default_rng(rng)
    p = params_CCS['p']
    delta = params_CCS['delta']
    N = len(image_list)
    distance = np.zeros((N, N))
    num_c = round(N * delta)
    J_ccs = rng.choice(N, num_c, replace=False)
    ubc = min(num_c * N, int(np.ceil(p * num_c * N)))
    C_obs_ind = rng.choice(num_c * N, ubc, replace=False)
    print(f"Sampling {ubc} column entries out of selected {num_c} columns")

    # Lists to store sampled positions and distances
    row_indices = []
    col_indices = []
    distances = []
    tic = time.perf_counter()

    for idx in C_obs_ind:
        i = idx // num_c
        j = idx % num_c
        j_full = J_ccs[j]
        if i != j_full:
            dist = compute_wasserstein_distance(image_list[i], image_list[j_full], reg=reg)
            computed_dist = dist if squared else np.sqrt(dist)
            distance[i, j_full] = computed_dist
            # print(f"Row:{i} | Column:{j_full} | Distance:{computed_dist:.2f}")  # -----Debug  
            row_indices.append(i)
            col_indices.append(j_full)
            distances.append(computed_dist)
        else:
            distance[i, j_full] = 0.0
            distance[j_full, i] = 0.0
    
    toc = time.perf_counter()
    total_time = (toc - tic)/60
    computed_entries = np.count_nonzero(distance)
    print(f"Computed {computed_entries}/{N*N} distances ({computed_entries/(N*N)*100:.2f}%) in {total_time:.2f} mins")
    # Combine indices and distances into Nx3 array
    sampled_positions = np.column_stack((row_indices, col_indices, distances))

    return distance, J_ccs, sampled_positions


# ------------ Helper function for ICURC ----------- # 
def set_default_params_ICURC(s):
    if 'TOL' not in s:
        s['TOL'] = 1e-4
    if 'max_ite' not in s:
        s['max_ite'] = 500
    if 'eta' not in s:
        s['eta'] = [1, 1, 1]
        s['steps_are1'] = True
    elif s['eta'] == [1, 1, 1]:
        s['steps_are1'] = True
    else:
        s['steps_are1'] = False
    return s


# ------------ Column only ICURC function ----------- # 
def ICURC(X_Omega, J_ccs, r, params_ICURC):
    params_ICURC = set_default_params_ICURC(params_ICURC)
    eta = params_ICURC['eta']
    #print(f'Using stepsize eta_C = {eta[0]:.6f}.')
    #print(f'Using stepsize eta_R = {eta[1]:.6f}.')
    #print(f'Using stepsize eta_U = {eta[2]:.6f}.')
    eta = params_ICURC['eta']
    TOL = params_ICURC['TOL']
    max_ite = params_ICURC['max_ite']
    steps_are1 = params_ICURC['steps_are1']

    Obs_U = X_Omega[np.ix_(J_ccs, J_ccs)]
    Obs_C = X_Omega[:, J_ccs]
    Smp_C = (Obs_C != 0)
    Smp_U = (Obs_U != 0)

    Omega_col = np.where(Smp_C.flatten())[0]
    Omega_U = np.where(Smp_U.flatten())[0]
    L_obs_col_vec = Obs_C.flatten()[Omega_col]
    L_obs_U_vec = Obs_U.flatten()[Omega_U]
    normC_obs = np.linalg.norm(L_obs_col_vec)
    normU_obs = np.linalg.norm(L_obs_U_vec)
    col_norm_sum = normC_obs + normU_obs 

    if col_norm_sum == 0:
        col_norm_sum = 1.0

    U = Obs_U.copy()
    u, s, vh = np.linalg.svd(U, full_matrices=False)
    u = u[:, :r]
    vh = vh[:r, :]
    s = np.diag(s[:r])
    U = u @ s @ vh

    C = Obs_C.copy()

    fct_time = time.time()
    for ICURC_ite in range(1, max_ite + 1):
        ite_time = time.time()

        C = C @ (vh.T @ vh)

        U_flat = U.flatten()
        C_flat = C.flatten()
        New_Error =  (np.linalg.norm(C_flat[Omega_col] - L_obs_col_vec) +
                     np.linalg.norm(U_flat[Omega_U] - L_obs_U_vec)) / col_norm_sum


        if New_Error < TOL or ICURC_ite == max_ite:
            ICURC_time = time.time() - fct_time
            C_flat = C_flat.copy()
            C_flat[Omega_col] = L_obs_col_vec
            C = C_flat.reshape(C.shape)
            U_flat = U_flat.copy()
            U_flat[Omega_U] = L_obs_U_vec
            U = U_flat.reshape(U.shape)

            u, s, vh = np.linalg.svd(U, full_matrices=False)
            u = u[:, :r]
            vh = vh[:r, :]
            s = np.diag(s[:r])
            U_pinv = vh.T @ np.linalg.pinv(s) @ u.T

            #print(f'ICURC finished in {ICURC_ite} iterations, final error: {New_Error:.2e}, total runtime: {ICURC_time:.2f}s')
            return C, U_pinv, ICURC_time

        if not steps_are1:
            C_flat = C_flat.copy()
            C_flat[Omega_col] = (1 - eta[0]) * C_flat[Omega_col] + eta[0] * L_obs_col_vec
            C = C_flat.reshape(C.shape)

            U_flat = U_flat.copy()
            U_flat[Omega_U] = (1 - eta[2]) * U_flat[Omega_U] + eta[2] * L_obs_U_vec
            U = U_flat.reshape(U.shape)

        else:
            C_flat = C_flat.copy()
            C_flat[Omega_col] = L_obs_col_vec
            C = C_flat.reshape(C.shape)

            U_flat = U_flat.copy()
            U_flat[Omega_U] = L_obs_U_vec
            U = U_flat.reshape(U.shape)

        u, s, vh = np.linalg.svd(U, full_matrices=False)
        u = u[:, :r]
        vh = vh[:r, :]
        s = np.diag(s[:r])
        U = u @ s @ vh

        #print(f'Iteration {ICURC_ite}: error: {New_Error:.2e}, timer: {time.time() - ite_time:.2f}s')