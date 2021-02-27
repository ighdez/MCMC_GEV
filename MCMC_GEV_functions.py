# Functions to generate GEV random numbers using an MCMC
# Based in McFadden (1995) procedure, theorem 3
# Written by Jose Ignacio Hernandez
# February 2021

# Load packages
import numpy as np

# The GEV joint density function
def gev_pdf(x,y,s):
    h_int = np.exp(-x/s) + np.exp(-y/s)
    f1 = np.exp(-(h_int**s) - (x/s) - (y/s)) * (h_int**(2*s - 2))
    f2 = ((1-s)/s)*np.exp(-(h_int**s) - (x/s) - (y/s)) * (h_int**(s-2))
    f_gev = f1 + f2
    return(f_gev)

# The density of an iid extreme value vector
def gev_iid_pdf(x,y):
    exp_e_x = np.exp(-np.exp(-x))
    exp_e_y = np.exp(-np.exp(-y))
    g_gev = (np.exp(-x)*exp_e_x)*(np.exp(-y)*exp_e_y)
    return(g_gev)

# The g_t function, as in McFadden (1995), Theorem 3
def gev_gt(x,y):
    log_x = np.log(x)
    log_y = np.log(y)
    g_gev = (-x*log_x)*(-y*log_y)
    return(g_gev)

# The MCMC GEV random number generator
def gev_rand(N,s,burn_ratio=0.1):
    
    # Generate N + burn observations
    burn = int(N*burn_ratio)
    N_plus_burn = N + burn

    # Initialize draws matrix
    draws = np.zeros((N_plus_burn,2))

    # Initialize expresions of the MCMC for the n = 0
    zeta_old = np.random.uniform(size=2)
    e_old = -np.log(-np.log(zeta_old))
    g_old = gev_gt(zeta_old[0],zeta_old[1])
    f_old = gev_pdf(e_old[0],e_old[1],s)

    # Set first two draws as an iid Gumbel
    draws[0,:] = e_old

    # MCMC Algorithm
    for n in range(1,N_plus_burn):

        # Step a: Draw eta^t and a 2-element vector zeta^t from a uniform distribution
        eta_t = np.random.uniform()
        zeta_t = np.random.uniform(size=2)
        
        # Step b: Define candidate e_t, compute g_t and f_t.
        e_t = -np.log(-np.log(zeta_t))
        g_t = gev_gt(zeta_t[0],zeta_t[1])
        f_t = gev_pdf(e_t[0],e_t[1],s)
        
        # Step c: Define the accept/reject condition
        r_t = (f_t*g_old)/(f_old*g_t)
        
        # Step d: if eta_t <= r_t, then accept. Else, use previous epsilon candidate
        if eta_t <= r_t:
            draws[n,:] = e_t
        else:
            draws[n,:] = draws[(n-1),:]
        
        # Store f and g as old values for the next iteration
        f_old = np.copy(f_t)
        g_old = np.copy(g_t)

    # "Burn" the first N*burn_ratio observations of the draws matrix
    burned_draws = draws[burn:,:]

    # Return the "burned" draws
    return(burned_draws)