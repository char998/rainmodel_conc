import numpy as np
from scipy.special import gamma, gammaincc
from scipy.optimize import root_scalar


# --- Parameters of empirical mu(Lambda) relation ---
def mu_from_lambda(Lambda):
    """
    Function of μ given Λ[1/mm] (from Zhang et al. 2001)
    
    Parameters:
        Lambda (float): Gamma slope parameter [1/cm]
    
    Returns
        mu (float): Gamma shape parameter"""
    
    return -0.016*(Lambda/10)**2 + 1.213*(Lambda/10) -1.957

# --- Gamma function helper ---
def incomplete_gamma(n, x):
    return gamma(n) * gammaincc(n, x)

# --- N0 from μ ---
def N0_from_mu(mu):
    """
    The N0 parameter of the Gamma as a function of μ (from Ulbrich 1983)

    Parameters:
        mu (float): Gamma shape parameter
    
    Returns:
        N0 [m-3 cm^(-1-μ)]"""
    
    return 6e4 * np.exp(3.2 * mu)

# --- Terminal drop size limit ---
def D_prime(vb, a):
    """
    Minimum raindrop size that exits the cloud
    
    Parameters:
        vb (float): Cloud bottom velocity [cm/s]
        a (float): Terminal velocity coefficient [1/s]
    
    Returns:
        D_prime [cm]"""

    return vb / a

# --- Full equation: R_model - R_target ---
def rainfall_equation_lambda_only(Lambda, R_target, a, vb):
    """
    Defining the equation to solve numerically for finding Lambda

    Parameters:
        Lambda (float): Gamma slope parameter [1/cm]
        R_target (float): Target rainfall rate [mm/hr]
        a (float): Terminal velocity coefficient [1/s]
        vb (float): Cloud bottom velocity [cm/s]

    Returns:
        R_model - R_target: Residual in [m/s]
    """
    v = vb            # [cm/s]
    mu = mu_from_lambda(Lambda)
    Dp = D_prime(v, a)
    N0 = N0_from_mu(mu)

    G1 = incomplete_gamma(5 + mu, Lambda * Dp)
    G2 = incomplete_gamma(4 + mu, Lambda * Dp)

    R_model = (np.pi / 6) * N0 * (
        a * G1 / Lambda**(5 + mu) - (v) * G2 / Lambda**(4 + mu)
    )*1e-5 ## 1e-5 = rho_w * 1e-8
    return R_model - R_target

# --- Solver for Lambda ---
def solve_lambda_mu_dependent(R_target, a, vb, bracket=(0.01, 500)):
    """
    Solves the equation for Lambda numerically

    Parameters:
        R_target (float): Target rainfall rate [mm/hr]
        a (float): Terminal velocity coefficient [1/s]
        vb (float): Cloud bottom velocity [m/s]
        bracket (tuple): Bracket for the root finding

    Returns:
        Lambda (float): Gamma slope parameter [1/cm]
    """

    result = root_scalar(
        rainfall_equation_lambda_only,
        args=(R_target, a, vb),
        bracket=bracket,
        method='brentq'
    )
    if result.converged:
        return result.root
    else:
        raise RuntimeError("Lambda root-finding did not converge")
    
def ground_precipitation_rate(R, mu, Lambda, Dc,vb, a):
    """
    Compute precipitation rate at the ground using gamma DSD.

    Parameters:
        R (float): Rainfall rate [mm/hr]
        mu (float): Gamma shape parameter
        Lambda (float): Gamma slope parameter [1/cm]
        Dc (float): Critical drop diameter [cm]
        a (float): Terminal velocity coefficient [1/s]
        Vb (float): Updraft velocity [m/s]
        rho_w (float): Water density [kg/m^3]

    Returns:
        P (float): Precipitation rate at the ground [kg/m^2/s]
    """
    v = vb           # [cm/s]
    N0 = N0_from_mu(mu)
    pi_over_6 = np.pi / 6
    D3 = Dc**3
    LambdaDc = Lambda * Dc
    LambdaVb = Lambda * v / a
    rho_w = 1000 #kg/m^3
    
    if v / a > Dc:
        term1 = a * incomplete_gamma(mu + 2, LambdaVb) / Lambda**(mu + 2)
        term2 = v * incomplete_gamma(mu + 1, LambdaVb) / Lambda**(mu + 1)
        P = R - (pi_over_6 * rho_w * N0 * D3 * (term1 - term2))*10**(-8)
    else:
        t1 = a  * incomplete_gamma(mu + 5, LambdaDc) / Lambda**(mu + 5)
        t2 = v * incomplete_gamma(mu + 4, LambdaDc) / Lambda**(mu + 4)
        t3 = a  * incomplete_gamma(mu + 2, LambdaDc) / Lambda**(mu + 2)
        t4 = v * incomplete_gamma(mu + 1, LambdaDc) / Lambda**(mu + 1)
        P = (pi_over_6 * rho_w *  N0 * (t1 - t2 - D3*t3 + D3*t4))*10**(-8)

    return P  # in [kg/m^2/s], which is mm/s because ρw = 1000kg/m^3