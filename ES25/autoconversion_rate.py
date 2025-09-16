from constants import cpd,T_star,p_star,R_v, A_2,C_1
from physics_equations import e_s,L,w_sat
from scipy.optimize import fsolve


LWC0 = 0.0005 #kg/m^3 -from Kessler (1969)
a = 0.001   #autoconversio rate s-1 -from Kessler (1969)
rho_a = 1.2 #kg/m^3
#N_c = 3e7       #cloud droplet concentration

def cloud_water_autoconversion_rate(X,Z_c,thresh):
    """
    Kessler autoconversion rate (Kessler 1969)

    Parameters
        X (float): Cloud water content [kg/m2]
        Z_c (float): Cloud depth [m]
        thresh (float): Threshold cloud water content [kg/m3]

    Returns
        R (float): Rainrate exiting the cloud [kg/m2/s]
    """
    
    R = a*(X - LWC0*Z_c)
    if X < LWC0*Z_c:
        return 0
    else: 
        return R

def cloud_water_autoconversion_rate_KK(X,Z_c,rho):
    """
    Kharutinov-Kogan autoconversion rate (Kharutinov and Kogan 2004)
    """
    if Z_c == 0 or X == 0:
        return 0
    else:
        N_c = 2e8*((X/Z_c/rho)/0.0005)**0.6
        print(N_c)
        R = 1350 * (X/Z_c/rho)**2.47 * (N_c*1e-6)**(-1.79)
        return R

class critical_diameter:
    
    """
    Class for calculating the critical diameter for evaporation (from Georgakakos and Bras 1984a)
    
    Parameters:
        T_0 (float): surface temperature [K]
        p_0 (float): surface pressure [Pa]
        T_d (float): surface dew point temperature [K]
        Z_b (float): Cloud bottom height [m]
        Z_c (float): Cloud depth [m]
        v (float):   velocity [m/s]
    
    """
    def __init__(self, T_0, p_0, T_d, Z_b, Z_c,v):
        self.T_0 = T_0
        self.p_0 = p_0
        self.T_d = T_d
        self.Z_b = Z_b
        self.Z_c = Z_c
        self.v = v

        self.T_w = self.solve_Tw()

    def equation_Tw(self, T):
         """
         Wet bulb temperature T [K] equation (from Georgakakos 1986)
         """
         return T + L(self.T_0)/cpd*(e_s(T)/self.p_0 - w_sat(self.T_d,self.p_0)) - self.T_0
     
    def solve_Tw(self):
         """
         Solves the equation for Tw numerically"""

         T_w_initial = 290
         return fsolve(self.equation_Tw, T_w_initial)
     
     
    def calculate_evaporation_critical_diameter(self):
        """
        Calculate the critical diameter for evaporation D_c [m]
        """

        #diffusivity of water vapor in air [m2/s] (from Georgakakos 1986)
        D_AB = A_2*(self.T_0/T_star)**1.94*(p_star/self.p_0)
        
        #critical diameter for evaporation  (from Georgakakos 1986)
        D_c = (1/C_1*4*D_AB/R_v*self.Z_b*abs(e_s(self.T_w)/self.T_w - e_s(self.T_d)/self.T_0))**(1/3)

        return D_c
