##The code for the authentic GB84 model

# In[58]:


import numpy as np
from scipy.optimize import fsolve,least_squares
import xarray as xr
from sympy import Symbol, Eq,exp, sqrt, lambdify
from sympy import exp, sqrt, Max
import matplotlib.pyplot as plt
from kalman_filter import kalman_filter as kf

# Constants (taken from A Generalized Stochastic Hydrometeorological Model 
#for Flood and Flash-Flood Forecasting, Georgakakos 1986)

epsilon = 0.622  # unitless
A = 2.5e6  # (J/kg)
B = 2.38e3  # (J/(kg K))
A_1 = 8e-4  # (kg/(m·s²·K^3.5))
A_2 = 2.11e-5  # (m²/s) 
T_star = 273.15  # (K)
p_star = 101325  # (kg/(m·s²))
p_n = 1e5  # nominal pressure - (kg/(m·s²))
g = 9.80  # (m/s²)
R = 287  # (J/(kg·K))
R_v = 461  # (J/(kg·K))
c_p = 1004  # (J/(kg·K))
p_l = 2e4  # lowest possible cloud top pressure - (kg/(m·s²))
alpha_rain = 3500  # (1/s) for rain
alpha_snow = 1500  # (1/s) for snow
c1_rain = 7e5  # (kg/(m³·s)) for rain
c1_snow = 1.4e5  # (kg/(m³·s)) for snow
C_1 = c1_rain
alpha = alpha_rain



# In[61]:


# storm invariant parameters from Georgakakos 1986
epsilon_1 = 1.65e-3 # unitless
epsilon_2 = 5e4  # highest possible cloud top pressure - (kg/(m/s²))
epsilon_3 = 1  # (s/m)
epsilon_4 = 5.5e-5  # nominal hydrometeor diameter - (m)
gamma = 1  # unitless
beta = 1  # unitless
m = 0  # unitless
delta = 1/3*(1/gamma + 1/gamma**2 + 1/gamma**3)


# #### Functions

# In[62]:

#-------------------------------------thermodynamical equations--------------------------------------------------------------
def w(T,P):
    """
    Function for calculating the saturation mixing ratio at a given temperature and pressure  (Georgakakos and Bras 1984a))
    
    Parameters:
        T: temperature [K]
        P: pressure [Pa]
        
    Returns:
        w: saturation mixing ratio [kg/kg]
    """
    return epsilon*A_1*(T - 223.15)**3.5/P

def L(T):
    """
    Function for calculating the latent heat of condensation at a given temperature (from Eagleson)
    
    Parameters:
        T: temperature [K]
        
    Returns:
        L: latent heat of condensation [J/kg]
    """
    #latent heat of condensation
    return A - B*(T - 273.15)

def e_s(T):
    """
    Function for calculating the saturation vapor pressure at a given temperature
    
    Parameters:
        T: temperature [K]
        
    Returns:
        L: latent heat of condensation [J/kg]
    """
    return A_1*(T - 223.15)**3.5

#-------------------------------------cloud heights and convection--------------------------------------------------------------
class calculations_for_convection:
    def __init__(self, T_0, T_d, p_0, obs):
        self.T_0 = T_0
        self.T_d = T_d
        self.p_0 = p_0
        self.obs = obs
        self.p_s, self.T_s = self.calculate_cloud_base()
        self.Theta_e = self.calculate_Theta_e(self.T_s,self.p_s)
        
    def calculate_cloud_base(self):

        # cloud base pressure, temperature
        p_s = (1/((self.T_0 - self.T_d)/223.15 + 1))**3.5 * self.p_0
        T_s = (1/((self.T_0 - self.T_d)/223.15 + 1))* self.T_0
        return p_s, T_s
    

    def calculate_Theta_e(self, T_s, p_s):
        # constant equivalent potential temperature, calculated at the LCL
        Theta_e = T_s*(p_n/p_s)**0.286*np.exp(L(T_s)*w(T_s,p_s)/(c_p*T_s))
        return Theta_e
#--------------------------------------------------------------------------------------------------------------------------
    def calculate_cloud_top(self,T_0,p_0,p_s,Theta_e):
        if self.obs:
            #if they are available from observations, obs=True
            T_t = T_t
            p_t = p_t
        else:
            p_t = Symbol('p_t')
            T_m = Symbol('T_m')
            T_t = Symbol('T_t')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # solve the two equation system for T_m and p_t

            #  equation for parametrization of cloud top pressure with v_updr and constants
            f1 = Eq(p_t, p_l + (epsilon_2 - p_l) / (1 + epsilon_3 * epsilon_1 *sqrt(Max(1e-7,c_p * (T_m - T_0 / (p_0/(3/4 * p_s + 1/4 * p_t))**0.286)))))
            # equation for solving Theta_e for T_m
            f2 = Eq(Theta_e,T_m * (p_n / (3/4 * p_s + 1/4 * p_t))**0.286 * exp((A - B * (T_m - 273.15)) #replaced L(T) with the whole expression for fsolve
                                * (epsilon*A_1*abs(T_m - 223.15)**3.5/(3/4 * p_s + 1/4 * p_t))/ (c_p * T_m)) ) 

            # Convert the symbolic equations to numerical functions using lambdify
            f1_func = lambdify((p_t, T_m), f1.lhs - f1.rhs, 'numpy')
            f2_func = lambdify((p_t, T_m), f2.lhs - f2.rhs, 'numpy')

            # Define the system of equations for fsolve
            def system(vars):
                p_t_val, T_m_val = vars
                return np.array([f1_func(p_t_val, T_m_val), 
                        f2_func(p_t_val, T_m_val)])

            # Set your initial guesses
            initial_guesses = [35000, 235]  
            bounds_lower = [20000, 223.15]
            bounds_upper = [50000, T_0]
            # Solve the system using fsolve
            solution = least_squares(system, initial_guesses, bounds=(bounds_lower, bounds_upper))

            # Extract solution
            p_t, T_m = solution.x
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # solve the non-linear equation for T_t
            def find_T_t(T_t):
                return Theta_e - T_t * (p_n / p_t)**0.286 * np.exp((A - B * (T_t - 273.15)) * (epsilon*A_1*abs(T_t - 223.15)**3.5/(p_t)) / (c_p * T_t))
            #f3_func = lambdify(T_t, f3.lhs - f3.rhs, 'numpy')
            T_t = fsolve(find_T_t, 240)[0]

            # find the ambient air temperature and pressure
            p_s_up = 3/4 * p_s + 1/4 * p_t  #p_s'
            T_s_up = T_0 / (p_0/(p_s_up))**0.286 #T_s'

        return p_t, T_m, T_t,T_s_up, p_s_up
#--------------------------------------------------------------------------------------------------------------------------
    def run(self):
        p_s,T_s = self.p_s,self.T_s
        p_t, T_m, T_t,T_s_up, p_s_up = self.calculate_cloud_top(self.T_0,self.p_0,self.p_s,self.Theta_e)
        return p_s,T_s,p_t, T_m, T_t,T_s_up, p_s_up#,self.Theta_e


# In[68]:
class variables_2:
    ## alteration of the original class, for directly calculating p_t when having already calculated v_updr
    def __init__(self, T_0, T_d, p_0,v_updr, obs):
        self.T_0 = T_0
        self.T_d = T_d
        self.p_0 = p_0
        self.obs = obs
        self.v_updr = v_updr    
        self.p_s, self.T_s = self.calculate_cloud_base(T_0, T_d, p_0)
        self.Theta_e = self.calculate_Theta_e(self.T_s,self.p_s)
        
    @staticmethod
    def calculate_cloud_base(T_0, T_d, p_0):
        # cloud base pressure, temperature and specific humidity
        p_s = (1/((T_0 - T_d)/223.15 + 1))**3.5 *p_0
        T_s = (1/((T_0 - T_d)/223.15 + 1))* T_0
        return p_s, T_s
    

    def calculate_Theta_e(self, T_s, p_s):
        # equivalent potential temperature inside the cloud
        Theta_e = T_s*(p_n/p_s)**0.286*np.exp(L(T_s)*w(T_s,p_s)/(c_p*T_s))
        return Theta_e
#--------------------------------------------------------------------------------------------------------------------------
    def calculate_cloud_top(self,T_0,p_0,p_s,Theta_e,v_updr):
        if self.obs:
            #if they are available from observations, obs=True
            T_t = T_t
            p_t = p_t
        else:
            #p_t = Symbol('p_t')
            p_t = p_l + (epsilon_2 - p_l) / (1 + epsilon_3 *v_updr)
            T_m = Symbol('T_m')
            T_t = Symbol('T_t')


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
            # solve only for T_m 
           
            f2 = Eq(Theta_e,T_m * (p_n / (3/4 * p_s + 1/4 * p_t))**0.286 * exp((A - B * (T_m - 273.15)) #replaced L(T) with the whole expression for fsolve
                                * (epsilon*A_1*abs(T_m - 223.15)**3.5/(3/4 * p_s + 1/4 * p_t))/ (c_p * T_m))
                                
                                
                                ) # whole expression here instead of p'

            # Convert the symbolic equations to numerical functions using lambdify
            f2_func = lambdify(T_m, f2.lhs - f2.rhs, 'numpy')

            # Set your initial guesses
            initial_guesses = [240]  
            bounds_lower = [223.15]
            bounds_upper = [T_0]
            # Solve the system using fsolve
            solution = least_squares(f2_func, initial_guesses, bounds=(bounds_lower, bounds_upper))

            # Extract solution
            T_m = solution.x
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
            # solve the non-linear equation for T_t
            f3 = Eq(Theta_e,T_t * (p_n / p_t)**0.286 * exp((A - B * (T_t - 273.15)) * (epsilon*A_1*abs(T_t - 223.15)**3.5/(p_t)) / (c_p * T_t)))
            f3_func = lambdify(T_t, f3.lhs - f3.rhs, 'numpy')
            T_t = fsolve(f3_func, 240)

            # find the ambient air temperature and pressure
            p_s_up = 3/4 * p_s + 1/4 * p_t  #p_s'
            T_s_up = T_0 / (p_0/(p_s_up))**0.286 #T_s'

        return p_t, T_m, T_t,T_s_up, p_s_up
#--------------------------------------------------------------------------------------------------------------------------
    def run(self):
        p_s,T_s = self.p_s,self.T_s
        p_t, T_m, T_t,T_s_up, p_s_up = self.calculate_cloud_top(self.T_0,self.p_0,self.p_s,self.Theta_e,self.v_updr)
        return p_s,T_s,p_t, T_m, T_t,T_s_up, p_s_up


def cloud_heights(T_s,T_t,T_0,p_s,p_t,p_0):
    """
    Function for calculating the cloud depth and cloud base height (from Georgakakos and Bras 1984a)
    
    Parameters:
        T_s: temperature at cloud base[K]
        T_t: cloud top temperature[K]
        T_0: surface temperature[K]
        p_s: pressure at cloud base[Pa]
        p_t: cloud top pressure[Pa]
        p_0: surface pressure [Pa]
        
    Returns:
        Z_c: cloud depth [m]
        Z_b: cloud base height [m]
    """


    Z_c = R*(T_s + T_t)/(2*g)*np.log(p_s/p_t)

    Z_b = R*(T_s + T_0)/(2*g)*np.log(p_0/p_s)

    return Z_c,Z_b

    


# In[66]:





# #### Fluxes, state and precipitation

# In[67]:

#------------------------------------------condentation fluxes------------------------------------------------------------------------------------------------

def v(T_m,T_s_prime):
    """
    Function for calculating the in-cloud average updraft velocity (from Georgakakos and Bras 1984a) 
    
    Parameters:
        T_m: cloud temperature at level of average updraft velocity[K]
        T_s_prime: ambient air temperature at level of average updraft velocity[K]
        
    Returns:
        v: in-cloud vertically averaged updraft velocity [m/s]
    """
    v = epsilon_1*np.sqrt(c_p*(T_m - T_s_prime))
    return v

def rho_m(T_s,T_t,p_s,p_t):
    """
    Function for calculating the in-cloud average air density (from Georgakakos and Bras 1984a) 
    
    Parameters:
        T_s: temperature at cloud base[K]
        T_t: cloud top temperature[K]
        p_s: pressure at cloud base[Pa]
        p_t: cloud top pressure[Pa]
        
    Returns:
        rho_m: in-cloud vertically averaged air density [kg/m3]
    """
    rho_m = (p_s/(R*T_s) + p_t/(R*T_t))/2
    return rho_m


# In[64]:

#------------------------------------
def f(T_d,p_0,p_t,T_t,rho,v):
    """
    Function for calculating the precipitation flux (from Georgakakos and Bras 1984a)
    
    Parameters:
        T_d: surface dew point temperature[K]
        p_0: surface pressure [Pa] 
        p_t: cloud top pressure[Pa]
        T_t: cloud top temperature[K]
        rho: in-cloud vertically averaged air density [kg/m3]
        v: in-cloud vertically averaged updraft velocity [m/s]
        
    Returns:
        f: precipitation flux at cloud base [kg/m2/s]   
    """
    w_0 = w(T_d,p_0)
    w_s = w(T_t,p_t)

    return (w_0 - w_s)*rho*v


#------------------------------------outflux----------------------------------------------------------------------
def non_dim_numbers(v):
    """
    Parameters for the calculation of fluxes

    Parameters:
        v: in-cloud vertically averaged updraft velocity [m/s]
        
    Returns:
        V_p: outflux velocity precipitation flux [m/s]
        N_v: non-dimensional number for evaporation flux
    """
    V_p = 4*alpha*epsilon_4*v**m
    N_v = beta*v**(1-m)/(alpha*epsilon_4)
    
    return V_p,N_v

class h_out:
    # from Georgakakos 1984a - moisture output flux equations
    def __init__(self,v,Z_c):
        self.V_p = non_dim_numbers(v)[0]
        self.N_v = non_dim_numbers(v)[1]
        self.Z_c = Z_c
        self.O_b = self.O_b()
        self.O_t = self.O_t()
        
    #cloud bottom outflow
    def O_b(self):
        return (1 + 3/4*self.N_v + self.N_v**2/4 + self.N_v**3/24)/np.exp(self.N_v)

    #cloud top outflow 
    def O_t(self):
        return(1 + 3/4*(gamma*self.N_v) + (gamma*self.N_v)**2/4 + (gamma*self.N_v)**3/24)\
                                        /(gamma**5*np.exp(gamma*self.N_v)) + self.N_v/(4*gamma**4) + 1/gamma**5

    #total outflow
    def h_v(self):
        return self.V_p/(self.Z_c*delta)*(self.O_t + self.O_b)

    def run(self):
        return self.O_b,self.O_t,self.h_v()


# In[69]:


#------------------------------------precipitation----------------------------------------------------------------------
class phi:
     #state translation to precipitation equation Φ
     def __init__(self, T_0, p_0, T_d, Z_b, Z_c,v):
        self.T_0 = T_0
        self.p_0 = p_0
        self.T_d = T_d
        self.Z_b = Z_b
        self.V_p = non_dim_numbers(v)[0]
        self.Z_c = Z_c
        self.N_v = non_dim_numbers(v)[1]
        self.v = v
        self.O_b = h_out(v, Z_c).O_b
        self.T_w = self.solve_Tw()

     def equation_Tw(self, T):
        #wet bulb temperature
         return T + L(self.T_0)/c_p*(epsilon*A_1*(T - 223.15)**3.5/self.p_0 - w(self.T_d,self.p_0)) - self.T_0
     
     def solve_Tw(self):
        #numerical solution of wet bulb temperature
         T_w_initial = 290
         return fsolve(self.equation_Tw, T_w_initial)
     
     
     def calculate_phi(self):

        #diffusivity of water vapor in air
        D_AB = A_2*(self.T_0/T_star)**1.94*(p_star/self.p_0)

        #critical diameter for evaporation
        D_c = (1/C_1*4*D_AB/R_v*self.Z_b*(e_s(self.T_w)/self.T_w - e_s(self.T_d)/self.T_0))**(1/3)

        # non-dimensional number indicative of diffusional (evaporation) losses of droplets
        N_D = D_c/(epsilon_4*self.v**m)

        #calculating phi
        if (N_D/self.N_v) >=1:
            phi_t = self.V_p/(self.Z_c*delta)* ((1 - self.N_v/4)*(1 + N_D + N_D**2/2) + N_D**3/8) / np.exp(N_D)
        else:
            phi_t = self.V_p/(self.Z_c*delta)*(self.O_b - 1/24*N_D**3/ np.exp(self.N_v))

        return phi_t


# In[70]:


class state:
    def __init__(self,X,T_d,T_0,p_0,Z_c,Z_b,dt,p_t,T_t,rho,v):
        self.X = X
        self.dt = dt
        self.T_d = T_d
        self.T_0 = T_0
        self.p_0 = p_0
        self.Z_c = Z_c
        self.Z_b = Z_b
        self.p_t = p_t
        self.T_t = T_t
        self.rho = rho
        self.v = v
        self.V_p = non_dim_numbers(self.v)[0]
        self.N_v = non_dim_numbers(self.v)[1]

    def state_evol(self):
        X_new = self.X + self.dt*(f(self.T_d,self.p_0,self.p_t,self.T_t,self.rho,self.v) \
                                  - h_out(self.v,self.Z_c).run()[2]*self.X)
        if X_new<0:
         X_new = 0
        return X_new

