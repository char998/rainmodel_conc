#GB84 parameters and constants


epsilon = 0.622  # unitless
A = 2.5e6  # (J/kg)
B = 2.38e3  # (J/(kg K))
A_1 = 8e-4  # (kg/(m·s²·K^3.5))
A_2 = 2.11e-5  # (m²/s) 
T_star = 273.15  # (K)
p_star = 101325  # (kg/(m·s²))
p_n = 1e5  # nominal pressure - (kg/(m·s²))
g = 9.80  # (m/s²)
R_dry = 287  # (J/(kg·K))
R_v = 461  # (J/(kg·K))
cpd = 1004  # (J/(kg·K))
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

#constants for potential temperatures

cpv = 2010.0      # Specific heat of water vapor [J/kg/K]
cl = 4218.0       # Specific heat of liquid water [J/kg/K]
Lv0 = 2.5e6       # Latent heat of vaporization [J/kg]
#epsilon = R_dry / R_v  # Ratio of gas constants