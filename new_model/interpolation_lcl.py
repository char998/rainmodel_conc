import numpy as np
from constants import R_dry,g


def insert_lcl_values(temp_parcel,p,Z,lcl_index,T_s_value,p_s_value,my_temp,q_v):
    """
    function to insert the exact LCL height and meteorological inputs' values into the parcel profile

    Parameters:
        temp_parcel: parcel temperature profile [K]
        p: pressure levels [Pa]
        Z: parcel height profile [m]
        lcl_index: index of the LCL
        T_s_value: temperature at the LCL [K]
        p_s_value: pressure at the LCL [Pa]
        my_temp: air temperature profile [K]
        RH_env: relative humidity profile [0-1]
        q_v: vapor specific humidity profile [kg/kg]
    
    Returns:
        temp_parcel: updated parcel temperature profile [K]
        p: updated pressure levels [Pa]
        Z: updated parcel height profile [m
        lcl_index: updated index of the LCL
        my_temp: updated air temperature profile [K]
        q_v: updated vapor specific humidity profile [kg/kg]
    """

    #when we have the lcl height exactly
    if abs(p[lcl_index] - p_s_value) < 0.1:
        # We already have that level in p.  But to keep output shapes consistent,
        # we still insert a duplicate at that same index—so length → length + 1.
        insert_idx = lcl_index

        # Insert T_s_value (which should be ≈ temp_parcel[lcl_index]) 
        temp_parcel = np.insert(temp_parcel, insert_idx, T_s_value)
        # Insert p_s_value (≈ p[lcl_index])
        p          = np.insert(p, insert_idx, p_s_value)
        # Compute Z_lcl = Z[lcl_index], so we insert the same height again
        Z_lcl      = Z[lcl_index]
        Z          = np.insert(Z, insert_idx, Z_lcl)

        # After insertion, our new LCL index is still at insert_idx
        new_lcl_index = insert_idx

        # Also insert the same ambient‐air values at that index
        my_t_lcl = my_temp[lcl_index]
        qv_lcl   = q_v[lcl_index]

        my_temp = np.insert(my_temp, insert_idx, my_t_lcl)
        q_v     = np.insert(q_v, insert_idx, qv_lcl)

        return temp_parcel, p, Z, new_lcl_index, my_temp, q_v
    
    #when we don't have the lcl height exactly
    elif (p[lcl_index] - p_s_value)>0:
        #when the lcl pressure is below our closest pressure level
        #and so the lcl height is above our closest height level

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #parcel profile update
        temp_parcel = np.insert(temp_parcel,lcl_index+1,T_s_value)  #insert the LCL temperature (Ts)
        p = np.insert(p,lcl_index+1,p_s_value)                      #insert the LCL pressure (ps)
        Z_lcl = R_dry*np.mean(temp_parcel[0:lcl_index+2])*np.log(p[0]/p[lcl_index+1])/g #interpolation to Find the exact LCL height
        Z = np.insert(Z,lcl_index+1,Z_lcl)                          #insert the LCL height
        lcl_index = np.argwhere(Z == Z_lcl).item()                  #update the lcl index
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #air profiles update
        #find the corresponding air temperature and specific humidity (interpolations)
        my_temp_lcl = my_temp[lcl_index-1] + (
            (my_temp[lcl_index +1] - my_temp[lcl_index-1]) /
                (p[lcl_index +1] - p[lcl_index-1])
                    ) * (p_s_value - p[lcl_index-1])

# Linear interpolation of q_v at the LCL
        qv_lcl = q_v[lcl_index-1] + (
            (q_v[lcl_index + 1] - q_v[lcl_index-1]) /
            (p[lcl_index + 1] - p[lcl_index-1])
        ) * (p_s_value - p[lcl_index-1])
        
        my_temp = np.insert(my_temp,lcl_index,my_temp_lcl.item())   #insert the ambient air LCL height temperature
        q_v = np.insert(q_v,lcl_index,qv_lcl.item())
        return temp_parcel, p, Z, lcl_index, my_temp, q_v
        #-----------------------------------------------------------------------------------------------------------------------------------------------

    else:
        #when the lcl pressure is above our closest pressure level
        #and so the lcl height is below our closest height level

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #parcel profile update
        temp_parcel = np.insert(temp_parcel,lcl_index,T_s_value)    #insert the LCL temperature (Ts)
        p = np.insert(p,lcl_index,p_s_value)                        #insert the LCL pressure (ps)
        Z_lcl = R_dry*np.mean(temp_parcel[0:lcl_index+1])*np.log(p[0]/p[lcl_index])/g   #interpolation to Find the exact LCL height
        Z = np.insert(Z,lcl_index,Z_lcl)    #insert the LCL height
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #air profiles update
        #find the corresponding air temperature and specific humidity (interpolations)
        my_temp_lcl = my_temp[lcl_index-1] + (
            (my_temp[lcl_index +1] - my_temp[lcl_index-1]) /
                (p[lcl_index +1] - p[lcl_index-1])
                    ) * (p_s_value - p[lcl_index-1])

# Linear interpolation of q_v at the LCL
        
        qv_lcl = q_v[lcl_index-1] + (
            (q_v[lcl_index + 1] - q_v[lcl_index-1]) /
            (p[lcl_index + 1] - p[lcl_index-1])
        ) * (p_s_value - p[lcl_index-1])

        my_temp = np.insert(my_temp,lcl_index,my_temp_lcl.item())

        q_v = np.insert(q_v,lcl_index,qv_lcl.item())
        return temp_parcel, p, Z, lcl_index, my_temp, q_v