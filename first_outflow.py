
#first outflow try
#def O_t(X,top_index,Z_b,Z,w_parcel_max,lfc_index,dt):

    #Z_t = Z[top_index]

    #if Z_t <= Z_b:
       # return 0

    #else:
       # w_parcel_prof = w_profile(w_parcel_max,Z,lfc_index,top_index)
       # D_cloud = Dc_func(X,Z_t-Z_b)
       # min_index = np.argmax((w_parcel_prof - alpha*D_cloud)*dt > Z_t - Z[lfc_index:top_index+1] )
       # max_index = np.argmax(w_parcel_prof[min_index:] < alpha*D_cloud) + min_index
       # Z_middle = (Z_t + Z[lfc_index])/2
       # print(min_index,max_index) 
       # flux_term = w_parcel_max*np.pi/6*rho_w*D_cloud**3
       # if Z[min_index] < Z_middle:


          #  O_t = flux_term*N_c/2*(1/(Z_middle - Z[lfc_index]))**3*((-Z_middle**3/3 +Z_middle**2/2*(Z_middle + Z[lfc_index]) - Z_middle**2*Z[lfc_index])\
             #               -(-Z[min_index]**3/3 +Z[min_index]**2/2*(Z_middle + Z[lfc_index]) - Z_middle*Z[lfc_index]**2))\
            #    +flux_term*N_c/2*(1/(Z[max_index] - Z_middle))*(1/(Z_t - Z_middle))**2*((-Z[max_index]**3/3 +Z[max_index]**2/2*(Z[max_index] - Z_middle) - Z[max_index]**2*Z_middle)\
              #              -(-Z_middle**3/3 +Z_middle**2/2*(Z[max_index] - Z_middle) - Z[max_index]*Z_middle**2))
            #return O_t
       # else:

           # O_t = flux_term*N_c/2*(1/(Z[max_index] - Z[min_index]))*(1/(Z_t - Z_middle))**2*((-Z[max_index]**3/3 +Z[max_index]**2/2*(Z[max_index] - Z[min_index]) - Z[max_index]**2*Z[min_index])\
                      #      -(-Z[min_index]**3/3 +Z[min_index]**2/2*(Z[max_index] - Z[min_index]) - Z[max_index]*Z[min_index]**2))
   # print(O_t)  
   # return O_t