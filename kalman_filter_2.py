Gamma = 1 #process noise mapping - how sensitive the state is on noise (Γ in the paper)
Q = 0.32 #process noise covariance - variance of white noise
R = 0.38/3600 #measurement noise covariance - covariance of white noise

class kalman_filter:
    def __init__(self, X_initial, Sigma_initial, dt,h,Phi,f,P_meas):
        self.X_apriori = X_initial
        self.Sigma_apriori = Sigma_initial
        self.dt = dt
        self.h = h
        self.Phi = Phi
        self.f = f
        self.P_meas = P_meas   

    def update(self):
        #X_apriori = (1 - self.h*self.dt)*self.X_initial + self.f*self.dt
        
        #Sigma_apriori = (1 + 2*self.h*self.dt)*self.Sigma_initial + Gamma**2*Q*self.dt
        
        K_gain = self.Sigma_apriori*self.Phi/(self.Phi*self.Sigma_apriori*self.Phi + R/self.dt)
        #print(K_gain)

        X_posterior = self.X_apriori + K_gain*(self.P_meas/3600 - self.Phi*self.X_apriori)
        #print(X_posterior,self.X_apriori)
        Cov_posterior = (1 - K_gain*self.Phi)*self.Sigma_apriori

        return X_posterior, Cov_posterior,self.Sigma_apriori