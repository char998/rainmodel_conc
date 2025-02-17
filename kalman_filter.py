G = 1 #process noise mapping - how sensitive the state is on noise
Q = 10 #process noise covariance - covariance of white noise
R = 10/3600 #measurement noise covariance - covariance of white noise

class kalman_filter:
    def __init__(self, X_initial, Sigma_initial, dt,h,Phi,f,P_meas):
        self.X_initial = X_initial
        self.Sigma_initial = Sigma_initial
        self.dt = dt
        self.h = h
        self.Phi = Phi
        self.f = f
        self.P_meas = P_meas   

    def update(self):
        X_apriori = (1 - self.h*self.dt)*self.X_initial + self.f*self.dt
        Sigma_apriori = (1 - self.h*self.dt)*self.Sigma_initial*(1 - self.h*self.dt) + G**2*Q
        K_gain = Sigma_apriori*self.Phi/(self.Phi*Sigma_apriori*self.Phi + R)


        X_posterior = X_apriori + K_gain*(self.P_meas/3600 - self.Phi*X_apriori)
        Cov_posterior = (1 - K_gain*self.Phi)*Sigma_apriori

        return X_posterior, Cov_posterior