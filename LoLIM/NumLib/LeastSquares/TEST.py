#!/usr/bin/env python3


import numpy as np
from matplotlib import pyplot as plt
from LoLIM.NumLib.LeastSquares import GSL_LeastSquares



def model(X, time, f_out):
    A = X[0]
    L = X[1]
    B = X[2]

    f_out[:] = time
    f_out *= -L
    np.exp(f_out, out=f_out)
    f_out *= A
    f_out += B

class fit_tester:
    def __init__(self, T, data):
        self.T = T
        self.data = data


    def objective_function(self, X, f_out, additional_info):

        model(X, self.T, f_out)

        f_out -= self.data


    def jacobian(self, X, J_out, additional_info):
        ##J[i,j] where i refers to the sample, and j to the parameter
        A = X[0]
        L = X[1]
        B = X[2]


        ## derivative with respect to A
        J_out[:,0] = self.T
        J_out[:,0] *= -L

        np.exp(J_out, out=J_out)

        ## derivative with respect to L
        J_out[:,1] = J_out[:,0] ## reuse calculation
        J_out[:,1] *= -1
        J_out[:,1] *= self.T

        ## with respect to B
        J_out[:,2]  = 1




if __name__ == "__main__":
    correct_A = 5.0
    correct_L = 1.5
    correct_B = 1.0

    num_points = 100
    max_T = 3


    time = np.linspace(0,max_T,num_points)

    data = np.empty( num_points )
    model([correct_A, correct_L, correct_B], time, data)


    noise = 0.1
    jitters = np.random.normal(loc=0, scale=noise, size=len(data))
    data += jitters



    fitData = fit_tester(time, data)


    fitter = GSL_LeastSquares( 3, num_points, fitData.objective_function, fitData.jacobian )

    guess_location = np.array([1.0, 1.0, 0.0])
    weights = np.array(  [  1/noise**2 ]*num_points )
    fitter.reset( guess_location, weights )

    code, text = fitter.run(2, max_itters=1000, xtol=1e-50, gtol=1e-16, ftol=1e-16)
    code, text = fitter.run(2, max_itters=1000, xtol=1e-50, gtol=1e-16, ftol=1e-16)
    print(code, text)

    resultX = fitter.get_X()
    print(resultX)
    print('num_iters', fitter.get_num_iters() )
    print('rc2', fitter.get_reduced_chi_squared() )

    ModelResult = np.empty( num_points )
    model( resultX,  time, ModelResult)

    plt.scatter( time, data )
    plt.plot(time, ModelResult , 'r')
    plt.show()



    cov_mat = fitter.get_covariance_matrix()
    print('sqrt covariance matrix:')
    print(np.sqrt(cov_mat))


