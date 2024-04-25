#cython: language_level=3 
#cython: cdivision=True
#cython: boundscheck=True
#cython: linetrace=False
#cython: binding=False
#cython: profile=False


cimport numpy as np
import numpy as np
from libcpp cimport bool
from libc.math cimport sqrt, fabs, isfinite
from libc.stdlib cimport malloc, free






#cdef extern from "gsl/gsl_math.h" nogil:
#    int gsl_finite(const double x)
#    double gsl_hypot(const double x, const double y)
    
cdef extern from "gsl/gsl_vector.h" nogil:
    
    ctypedef struct gsl_vector:
        size_t size
        size_t stride
        double* data
        void* block
        int owner
    
    gsl_vector* gsl_vector_alloc(size_t n)
    void gsl_vector_free(gsl_vector* v)
    
    double gsl_vector_get(const gsl_vector* v, const size_t i)
    void gsl_vector_set(gsl_vector* v, const size_t i, double x)
    double* gsl_vector_ptr(gsl_vector* v, size_t i)
    
    void gsl_vector_set_all(gsl_vector* v, double x)



    ctypedef struct gsl_vector_view:
        gsl_vector vector

    gsl_vector_view gsl_vector_view_array(double *base, size_t n)
    
cdef extern from "gsl/gsl_matrix.h" nogil:

    ctypedef struct gsl_matrix:
        size_t size1   ##num rows
        size_t size2   ## num collums
        size_t tda
        double* data
        void* block
        int owner
    
    gsl_matrix* gsl_matrix_alloc(size_t n1, size_t n2)
    void gsl_matrix_free(gsl_matrix* m)
    
    double gsl_matrix_get(const gsl_matrix * m, const size_t i, const size_t j)
    void gsl_matrix_set(gsl_matrix* m, const size_t i, const size_t j, double x)
    
    void gsl_matrix_set_all(gsl_matrix* m, double x)


    ctypedef struct gsl_matrix_view:
        gsl_matrix matrix

    gsl_matrix_view gsl_matrix_view_array(double *base, size_t n1, size_t n2)
    
cdef extern from "gsl/gsl_multifit_nlinear.h" nogil:
    int GSL_SUCCESS
    int GSL_EMAXITER
    int GSL_ENOPROG
    
    ctypedef struct gsl_multifit_nlinear_fdf:
        int (* f) ( const gsl_vector* x,  void* params,  gsl_vector* residuals)
        int (* df) ( const gsl_vector* x, void * params, gsl_matrix* jaccobian)
        int (* fvv) (const gsl_vector* x, const void* v, void* params, void* fvv)
        size_t n # number residuals, returned by f
        size_t p # number of variables to fit, length of x
        void* params
        size_t nevalf # counts num function evalutions, set by init
        size_t nevaldf # counts df evaluations, set by init
        size_t nevalfvv # counts fvv evaluations, set by init
        
        
    ctypedef struct gsl_multifit_nlinear_parameters:
        pass
    
    gsl_multifit_nlinear_parameters gsl_multifit_nlinear_default_parameters()
 
    void* gsl_multifit_nlinear_trust ## type of minimizer
    
    void* gsl_multifit_nlinear_alloc(const void* min_type, const gsl_multifit_nlinear_parameters* params, const size_t n, const size_t p)
    
    int gsl_multifit_nlinear_init(const gsl_vector* initial_x, gsl_multifit_nlinear_fdf* fdf, void* workspace)
    int gsl_multifit_nlinear_winit(const gsl_vector *x, const gsl_vector *wts, gsl_multifit_nlinear_fdf* fdf, void* w)
    
    void gsl_multifit_nlinear_free(void* workspace)
    
    gsl_vector* gsl_multifit_nlinear_position(const void* workspace)
    gsl_vector* gsl_multifit_nlinear_residual(const void* workspace)
    

    int gsl_multifit_nlinear_iterate(void *w)

    int gsl_multifit_nlinear_driver(const size_t maxiter, const double xtol, const double gtol, const double ftol, void* callback, void* callback_params, int* info, void* workspace)
    
    size_t gsl_multifit_nlinear_niter(const void* workspace)


    gsl_matrix *gsl_multifit_nlinear_jac(const void *w)

    int gsl_multifit_nlinear_covar(const gsl_matrix *J, const double epsrel, gsl_matrix *covar)





cdef int call_F ( const gsl_vector * x,    void* data,   gsl_vector* F) noexcept:


    cdef size_t Xn = x.size
    cdef double[:] Xview = <double[:Xn]> x.data
    X_array = np.asarray(Xview)

    cdef GSL_LeastSquares LSI = <GSL_LeastSquares> data

    cdef size_t Fn = F.size
    cdef double[:] Fview = <double[:Fn]> F.data
    F_array = np.asarray(Fview)


    try:
        #objective_function( X_array,  F_array)
        LSI.objective_function( X_array,  F_array, LSI.additional_info )
    except Exception as e:
        print('exception in objective function')
        print('   ', e)
        return GSL_SUCCESS+1
        
    return GSL_SUCCESS





cdef int call_df ( const gsl_vector * x,    void* data,   gsl_matrix* J) noexcept:

    cdef size_t Xn = x.size
    cdef double[:] Xview = <double[:Xn]> x.data
    X_array = np.asarray(Xview)


    cdef GSL_LeastSquares LSI = <GSL_LeastSquares> data

    cdef size_t Jn = J.size1 * J.size2
    cdef double[:] Jview = <double[:Jn]> J.data
    J_array = np.asarray(Jview)
    J_array = J_array.reshape( (J.size1, J.size2) )


    try:
        LSI.jacobian( X_array,  J_array, LSI.additional_info)
    except Exception as e:
        print('exception in jacobian')
        print('  ', e)
        return GSL_SUCCESS+1
    
        
    return GSL_SUCCESS


cdef copy_GSLVector_toNumpyArray(gsl_vector* A, np.ndarray[double, ndim=1] B):
    
    cdef int i
    for i in range(len(B)):
        B[i] = gsl_vector_get(A,  i)

cdef copy_NumpyArray_to_GSLVector(np.ndarray[double, ndim=1] A, gsl_vector* B):
    
    cdef int i
    for i in range(len(A)):
        gsl_vector_set(B,  i,  A[i] )


cdef gsl_vector get_GLSVector_view_of_NumpyArray( np.ndarray[double, ndim=1] A ):
    cdef gsl_vector_view VV = gsl_vector_view_array( &A[0], len(A) )
    return VV.vector



cdef class GSL_LeastSquares:

    cdef int x_length
    cdef int f_length

    cdef gsl_multifit_nlinear_fdf fit_function
    cdef gsl_multifit_nlinear_parameters fit_parameters
    cdef void* fit_workspace
    cdef gsl_vector* vector_X
    # cdef gsl_vector* vector_F

    cdef object objective_function
    cdef object jacobian
    cdef object additional_info

    cdef int is_weighted

    def __init__(self, int x_length, int f_length, objective_function, jacobian=None, additional_info=None):
        """ x_length is length of parameters to find. f_length is length of the residuals.
        objective_function should have signiture (x, residuals, additional_info)
            where x and residuals are 1D numpy arrays of doubles. objective_function should use x to fill residuals.


        jacobian should have signiture (x, jacobian_out, additional_info)
            where x is 1D numpy arrays of doubles.  jacobian_out[ i, j] is 2d matrix
            this should fill jacobian_out[i, j]  with d residual[i] / d X[j].  


        additional_info is passed to objective_function and jacobian """

        self.x_length = x_length
        self.f_length = f_length
        self.objective_function = objective_function
        self.jacobian = jacobian
        self.additional_info = None



        self.fit_parameters = gsl_multifit_nlinear_default_parameters()
        
        self.fit_function.f = &call_F
        if jacobian is None:
            self.fit_function.df = NULL
        else:
            self.fit_function.df = &call_df

        self.fit_function.fvv = NULL
        self.fit_function.n = self.f_length
        self.fit_function.p = self.x_length

        self.fit_function.params = <void*>self


        self.fit_workspace = NULL 
                                 


        self.vector_X = NULL
        self.is_weighted = 0

        
        # self.vector_F = gsl_multifit_nlinear_residual( self.fit_workspace )
        


    def __dealloc__(self):
        if self.fit_workspace:
            gsl_multifit_nlinear_free( self.fit_workspace )



    def reset(self, np.ndarray[double, ndim=1] X,  np.ndarray[double, ndim=1] weights=None):
        """this must be called before running fitter. May be called multiple times to fit different problems
        X : np.ndarray[double, ndim=1]   first guess of X
        weights : np.ndarray[double, ndim=1]    weights of the residuals. Optional (assumed to be 1 if not given)
            if errors are gaussian, then weights should be 1/error^2
        """ 



        if self.fit_workspace:
            gsl_multifit_nlinear_free( self.fit_workspace )

        self.fit_workspace = gsl_multifit_nlinear_alloc(gsl_multifit_nlinear_trust, &self.fit_parameters,
                                                       self.f_length, self.x_length)



        if not self.fit_workspace:
            print('cannot allocate memory!')
            return


        self.vector_X = gsl_multifit_nlinear_position( self.fit_workspace )
        copy_NumpyArray_to_GSLVector( X, self.vector_X )
        cdef gsl_vector weight_vector


        cdef int R
        if weights is None:
            R = gsl_multifit_nlinear_init(self.vector_X,   &self.fit_function,   self.fit_workspace )
            self.is_weighted = 0

        else:
            weight_vector = get_GLSVector_view_of_NumpyArray( weights )
            self.is_weighted = 1

            R = gsl_multifit_nlinear_winit(self.vector_X, &weight_vector, &self.fit_function,   self.fit_workspace )


        if R != GSL_SUCCESS:
            print('cannot initialize fitter')
            gsl_multifit_nlinear_free( self.fit_workspace )
            self.fit_workspace = NULL


  


    def run(self, int min_itters, int max_itters, double xtol, double gtol, double ftol):
        """ run fitter untill conditions are met.
            returns code, text
            where code is 1-5 of reason for return. text is string explanation

            codes: 1 if xtol reached, 2 if gtol reached, 3 if ftol reached
            4 if max_itters reached, and 5 if fitter can't make progress

            This can be called multiple times for more iterations. """


        if not self.fit_workspace:
            print('fitter is not initialized! call reset first')
            return None


        cdef int i=0
        cdef int ret
        for i in range(min_itters):
            ret = gsl_multifit_nlinear_iterate( self.fit_workspace )

                    
                
        cdef int info=0
        ret = gsl_multifit_nlinear_driver(max_itters, xtol, gtol, ftol, 
                                          NULL,NULL , &info, self.fit_workspace)

        if ret==GSL_SUCCESS:
            if info == 0:
                return 0, "unknown result"
            elif info == 1: 
                return 1, "xtol met"
            elif info == 2: 
                return 2, "gtol met"
            elif info == 3: 
                return 3, "ftol met"
            else:
                return 6, 'unknown result'

        elif ret==GSL_EMAXITER:
            return 4, 'max iterations reached'
        elif ret==GSL_ENOPROG:
            return 5, 'fitter cannot progress'
        else:
            return 6, 'unknown result'

    def get_num_iters(self):
        """ get number of iterations """

        if not self.fit_workspace:
            print('fitter is not initialized! call reset first')
            return 0

        return gsl_multifit_nlinear_niter( self.fit_workspace )


    def get_X(self, np.ndarray[double, ndim=1] X=None):
        """ get best X location"""

        if not self.fit_workspace:
            print('fitter is not initialized! call reset first')
            return None

        if X is None:
            X = np.empty( self.x_length, dtype=np.double )
    


        cdef gsl_vector* vector_X = gsl_multifit_nlinear_position( self.fit_workspace )
        copy_GSLVector_toNumpyArray( vector_X, X)
        return X

    def get_residual(self, np.ndarray[double, ndim=1] F=None):
        """get residual at best X. NOTE: is multiplied by sqrt(weights)"""

        if not self.fit_workspace:
            print('fitter is not initialized! call reset first')
            return None

        if F is None:
            F = np.empty( self.f_length, dtype=np.double )
    

        
        cdef gsl_vector* vector_F = gsl_multifit_nlinear_residual( self.fit_workspace )
        copy_GSLVector_toNumpyArray( vector_F, F)
        return F

    def get_reduced_chi_squared(self):


        cdef gsl_vector* vector_F = gsl_multifit_nlinear_residual( self.fit_workspace )

        cdef double C2 = 0
        cdef int i = 0

        for i in range( self.f_length ):
            C2 += gsl_vector_get(vector_F, i)**2

        return C2/(self.f_length-self.x_length)

    def get_covariance_matrix(self, double epsrel=0, multiply_by_chi2=False, np.ndarray[double, ndim=2] cov_out=None):
        """ epsrel : remove dependent rows for inversion

        multiply_by_chi2 : whether to multiply the covariance matrix by reduced chi-squared. Is automatically true if weights are not provided"""

        if not self.fit_workspace:
            print('fitter is not initialized! call reset first')
            return None

        if not self.is_weighted:
            multiply_by_chi2 = True

        if cov_out is None:
            cov_out = np.empty( (self.x_length, self.x_length) )

        cov_out[:,:] = 1



        cdef gsl_matrix *J = gsl_multifit_nlinear_jac( self.fit_workspace )



        cdef gsl_matrix_view MV = gsl_matrix_view_array( &cov_out[0,0],   self.x_length,   self.x_length)

        cdef gsl_matrix* covMat = &MV.matrix
    
        cdef int R = gsl_multifit_nlinear_covar(J, epsrel, covMat)


        if R != GSL_SUCCESS:
            print('could not calculate covariance matrix. Consider increasing epsrel')
            return None


        if multiply_by_chi2:
            chi2 = self.get_reduced_chi_squared()
            cov_out *= chi2

        return cov_out
    
    