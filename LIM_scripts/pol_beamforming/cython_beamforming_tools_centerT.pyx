#!/usr/bin/env python3

#cython: language_level=3 
#cython: cdivision=False
#cython: boundscheck=True

#cython: linetrace=False
#cython: binding=False
#cython: profile=False

#### wiki spherical vector fields: https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport sqrt, log, sin, cos, acos, atan2, M_PI
from libc.stdlib cimport malloc, free
cimport scipy.linalg.cython_lapack as lp
cimport scipy.linalg.cython_blas as bs

from libc.stdlib cimport malloc, free

cdef extern from "complex.h" nogil:
    double complex cexp(double complex)
    double cabs(double complex z)
    double cimag(double complex)
    double creal(double complex)
    double carg(double complex arg) ## returns phase in radians
    double complex conj( double complex z )
    
    
    
cdef extern from "gsl/gsl_fft_complex.h" nogil:
    void* gsl_fft_complex_wavetable_alloc(size_t n)
    void gsl_fft_complex_wavetable_free(void* wavetable)
    void* gsl_fft_complex_workspace_alloc(size_t n)
    void gsl_fft_complex_workspace_free(void* workspace)
    int gsl_fft_complex_forward(double* data, size_t stride, size_t n, const void* wavetable, void* work)
    int gsl_fft_complex_inverse(double* data, size_t stride, size_t n, const void* wavetable, void* work)

cdef double c_air_inverse = 1.000293/299792458.0


#### SVD functions
# note that these are collumn-major order!!

## some lapack documentation

# ZGESDD
# *  JOBZ    (input) CHARACTER*1
# *          Specifies options for computing all or part of the matrix U:
# *          = 'A':  all M columns of U and all N rows of V**H are
# *                  returned in the arrays U and VT;
# *          = 'S':  the first min(M,N) columns of U and the first
# *                  min(M,N) rows of V**H are returned in the arrays U
# *                  and VT;
# *          = 'O':  If M >= N, the first N columns of U are overwritten
# *                  in the array A and all rows of V**H are returned in
# *                  the array VT;
# *                  otherwise, all columns of U are returned in the
# *                  array U and the first M rows of V**H are overwritten
# *                  in the array A;
# *          = 'N':  no columns of U or rows of V**H are computed.
# *
# *  M       (input) INTEGER
# *          The number of rows of the input matrix A.  M >= 0.
# *
# *  N       (input) INTEGER
# *          The number of columns of the input matrix A.  N >= 0.
# *
# *  A       (input/output) COMPLEX*16 array, dimension (LDA,N)
# *          On entry, the M-by-N matrix A.
# *          On exit,
# *          if JOBZ = 'O',  A is overwritten with the first N columns
# *                          of U (the left singular vectors, stored
# *                          columnwise) if M >= N;
# *                          A is overwritten with the first M rows
# *                          of V**H (the right singular vectors, stored
# *                          rowwise) otherwise.
# *          if JOBZ .ne. 'O', the contents of A are destroyed.
# *
# *  LDA     (input) INTEGER
# *          The leading dimension of the array A.  LDA >= max(1,M).
# *
# *  S       (output) DOUBLE PRECISION array, dimension (min(M,N))
# *          The singular values of A, sorted so that S(i) >= S(i+1).
# *
# *  U       (output) COMPLEX*16 array, dimension (LDU,UCOL)
# *          UCOL = M if JOBZ = 'A' or JOBZ = 'O' and M < N;
# *          UCOL = min(M,N) if JOBZ = 'S'.
# *          If JOBZ = 'A' or JOBZ = 'O' and M < N, U contains the M-by-M
# *          unitary matrix U;
# *          if JOBZ = 'S', U contains the first min(M,N) columns of U
# *          (the left singular vectors, stored columnwise);
# *          if JOBZ = 'O' and M >= N, or JOBZ = 'N', U is not referenced.
# *
# *  LDU     (input) INTEGER
# *          The leading dimension of the array U.  LDU >= 1; if
# *          JOBZ = 'S' or 'A' or JOBZ = 'O' and M < N, LDU >= M.
# *
# *  VT      (output) COMPLEX*16 array, dimension (LDVT,N)
# *          If JOBZ = 'A' or JOBZ = 'O' and M >= N, VT contains the
# *          N-by-N unitary matrix V**H;
# *          if JOBZ = 'S', VT contains the first min(M,N) rows of
# *          V**H (the right singular vectors, stored rowwise);
# *          if JOBZ = 'O' and M < N, or JOBZ = 'N', VT is not referenced.
# *
# *  LDVT    (input) INTEGER
# *          The leading dimension of the array VT.  LDVT >= 1; if
# *          JOBZ = 'A' or JOBZ = 'O' and M >= N, LDVT >= N;
# *          if JOBZ = 'S', LDVT >= min(M,N).
# *
# *  WORK    (workspace/output) COMPLEX*16 array, dimension (MAX(1,LWORK))
# *          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
# *
# *  LWORK   (input) INTEGER
# *          The dimension of the array WORK. LWORK >= 1.
# *          if JOBZ = 'N', LWORK >= 2*min(M,N)+max(M,N).
# *          if JOBZ = 'O',
# *                LWORK >= 2*min(M,N)*min(M,N)+2*min(M,N)+max(M,N).
# *          if JOBZ = 'S' or 'A',
# *                LWORK >= min(M,N)*min(M,N)+2*min(M,N)+max(M,N).
# *          For good performance, LWORK should generally be larger.
# *
# *          If LWORK = -1, a workspace query is assumed.  The optimal
# *          size for the WORK array is calculated and stored in WORK(1),
# *          and no other work except argument checking is performed.
# *
# *  RWORK   (workspace) DOUBLE PRECISION array, dimension (MAX(1,LRWORK))
# *          If JOBZ = 'N', LRWORK >= 5*min(M,N).
# *          Otherwise, LRWORK >= 5*min(M,N)*min(M,N) + 7*min(M,N)
# *
# *  IWORK   (workspace) INTEGER array, dimension (8*min(M,N))
# *
# *  INFO    (output) INTEGER
# *          = 0:  successful exit.
# *          < 0:  if INFO = -i, the i-th argument had an illegal value.
# *          > 0:  The updating process of DBDSDC did not converge.


## some psuedo inverse code!
@cython.final
cdef class SVD_psuedoinversion:
    cdef int M 
    cdef int N 
    cdef int len_s # min(M,N)
    
    cdef double cond
    cdef double cond_factor
    
    cdef int lwork
    
    cdef int[:] iwork
    cdef double[:] rwork
    cdef double complex[:] work
    
    cdef double complex[:,:] A_tmp
    
    cdef double[:] S
    cdef double complex[:,:] U
    cdef double complex[:,:] U_over_S
    cdef double complex[:,:] Vt
    
    cdef int matrix_set
    cdef int default_rank
    
    
    def __init__(self, M, N, cond=None, rcond=None):
        """setup data for inverting MxN matrix. cond sets minimum singular value. If cond is None, then rcond sets the maximum ratio of max over min singular value. 
        If both are none, rcond is set to something like machine precision."""
        
        self.M = M
        self.N = N
        
        self.len_s = self.M
        if self.N < self.M:
            self.len_s = self.N
        
        
        self.A_tmp = np.empty( (self.N,self.M), dtype=np.cdouble )
        
        self.S = np.empty( self.len_s, dtype=np.double )
        self.U = np.empty( (self.len_s, self.M), dtype=np.cdouble )
        self.U_over_S = np.empty( (self.len_s, self.M), dtype=np.cdouble )
        self.Vt = np.empty( (self.N, self.len_s), dtype=np.cdouble )
        
        
        self.rwork = np.empty( 5*self.len_s*self.len_s + 7*self.len_s, dtype=np.double )
        self.iwork = np.empty( 8*self.len_s, dtype=np.intc )
        
        self.lwork = -1
        cdef int info
        cdef double complex wkopt
        lp.zgesdd('S', &self.M, &self.N, &self.A_tmp[0,0], &self.M, &self.S[0], &self.U[0,0], &self.M, &self.Vt[0,0], &self.len_s, 
                  &wkopt, &self.lwork, &self.rwork[0], &self.iwork[0], &info)
        
                             
        self.lwork = int( wkopt.real )
        self.work = np.empty( self.lwork, dtype=np.cdouble )
        
        if cond==None:
            if rcond == None:
                self.cond = -1
                # t = np.cdouble.char.lower()
                self.cond_factor = np.max([self.M, self.N])* np.finfo( np.cdouble ).eps
            else:
                self.cond = -1
                self.cond_factor = rcond
        else:
            self.cond = cond
            
        self.matrix_set = 0
        
        
    
    @cython.wraparound (False) #turn off negative indexing
    @cython.boundscheck(False) # turn off bounds-checking
    cdef int set_matrix_C(self, double complex[:,:] A, int can_destroy_A=False, int A_is_collumnMajor=False) nogil:
        cdef double complex[:,:] A_to_use
        
    ### first copy-in matrix, flip axes if necisary
        cdef int i
        cdef int j
        if A_is_collumnMajor:
            if A.shape[1] != self.M or A.shape[0] != self.N:
                self.matrix_set = 0
                return -1000
            
            if can_destroy_A:
                A_to_use = A
            else:
                A_to_use = self.A_tmp
                for i in range(self.N):
                    for j in range(self.M):
                        self.A_tmp[i,j] = A[i,j]
        else:
            if A.shape[0] != self.M or A.shape[1] != self.N:
                self.matrix_set = 0
                return -1000
            A_to_use = self.A_tmp
            for i in range(self.N):
                for j in range(self.M):
                    self.A_tmp[i,j] = A[j,i]
            
            
            
            
    ## do SVD
        cdef int info
        lp.zgesdd('S', &self.M, &self.N, &A_to_use[0,0], &self.M, &self.S[0], &self.U[0,0], &self.M, &self.Vt[0,0], &self.len_s, 
                  &self.work[0], &self.lwork, &self.rwork[0], &self.iwork[0], &info)
        
        if info <= 0:
            self.matrix_set = 1
        else:
            self.matrix_set = 0
            return info
        
        
        
    ## calc default rank
        cdef double cond = self.cond
        if self.cond < 0: ## use cond_factor
            cond = 0.0
            for i in range( self.len_s ):
                if self.S[i]>cond:
                    cond = self.S[i]
            cond *= self.cond_factor
                
        self.default_rank = 0
        for i in range( self.len_s ):
            if self.S[i] > cond:
                self.default_rank += 1
            
        
        return info
    
    
    
    
    @cython.wraparound (False) #turn off negative indexing
    @cython.boundscheck(False) # turn off bounds-checking
    cdef int do_psuedoinverse_C(self, double complex[:,:] out, int override_cond=-1 ) nogil:
        if self.matrix_set == 0:
            return 0
        
        
        cdef int i
        cdef int j
        
    ## calculate rank
        cdef int rank = 0
        if override_cond<0: ## we use the built-in info
            rank = self.default_rank
        else:
            rank = self.len_s - override_cond
        
        
    ## divide U by S
        for i in range( rank ):
            for j in range( self.M ):
                self.U_over_S[i, j] = self.U[i, j]/self.S[ i ]
        
            
    ## multiply matrices and copy to out, Note the transpose 
        cdef double complex Q
        cdef int k

        for i in range(self.M):
            for j in range(self.N):
                
                Q = 0
                for k in range(rank):
                    Q += self.U_over_S[k,i]*self.Vt[j,k]
                    
                out[j,i].real = Q.real
                out[j,i].imag = -Q.imag
            
        return rank
            
    
    
    
    @cython.wraparound (False) #turn off negative indexing
    @cython.boundscheck(False) # turn off bounds-checking
    cdef int solve_C(self, double complex[:] b, double complex[:] out, int override_cond=-1 ) nogil:
        if self.matrix_set == 0:
            return 0
        
        
        
    ## calculate rank
        cdef int rank = 0
        if override_cond<0: ## we use the built-in info
            rank = self.default_rank
        else:
            rank = self.len_s - override_cond
        
        
        
            
    ## multiply matrices and copy to out
        cdef double complex Q
        cdef double complex A
        
        cdef int i
        cdef int j
        cdef int k
        
        for j in range(self.N):
            out[j] = 0.0
            
        for k in range(rank):
            
            ## dot with U
            Q = 0
            for i in range(self.M):
                Q += self.U[k,i]*b[i]
                
            Q /= self.S[ k ]
                
            ## multipy by V
            for j in range(self.N):
                A = self.Vt[j,k]*Q
                out[j].real += A.real
                out[j].imag += -A.imag
            
        return rank
    
    def set_matrix(self, np.ndarray[double complex, ndim=2] A):
        """do SVD. Assumes A is row-major (normal C/numpy stuff). Returns status. 0 if succsesfull, unsucsesful otherwise"""
        
        return self.set_matrix_C(A, can_destroy_A=False, A_is_collumnMajor=False)
        
    def get_psuedoinverse(self, np.ndarray[double complex, ndim=2] out, override_cond=None ):
        """get psuedoinverse with SVD. if override_cond is None, then cond is respected. Else, number singular values used = number total - override_cond. Returns final rank."""
        
        if self.matrix_set == 0:
            print("SVD matrix not set")
            return 0
    
        cdef int OC = -1
        if override_cond is not None:
            OC = <int> override_cond
    
        return self.do_psuedoinverse_C(out, override_cond=OC )
    
    def solve(self, np.ndarray[double complex, ndim=1] B, np.ndarray[double complex, ndim=1] out, override_cond=None ):
        """solves matrix equation. if override_cond is None, then cond is respected. Else, number singular values used = number total - override_cond. Returns final rank."""
        
        if self.matrix_set == 0:
            print("SVD matrix not set")
            return 0
    
        cdef int OC = -1
        if override_cond is not None:
            OC = <int> override_cond
    
        if out is None:
            out = np.empty( self.N, dtype=np.cdouble )
    
        return self.solve_C(B, out, OC)
    
    def rank_to_CN(self, rank):
        """given rank, returns ratio of singular values"""
        
        if self.matrix_set == 0:
            print("SVD matrix not set")
            return 0
        
        cdef long R = rank
        
        return self.S[ 0 ]/self.S[R-1]
    
    
    
    
    
    
##### THE BEAMFORMER!!!   

cdef class beamform_engine3D:
    
## inputs
    cdef long num_antennas
    cdef long num_X
    cdef long num_Y
    cdef long num_Z
    
    cdef double[:] reference_XYZ
    
    cdef double[:] center_XYZ
    cdef double[:] X_array
    cdef double[:] Y_array
    cdef double[:] Z_array
    
    cdef double[:,:] antenna_locations
    cdef double[:] antenna_startTimes
    cdef int[:] antenna_polarizations
    
## caluclated
    cdef double[:] geometric_delays ## geometric delays to center pixel. Larger means pulse arrives later
    cdef double[:,:,:] relative_geo_delayDif ## geo-delay towards reference location per Xi, Yi, Zi voxel. Diff to center pixel subtracted off
    cdef double[:] cal_shifts ## add to TOTAL geo delay to get TOTAL time should shift antenna
    
    cdef double[:] max_DTs ## max time shift per antenna
    cdef double[:] min_DTs 
    cdef long[:] index_shifts ## number of samples to shift each antenna (relative to an antenna at the reference position) to shift 
      # antenna data before setting
    
    #cdef double[:, :,:,:] distance_grid ## distance, in m, from pixel to antenna. indeces are:  antenna, X,Y,Z
    

    cdef long num_stations 
    cdef long[:] anti_to_stat_i
    cdef long[:] stati_to_anti  ## antennas range from stat_i to stat_i+1(not inclusive)
    
    cdef int[:] antenna_mask
    cdef double[:] antenna_weights
    
    cdef double complex[:,:] ant_data ## antenna, frequency (only used frequencies)
    cdef long num_data_frequencies ## ant_data will have maximum number of secondary or primary frequencies
        ## note, we assume primary has more frequencies, and is set first

#### primary frequency data
    
    cdef long trace_length
    cdef long num_used_frequencies ## freq_end_index-freq_start_index
    cdef long freq_start_index
    cdef long freq_end_index
    cdef double[:] all_frequencies ## we use between  freq_start_index and freq_end_index
    cdef double complex[:,:,:,:] cut_jones_matrices ## station, frequency(only used frequencies), dipole, ze/az, 
 
    cdef double complex[:,:,:,:] eField_inversion_jonesMarix ## station, direction(X,Y,Z), dipole (X,Y), frequency
    
    cdef void* FFT_wavetable
    cdef void* FFT_workspace
    
    cdef double complex[:] FFT_X_workspace
    cdef double complex[:] FFT_Y_workspace
    cdef double complex[:] FFT_Z_workspace
    cdef int FFT_workspace_set ## is bool
    
 
 #### secondary
    
    cdef long trace_length_secondary
    cdef long num_used_frequencies_secondary ## freq_end_index-freq_start_index
    cdef long freq_start_index_secondary
    cdef long freq_end_index_secondary
    cdef double[:] all_frequencies_secondary ## we use between  freq_start_index and freq_end_index
    cdef double complex[:,:,:,:] cut_jones_matrices_secondary ## station, frequency(only used frequencies), dipole, ze/az, 
    

    cdef double complex[:,:,:,:] eField_inversion_jonesMarix_secondary ## station, direction(X,Y,Z), dipole (X,Y), frequency
    
    cdef void* FFT_wavetable_secondary
    cdef void* FFT_workspace_secondary
    
    cdef double complex[:] FFT_X_workspace_secondary
    cdef double complex[:] FFT_Y_workspace_secondary
    cdef double complex[:] FFT_Z_workspace_secondary
    cdef int FFT_workspace_set_secondary ## is bool
    

    
    def __init__(self, np.ndarray[double, ndim=1] X_array, np.ndarray[double, ndim=1] Y_array, np.ndarray[double, ndim=1] Z_array, 
                 np.ndarray[double, ndim=1] center_XYZ, np.ndarray[double, ndim=1] reference_XYZ, np.ndarray[double, ndim=2] antenna_locs, 
                 np.ndarray[double, ndim=1] ant_startTimes, np.ndarray[int, ndim=1] antenna_polarizations, 
                 np.ndarray[long, ndim=1] anti_to_stat_i, np.ndarray[long, ndim=1] stati_to_anti, 
                 np.ndarray[double, ndim=1] geometric_delays_memory, np.ndarray[double, ndim=1] min_DTs_memory, 
                 np.ndarray[double, ndim=1] max_DTs_memory, np.ndarray[long, ndim=1] index_shifts_memory, 
                 np.ndarray[double, ndim=1] cal_shifts_memory):
    
        if not X_array.flags['C_CONTIGUOUS']:
            print('X_array error')
            quit()
        if not Y_array.flags['C_CONTIGUOUS']:
            print('Y_array error')
            quit()
        if not Z_array.flags['C_CONTIGUOUS']:
            print('Z_array error')
            quit()
        if not center_XYZ.flags['C_CONTIGUOUS']:
            print('center_XYZ error')
            quit()
        if not reference_XYZ.flags['C_CONTIGUOUS']:
            print('reference_XYZ error')
            quit()
        if not antenna_locs.flags['C_CONTIGUOUS']:
            print('antenna_locs error')
            quit()
        if not ant_startTimes.flags['C_CONTIGUOUS']:
            print('ant_startTimes error')
            quit()
        if not antenna_polarizations.flags['C_CONTIGUOUS']:
            print('antenna_polarizations error')
            quit()
        if not anti_to_stat_i.flags['C_CONTIGUOUS']:
            print('anti_to_stat_i error')
            quit()
        if not stati_to_anti.flags['C_CONTIGUOUS']:
            print('stati_to_anti error')
            quit()
            
        if not geometric_delays_memory.flags['C_CONTIGUOUS']:
            print('geometric_delays_memory error')
            quit()
        if not min_DTs_memory.flags['C_CONTIGUOUS']:
            print('min_DTs_memory error')
            quit()
        if not max_DTs_memory.flags['C_CONTIGUOUS']:
            print('max_DTs_memory error')
            quit()
        if not index_shifts_memory.flags['C_CONTIGUOUS']:
            print('index_shifts_memory error')
            quit()
        if not cal_shifts_memory.flags['C_CONTIGUOUS']:
            print('cal_shifts_memory error')
            quit()
        
        # test_psuedoinverse()
        #quit()
        
        self.FFT_wavetable = NULL
        self.FFT_workspace = NULL
        self.FFT_wavetable_secondary = NULL
        self.FFT_workspace_secondary  = NULL
        
        self.X_array = X_array
        self.Y_array = Y_array
        self.Z_array = Z_array
        self.center_XYZ = center_XYZ
        self.reference_XYZ = reference_XYZ
        
        self.antenna_locations = antenna_locs
        self.antenna_startTimes = ant_startTimes
        self.antenna_polarizations = antenna_polarizations
        
        self.num_antennas = len( self.antenna_locations )
        self.num_X = len( self.X_array )
        self.num_Y = len( self.Y_array )
        self.num_Z = len( self.Z_array )
        
        
        self.num_stations = len(stati_to_anti)-1
        self.anti_to_stat_i = anti_to_stat_i
        self.stati_to_anti = stati_to_anti
        
        
    ### set geometric delays
        self.geometric_delays = geometric_delays_memory#np.empty( self.num_antennas, dtype=np.double )
        self.max_DTs = max_DTs_memory #np.zeros( self.num_antennas, dtype=np.double )
        self.min_DTs = min_DTs_memory #np.zeros( self.num_antennas, dtype=np.double )
        
        
        self.relative_geo_delayDif = np.empty( (self.num_X ,self.num_Y ,self.num_Z ) )
        
        ## setup delays
        print('setting known delays')
        cdef int Xi
        cdef int Yi
        cdef int Zi
        cdef double dx
        cdef double dy
        cdef double dz
        cdef double R
        
        dx = self.reference_XYZ[ 0] - center_XYZ[0]
        dy = self.reference_XYZ[ 1] - center_XYZ[1]
        dz = self.reference_XYZ[ 2] - center_XYZ[2]
        R = sqrt( dx*dx + dy*dy + dz*dz )
        cdef double Ref_center_diff = R*c_air_inverse
        
        for Xi in range(self.num_X):
            for Yi in range(self.num_Y):
                for Zi in range(self.num_Z):
                
                    dx = self.reference_XYZ[ 0] - self.X_array[Xi]
                    dy = self.reference_XYZ[ 1] - self.Y_array[Yi]
                    dz = self.reference_XYZ[ 2] - self.Z_array[Zi]
                    
                    self.relative_geo_delayDif[Xi,Yi,Zi] = sqrt( dx*dx + dy*dy + dz*dz )*c_air_inverse - Ref_center_diff
            
                
                
        cdef int ant_i
        cdef double total_dt
        cdef double center_diff
        for ant_i in range(self.num_antennas):
            # print(' ', ant_i, '/', self.num_antennas)
            
            dx = self.antenna_locations[ant_i, 0] - center_XYZ[0]
            dy = self.antenna_locations[ant_i, 1] - center_XYZ[1]
            dz = self.antenna_locations[ant_i, 2] - center_XYZ[2]
            R = sqrt( dx*dx + dy*dy + dz*dz )
            center_diff = R*c_air_inverse
            
            self.geometric_delays[ ant_i ] = center_diff
            
            self.max_DTs[ ant_i ] = 0.0
            self.min_DTs[ ant_i ] = 0.0
            
            for Xi in range(self.num_X):
                for Yi in range(self.num_Y):
                    for Zi in range(self.num_Z):
            
                        dx = self.antenna_locations[ant_i, 0] - self.X_array[Xi]
                        dy = self.antenna_locations[ant_i, 1] - self.Y_array[Yi]
                        dz = self.antenna_locations[ant_i, 2] - self.Z_array[Zi]
                        total_dt = sqrt( dx*dx + dy*dy + dz*dz )*c_air_inverse - self.geometric_delays[ ant_i ]  - self.relative_geo_delayDif[Xi,Yi,Zi]
                        
                        if total_dt > self.max_DTs[ ant_i ]:
                            self.max_DTs[ ant_i ] = total_dt
                        if total_dt < self.min_DTs[ ant_i ]:
                            self.min_DTs[ ant_i ] = total_dt
                            
                        #self.geometric_DT_grid[ ant_i, Xi,Yi,Zi ] = total_dt
                        
        self.index_shifts = index_shifts_memory
        self.cal_shifts = cal_shifts_memory #np.empty( self.num_antennas, dtype=np.double )
        cdef double arrival_t
        cdef long index
        for ant_i in range(self.num_antennas):
            arrival_t = self.geometric_delays[ ant_i ] - ant_startTimes[ ant_i ]
            index = int( arrival_t/(5.0e-9) )
            self.index_shifts[ant_i] = index
            
            
            self.cal_shifts[ant_i] = -self.antenna_startTimes[ant_i] - index*(5.0e-9) ## add total geo delay to this to get total time to shift during beamforming. 
            
        cdef long min_index_shift = np.min( index_shifts_memory )
        index_shifts_memory -= min_index_shift
        
        ## I do not know how this works...
        #for ant_i in range(self.num_antennas):
        
        self.FFT_workspace_set = False
        self.FFT_workspace_set_secondary = False
        
        self.antenna_mask = np.empty( self.num_antennas, dtype=np.intc )
        self.antenna_weights = np.ones( self.num_antennas, dtype=np.double )
    
    def __del__(self):
        if( self.FFT_wavetable ):
            gsl_fft_complex_wavetable_free( self.FFT_wavetable )
        if( self.FFT_workspace ):
            gsl_fft_complex_workspace_free( self.FFT_workspace )
        if( self.FFT_wavetable_secondary ):
            gsl_fft_complex_wavetable_free( self.FFT_wavetable_secondary )
        if( self.FFT_workspace_secondary ):
            gsl_fft_complex_workspace_free( self.FFT_workspace_secondary )


    def set_antenna_functions(self, long trace_length, long freq_start_index, long freq_end_index, 
                          np.ndarray[double, ndim=1] all_frequencies, np.ndarray[complex, ndim=4] cut_jones_matrices,
                          int freq_mode=1 ):
        
        
        ### setup response matrix
        
        cdef long num_used_frequencies = freq_end_index - freq_start_index
        cdef double complex[:,:,:,:] invert_tmp ## for each station, for each frequency, covery X-antenna and Y-antenna voltages into X,Y, and Z electric fields
        
        if freq_mode == 1:
            self.trace_length = trace_length
            self.freq_start_index = freq_start_index
            self.freq_end_index = freq_end_index
            self.num_used_frequencies = num_used_frequencies
            self.all_frequencies = all_frequencies
            self.cut_jones_matrices = cut_jones_matrices
        
        
        
            #### set some memory
            self.FFT_wavetable = gsl_fft_complex_wavetable_alloc( self.trace_length )
            self.FFT_workspace = gsl_fft_complex_workspace_alloc( self.trace_length )
        
            self.ant_data = np.empty( (self.num_antennas, self.num_used_frequencies), dtype=np.cdouble )
            self.num_data_frequencies = self.num_used_frequencies ## assume primary is set first, and is larger
            
            self.eField_inversion_jonesMarix = np.empty((self.num_stations, 3, 2, self.num_used_frequencies), dtype=np.complex)
            invert_tmp =  self.eField_inversion_jonesMarix
            
        elif freq_mode == 2:
            
            self.trace_length_secondary = trace_length
            self.freq_start_index_secondary = freq_start_index
            self.freq_end_index_secondary = freq_end_index
            self.num_used_frequencies_secondary = num_used_frequencies
            self.all_frequencies_secondary = all_frequencies
            self.cut_jones_matrices_secondary = cut_jones_matrices
        
        
        
            #### set some memory
            self.FFT_wavetable_secondary = gsl_fft_complex_wavetable_alloc( trace_length )
            self.FFT_workspace_secondary = gsl_fft_complex_workspace_alloc( trace_length )
        
            #self.ant_data_secondary = np.empty( (self.num_antennas, num_used_frequencies), dtype=np.cdouble )

            
            self.eField_inversion_jonesMarix_secondary = np.empty((self.num_stations, 3, 2, num_used_frequencies), dtype=np.complex)
            invert_tmp =  self.eField_inversion_jonesMarix_secondary
            
        else:
            print('freq mode must be primary (1) or secondary (2)')
            quit()
        

        #### build the matrices
        cdef long stat_i
        cdef long ant_i
        cdef long freq_i
        
        cdef double ave_x
        cdef double ave_y
        cdef double ave_z
        
        cdef double stat_R
        cdef double stat_zenith
        cdef double stat_azimuth
        
        
        cdef double frequency
        cdef double complex J00
        cdef double complex J01
        cdef double complex J10
        cdef double complex J11
        
        
        cdef double complex deterimainant
        cdef double complex Ji00
        cdef double complex Ji01
        cdef double complex Ji10
        cdef double complex Ji11
        
        # print('building 3D jones inverse')
        for stat_i in range(self.num_stations):
            # print(' ', stat_i, '/', self.num_stations)
            
            ave_x = 0
            ave_y = 0
            ave_z = 0
            for ant_i in range(self.stati_to_anti[stat_i], self.stati_to_anti[stat_i+1]):
                ave_x += self.antenna_locations[ant_i, 0]
                ave_y += self.antenna_locations[ant_i, 1]
                ave_z += self.antenna_locations[ant_i, 2]
            ave_x /= (self.stati_to_anti[stat_i+1] - self.stati_to_anti[stat_i])
            ave_y /= (self.stati_to_anti[stat_i+1] - self.stati_to_anti[stat_i])
            ave_z /= (self.stati_to_anti[stat_i+1] - self.stati_to_anti[stat_i])
            
            ave_x -= self.center_XYZ[0] ## note, this is from SOURCE to STATION
            ave_y -= self.center_XYZ[1]
            ave_z -= self.center_XYZ[2]
            
            ave_x *= -1 ## but we want from STATION to SOURCE
            ave_y *= -1
            ave_z *= -1
            
            stat_R = np.sqrt( ave_x*ave_x + ave_y*ave_y + ave_z*ave_z )
            
            ## from station to source...
            stat_zenith = acos( ave_z/stat_R )
            stat_azimuth = atan2( ave_y, ave_x)
            
            sin_stat_azimuth = sin( stat_azimuth )
            cos_stat_azimuth = cos( stat_azimuth )
            sin_stat_zenith = sin( stat_zenith )
            cos_stat_zenith = cos( stat_zenith )
            
            for freq_i in range(num_used_frequencies):
                frequency = all_frequencies[freq_start_index + freq_i]
                
                # these take zeinthal-azimuthal fields to X-Y dipole votages
                J00 = cut_jones_matrices[ stat_i, freq_i, 0,0 ]
                J01 = cut_jones_matrices[ stat_i, freq_i, 0,1 ]
                J10 = cut_jones_matrices[ stat_i, freq_i, 1,0 ]
                J11 = cut_jones_matrices[ stat_i, freq_i, 1,1 ]
            
                ## invert to get zenithal-azimuthal fields from X-Y dipole voltags
                deterimainant = J00*J11 - J01*J10
                Ji00 = J11/deterimainant
                Ji01 = -J01/deterimainant
                Ji10 = -J10/deterimainant
                Ji11 = J00/deterimainant
            
                ## now convert to 3D
            ## response of X-field
                ## to X-dipole
                    ## to zenithal field
                invert_tmp[stat_i,0,0,freq_i] = Ji00*cos_stat_zenith*cos_stat_azimuth
                    ## azimuthal field
                invert_tmp[stat_i,0,0,freq_i] += -Ji10*sin_stat_azimuth
                
                ## to Y-dipole
                    ## to zenithal field
                invert_tmp[stat_i,0,1,freq_i] = Ji01*cos_stat_zenith*cos_stat_azimuth
                    ## azimuthal field
                invert_tmp[stat_i,0,1,freq_i] += -Ji11*sin_stat_azimuth
                
                
            ## response of Y-field
                ## to X-dipole
                    ## to zenithal field
                invert_tmp[stat_i,1,0,freq_i] = Ji00*cos_stat_zenith*sin_stat_azimuth
                    ## azimuthal field
                # invert_tmp[stat_i,1,0,freq_i] += -Ji10*sin_stat_azimuth
                invert_tmp[stat_i,1,0,freq_i] += Ji10*cos_stat_azimuth
                
                ## to Y-dipole
                    ## to zenithal field
                invert_tmp[stat_i,1,1,freq_i] = Ji01*cos_stat_zenith*sin_stat_azimuth
                    ## azimuthal field
                invert_tmp[stat_i,1,1,freq_i] += Ji11*cos_stat_azimuth
                
                
            ## response of Z-field
                ## to X-dipole
                    ## to zenithal field
                invert_tmp[stat_i,2,0,freq_i] = -Ji00*sin_stat_zenith
                    ## no azimuthal response
                
                ## to Y-dipole
                    ## to zenithal field
                invert_tmp[stat_i,2,1,freq_i] = -Ji01*sin_stat_zenith
                    ## no azimuthal response
                    
    
        
        
    def get_correctionMatrix(self, np.ndarray[double complex, ndim=2] out=None, np.ndarray[double, ndim=1] loc_to_use=None ):
        """note, returned matrix is stored as complex, despite being purly real"""
        
        cdef double[:] location
        if loc_to_use is None:
            location = self.center_XYZ
        else:
            location = loc_to_use
        
        if out is not None:
            if out.shape[0]!=3 or out.shape[1]!=3:
                print('shape error in get_centerOPol_matrix')
                quit()
            out[:,:] = 0.0
        else:
            out = np.zeros((3,3), dtype=np.cdouble)
            
        cdef long ant_i
        cdef double[:] unit_vector = np.empty(3, dtype=np.double)
        cdef double dx
        cdef double dy
        cdef double dz
        cdef double OPol_weight
        
        cdef double R2
        cdef double R
            
            
        for ant_i in range( self.num_antennas ):
            
            if not self.antenna_mask[ant_i]:
                continue
            
            if self.antenna_polarizations[ant_i] == 1: ## we really only want this for one antenna!!
                continue
            
            ## this is from SOURCE to ANTENNA
            dx = self.antenna_locations[ant_i, 0] - location[0]
            dy = self.antenna_locations[ant_i, 1] - location[1]
            dz = self.antenna_locations[ant_i, 2] - location[2]
            
            R2 = ( dx*dx + dy*dy + dz*dz )
            R = sqrt( R2 )
            
            unit_vector[0] = dx/R
            unit_vector[1] = dy/R
            unit_vector[2] = dz/R
            
            OPol_weight = self.antenna_weights[ant_i]/R2
            
            out[0,0] += OPol_weight
            out[1,1] += OPol_weight
            out[2,2] += OPol_weight
            
            for i in range(3):
                for j in range(3):
                    out[i,j] -= unit_vector[i]*unit_vector[j]*OPol_weight
            
        return out
    
    def set_antennaWeight(self, long ant_i, double ant_weight):
        self.antenna_weights[ant_i] = ant_weight
        
    def set_antennaData_zero(self, long ant_I):
        self.antenna_mask[ant_I] = 0
        
    def turn_on_all_antennas(self):
        cdef long ant_i
        for ant_i in range(self.num_antennas):
            self.antenna_mask[ant_i] = 1
        
    def turn_off_all_antennas(self):
        cdef long ant_i
        for ant_i in range(self.num_antennas):
            self.antenna_mask[ant_i] = 0
    
    def set_antennaData(self, long ant_I, np.ndarray[double complex, ndim=1] data, int freq_mode=1):
        self.antenna_mask[ant_I] = 1
        
        cdef long length
        cdef void* wavetable
        cdef void* workspace
        cdef long num_freqs
        cdef long freq_start_index
        
        if freq_mode == 1:
            length = self.trace_length
            wavetable = self.FFT_wavetable
            workspace = self.FFT_workspace
            num_freqs = self.num_used_frequencies 
            freq_start_index = self.freq_start_index
        elif freq_mode == 2:
            length = self.trace_length_secondary
            wavetable = self.FFT_wavetable_secondary
            workspace = self.FFT_workspace_secondary
            num_freqs = self.num_used_frequencies_secondary
            freq_start_index = self.freq_start_index_secondary
        else:
            
            length = 0
            # wavetable = 0
            # workspace = 0
            num_freqs = 0
            freq_start_index = 0
            
            print('bad freq mode')
            quit()
        
        cdef int info
        info = gsl_fft_complex_forward( <double*>&data[0], 1, length, wavetable, workspace)
        if info != 0:
            print('GSL FFT problem!', info)
            quit()
            
            
        cdef long Fi
        for Fi in range( num_freqs ):
            self.ant_data[ant_I,Fi] = data[num_freqs+Fi]
            
            
    def partial_inverse_FFT(self, np.ndarray[double complex, ndim=1] in_data, np.ndarray[double complex, ndim=1] out_data=None,
                            int freq_mode=1):
        
        cdef long length
        cdef void* wavetable
        cdef void* workspace
        cdef long num_freqs
        cdef long freq_start_index
        
        if freq_mode == 1:
            length = self.trace_length
            wavetable = self.FFT_wavetable
            workspace = self.FFT_workspace
            num_freqs = self.num_used_frequencies 
            freq_start_index = self.freq_start_index
        elif freq_mode == 2:
            length = self.trace_length_secondary
            wavetable = self.FFT_wavetable_secondary
            workspace = self.FFT_workspace_secondary
            num_freqs = self.num_used_frequencies_secondary
            freq_start_index = self.freq_start_index_secondary
        else:
            
            length = 0
            # wavetable = 0
            # workspace = 0
            num_freqs = 0
            freq_start_index = 0
            
            print('bad freq mode')
            quit()
        
        
        
        if out_data is None:
            out_data = np.empty(length, dtype=np.cdouble)
        else:
            if not out_data.flags['C_CONTIGUOUS']:
                print('C-contiguous eror in partial_inverse_FFT')
                quit()
            if  (out_data.shape[0] != length) or  (out_data.ndim != 1) :
                print('length output error in partial_inverse_FFT')
                quit()
            if (in_data.shape[0] != num_freqs) or  (in_data.ndim != 1) :
                print('input length error in partial_inverse_FFT')
                quit()
        
        out_data[:] = 0.0
        cdef long Fi
        for Fi in range( num_freqs ):
            out_data[freq_start_index+Fi] = in_data[Fi]
            
        cdef int info
        info = gsl_fft_complex_inverse( <double*>&out_data[0], 1, length, wavetable, workspace)
        if info != 0:
            print('GSL FFT problem!', info)
            quit()
            
        return out_data
    
    
    
    def full_image(self, np.ndarray[double complex, ndim=5] pol_image=None, int print_progress=0,
                            np.ndarray[double, ndim=1] frequency_weights=None, int freq_mode=1):
        """return F-vector for every pixel and frequency. i.e.: [xi,yi,zi,fi, component]"""
        
        cdef long num_pixels = self.num_X*self.num_Y*self.num_Z
        
        
        cdef long num_freqs
        cdef double complex[:,:,:,:] eField_inversion
        cdef double first_frequency
        cdef double delta_frequency
        
        if freq_mode == 1:
            num_freqs = self.num_used_frequencies 
            eField_inversion = self.eField_inversion_jonesMarix
            first_frequency = self.all_frequencies[self.freq_start_index]
            delta_frequency = self.all_frequencies[1] - self.all_frequencies[0]
            
        elif freq_mode == 2:
            num_freqs = self.num_used_frequencies_secondary
            eField_inversion = self.eField_inversion_jonesMarix_secondary
            first_frequency = self.all_frequencies_secondary[self.freq_start_index_secondary]
            delta_frequency = self.all_frequencies_secondary[1] - self.all_frequencies_secondary[0]
            
        else:
            num_freqs = 0
            # eField_inversion = 0
            first_frequency = 0
            delta_frequency = 0
            
            print('bad freq mode')
            quit()
        
        
        
        
        if pol_image is None:
            pol_image = np.zeros( (self.num_X,self.num_Y,self.num_Z, num_freqs,3), dtype=np.cdouble )
        else:
            if not pol_image.flags['C_CONTIGUOUS']:
                print('error A in total_Image!')
                quit()
            if (pol_image.shape[0] != self.num_X) or (pol_image.shape[1] != self.num_Y) or (pol_image.shape[2] != self.num_Z) or \
                (pol_image.shape[3] != num_freqs) or (pol_image.shape[4] != 3):
                print('shape error A in total_Image' )
                quit()
                
            pol_image[:] = 0.0
            
        if frequency_weights is not None:
            if not frequency_weights.flags['C_CONTIGUOUS']:
                print('frequency_weights must be c-contiguous!')
                quit()
            if frequency_weights.shape[0] != num_freqs:
                print('frequency_weights has wrong length')
                quit()
                
        
        cdef double complex* pol_image_accses = &pol_image[0,0,0,0,0]
        
        
        cdef double complex* Xtmp_inversion
        cdef double complex* Ytmp_inversion
        cdef double complex* Ztmp_inversion
        
        cdef double complex* ant_data_accses = &self.ant_data[0, 0]
        
        #cdef double* geo_delay_accses = &self.geometric_delays[0]
        cdef double* cal_shifts_accses = &self.cal_shifts[0]
        
        #cdef double* DT_grid_accses = &self.geometric_DT_grid[0,0,0,0]
        #cdef double* distance_accses = &self.distance_grid[0,0,0,0] # station, X,Y,Z
        
        cdef double* rel_geo_delay_accses = &self.relative_geo_delayDif[0,0,0]
        
        cdef double* antenna_loc_accses =  &self.antenna_locations[0,0]
        cdef double* XVoxel_accses = &self.X_array[0]
        cdef double* YVoxel_accses = &self.Y_array[0]
        cdef double* ZVoxel_accses = &self.Z_array[0]
        
        cdef long* stati_to_anti_accses = &self.stati_to_anti[0]    
        cdef int* ant_pol_accses = &self.antenna_polarizations[0]
    

        cdef long pixel_Xi
        cdef long pixel_Yi
        cdef long pixel_Zi
        
        cdef double ant_X
        cdef double ant_Y
        cdef double ant_Z
        
        cdef double pixel_dX_sq
        cdef double pixel_dY_sq
        cdef double pixel_dZ_sq
        
        
        cdef long pixel_i
        cdef long Fi
        cdef long stat_i
        cdef long ant_i
        
        cdef int pol
        cdef double center_delay
        cdef double total_delay
        cdef double complex phase
        cdef double complex delta_phase
        cdef double complex J2pi = 2j*M_PI
        #cdef double geo_delay
        #cdef double delay_delta
        
        cdef double complex phased_data
        cdef double antenna_distance_inv
        
        cdef long antenna_step
        cdef long pixel_step
        cdef long F_step
        
        
        # cdef long i
        # cdef long j
        
        for stat_i in range(self.num_stations):
                        
            
            for ant_i in range( stati_to_anti_accses[stat_i], stati_to_anti_accses[stat_i+1] ):
                
                if not self.antenna_mask[ant_i]:
                    continue
                
                if print_progress:
                    print(' ant', ant_i, '/', self.num_antennas, " "*10, end='\r')
                
                pol = ant_pol_accses[ant_i]
                #geo_delay = geo_delay_accses[ ant_i ] 
                #center_delay = geo_delay + cal_shifts_accses[ant_i]
                center_delay = cal_shifts_accses[ant_i]
                
                antenna_step = ant_i*self.num_data_frequencies
                
                Xtmp_inversion = &eField_inversion[stat_i, 0, pol, 0] #station, direction(X,Y,Z), dipole (X,Y), frequency
                Ytmp_inversion = &eField_inversion[stat_i, 1, pol, 0]
                Ztmp_inversion = &eField_inversion[stat_i, 2, pol, 0]
                
                ant_X = antenna_loc_accses[ ant_i*3 + 0 ]
                ant_Y = antenna_loc_accses[ ant_i*3 + 1 ]
                ant_Z = antenna_loc_accses[ ant_i*3 + 2 ]
                
                for pixel_Xi in range(self.num_X):
                    pixel_dX_sq = XVoxel_accses[pixel_Xi] - ant_X
                    pixel_dX_sq *= pixel_dX_sq
                    
                    for pixel_Yi in range(self.num_Y):
                        pixel_dY_sq = YVoxel_accses[pixel_Yi] - ant_Y
                        pixel_dY_sq *= pixel_dY_sq
                        
                        for pixel_Zi in range(self.num_Z):
                            pixel_dZ_sq = ZVoxel_accses[pixel_Zi] - ant_Z
                            pixel_dZ_sq *= pixel_dZ_sq
                            
                            pixel_i = pixel_Xi*self.num_Y*self.num_Z + pixel_Yi*self.num_Z + pixel_Zi
                            
                            antenna_distance_inv = sqrt( pixel_dX_sq + pixel_dY_sq + pixel_dZ_sq ) ## not inverted yet!
                            
                            #delay_delta = DT_grid_accses[ ant_i*num_pixels + pixel_i ]
                            #total_delay = center_delay + delay_delta
                            # total_delay = center_delay + antenna_distance_inv*c_air_inverse
                            total_delay = center_delay + antenna_distance_inv*c_air_inverse - rel_geo_delay_accses[ pixel_i ]
                            
                            # antenna_distance_inv = 1/distance_accses[ant_i*num_pixels + pixel_i]
                            antenna_distance_inv = 1.0/antenna_distance_inv
                                    
                            
                            ### sum data
                            pixel_step = pixel_i*num_freqs
                            
                            phase = self.antenna_weights[ant_i]*cexp( J2pi*first_frequency*total_delay )*antenna_distance_inv
                            delta_phase = cexp( J2pi*delta_frequency*total_delay )
                            for Fi in range( num_freqs ):
                                
                                phased_data = phase*ant_data_accses[ antenna_step + Fi ]
                                
                                F_step = (pixel_step + Fi)*3
                                pol_image_accses[F_step    ] += Xtmp_inversion[Fi]*phased_data
                                pol_image_accses[F_step + 1] += Ytmp_inversion[Fi]*phased_data
                                pol_image_accses[F_step + 2] += Ztmp_inversion[Fi]*phased_data
                                
                                # if (Fi == 9) and (pixel_i==505050):
                                #     print('anti', ant_i)
                                #     print('  delay', center_delay, delay_delta)
                                #     print('  pd, abs phase:', cabs(phased_data), carg(phased_data) )
                                #     print('  X:', Xtmp_inversion[Fi]*phased_data)
                                #     print('  Y:', Ytmp_inversion[Fi]*phased_data)
                                
                                phase *= delta_phase

        
        cdef double[:] weight_accses
        cdef double W
        if frequency_weights is not None:
            weight_accses = frequency_weights
            
            for Fi in range( num_freqs ):
                W = weight_accses[Fi]
                
                for pixel_i in range(num_pixels):
                    F_step = (pixel_i*num_freqs +Fi )*3
                    pol_image_accses[ F_step+0 ] *= W
                    pol_image_accses[ F_step+1 ] *= W
                    pol_image_accses[ F_step+2 ] *= W

                
        if print_progress:
            print(' done!', " "*10)
        return pol_image
                        
    def ChunkedIntensity_Image(self, long first_sample, long chunk_size, long number_chunks, np.ndarray[double, ndim=4] image=None, int print_progress=0,
                            np.ndarray[double, ndim=1] frequency_weights=None, int freq_mode=1):
        """return intensity for every pixel and chunk. i.e.: [xi,yi,zi, chunk]"""
        
        
        print('dooglebosh')
        
        cdef long length ## total length of the time trace
        cdef long num_freqs
        cdef double complex[:,:,:,:] eField_inversion
        cdef long freq_start_index
        cdef double first_frequency
        cdef double delta_frequency
        
        cdef double complex* FFT_X_accses
        cdef double complex* FFT_Y_accses
        cdef double complex* FFT_Z_accses
        
        cdef void* wavetable
        cdef void* workspace
        
        if freq_mode == 1:
            length = self.trace_length
            num_freqs = self.num_used_frequencies 
            freq_start_index = self.freq_start_index
            eField_inversion = self.eField_inversion_jonesMarix
            first_frequency = self.all_frequencies[ freq_start_index ]
            delta_frequency = self.all_frequencies[1] - self.all_frequencies[0]
            
            if not self.FFT_workspace_set:
                self.FFT_X_workspace = np.empty(self.trace_length, dtype=np.cdouble)
                self.FFT_Y_workspace = np.empty(self.trace_length, dtype=np.cdouble)
                self.FFT_Z_workspace = np.empty(self.trace_length, dtype=np.cdouble)
                self.FFT_workspace_set = True 
                
            FFT_X_accses = &self.FFT_X_workspace[0]
            FFT_Y_accses = &self.FFT_Y_workspace[0]
            FFT_Z_accses = &self.FFT_Z_workspace[0]
            
            wavetable = self.FFT_wavetable
            workspace = self.FFT_workspace
            
        elif freq_mode == 2:
            length = self.trace_length_secondary
            num_freqs = self.num_used_frequencies_secondary
            freq_start_index = self.freq_start_index_secondary
            eField_inversion = self.eField_inversion_jonesMarix_secondary
            first_frequency = self.all_frequencies_secondary[ freq_start_index ]
            delta_frequency = self.all_frequencies_secondary[1] - self.all_frequencies_secondary[0]
            
            if not self.FFT_workspace_set_secondary:
                self.FFT_X_workspace_secondary = np.empty(self.trace_length_secondary, dtype=np.cdouble)
                self.FFT_Y_workspace_secondary = np.empty(self.trace_length_secondary, dtype=np.cdouble)
                self.FFT_Z_workspace_secondary = np.empty(self.trace_length_secondary, dtype=np.cdouble)
                self.FFT_workspace_set_secondary = True 
                
            FFT_X_accses = &self.FFT_X_workspace_secondary[0]
            FFT_Y_accses = &self.FFT_Y_workspace_secondary[0]
            FFT_Z_accses = &self.FFT_Z_workspace_secondary[0]
            
            wavetable = self.FFT_wavetable_secondary
            workspace = self.FFT_workspace_secondary
            
        else:
            num_freqs = 0
            # eField_inversion = 0
            first_frequency = 0
            delta_frequency = 0
            
            print('bad freq mode')
            quit()
        
        
        if (first_sample + chunk_size*number_chunks) > length:
            print('WARNING: total length to sum over is longer then actual length in ChunkedIntensity_Image')
        
        
        if image is None:
            image = np.zeros( (self.num_X,self.num_Y,self.num_Z, number_chunks), dtype=np.double )
        else:
            if not image.flags['C_CONTIGUOUS']:
                print('error A in total_Image!')
                quit()
            if (image.shape[0] != self.num_X) or (image.shape[1] != self.num_Y) or (image.shape[2] != self.num_Z) or \
                (image.shape[3] != number_chunks):
                print('shape error A in total_Image' )
                quit()
                
            image[:] = 0.0
        
        cdef double* weight_accses
        cdef int do_weighting = 0
        if frequency_weights is not None:
            if not frequency_weights.flags['C_CONTIGUOUS']:
                print('frequency_weights must be c-contiguous!')
                quit()
            if frequency_weights.shape[0] != num_freqs:
                print('frequency_weights has wrong length')
                quit()
                
            weight_accses = &frequency_weights[0]
            do_weighting = 1

           
        
        cdef double* image_accses = &image[0,0,0,0]

        
        
        cdef double complex* Xtmp_inversion
        cdef double complex* Ytmp_inversion
        cdef double complex* Ztmp_inversion
        
        cdef double complex* ant_data_accses = &self.ant_data[0, 0]
        
        cdef double* cal_shifts_accses = &self.cal_shifts[0]
        
        cdef double* antenna_loc_accses =  &self.antenna_locations[0,0]
        cdef double* XVoxel_accses = &self.X_array[0]
        cdef double* YVoxel_accses = &self.Y_array[0]
        cdef double* ZVoxel_accses = &self.Z_array[0]
        
        cdef long* stati_to_anti_accses = &self.stati_to_anti[0]    
        cdef int* ant_pol_accses = &self.antenna_polarizations[0]
        

        cdef long pixel_Xi
        cdef long pixel_Yi
        cdef long pixel_Zi
        
        cdef double ant_X
        cdef double ant_Y
        cdef double ant_Z
        
        cdef double pixel_X
        cdef double pixel_Y
        cdef double pixel_Z
        
        cdef double pixel_dX_sq
        cdef double pixel_dY_sq
        cdef double pixel_dZ_sq
        
        cdef long pixel_i
        cdef long Fi
        cdef long stat_i
        cdef long ant_i
        
        cdef int pol
        cdef double total_delay
        cdef double complex phase
        cdef double complex delta_phase
        cdef double complex J2pi = 2j*M_PI
        cdef double delay_delta
        
        cdef double complex phased_data
        cdef double antenna_distance_inv
        
        cdef long antenna_step
        #cdef long pixel_step
        cdef long F_step
        
        
        # cdef long i
        # cdef long j
        
        cdef double W
        cdef int FFTinfo
        
        
        cdef long chunk_i
        cdef long CS_i
        cdef long sample_i
        # cdef long first_sample = <long>( (length-image_size)/2 )
        cdef double complex X_tmp
        cdef double complex Y_tmp
        cdef double complex Z_tmp
        cdef double rel_geo_delay
        for pixel_Xi in range(self.num_X):
            if print_progress:
                print(' Xi', pixel_Xi, '/', self.num_X, " "*10, end='\r')
            
            pixel_X = XVoxel_accses[pixel_Xi]
            
            for pixel_Yi in range(self.num_Y):
                pixel_Y = YVoxel_accses[pixel_Yi]
                
                for pixel_Zi in range(self.num_Z):
                    pixel_Z = ZVoxel_accses[pixel_Zi]
                    
                    pixel_i = pixel_Xi*self.num_Y*self.num_Z + pixel_Yi*self.num_Z + pixel_Zi
                    rel_geo_delay = self.relative_geo_delayDif[pixel_Xi, pixel_Yi, pixel_Zi]
                    
                    for Fi in range(length):
                        FFT_X_accses[Fi] = 0
                        FFT_Y_accses[Fi] = 0
                        FFT_Z_accses[Fi] = 0
            
                    for stat_i in range(self.num_stations):
                        for ant_i in range( stati_to_anti_accses[stat_i], stati_to_anti_accses[stat_i+1] ):
                            
                            if not self.antenna_mask[ant_i]:
                                continue
                            
                            pol = ant_pol_accses[ant_i]
                            
                            antenna_step = ant_i*self.num_data_frequencies
                            
                            Xtmp_inversion = &eField_inversion[stat_i, 0, pol, 0] #station, direction(X,Y,Z), dipole (X,Y), frequency
                            Ytmp_inversion = &eField_inversion[stat_i, 1, pol, 0]
                            Ztmp_inversion = &eField_inversion[stat_i, 2, pol, 0]
                            
                            
                            
                            ant_X = antenna_loc_accses[ ant_i*3 + 0 ]
                            ant_Y = antenna_loc_accses[ ant_i*3 + 1 ]
                            ant_Z = antenna_loc_accses[ ant_i*3 + 2 ]
                    
                            
                            
                            pixel_dX_sq = pixel_X-ant_X
                            pixel_dX_sq *= pixel_dX_sq
                            
                            pixel_dY_sq = pixel_Y-ant_Y
                            pixel_dY_sq *= pixel_dY_sq
                            
                            pixel_dZ_sq = pixel_Z-ant_Z
                            pixel_dZ_sq *= pixel_dZ_sq
                            
                            antenna_distance_inv = sqrt( pixel_dX_sq + pixel_dY_sq + pixel_dZ_sq )
                            total_delay = cal_shifts_accses[ant_i] + antenna_distance_inv*c_air_inverse - rel_geo_delay
                            antenna_distance_inv = 1.0/antenna_distance_inv
                                    
                            
                            ### sum data
                            #pixel_step = pixel_i*self.num_used_frequencies 
                            
                            phase = self.antenna_weights[ant_i]*cexp( J2pi*first_frequency*total_delay )*antenna_distance_inv
                            delta_phase = cexp( J2pi*delta_frequency*total_delay )
                            for Fi in range( num_freqs ):
                                
                                phased_data = phase*ant_data_accses[ antenna_step + Fi ]
                                
                                F_step = Fi + freq_start_index
                                
                                FFT_X_accses[F_step] += Xtmp_inversion[Fi]*phased_data
                                FFT_Y_accses[F_step] += Ytmp_inversion[Fi]*phased_data
                                FFT_Z_accses[F_step] += Ztmp_inversion[Fi]*phased_data
                                
                                phase *= delta_phase
                                
                                
                    if do_weighting:
                        for Fi in range( num_freqs ):
                            F_step = Fi + freq_start_index
                            W = weight_accses[Fi]
                            FFT_X_accses[F_step] *= W
                            FFT_Y_accses[F_step] *= W
                            FFT_Z_accses[F_step] *= W
                                
                    
                    FFTinfo = gsl_fft_complex_inverse( <double*>FFT_X_accses, 1, length, wavetable, workspace )
                    if FFTinfo != 0:
                        print('IFFT BADNESS A!')
                        quit()
                    
                    FFTinfo = gsl_fft_complex_inverse( <double*>FFT_Y_accses, 1, length, wavetable, workspace )
                    if FFTinfo != 0:
                        print('IFFT BADNESS B!')
                        quit()
                        
                    FFTinfo = gsl_fft_complex_inverse( <double*>FFT_Z_accses, 1, length, wavetable, workspace )
                    if FFTinfo != 0:
                        print('IFFT BADNESS C!')
                        quit()
        
        
                    
                    for chunk_i in range(number_chunks):
                        for CS_i in range(chunk_size):
                            sample_i = first_sample + chunk_i*chunk_size + CS_i
                            
                            X_tmp = FFT_X_accses[ sample_i ]
                            Y_tmp = FFT_Y_accses[ sample_i ]
                            Z_tmp = FFT_Z_accses[ sample_i ]
                            
                            image_accses[ pixel_i*number_chunks + chunk_i ] += X_tmp.real*X_tmp.real + X_tmp.imag*X_tmp.imag + Y_tmp.real*Y_tmp.real + \
                                Y_tmp.imag*Y_tmp.imag + Z_tmp.real*Z_tmp.real + Z_tmp.imag*Z_tmp.imag
         
                
        if print_progress:
            print(' done!', " "*10)
        return image
    
    def Image_at_Spot(self, double X, double Y, double Z, np.ndarray[double complex, ndim=2] image=None,
                            np.ndarray[double, ndim=1] frequency_weights=None, int freq_mode=1, 
                            np.ndarray[double complex, ndim=2] matrix=None):
        """return intensity for every pixel and chunk. i.e.: [xi,yi,zi, chunk]"""
        
        # cdef long length
        cdef long num_freqs
        cdef double complex[:,:,:,:] eField_inversion
        # cdef long freq_start_index
        cdef double first_frequency
        cdef double delta_frequency
        
        # cdef double complex* FFT_X_accses
        # cdef double complex* FFT_Y_accses
        # cdef double complex* FFT_Z_accses
        
        # cdef void* wavetable
        # cdef void* workspace
        
        if freq_mode == 1:
            # length = self.trace_length
            num_freqs = self.num_used_frequencies 
            # freq_start_index = self.freq_start_index
            eField_inversion = self.eField_inversion_jonesMarix
            first_frequency = self.all_frequencies[ self.freq_start_index ]
            delta_frequency = self.all_frequencies[1] - self.all_frequencies[0]
            
            # if not self.FFT_workspace_set:
            #     self.FFT_X_workspace = np.empty(self.trace_length, dtype=np.cdouble)
            #     self.FFT_Y_workspace = np.empty(self.trace_length, dtype=np.cdouble)
            #     self.FFT_Z_workspace = np.empty(self.trace_length, dtype=np.cdouble)
            #     self.FFT_workspace_set = True 
                
            # FFT_X_accses = &self.FFT_X_workspace[0]
            # FFT_Y_accses = &self.FFT_Y_workspace[0]
            # FFT_Z_accses = &self.FFT_Z_workspace[0]
            
            # wavetable = self.FFT_wavetable
            # workspace = self.FFT_workspace
            
        elif freq_mode == 2:
            # length = self.trace_length_secondary
            num_freqs = self.num_used_frequencies_secondary
            # freq_start_index = self.freq_start_index_secondary
            eField_inversion = self.eField_inversion_jonesMarix_secondary
            first_frequency = self.all_frequencies_secondary[ self.freq_start_index_secondary ]
            delta_frequency = self.all_frequencies_secondary[1] - self.all_frequencies_secondary[0]
            
            # if not self.FFT_workspace_set_secondary:
            #     self.FFT_X_workspace_secondary = np.empty(self.trace_length_secondary, dtype=np.cdouble)
            #     self.FFT_Y_workspace_secondary = np.empty(self.trace_length_secondary, dtype=np.cdouble)
            #     self.FFT_Z_workspace_secondary = np.empty(self.trace_length_secondary, dtype=np.cdouble)
            #     self.FFT_workspace_set_secondary = True 
                
            # FFT_X_accses = &self.FFT_X_workspace_secondary[0]
            # FFT_Y_accses = &self.FFT_Y_workspace_secondary[0]
            # FFT_Z_accses = &self.FFT_Z_workspace_secondary[0]
            
            # wavetable = self.FFT_wavetable_secondary
            # workspace = self.FFT_workspace_secondary
            
        else:
            num_freqs = 0
            # eField_inversion = 0
            first_frequency = 0
            delta_frequency = 0
            
            print('bad freq mode')
            quit()
        
        
        
        if image is None:
            image = np.zeros( (num_freqs,3), dtype=np.cdouble )
        else:
            if not image.flags['C_CONTIGUOUS']:
                print('error A in total_Image!')
                quit()
            if (image.shape[0] != num_freqs) or (image.shape[1] != 3):
                print('shape error A in total_Image' )
                quit()
                
            image[:] = 0.0
        
        
        
        cdef double* weight_accses
        cdef int do_weighting = 0
        if frequency_weights is not None:
            if not frequency_weights.flags['C_CONTIGUOUS']:
                print('frequency_weights must be c-contiguous!')
                quit()
            if frequency_weights.shape[0] != num_freqs:
                print('frequency_weights has wrong length')
                quit()
                
            weight_accses = &frequency_weights[0]
            do_weighting = 1

           
        cdef int do_matrix = 0
        cdef double complex M00
        cdef double complex M01
        cdef double complex M02
        cdef double complex M10
        cdef double complex M11
        cdef double complex M12
        cdef double complex M20
        cdef double complex M21
        cdef double complex M22
        if matrix is not None:
            if (matrix.shape[0]) != 3 or (matrix.shape[1] != 3)  :
                print('matrix is wrong shape!')
                quit()
            do_matrix = 1
            M00 = matrix[0,0]
            M01 = matrix[0,1]
            M02 = matrix[0,2]
            M10 = matrix[1,0]
            M11 = matrix[1,1]
            M12 = matrix[1,2]
            M20 = matrix[2,0]
            M21 = matrix[2,1]
            M22 = matrix[2,2]


        
        cdef double complex* image_accses = &image[0,0]

        cdef double complex* Xtmp_inversion
        cdef double complex* Ytmp_inversion
        cdef double complex* Ztmp_inversion
        
        cdef double complex* ant_data_accses = &self.ant_data[0, 0]
        
        cdef double* cal_shifts_accses = &self.cal_shifts[0]
        
        cdef double* antenna_loc_accses =  &self.antenna_locations[0,0]
        cdef double* XVoxel_accses = &self.X_array[0]
        cdef double* YVoxel_accses = &self.Y_array[0]
        cdef double* ZVoxel_accses = &self.Z_array[0]
        
        cdef long* stati_to_anti_accses = &self.stati_to_anti[0]    
        cdef int* ant_pol_accses = &self.antenna_polarizations[0]
        

        cdef long pixel_Xi
        cdef long pixel_Yi
        cdef long pixel_Zi
        
        cdef double ant_X
        cdef double ant_Y
        cdef double ant_Z
        
        cdef double pixel_X
        cdef double pixel_Y
        cdef double pixel_Z
        
        cdef double pixel_dX_sq
        cdef double pixel_dY_sq
        cdef double pixel_dZ_sq
        
        cdef long pixel_i
        cdef long Fi
        cdef long stat_i
        cdef long ant_i
        
        cdef int pol
        cdef double total_delay
        cdef double complex phase
        cdef double complex delta_phase
        cdef double complex J2pi = 2j*M_PI
        cdef double delay_delta
        
        cdef double complex phased_data
        cdef double antenna_distance_inv
        
        cdef long antenna_step
        #cdef long pixel_step
        cdef long F_step
        
        
        
        
        cdef double dx = self.reference_XYZ[ 0] - self.center_XYZ[0]
        cdef double dy = self.reference_XYZ[ 1] - self.center_XYZ[1]
        cdef double dz = self.reference_XYZ[ 2] - self.center_XYZ[2]
        cdef double center_R = sqrt( dx*dx + dy*dy + dz*dz )
        
        dx = self.reference_XYZ[ 0] - X
        dy = self.reference_XYZ[ 1] - Y
        dz = self.reference_XYZ[ 2] - Z
        cdef double rel_geo_delay = ( sqrt( dx*dx + dy*dy + dz*dz ) - center_R )*c_air_inverse
        
        



        for stat_i in range(self.num_stations):
            for ant_i in range( stati_to_anti_accses[stat_i], stati_to_anti_accses[stat_i+1] ):
                
                if not self.antenna_mask[ant_i]:
                    continue
                
                pol = ant_pol_accses[ant_i]
                
                antenna_step = ant_i*self.num_data_frequencies
                
                Xtmp_inversion = &eField_inversion[stat_i, 0, pol, 0] #station, direction(X,Y,Z), dipole (X,Y), frequency
                Ytmp_inversion = &eField_inversion[stat_i, 1, pol, 0]
                Ztmp_inversion = &eField_inversion[stat_i, 2, pol, 0]
                
                
                
                ant_X = antenna_loc_accses[ ant_i*3 + 0 ]
                ant_Y = antenna_loc_accses[ ant_i*3 + 1 ]
                ant_Z = antenna_loc_accses[ ant_i*3 + 2 ]
        
                
                
                pixel_dX_sq = X-ant_X
                pixel_dX_sq *= pixel_dX_sq
                
                pixel_dY_sq = Y-ant_Y
                pixel_dY_sq *= pixel_dY_sq
                
                pixel_dZ_sq = Z-ant_Z
                pixel_dZ_sq *= pixel_dZ_sq
                
                antenna_distance_inv = sqrt( pixel_dX_sq + pixel_dY_sq + pixel_dZ_sq )
                total_delay = cal_shifts_accses[ant_i] + antenna_distance_inv*c_air_inverse - rel_geo_delay
                antenna_distance_inv = 1/antenna_distance_inv
                        
                
                ### sum data
                #pixel_step = pixel_i*self.num_used_frequencies 
                
                phase = self.antenna_weights[ant_i]*cexp( J2pi*first_frequency*total_delay )*antenna_distance_inv
                delta_phase = cexp( J2pi*delta_frequency*total_delay )
                for Fi in range( num_freqs ):
                    
                    phased_data = phase*ant_data_accses[ antenna_step + Fi ]
                    
                    F_step = Fi*3
                    
                    image_accses[F_step + 0] += Xtmp_inversion[Fi]*phased_data
                    image_accses[F_step + 1] += Ytmp_inversion[Fi]*phased_data
                    image_accses[F_step + 2] += Ztmp_inversion[Fi]*phased_data
                    
                    phase *= delta_phase
                    
             
        cdef double W
        cdef double complex A
        cdef double complex B
        cdef double complex C
        if do_weighting or do_matrix:
            for Fi in range( num_freqs ):
                F_step = Fi*3
                
                A = image_accses[F_step + 0]
                B = image_accses[F_step + 1]
                C = image_accses[F_step + 2]
                
                if do_weighting:
                    W = weight_accses[Fi]
                    A *= W
                    B *= W
                    C *= W
                    
                if do_matrix:
                    image_accses[F_step + 0] = M00*A + M01*B + M02*C
                    image_accses[F_step + 1] = M10*A + M11*B + M12*C
                    image_accses[F_step + 2] = M20*A + M21*B + M22*C
                else:
                    image_accses[F_step + 0] = A
                    image_accses[F_step + 1] = B
                    image_accses[F_step + 2] = C
                    
                                
                
        # if print_progress:
            # print(' done!', " "*10)
        return image


##### SOME TOOLS ####

def get_total_emistivity(np.ndarray[double complex, ndim=5] total_image, np.ndarray[double, ndim=3] image_out=None):
    """get power per pixel. Add squares of all frequencies and polarizations. And the location of the maximum"""

    
    cdef long Nx = total_image.shape[0]
    cdef long Ny = total_image.shape[1]
    cdef long Nz = total_image.shape[2]
    cdef long Nf = total_image.shape[3]
    cdef long Np = total_image.shape[4]
    
    if image_out is None:
        image_out = np.empty( (Nx, Ny, Nz), dtype=np.double )
        
    cdef long best_Xi = 0
    cdef long best_Yi = 0
    cdef long best_Zi = 0
    cdef double best_I = 0
        
    cdef long Xi
    cdef long Yi
    cdef long Zi
    cdef long Fi
    cdef long Pi
    
    cdef double current_sum
    cdef double complex im
    for Xi in range(Nx):
        for Yi in range(Ny):
            for Zi in range(Nz):
                
                current_sum = 0.0
                
                for Fi in range(Nf):
                    
                    for Pi in range(Np):
                        im = total_image[Xi,Yi,Zi,Fi,Pi]
                        current_sum += im.real*im.real + im.imag*im.imag
                    
                if current_sum > best_I:
                    best_I = current_sum
                    best_Xi = Xi
                    best_Yi = Yi
                    best_Zi = Zi
                    
                image_out[Xi,Yi,Zi] = current_sum
                
    cdef np.ndarray[int, ndim=1] peak = np.array([best_Xi,best_Yi,best_Zi], dtype=np.int32)
    return image_out, peak


def get_peak_loc(np.ndarray[double, ndim=3] image):
    """get power per pixel. Add squares of all frequencies and polarizations. And the location of the maximum"""

    cdef long Nx = image.shape[0]
    cdef long Ny = image.shape[1]
    cdef long Nz = image.shape[2]
        
    cdef long best_Xi = 0
    cdef long best_Yi = 0
    cdef long best_Zi = 0
    cdef double best_I = 0
        
    cdef long Xi
    cdef long Yi
    cdef long Zi
    
    for Xi in range(Nx):
        for Yi in range(Ny):
            for Zi in range(Nz):
                
                I = image[Xi, Yi, Zi]
                if I > best_I:
                    best_Xi = Xi
                    best_Yi = Yi
                    best_Zi = Zi
                    best_I = I
                
    cdef np.ndarray[long, ndim=1] peak = np.array([best_Xi,best_Yi,best_Zi], dtype=np.int)
    return peak

def total_2dnorm_squared(np.ndarray[double complex, ndim=2] total_image):
    cdef long N1 = total_image.shape[0]
    cdef long N2 = total_image.shape[1]
    
    cdef long i
    cdef long j
    cdef double out = 0
    cdef double R = 0
    cdef double I = 0
    for i in range(N1):
        for j in range(N2):
            R = total_image[i,j].real
            I = total_image[i,j].imag
            out += R*R + I*I
    return out

def total_1dnorm_squared(np.ndarray[double complex, ndim=1] total_image):
    cdef long N1 = total_image.shape[0]
    
    cdef long i
    cdef double out = 0
    cdef double R = 0
    cdef double I = 0
    for i in range(N1):
        R = total_image[i].real
        I = total_image[i].imag
        out += R*R + I*I
    return out

def apply_matrix_in_place(np.ndarray[double complex, ndim=2] image, np.ndarray[double complex, ndim=2] matrix):
    ## for fast accses
    cdef double complex M00 = matrix[0,0]
    cdef double complex M01 = matrix[0,1]
    cdef double complex M02 = matrix[0,2]
    cdef double complex M10 = matrix[1,0]
    cdef double complex M11 = matrix[1,1]
    cdef double complex M12 = matrix[1,2]
    cdef double complex M20 = matrix[2,0]
    cdef double complex M21 = matrix[2,1]
    cdef double complex M22 = matrix[2,2]
    
    cdef long Fi
    cdef double complex A
    cdef double complex B
    cdef double complex C
    for Fi in range(image.shape[0]):
        A = image[Fi,0]
        B = image[Fi,1]
        C = image[Fi,2]
        
        image[Fi,0] = M00*A + M01*B + M02*C
        image[Fi,1] = M10*A + M11*B + M12*C
        image[Fi,2] = M20*A + M21*B + M22*C
    

def weighted_coherency(np.ndarray[double complex, ndim=2] image, np.ndarray[double, ndim=1] weights):
    
    cdef np.ndarray[double complex, ndim=2] out = np.zeros( (3,3), dtype=np.cdouble )
    
    
    cdef np.ndarray[double complex, ndim=1] A = np.zeros( 3, dtype=np.cdouble )
    cdef np.ndarray[double complex, ndim=1] B = np.zeros( 3, dtype=np.cdouble )
    
    cdef int Fi
    cdef double W
    cdef double W2_sum = 0
    for Fi in range( image.shape[0] ):
        A[0] = image[Fi,0]
        A[1] = image[Fi,1]
        A[2] = image[Fi,2]
        W = weights[Fi]
        A *= W
        W2_sum += W*W
        B[0] = conj(A[0])
        B[1] = conj(A[1])
        B[2] = conj(A[2])
        
        out[0,0] += A[0]*B[0]
        out[0,1] += A[0]*B[1]
        out[0,2] += A[0]*B[2]
        
        out[1,0] += A[1]*B[0]
        out[1,1] += A[1]*B[1]
        out[1,2] += A[1]*B[2]
        
        out[2,0] += A[2]*B[0]
        out[2,1] += A[2]*B[1]
        out[2,2] += A[2]*B[2]
        
    out *= 1.0/W2_sum
    return out


cdef class independant_FFT:
    """for taking inverse FFT of the data"""
    
    cdef long trace_length
    cdef long freq_start_index
    cdef long freq_end_index
    
    cdef void* FFT_wavetable
    cdef void* FFT_workspace
    
    def __init__(self, long trace_length, long freq_start_index, long freq_end_index):
        self.trace_length = trace_length
        self.freq_start_index = freq_start_index
        self.freq_end_index = freq_end_index
        
        
        self.FFT_wavetable = gsl_fft_complex_wavetable_alloc( self.trace_length )
        self.FFT_workspace = gsl_fft_complex_workspace_alloc( self.trace_length )
        
    def __del__(self):
        if( self.FFT_wavetable ):
            gsl_fft_complex_wavetable_free( self.FFT_wavetable )
        if( self.FFT_workspace ):
            gsl_fft_complex_workspace_free( self.FFT_workspace )
    
    def partial_inverse_FFT(self, np.ndarray[double complex, ndim=1] in_data, np.ndarray[double complex, ndim=1] out_data=None):
        
        if out_data is None:
            out_data = np.empty(self.trace_length, dtype=np.cdouble)
        else:
            if not out_data.flags['C_CONTIGUOUS']:
                print('C-contiguous eror in partial_inverse_FFT')
                quit()
            if  (out_data.shape[0] != self.trace_length) or  (out_data.ndim != 1) :
                print('length output error in partial_inverse_FFT')
                quit()
            if (in_data.shape[0] != self.freq_end_index-self.freq_start_index) or  (in_data.ndim != 1) :
                print('input length error in partial_inverse_FFT')
                quit()
        
        out_data[:] = 0.0
        cdef long Fi
        for Fi in range( self.freq_end_index-self.freq_start_index ):
            out_data[self.freq_start_index+Fi] = in_data[Fi]
            
        cdef int info
        info = gsl_fft_complex_inverse( <double*>&out_data[0], 1, self.trace_length, self.FFT_wavetable, self.FFT_workspace )
        if info != 0:
            print('GSL FFT problem!', info)
            quit()
            
        return out_data
    
    def full_polarized_inverse_FFT(self, np.ndarray[double complex, ndim=5] total_image, np.ndarray[double complex, ndim=5] image_out=None, long extracted_trace_length=-1):
        """take FFT of every pixel, and polarizaiton. In should have dimensions [X,Y,Z,F,P]. Out has indeces[X,Y,Z,T,P]."""
        
        ### dimensions are: X,Y,Z, F/T,  pol
        
        
        cdef int Nx = total_image.shape[0]
        cdef int Ny = total_image.shape[1]
        cdef int Nz = total_image.shape[2]
        
        cdef int N_pol = total_image.shape[4]
        
        if extracted_trace_length<=0:
            extracted_trace_length = self.trace_length
        
        if image_out is None:
            image_out = np.empty((Nx,Ny,Nz,extracted_trace_length,N_pol), dtype=np.cdouble)
        # else:
        #     if not out_data.flags['C_CONTIGUOUS']:
        #         print('C-contiguous eror in partial_inverse_FFT')
        #         quit()
        #     if  (out_data.shape[0] != self.trace_length) or  (out_data.ndim != 1) :
        #         print('length output error in partial_inverse_FFT')
        #         quit()
        #     if (in_data.shape[0] != self.freq_end_index-self.freq_start_index) or  (in_data.ndim != 1) :
        #         print('input length error in partial_inverse_FFT')
        #         quit()
        
        
        cdef double complex[:] tmp_data = np.zeros(self.trace_length, dtype=np.cdouble)
        
        cdef long initial_time_index = int((self.trace_length - extracted_trace_length)/2)
        
        cdef long xi
        cdef long yi
        cdef long zi
        cdef long fi
        cdef long ti
        cdef long pi
        cdef int info
        for xi in range(Nx):
            for yi in range(Ny):
                for zi in range(Nz):
                    for pi in range(N_pol):
                    
                        for fi in range( self.freq_end_index-self.freq_start_index ):
                            tmp_data[self.freq_start_index+fi] = total_image[xi, yi, zi, fi, pi]
                            
                        info = gsl_fft_complex_inverse( <double*>&tmp_data[0], 1, self.trace_length, self.FFT_wavetable, self.FFT_workspace )
                        if info != 0:
                            print('GSL FFT problem!', info)
                            quit()
                            
                        for ti in range(extracted_trace_length):
                            image_out[xi, yi, zi, ti, pi] = tmp_data[initial_time_index+ti]
            
        return image_out
    
    
    
    
    
    
####### trilinear interpolation #######
# double 3D
cdef struct doubleArray_3D:
    double* data
    long D1_stride
    long D1_length
    long D2_stride
    long D2_length
    long D3_stride
    long D3_length
    
cdef void set_double3D(doubleArray_3D* data, np.ndarray[double, ndim=3] nparray):
    data.data = &nparray[0,0,0]
    data.D1_stride = int( nparray.strides[0]/nparray.itemsize )
    data.D1_length = nparray.shape[0]
    data.D2_stride = int( nparray.strides[1]/nparray.itemsize )
    data.D2_length = nparray.shape[1]
    data.D3_stride = int( nparray.strides[2]/nparray.itemsize )
    data.D3_length = nparray.shape[2]
    
cdef inline double* get_double3D(doubleArray_3D* data, long i, long j, long k) nogil:
    return &data.data[ i*data.D1_stride + j*data.D2_stride + k*data.D3_stride  ]
    

## trilinear interpolation
# optimized for one point to be applied to many equaly size grids
cdef struct trilinear_info:
    
    double c000_weight
    double c001_weight
    double c010_weight
    double c100_weight
    double c011_weight
    double c101_weight
    double c110_weight
    double c111_weight
    
    double c000_dx_weight
    double c001_dx_weight
    double c010_dx_weight
    double c100_dx_weight
    double c011_dx_weight
    double c101_dx_weight
    double c110_dx_weight
    double c111_dx_weight
    
    double c000_dy_weight
    double c001_dy_weight
    double c010_dy_weight
    double c100_dy_weight
    double c011_dy_weight
    double c101_dy_weight
    double c110_dy_weight
    double c111_dy_weight
    
    double c000_dz_weight
    double c001_dz_weight
    double c010_dz_weight
    double c100_dz_weight
    double c011_dz_weight
    double c101_dz_weight
    double c110_dz_weight
    double c111_dz_weight
    
cdef inline void set_weights(trilinear_info* data_struct, double X, double Y, double Z) nogil:
    X -= <long>X
    Y -= <long>Y
    Z -= <long>Z
    
    data_struct.c000_weight = 1 - X - Y - Z + X*Y + X*Z + Y*Z - X*Y*Z
    data_struct.c001_weight = Z - X*Z - Y*Z + X*Y*Z
    data_struct.c010_weight = Y - X*Y - Y*Z + X*Y*Z
    data_struct.c100_weight = X - X*Y - X*Z + X*Y*Z
    data_struct.c011_weight = Y*Z - X*Y*Z
    data_struct.c101_weight = X*Z - X*Y*Z
    data_struct.c110_weight = X*Y - X*Y*Z
    data_struct.c111_weight = X*Y*Z
    
    
cdef inline void set_derivative_weights(trilinear_info* data_struct, double X, double Y, double Z) nogil:
    X -= <long>X
    Y -= <long>Y
    Z -= <long>Z
    
    data_struct.c000_dx_weight = -1 + Y + Z - Y*Z
    data_struct.c001_dx_weight = -Z + Y*Z
    data_struct.c010_dx_weight = -Y + Y*Z
    data_struct.c100_dx_weight = 1 - Y - Z + Y*Z
    data_struct.c011_dx_weight = -Y*Z
    data_struct.c101_dx_weight = Z - Y*Z
    data_struct.c110_dx_weight = Y - Y*Z
    data_struct.c111_dx_weight = Y*Z
    
    data_struct.c000_dy_weight = -1 + X + Z - X*Z
    data_struct.c001_dy_weight = -Y + X*Z
    data_struct.c010_dy_weight = 1 - X - Z + X*Z
    data_struct.c100_dy_weight = -X + X*Z
    data_struct.c011_dy_weight = Z - X*Z
    data_struct.c101_dy_weight = -X*Z
    data_struct.c110_dy_weight = X - X*Z
    data_struct.c111_dy_weight = X*Z
    
    data_struct.c000_dz_weight = -1 + X + Y - X*Y
    data_struct.c001_dz_weight = 1 - X - Y + X*Y
    data_struct.c010_dz_weight = -Y + X*Y
    data_struct.c100_dz_weight = -X + X*Y
    data_struct.c011_dz_weight = Y - X*Y
    data_struct.c101_dz_weight = X - X*Y
    data_struct.c110_dz_weight = -X*Y
    data_struct.c111_dz_weight = X*Y
    
    
cdef inline double trilinear_interpolate( trilinear_info* data_struct, long xi, long yi, long zi, doubleArray_3D* array_data ) nogil:
    cdef double F = 0
    F += data_struct.c000_weight * get_double3D(array_data, xi,  yi,  zi  )[0]
    
    F += data_struct.c001_weight * get_double3D(array_data, xi,  yi,  zi+1 )[0]
    F += data_struct.c010_weight * get_double3D(array_data, xi,  yi+1,zi   )[0]
    F += data_struct.c100_weight * get_double3D(array_data, xi+1,yi  ,zi   )[0]
    
    F += data_struct.c101_weight * get_double3D(array_data, xi+1,yi  ,zi+1 )[0]
    F += data_struct.c011_weight * get_double3D(array_data, xi  ,yi+1,zi+1 )[0]
    F += data_struct.c110_weight * get_double3D(array_data, xi+1,yi+1,zi   )[0]
    
    F += data_struct.c111_weight * get_double3D(array_data, xi+1,yi+1,zi+1 )[0]

    return F

cdef inline void trilinear_interpolate_derivative( trilinear_info* data_struct, long xi, long yi, long zi, 
                               double* f_out, double* dx_out, double* dy_out, double* dz_out, 
                               doubleArray_3D* array_data) nogil:
    
    cdef double W000 = get_double3D(array_data, xi,  yi,  zi  )[0]
    
    cdef double W001 = get_double3D(array_data, xi,  yi,  zi+1 )[0]
    cdef double W010 = get_double3D(array_data, xi,  yi+1,zi   )[0]
    cdef double W100 = get_double3D(array_data, xi+1,yi  ,zi   )[0]
    
    cdef double W011 = get_double3D(array_data, xi  ,yi+1,zi+1 )[0]
    cdef double W110 = get_double3D(array_data, xi+1,yi+1,zi   )[0]
    cdef double W101 = get_double3D(array_data, xi+1,yi  ,zi+1 )[0]
    
    cdef double W111 = get_double3D(array_data, xi+1,yi+1,zi+1 )[0]
    
    f_out[0] = data_struct.c000_weight*W000  + data_struct.c001_weight*W001 + data_struct.c010_weight*W010 + data_struct.c100_weight*W100 + \
        data_struct.c011_weight*W011 + data_struct.c110_weight*W110 + data_struct.c101_weight*W101 + data_struct.c111_weight*W111
    
    dx_out[0] = data_struct.c000_dx_weight*W000  + data_struct.c001_dx_weight*W001 + data_struct.c010_dx_weight*W010 + data_struct.c100_dx_weight*W100 + \
        data_struct.c011_dx_weight*W011 + data_struct.c110_dx_weight*W110 + data_struct.c101_dx_weight*W101 + data_struct.c111_dx_weight*W111
    
    dy_out[0] = data_struct.c000_dy_weight*W000  + data_struct.c001_dy_weight*W001 + data_struct.c010_dy_weight*W010 + data_struct.c100_dy_weight*W100 + \
        data_struct.c011_dy_weight*W011 + data_struct.c110_dy_weight*W110 + data_struct.c101_dy_weight*W101 + data_struct.c111_dy_weight*W111
    
    dz_out[0] = data_struct.c000_dz_weight*W000  + data_struct.c001_dz_weight*W001 + data_struct.c010_dz_weight*W010 + data_struct.c100_dz_weight*W100 + \
        data_struct.c011_dz_weight*W011 + data_struct.c110_dz_weight*W110 + data_struct.c101_dz_weight*W101 + data_struct.c111_dz_weight*W111
    



def interpolate_image_full( np.ndarray[double complex, ndim=5] image, double Xi, double Yi, double Zi, np.ndarray[double complex, ndim=2] out ):
    """given a full image[X,Y,Z, F, p], (where p=3), return the 3 complex polarizations interpolated between pixels at a specific frequency"""
    
    
    cdef trilinear_info trilinear_data
    set_weights( &trilinear_data, Xi, Yi, Zi )
    
    cdef long stride = image.strides[4]/image.itemsize
    
    cdef doubleArray_3D data_array
    cdef double P0_real
    cdef double P0_imag
    cdef double P1_real
    cdef double P1_imag
    cdef double P2_real
    cdef double P2_imag
    
    cdef long Fi
    for Fi in range( image.shape[3] ):
        
        set_double3D( &data_array,  image[:,:,:,Fi,0].real )
        
        P0_real = trilinear_interpolate( &trilinear_data, <long>Xi, <long>Yi, <long>Zi, &data_array  )
        
        data_array.data += 1
        P0_imag = trilinear_interpolate( &trilinear_data, <long>Xi, <long>Yi, <long>Zi, &data_array  )
        
        data_array.data += 2*stride - 1
        P1_real = trilinear_interpolate( &trilinear_data, <long>Xi, <long>Yi, <long>Zi, &data_array  )
        
        data_array.data += 1
        P1_imag = trilinear_interpolate( &trilinear_data, <long>Xi, <long>Yi, <long>Zi, &data_array  )
        
        data_array.data += 2*stride - 1
        P2_real = trilinear_interpolate( &trilinear_data, <long>Xi, <long>Yi, <long>Zi, &data_array  )
        
        data_array.data += 1
        P2_imag = trilinear_interpolate( &trilinear_data, <long>Xi, <long>Yi, <long>Zi, &data_array  )
        
        out[Fi,0] = P0_real+1j*P0_imag
        out[Fi,1] = P1_real+1j*P1_imag
        out[Fi,2] = P2_real+1j*P2_imag
        
    return out
    
    
    
    

    