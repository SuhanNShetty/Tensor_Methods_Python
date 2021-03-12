"""
Reference:

1. "Tensor Decompositions and Applications", Tamara Kolda, Brett W. Bader, 
2. Multi-way Analysis by Age Smilde, R Bros 
Author: Suhan Shetty suhan.n.shetty@gmail.com

Date: 29th July 2019

This is an implementation of common tools for 3-way array

- Generate a 3-way tensor data {Input(X),Output(Y)}
- Tensor Operations
- Tensor Decomposition Techniques
- Tensor-Tensor Regression
- Array-normal distribution

"""

import numpy as np
from numpy.linalg import multi_dot as mat_mul
from numpy.random import random_sample 
import pdb
from numpy.linalg import norm
from scipy.stats import multivariate_normal
import random
"""   
A note about convention used:
numpy array index starts from 0. However, we use for the mode numbering startion from 1.
So, axis 0 in a numpy array corresponds to mode 1.
"""

class array3:
    
#-----------------------------------------------------------------------------------------------------------------
    # Genrate a random 3-way array
    def random_array3(self,tensor_shape):
        #tensor_shape.reverse()
        t_shape = [tensor_shape[2],tensor_shape[0],tensor_shape[1]]
        X = 0
        R = np.random.randint(10)+10 # Randomize R, number of true components
        for p in range(R): # Add some true components to the tensor
            a = random_sample((tensor_shape[0],))
            b = random_sample((tensor_shape[1],))
            c = random_sample((tensor_shape[2],))
            X = X + self.outer_product(a,b,c) 
        X = X + np.random.random(t_shape) # Add some noise over the true component
        return X


# Shape of the tensor --------------------------------------------------------------------------------------------

    def shape(self,X): #
        return (X.shape[1],X.shape[2],X.shape[0])

# Compute kronecker product of the set of matrices ---------------------------------------------------------------
    
    def kron(self,A): # Here, A=[A1,A2,A3,..,AN]
        N = len(A)
        kp = np.kron(A[0],A[1])
        k = 2
        while k<N:
            kp = np.kron(kp,A[k])
            k=k+1
        return kp
    
# Unfold the tensor along the given mode --------------------------------------------------------------------------
   
    def unfold(self,X,n):
        if n==2:
            X = np.swapaxes(X,1,2) 
        elif n==3:
            X = np.swapaxes(X,0,2)
            X = np.swapaxes(X,1,2)
        depth = X.shape[0]
        Xn = X[0,:,:] 
        for k in range(depth-1):
            tmp = X[k+1,:,:]
            Xn = np.concatenate((Xn,tmp),axis=1)
        return Xn
    
# Unfold the tensor along the given mode for 4-way array ---------------------------------------------------------
   
    def unfold4(self,X,n):
        if n==1:
            m = X.shape[2]
            X = np.swapaxes(X,0,2)
            X = np.swapaxes(X,1,2)
        elif n==2:
            m = X.shape[3]
            X = np.swapaxes(X,0,3)
            X = np.swapaxes(X,1,3)
            X = np.swapaxes(X,2,3)
        elif n==3:
            m = X.shape[1]
            X = np.swapaxes(X,0,1)
            
        elif n==4:
            m = X.shape[0]

       
        Xn = X.flatten('C')
        l = len(Xn)
        Xn = np.reshape(Xn,[m,int(l/m)])
        return Xn
    
# mode-n product of the tensor with a matrix-----------------------------------------------------------------------
   
    # For 3-way tensor
    def mode_n_product(self,X,M,n): # T_out = T_in * M
        if n==2:
            X = np.einsum('ijl,kl',X,M)
        elif n==1:
            X = np.einsum('ilk,jl',X,M)
        elif n==3:
            X = np.einsum('ljk,il',X,M)
        return X
    # For 4-way tensor
    def mode_n_product4(self,X,M,n): # T_out = T_in * M
        if n==2:
            X = np.einsum('hijl,kl',X,M)
        elif n==1:
            X = np.einsum('hilk,jl',X,M)
        elif n==3:
            X = np.einsum('hljk,il',X,M)
        elif n==4:
            X = np.einsum('lijk,hl',X,M)
        return X



# Compute the Frobenius norm -------------------------------------------------------------------------------------
   
    def norm(self,X):
        return np.linalg.norm(X.flatten())

# Vectorize a tensor----------------------------------------------------------------------------------------------
   
    def vectorize(self,X):
        return X.flatten()
            
# Khatri-Rao product of two matrices------------------------------------------------------------------------------
    
    def khatri_rao(self,M):
        X = M[0]
        Y = M[1]
        assert X.shape[1]==Y.shape[1],"Number of columns must match"
        I = X.shape[0]*Y.shape[0]
        J = X.shape[1]
        Z = np.empty([I,J])      
        for k in range(J):
            Z[:,k] = np.kron(X[:,k],Y[:,k])
        return Z

# Hadamard Product -----------------------------------------------------------------------------------------------

    def hadamard(self,X,Y):
        assert X.shape==Y.shape,"Dimensions must match"
        return X*Y

# Outer Product ---------------------------------------------------------------------------------------------------

    def outer_product(self,a,b,c): #outer product of three vectors
        assert a.ndim == 1 and b.ndim==1 and c.ndim==1, "Input has to be 1d arrays"
        return np.einsum('i,k,j',c,b,a) # or np.einsum('i,j,k',a,b,c).swapaxes(0,2)


# CP/CANDECOMP/PARAFAC decompostion of a tensor ------------------------------------------------------------------
    # Uses ALS technique to decompose the tensor into R rank-1 tensors
    def cp(self,X,R):
        I = self.shape(X)
        r = min([I[0]*I[1], I[1]*I[2], I[0]*I[2]])
        assert R<r, "Reduce R. Its too large comapred to the dimension of the data"
             
        # Initialization of factor matrices
        A = [np.random.random((I[k],R)) for k in range(3)]
        #print("Shape of A: ", A[0].shape, A[1].shape, A[2].shape)
        # Unfold X along every mode
        Xn = [self.unfold(X,k+1) for k in range(3)]
        # Alternative way to initialize A is to take first R left singular vectors from unfolded tensors
        #for k in range(3):
            #U,S,V = np.linalg.svd(Xn[k])
            #A[k] = U[:,0:R]
        # Iterative step of ALS
        for reps in range(10000):
            tmp = A[:]
            for k in range(3):# iterate for each mode
                idx = [0,1,2]
                idx.remove(k)
                #print("Shape 0 and 1: ",A[idx[0]].shape, A[idx[1]].shape )
                Z = self.khatri_rao([A[idx[1]],A[idx[0]]])
                W = self.hadamard(mat_mul([A[idx[0]].T,A[idx[0]]]),mat_mul([A[idx[1]].T,A[idx[1]]]))
                Winv = np.linalg.pinv(W,rcond=0.001)
                A[k] = mat_mul([Xn[k],Z,Winv])         
            if (reps+1)%100==0:
                err_ = sum([np.linalg.norm(A[p]-tmp[p]) for p in range(3)])
                if err_<0.001:
                    print("The algorithm has converged. Number of iterations in ALS: ",reps)
                    break
                else:
                    tmp = A[:]  
        return A # Return A = [[a1,a2,..,aR], [b1,b2,...,bR],[c1,c2,...,cR]]
    
# Tucker product--------------------------------------------------------------------------------------------------

    # Tucker product of a tensor with a list of factor matrices
    def tucker_product(self,X, M, I): # M = [Mp,Mq,Mr,..,Mn] , I: Index for the corresponding mode I = [p,q,..n]
        if len(X.shape)==3:
            for i in range(len(I)):
                X = self.mode_n_product(X,M[i],I[i])
        elif len(X.shape)==4:
            for i in range(len(I)):
                X = self.mode_n_product4(X,M[i],I[i])
                
        return X

# Tucker Matrix Product ------------------------------------------------------------------------------------------
    
    def tucker_matrix_product(self,X, M_,n): # M = [M1,M2,M3,..,MN] 
        M = M_[:]
        Mn = M[n-1]
        M.pop(n-1)
        M.reverse()
        M = [M[k].T for k in range(len(M))]
        Z = self.kron(M)
        Xn = self.unfold(X,n)
        Yn = mat_mul([Mn,Xn, Z])
        return Yn # Returns the unfolded tensor along mode n after taking tucker product    
      
# Tucker Decompostion---------------------------------------------------------------------------------------------

    # Technique: HOOI(Higher-order orthogonal iteration)    
    def tucker3(self,X,R):# Find the decompostion X = G x {A1,A2,A3}
            assert len(R)==3, "Input list should have exactly three elements"
            I = self.shape(X)
            A = [np.random.random((I[k],R[k])) for k in range(3)] #Initialize the factors 
            Xn = [self.unfold(X,n+1) for n in range(3)]
            # Iterate till convergence or max_step
            max_step = 1000
            for reps in range(max_step):
                tmp = A[:]
                for k in range(3): # for each mode
                    idx = [0,1,2]
                    idx.remove(k)
                    Z = self.kron([A[idx[1]], A[idx[0]]])
                    Yk = mat_mul([Xn[k],Z])
                    U,S,V = np.linalg.svd(Yk)
                    A[k] = (U[:,0:R[k]])# Take the R[k] leading left singular vectors of Yk
                if (reps+1)%10==0: # Check for convergence
                    err_ = sum([np.linalg.norm(A[p].flatten()-tmp[p].flatten()) for p in range(3)])
                    if err_<= 0.001:
                        print("The algorithm has converged. Number of iterations: ",reps)
                        break
                    else:
                        tmp = A[:]   
            At = [A[k].T for k in range(len(A))]
            G = self.tucker_product(X,At,[1,2,3])
            return (A,G)

# Partial Tucker--------------------------------------------------------------------------------------------------------#
   
    # Find the decompostion X = G x {A1,A2,A3} given one of the factors
    def partial_tucker3(self,X,R,A_,mode):
            assert len(R)==3, "Input list should have exactly three elements"
            I = self.shape(X)
            A = [np.random.random((I[k],R[k])) for k in range(3)] #Initialize the factors 
            A[mode-1] = A_
            Xn = [self.unfold(X,n+1) for n in range(3)]
            # Iterate till convergence or max_step
            max_step = 1000
            id_ = [0,1,2]
            id_.remove(mode-1)
            for reps in range(max_step):
                tmp = A[:]
                for k in id_: # for each mode
                    idx = [0,1,2]
                    idx.remove(k)
                    Z = self.kron([A[idx[1]], A[idx[0]]])
                    Yk = mat_mul([Xn[k],Z])
                    U,S,V = np.linalg.svd(Yk)
                    A[k] = (U[:,0:R[k]])# Take the R[k] leading left singular vectors of Yk

                if (reps+1)%10==0: # Check for convergence
                    err_ = sum([np.linalg.norm(A[p].flatten()-tmp[p].flatten()) for p in range(3)])
                    if err_<= 0.001:
                        #print("The algorithm has converged. Number of iterations: ",reps)
                        break
                    else:
                        tmp = A[:]   
                        
            At = [A[k].T for k in range(len(A))]
            G = self.tucker_product(X,At,[1,2,3])
            #A = [At[k].T for k in range(3)]
            return (A,G)
            
# Covariate Tensor-Tensor Regression------------------------------------------------------------------------------
   
    # Assumption: The samples are stacked along the third mode of the tensor 
    def cov_regression(self,X_Data,Y_Data, alpha, R_X, R_Y):
        # alpha is between 0 and 1. alpha is approximately (input fitting)/(output correlation) coefficient
        # R_X and R_Y are dimensions of tucker factors for X_Data and Y_Data respectively
        I_X = self.shape(X_Data)
        I_Y = self.shape(Y_Data)
        assert I_X[2]==I_Y[2] and R_X[2]==R_Y[2], "The number of samples must equal the dimension in the third mode"
        C = np.random.random((I_X[2],R_X[2])) # C_X = C_Y
        C_old = np.copy(C)
        for reps in range(1000):
            (Ax,Gx_)= self.partial_tucker3(X_Data,R_X, C, mode=3)
            (Ay,Gy_)= self.partial_tucker3(Y_Data,R_Y, C, mode=3)
            Gx = self.unfold(Gx_,3)
            Gy = self.unfold(Gy_,3)
            X_flat = self.unfold(X_Data,3)
            Y_flat = self.unfold(Y_Data,3)
            Px = mat_mul([Gx,self.kron([Ax[1].T,Ax[0].T])])
            Py = mat_mul([Gy,self.kron([Ay[1].T,Ay[0].T])])
            X_pinv = np.linalg.pinv(X_flat,rcond=0.001)
            Z = np.sqrt(alpha)*mat_mul([X_flat,Px.T])+np.sqrt(1-alpha)*mat_mul([Y_flat,Py.T])
            #Alternatively, U = np.sqrt(alpha)*mat_mul([Px,Px.T])+np.sqrt(1-alpha)*mat_mul([Py,Py.T])
            U = np.sqrt(alpha)*mat_mul([Gx,Gx.T])+np.sqrt(1-alpha)*mat_mul([Gy,Gy.T])
            X_pinv = np.linalg.pinv(X_flat)
            U_pinv = np.linalg.pinv(U)
            W = mat_mul([X_pinv,Z,U_pinv])
            C = mat_mul([X_flat,W])
            if (reps+1)%500==0:
                if np.linalg.norm(C-C_old)<0.1:
                    print("The algorithm has converged. Number of iterations: ",reps)
                    break
        X_fit = mat_mul([C,Px])
        Error = np.linalg.norm(X_fit-X_flat)
        print("Error in input data fit: ", Error, "and norm of the input data: ",np.linalg.norm(X_flat) )     
        return (Ax,Gx,Ay,Gy,W,Py)

# Compute Matrix root--------------------------------------------------------------------------------------
    # Compute A's from Cov: Cov = U*S*Vh, A = U*Sqrt(S)
    def mat_root(self,Cov_):
#         U,s,Vh = np.linalg.svd(Cov)
#         s_r = np.sqrt(s)
#         if s_r.any()==0:
#             print("Error: Cov is not conditioned well")
# #         s_inv_r = 1/s_r
# #         s_inv_r[s_inv_r>1000000] = 0
# #         s_inv_r[s_inv_r<1000000] = 0
#         S_r = np.diag(s_r)
#         S_inv_r = np.linalg.inv(S_r)#np.diag(s_inv_r) 
#         A = mat_mul([U,S_r])
#         A_inv = mat_mul([S_inv_r, U.T])
        A = np.linalg.cholesky(Cov_)
        A_inv = np.linalg.inv(A)
        return (A, A_inv)
    
    def init_covariance(self, T):
#             for i in range(T-1):
#                 a = 0
#                 m = 0
#                 for k in range(i,T):
#                     a = a + CovTime[m,k]
#                     m = m+1
#                 a = a/(T-i)
#                 corr_[i] = a
#             print("Toeplitz: ", corr_)    
#             Cov[2][0,:] = np.array(corr_)
#             #pdb.set_trace()
#             K = 3
#             centr = np.linspace(0,T,K)
#             sig = T/(0.1*K)
#             # form Phi matrix
#             Phi = np.empty([T,K])
#             for j in range(T):
#                 for k in range(K):
#                     Phi[j,k] = np.exp(-(j-centr[k])**2/sig**2)
#             w = mat_mul([np.linalg.pinv(Phi),np.array(corr_)])
#            corr_ = list(mat_mul([Phi,w]))
#             for j in range(I[2]-1):
#                 el = corr_.pop()
#                 corr_.insert(0,0)
#                 Cov[2][j+1,:] = np.array(corr_)
#             Cov[2] = Cov[2] + Cov[2].T
                
#             print("Appr Toeplitz: ", corr_)  
        pass

## Array-Normal Distribution : Dutilleuls's paper on MLE estimation for tensor normal distribution------------------
  
    # Separable Covariance estimation for 3-way array data using MLE
    def anormal_old(self, X, coef=0.9, var=1, pow_ = 1, constraint=False, normalised=False, 
                for_mix_model=False,gamma_k=None,Cov_Old=[]): # X = {X1, X2,...,Xn}
        I = self.shape(X[0])
        N = len(X)
        Nk = N*1
        # Compute the mean:
        M = 0
        if normalised == False:
            for X_ in X:
                M = M + X_
            M = M/N
            #print("M:",M.shape)
            X_n = [X_-M for X_ in X] # Mean subtracted data  
        else:
            X_n = X[:]
        
        # X_i ~ M + Z x {A1, A2, A3}, and CovK = Ak*Ak'
        # Intialize the covariance matrices (mode-1, mode-2,mode-3)
        Cov = [np.identity(I[i]) for i in range(3)]
        ar = 3
        
        if for_mix_model==True: 
            Nk = np.sum(gamma_k)
            X_n = [np.sqrt(gamma_k[i])*X[i]   for i in range(N)]
            
       
            
            
        if constraint==True:
            # Give auto-regressive structure to the longituduinal covariance matrix
            ar = 2
            
#             # Logitudinal Covariance from data
#             sum_ = 0.
#             for j in range(N):
#                 X_k = self.unfold(X_n[j],3)
#                 sum_ = sum_ + mat_mul([X_k,X_k.T]) 
#             n2 = Nk*np.prod(I)/I[2]
#             Cov_ = sum_/n2
#             Cov_d = np.diagonal(sum_/n2)
#             print(Cov_d)
            ##
            
            # Approximate toeplitz structure from AR(1)
            T = I[2]
            #cof = sum(Cov_d)/T
            corr_ = [coef**(pow_*j)for j in range(T)] #[coef**(pow_*j)for j in range(T)]
            Cov[2][0,:] = np.array(corr_)
            #print("Corr_: ",corr_)
            for j in range(I[2]-1):
                corr_.pop()
                corr_.insert(0,0)
                Cov[2][j+1,:] = np.array(corr_)
            Cov[2] = Cov[2] + Cov[2].T
            
            #print(np.diagonal(Cov[2]))
            np.fill_diagonal(Cov[2], 1)
#             print(np.diagonal(Cov[2]))
            Cov[2] = var*Cov[2]
#             for k in range(Cov[2].shape[0]):
#                 if k<35 and k >25:
#                     Cov[2][k,k] = 10
#                 else:
#                     Cov[2][k,k] = 0.01
                        
           

    
            Cov_struct = 1.0*Cov[2]
            det_struct = np.linalg.det(Cov_struct)#             corr_[T-1] = CovTime[0,T-1]
            #print("Cov_Struct: \n", Cov_struct)
            
            
        eps = 0.00001        
        cov_norm = np.array([np.linalg.norm(Cov[j]) for j in range(3)])
        for reps in range(100):
            tmp = Cov[:]
            for k in range(ar): #iterate over each mode
                if cov_norm.any()<eps:
                    print("Atleast one Cov matrix is zero,. Verify the data")
                    break
                idx = list(range(3))
                idx.pop(k)
                #print("before inv norm: ",norm(np.kron(Cov[idx[1]],Cov[idx[0]])))
                try:
                    Z = np.linalg.inv(np.kron(Cov[idx[1]],Cov[idx[0]]))
                except:
                    print("The Z matrix in anormal became singular")
                    Z_arg = np.kron(Cov[idx[1]],Cov[idx[0]])
                    Z = np.linalg.inv(Z_arg+0.01*np.identity(Z_arg.shape[0])) 
                
                sum_ = 0.
                for j in range(N):
                    X_k = self.unfold(X_n[j],k+1)
                    sum_ = sum_ + mat_mul([X_k,Z,X_k.T]) 
                nk = Nk*np.prod(I)/I[k]
                Cov[k] = sum_/nk
                cov_norm = np.array([np.linalg.norm(Cov[j]) for j in range(3)])
            
            err_ = np.linalg.norm(self.kron(Cov)-self.kron(tmp))
            #print(err_)
            if (reps+1)%2==0:
                #print("cov error in anormal: ",err_)
            #print([np.linalg.norm(Cov[i]) for i in range(3)])
                if err_ < eps or np.linalg.norm(self.kron(Cov))<eps :
                    if for_mix_model==False:
                        print("MLE has converged in ", reps, " steps")
                    if np.linalg.norm(self.kron(Cov))<eps:
                        print("The Cov matrices are close to nonsingular ")
                    break
                else:
                    tmp = Cov[:]
        Cov = [Cov_ for Cov_ in Cov]
        A = [self.mat_root(Cov[j]) for j in range(3)]
        return (M,Cov,A)    

    

## Array-Normal Distribution : Dutilleuls's paper on MLE estimation for tensor normal distribution------------------
  
    # Separable Covariance estimation for 3-way array data using MLE
    def anormal(self, X, coef=0.9, var=1, pow_ = 1, constraint=False, normalised=False, 
                for_mix_model=False,gamma_k=None,Cov_Old=[]): # X = {X1, X2,...,Xn}
        
        I = self.shape(X[0])
        N = len(X)
        Nk = N*1
        
        # Compute the mean:
        M = 0
        t_data = np.linspace(0,1,I[2]) 
        if normalised == False:
            for X_ in X:
                M = M + X_
            M = M/N
            #print("M:",M.shape)
            X_n = [X_-M for X_ in X] # Mean subtracted data  
        else:
            X_n = X[:]
        
        # X_i ~ M + Z x {A1, A2, A3}, and CovK = Ak*Ak'
        # Intialize the covariance matrices (mode-1, mode-2,mode-3)
        Cov = [np.identity(I[i]) for i in range(3)]
        ar = 3
        
        if for_mix_model==True: 
            Nk = np.sum(gamma_k)
            X_n = [np.sqrt(gamma_k[i])*X[i]   for i in range(N)]
            
            
        # Construct the Basis Function Phi (x = Phi*w)
        Nb = 10
        delta = 0.0001
        t_basis = np.linspace(0,1,Nb) # Centre of the basis
        sig = 2.0/Nb ;
        Phi = np.empty([I[2],Nb])
#         print("t_basis: ", t_basis.shape)
#         print("t_data: ", t_data.shape)
#         print("Phi_shape: ", Phi.shape)
        for i in range(Nb): # over t_basis
            for j in range(I[2]): # over t_data
                Phi[j,i] = np.exp(-(t_basis[i]-t_data[j])**2/sig**2)

        PhiT = np.transpose(Phi)
        invPhi = np.linalg.pinv(Phi)         
        invPhiT = np.transpose(invPhi)

        # Assume a covariance matrix for weights w: Uw

        Uw = np.identity(Nb)
        Reg = delta*np.identity(I[2])

        Cov[2] = mat_mul([Phi, Uw, PhiT]) + Reg
            
#             # Logitudinal Covariance from data
#             sum_ = 0.
#             for j in range(N):
#                 X_k = self.unfold(X_n[j],3)
#                 sum_ = sum_ + mat_mul([X_k,X_k.T]) 
#             n2 = Nk*np.prod(I)/I[2]
#             Cov_ = sum_/n2
#             Cov_d = np.diagonal(sum_/n2)
#             print(Cov_d)
            ##

        Cov_struct = 1.0*Cov[2]
        det_struct = np.linalg.det(Cov_struct)#             corr_[T-1] = CovTime[0,T-1]
            
        eps = 0.00001        
        cov_norm = np.array([np.linalg.norm(Cov[j]) for j in range(3)])
        
        for reps in range(100):
            tmp = Cov[:]
            for k in range(3): #iterate over each mode
                if cov_norm.any()<eps:
                    print("Atleast one Cov matrix is zero,. Verify the data")
                    break
                idx = list(range(3))
                idx.pop(k)
                #print("before inv norm: ",norm(np.kron(Cov[idx[1]],Cov[idx[0]])))
                try:
                    Z = np.linalg.inv(np.kron(Cov[idx[1]],Cov[idx[0]]))
                except:
                    print("The Z matrix in anormal became singular")
                    Z_arg = np.kron(Cov[idx[1]],Cov[idx[0]])
                    Z = np.linalg.inv(Z_arg+0.001*np.identity(Z_arg.shape[0])) 
                
                sum_ = 0.
                for j in range(N):
                    X_k = self.unfold(X_n[j],k+1)
                    sum_ = sum_ + mat_mul([X_k,Z,X_k.T]) 
                nk = Nk*np.prod(I)/I[k]
                Cov[k] = sum_/nk         
                
                if k==2:
                    Uw = mat_mul([invPhi,Cov[2],invPhiT])
                    Cov[2] = mat_mul([Phi,Uw,PhiT])+Reg
                
                cov_norm = np.array([np.linalg.norm(Cov[j]) for j in range(3)])
            
            err_ = np.linalg.norm(self.kron(Cov)-self.kron(tmp))
#             print("Error EM: ", err_)
            if (reps+1)%2==0:
                if err_ < eps or np.linalg.norm(self.kron(Cov))<eps :
                    if for_mix_model==False:
                        print("MLE has converged in ", reps, " steps")
                    if np.linalg.norm(self.kron(Cov))<eps:
                        print("The Cov matrices are close to nonsingular ")
                    break
                else:
                    tmp = Cov[:]
        Cov = [Cov_ for Cov_ in Cov]
        A = [self.mat_root(Cov[j]) for j in range(3)]
        return (M,Cov,A)      
## Array-Normal Distribution : Dutilleuls's paper on MLE estimation for tensor normal distribution------------------
  
    # Separable Covariance estimation for 3-way array data using MLE
    def anormal2D(self,X,Nb=10, rbf=False): # X = {X1, X2,...,Xn}
        
        N = len(X) # Number of data points
        Nk = N*1
        N1 = X[0].shape[0]
        N2 = X[0].shape[1]
        # Compute the mean:
        M = 0
        for X_ in X:
            M = M + X_
        M = M/N
        #print("M:",M.shape)
        X_n = [Xs-M for Xs in X] # Mean subtracted data  
#         else:
#             X_n = X[:]
        
        # X_i ~ M + Z x {A1, A2}, and CovK = Ak*Ak'
        # Intialize the covariance matrices (mode-1, mode-2)
        Cov = [np.identity(N1),np.identity(N2)]
  
        # Construct the Basis Function Phi (x = Phi*w)
        delta = 0.0000001
        t_data = np.linspace(0,1,N2) 
        t_basis = np.linspace(0,1,Nb) # Centre of the basis
        sig = 1.0/Nb ;
        Phi = np.empty([N2,Nb])

        for i in range(Nb): # over t_basis
            for j in range(N2): # over t_data
                Phi[j,i] = np.exp(-(t_basis[i]-t_data[j])**2/sig**2)

        PhiT = np.transpose(Phi)
        invPhi = np.linalg.pinv(Phi)         
        invPhiT = np.transpose(invPhi)

        # Assume a covariance matrix for weights w: Uw
        Uw = np.identity(Nb)
        Reg = delta*np.identity(N2) # Regularisation
        Cov[0] = np.random.randn(N1,N1)#np.identity(N1)
        Cov[1] = mat_mul([Phi, Uw, PhiT]) + Reg
        if rbf==False:
            Cov[1] = np.random.randn(N2,N2) #np.identity(N2)
            
        eps = 0.00001        
        cov_norm = np.array([np.linalg.norm(Cov[j]) for j in range(2)])
        
        for reps in range(1000000):
            tmp = Cov[:]
            
            
            # Compute Uw and U2 given U1
            try:
                Z2 = np.linalg.inv(Cov[0])
            except:
                print("The Cov1 matrix in anormal became singular")
                Z2 = np.linalg.inv(Cov[0]+eps*np.identity(N1)) 
                
            sum_ = 0.
            for j in range(N):
                X_k = X_n[j]
                sum_ = sum_ + mat_mul([X_k.T,Z2,X_k]) 
            nk = Nk*N1
            Cov2 = sum_/nk 
            Uw = mat_mul([invPhi,Cov2,invPhiT])#- delta* np.identity(Nb)
            if rbf==True:
                Cov[1]=mat_mul([Phi,Uw,PhiT])+ eps*np.identity(N2)
            else:
                Cov[1] = Cov2+eps*np.identity(N2)    
            #Cov[1] = (1/Cov[1][0,0])*Cov[1]
            #pdb.set_trace()
            if cov_norm.any()<eps:
                print("Atleast one Cov matrix is zero,. Verify the data")
                break
                
            # Compute U1 given Uw/U2
            try:
                Z1 = np.linalg.inv(Cov[1])
            except:
                print("The Cov1 matrix in anormal became singular")
                Z1 = np.linalg.inv(Cov[1]+eps*np.identity(N1)) 

            sum_ = 0.
            for j in range(N):
                X_k = X_n[j] 
                sum_ = sum_ + mat_mul([X_k,Z1,X_k.T]) 
            nk = Nk*N2
            Cov[0] = sum_/nk + eps*np.identity(N1)  
            Cov[0] = (1/Cov[0][0,0])*Cov[0]
            
            cov_norm = np.array([np.linalg.norm(Cov[j]) for j in range(2)])
            
            err_ = np.linalg.norm(self.kron(Cov)-self.kron(tmp))
            
            print("Error EM: ", err_)
            #print(np.linalg.det(Cov[0]), np.linalg.det(Cov[1]))
            if (reps+1)%10==0:
                if err_ < eps or np.linalg.norm(self.kron(Cov))<eps :
                    print("MLE has converged in ", reps, " steps")
#                     if for_mix_model==False:
#                         print("MLE has converged in ", reps, " steps")
                    if np.linalg.norm(self.kron(Cov))<eps:
                        print("The Cov matrices are close to nonsingular ")
                    break
                else:
                    tmp = Cov[:]
        #Cov = [Cov_ for Cov_ in Cov]
        A = [self.mat_root(Cov[j]) for j in range(2)]
        return (M,Cov,A, Uw, Phi)          
    
    
# Array-Normal Distribution--------------------------------------------------------------------------------------
  
    # Separable Covariance estimation for 3-way array data using MLE
    def anormal_hoff(self, X, coef =0.9,var = 1,pow_=1, constraint=False): # X = {X1, X2,...,Xn}
        I = self.shape(X[0])
        N = len(X)
        # Compute the mean:
        M = 0
        for X_ in X:
            M = M + X_
        M = M/N
        shape_ = list(X[0].shape)
        shape_.insert(0,N)
        M_ext = np.empty(shape_)
        X_ext = np.empty(shape_)
        
        # Extended array mean and array
        for i in range(N):
            M_ext[i,:,:,:] = M 
            X_ext[i,:,:,:] = X[i]
            
        # Residual:
        E = X_ext - M_ext
        
        # X_i ~ M + Z x {Cov1, Cov2, Cov3}, and CovK = Ak*Ak'
        
        # Intialize the covariance matrices (mode-1, mode-2,mode-3)
        # mode-1 covariance = column cov, mode-2 cov= row covariance, mode-3 is pipe cov
        Cov = [np.identity(I[i]) for i in range(3)]
        ar = 3
        
        if constraint==True:
            # Give auto-regressive structure to the longituduinal covariance matrix
            ar = 2
            corr_ = [coef**(pow_*j) for j in range(I[2])]
            Cov[2][0,:] = np.array(corr_)
            for j in range(I[2]-1):
                el = corr_.pop()
                corr_.insert(0,coef**(j+1))
                Cov[2][j+1,:] = np.array(corr_)
            Cov[2] = var*Cov[2]
                   
        # Compute matrix-square-root of the covariances
        A = Cov[:]
        A_inv = Cov[:]
   
        A_inv_ext = A_inv[:]
        A_inv_ext.append(np.identity(N))
        eps = 0.01
        cov_norm = np.array([np.linalg.norm(Cov[j]) for j in range(3)])
        for reps in range(10000):
            tmp = Cov[:]
            for k in range(ar): #iterate over each mode
                if cov_norm.any()<eps:
                    print("One of the covariances is Zero. Terminating MLE at k= ", k)
                    break
                idx = list(range(ar))
                idx.pop(k)
                for j in idx:
                    (A[j], A_inv_ext[j]) = self.mat_root(Cov[j])
                A_inv_ext[k] = np.identity(I[k])
                E_ = self.tucker_product(E,A_inv_ext,[1,2,3,4])
                E_k = self.unfold4(E_,k+1)
                S = mat_mul([E_k,E_k.T])
                nk = N*np.prod(I)/I[k]
                Cov[k] = S/nk
                cov_norm[k] = np.linalg.norm(Cov[k])
                
            err_ = np.linalg.norm(self.kron(Cov)-self.kron(tmp))
            print(err_)
            if err_ < eps or np.linalg.norm(self.kron(Cov))<eps :
                print("MLE has converged in ", reps, " steps")
                break
            else:
                tmp = Cov[:]
        A = [self.mat_root(Cov[j])[0] for j in range(3)]
        return (M,Cov,A)
    
# Array-Normal Distribution--------------------------------------------------------------------------------------
  
    # Separable Covariance estimation for 3-way array data using MLE
    def anormal_hoff3(self, X, coef =0.9, constraint=False): # X = {X1, X2,...,Xn}
        I = self.shape(X[0])
        N = len(X)
        # Compute the mean:
        M = 0
        for X_ in X:
            M = M + X_
        M = M/N
        shape_ = list(X[0].shape)
        shape_.insert(0,N)
        M_ext = np.empty(shape_)
        X_ext = np.empty(shape_)
        
        # Extended array mean and array
        for i in range(N):
            M_ext[i,:,:,:] = M 
            X_ext[i,:,:,:] = X[i]
            
        # Residual:
        E = X_ext - M_ext
        
        # X_i ~ M + Z x {Cov1, Cov2, Cov3}, and CovK = Ak*Ak'
        
        # Intialize the covariance matrices (mode-1, mode-2,mode-3)
        # mode-1 covariance = column cov, mode-2 cov= row covariance, mode-3 is pipe cov
        Cov = [np.random.rand(I[i],I[i]) for i in range(3)]
        Cov = [mat_mul([Cov[i],Cov[i].T]) for i in range(3)]
        ar = 3
        if constraint==True:
            # Give auto-regressive structure to the longituduinal covariance matrix
            corr_ = [coef**j for j in range(I[2])]
            Cov[2][0,:] = np.array(corr_)
            ar = 2
            for j in range(I[2]-1):
                el = corr_.pop()
                corr_.insert(0,coef**(j+1))
                Cov[2][j+1,:] = np.array(corr_)
              
                
        # Compute matrix-square-root of the covariances
        A = [None]*3
        A_inv = [None]*3
        for j in range(3):
            (A[j],A_inv[j]) = self.mat_root(Cov[j])
        A_inv_ext = A_inv[:]
        A_inv_ext.append(np.identity(N))
        
        for reps in range(1000):
            tmp = Cov[:]
            for k in range(ar): #iterate over each mode
                idx = list(range(ar))
                idx.pop(k)
                for j in idx:
                    (A[j], A_inv_ext[j]) = self.mat_root(Cov[j])
                A_inv_ext[k] = np.identity(I[k])
                E_ = self.tucker_product(E,A_inv_ext,[1,2,3,4])
                E_k = self.unfold4(E_,k+1)
                S = mat_mul([E_k,E_k.T])
                nk = N*np.prod(I)/I[k]
                Cov[k] = S/nk
                
            if (reps+1)%10==0:
                err_ = sum([np.linalg.norm(Cov[p]-tmp[p]) for p in range(3)])
                print(err_)
                if err_ < 0.1:
                    print("MLE converged in ", reps, " steps")
                    break
                else:
                    tmp = Cov[:]
        return (M,Cov,A)

#  Array-normal Conditioning--------------------------------------------------------------------------------------
   
    # 3-way array-normal conditioning
    def anormal_condition(self,M,Cov,Ia,X_a,slice_):
       # pdb.set_trace()
        # mode: along which mode unfolding each column is partially known
        # Ia: Index of the know data along the given mode
        # X_a: data slice at index Ia
        # mode = 1,2,3 => incomplete info along each row, columns, pipe respectively
        Ix = M.shape
        if slice_ == 1:# incomplete info along each row when unfoded along mode 2 OR slice given along mode 1
            mode = 2
            Ib = set(range(Ix[2]))
            Ib = list(Ib - set(Ia)) #index of unknown row elements
            Cov_mode = Cov[mode-1] # row covariance i.e. covariance along mode 2 unfolding
            M_b = M[:,:,Ib] 
            M_a = M[:,:,Ia]
        elif slice_ == 2:# incomplete info along each column when unfolded along mode 1 OR slice given along mode 2
            mode =  1
            Ib = set(range(Ix[1]))# index of unknown columns
            Ib = list(Ib - set(Ia))
            Cov_mode = Cov[mode-1] # row covariance  
            M_b = M[:,Ib,:]
            M_a = M[:,Ia,:]
        elif slice_==3:# incomplete info along each pipe
            mode = 3
            Ib = set(range(Ix[0]))# index of unknown columns
            Ib = list(Ib - set(Ia))
            Cov_mode = Cov[mode-1] # row covariance  
            M_b = M[Ib,:, :]
            M_a = M[Ia,:,:]
        else:
            raise ValueError('Invalid mode passed')
        Cov_aa = Cov_mode[np.ix_(Ia,Ia)]
        Cov_ba = Cov_mode[np.ix_(Ib,Ia)]
        Cov_bb = Cov_mode[np.ix_(Ib,Ib)]
        Cov_ab =  Cov_mode[np.ix_(Ia,Ib)]
        inv_Cov_aa = np.linalg.inv(Cov_aa)
        update_coef = mat_mul([Cov_ba,inv_Cov_aa]) 
        Cov_o = Cov[:] 
        Cov_a = Cov[:]
        Cov_o[mode-1] = Cov_bb - mat_mul([update_coef,Cov_ba.T])
        update_M = self.mode_n_product((X_a-M_a),update_coef,mode) 
        M_o = M_b + update_M
        Cov_a[mode-1] = Cov_aa[:,:]
        return (M_o,Cov_o, M_a, Cov_a)

#  Array-normal-mix Conditioning--------------------------------------------------------------------------------------
    def anormal_mix_condition(self,p_mix,M_mix,Cov_mix,Ia,X_a,slice_):
        K = len(p_mix)
        M_mix_o = [None]*K
        M_mix_i = [None]*K
        p_mix_o = [None]*K
        Cov_mix_o = Cov_mix[:]
        Cov_mix_i = Cov_mix[:]
        for k in range(K):
            (M_mix_o[k],Cov_mix_o[k], M_mix_i[k], Cov_mix_i[k])= self.anormal_condition(M_mix[k],Cov_mix[k],Ia,X_a,slice_)

        p_mix_o = self.anormal_mix_gamma(X_a,p_mix,M_mix_i,Cov_mix_i)
        return (p_mix_o, M_mix_o, Cov_mix_o)
    
    
# Sampling from array-normal distribution---------------------------------------------------------------------------
    
    def anormal_sampling(self,M,Cov):
        A = [None]*3 #Cov[:]
        # Compute the matrix-square-root (Cov = A*A') of covariance matrices
        for j in range(3):             
            A[j],_ = self.mat_root(Cov[j])
        Z = np.random.randn(*M.shape)
        X_ = M + self.tucker_product(Z,A,[1,2,3])
        return X_
    
# Sampling from array-normal-2D distribution---------------------------------------------------------------------------
    
    def anormal_sampling_2D(self,M,Cov):
        A = [None]*2 #Cov[:]
        # Compute the matrix-square-root (Cov = A*A') of covariance matrices
        for j in range(2):             
            A[j],_ = self.mat_root(Cov[j])
        Z = np.random.standard_normal(tuple(M.shape))
        X_ = M + mat_mul([A[0],Z, A[1].T])#mat_mul([Z, A[1].T])# 
#         x = np.random.multivariate_normal(M.flatten(), np.kron(Cov[1],Cov[0]))
#         X_ = x.reshape(M.shape)
        return X_

 # Condition from array-normal-2D distribution---------
    
    def anormal_condition_2D(self,M,Cov,Ia,X_a):
        Ib = list(set(range(M.shape[1]))- set(Ia)) #index of unknown columns
        M_b = M[:,Ib] 
        M_a = M[:,Ia]
        #print(M_a.shape)
        #X_a = X_a.reshape(3,1)
        Cov_mode = Cov[1]
        Cov_aa = Cov_mode[np.ix_(Ia,Ia)]
        Cov_ba = Cov_mode[np.ix_(Ib,Ia)]
        Cov_bb = Cov_mode[np.ix_(Ib,Ib)]
        Cov_ab =  Cov_mode[np.ix_(Ia,Ib)]
        inv_Cov_aa = np.linalg.inv(Cov_aa)
        update_coef = mat_mul([Cov_ba,inv_Cov_aa]) 
        Cov_o = Cov[:] 
        Cov_a = Cov[:]
        Cov_o[1] = Cov_bb - mat_mul([update_coef,Cov_ba.T])
        #pdb.set_trace()
        update_M = mat_mul([(X_a-M_a),update_coef.T]) 
        M_o = M_b + update_M
        Cov_a[1] = Cov_aa[:,:]
        return (M_o,Cov_o, M_a, Cov_a)
                  
        
        
 
 # Sampling from array-normal mixture distribution---------------------------------------------------------------
    
    def anormal_mix_sampling(self,M_mix,Cov_mix, pi_mix):
        K = len(M_mix)
        I = M_mix[0].shape
        A_mix = [[None]*3 for k in range(K)] #Cov[:]
        # Compute the matrix-square-root (Cov = A*A') of covariance matrices
        for k in range(K):
            for j in range(3):             
                A_mix[k][j],_ = self.mat_root(Cov_mix[k][j])
        Z_mix = [np.random.randn(*I) for k in range(K)]
        X_ = 0
        for k in range(K):
            X_ = X_ + pi_mix[k]*(M_mix[k] + self.tucker_product(Z_mix[k],A_mix[k],[1,2,3]))
        return X_
    
    
    
# array-normal distribution---------------------------------------------------------------------------
    def anormal_mix_gamma(self,X,p_mix,M_mix,Cov_mix):
        #pdb.set_trace()
        K = len(p_mix)
        X_0 = self.unfold(X,1)
        x = X_0.flatten('F') # verify this
        Cov = [None]*K
        for k in range(K):
            cov_k = Cov_mix[k]
            cov_k.reverse()
            Cov[k] = self.kron(cov_k) 
        M = [self.unfold(M_mix[k],1).flatten('F') for k in range(K)]
        d = max(M[0].shape)
        v = [-0.5*mat_mul([(x-M[k]).T,np.linalg.inv(Cov[k]), (x-M[k])]) for k in range(K)]
        ln_w = [None]*K
        for k in range(K):
            ln_w[k]=np.log(p_mix[k])+ np.log(0.00001+np.linalg.det(Cov[k]))*(-1/2.0)+ np.log(2*np.pi)*(-d/2.0)
        #print(np.array(v))
        #print(np.array(ln_w))
        #print(np.array(ln_w)+np.array(v))
        a = np.max(np.array(v)+np.array(ln_w)) 
       # print(a)
        tmp = [np.exp(v[k]+ln_w[k]-a) for k in range(K)]
        normalise = sum(tmp)
        #print("normalise: ", normalise)
        gamma_ = [tmp[k]/normalise for k in range(K)]
        if abs(1-sum(gamma_))>0.01:
            print("gamma is being computed wrong. Instead of 1.0, gamma adds to: ", sum(gamma_))
        return gamma_
    
    
    def anormal_prob(self,X_,M_,Cov_):
        X = self.unfold(X_,1)
        M = self.unfold(M_,1)
        Cov_.reverse()
        Cov = self.kron(Cov_)
        Cov = Cov + 0.0*np.identity(Cov.shape[0])
        p = multivariate_normal.pdf(X.flatten('F'), M.flatten('F'),Cov) 
        return p
        
# Tensor-variate GMM ---------------------------------------------------------------------------

    def anormal_mix(self,X,K,coef =0.9,var=1,pow_=1,constraint=True):
        # K is number of components
        N = len(X) # Number of data points
        I = self.shape(X[0])
        
        # Extract mean 
        M = 0 
        for X_ in X:
            M = M+X_
        M = M/N

        # initialize the mean of the mixure components
        #M_mix = [M+0.1*random_sample(M.shape) for k in range(K)] 
        
        # initialise the covariance of the mixture components
        Cov_mix = [[None]*3 for k in range(K)]
        M_mix = [None]*K
        for k in range(K):# for each anormal component  
            M_mix[k] = 0.8*M+0.2*(1/int(N/K))*sum([random.choice(X) for j in range(int(N/K))])
            for j in range(3):# for each mode
                Cov_mix[k][j] = np.identity(I[j])
#                 Cov_mix[k][j] = random_sample([I[j],I[j]])
#                 Cov_mix[k][j] = 100*mat_mul([Cov_mix[k][j],Cov_mix[k][j].T])  
        
#         print("len of M_mix: ",len(M_mix))
#         print(M_mix)
        # intialize the cluster assignment probabilities
        p_mix = [1.0/K for k in range(K)]
        
        # holder for responsibility params
        gamma_ = np.empty([N,K])
        #pdb.set_trace()
        eps = 0.0001
        for reps in range(10000):
            #print("EM rep: ", reps)
            #print("Cov_mix[k]:",Cov_mix[0])
            #pdb.set_trace()
            all_cov_o = [self.kron(Cov_mix[k]) for k in range(K)]
            all_mean_o = [M_mix[k] for k in range(K)]
            all_pi_o = [p_mix[k] for k in range(K)]
            #print("At the E stap")
            # E-step
            #pdb.set_trace()
            for i in range(N):
                #print(Cov_mix)
                #pdb_set_trace()
                gamma_[i,:] = np.array(self.anormal_mix_gamma(X[i],p_mix, M_mix, Cov_mix))
                #print("gamma_[i,:] with i = ", i , " is : ", gamma_[i,:])
                #print("gamma_",i,"_",k,"\n", gamma_)
                #p_i = [self.anormal_prob(X[i],M_mix[k],Cov_mix[k]) for k in range(K)] #probability of observing ith data under each comp 
                
                #print("p_i at i = ", i, " is : \n", p_i)
                
#                 normalise_ = 0.
#                 for k in range(K):
#                     normalise_ = normalise_ + p_i[k]*p_mix[k] 
                    
#                 print("normalise: ", normalise_)
#                 for k in range(K):
#                     #Cov_ = Cov_mix[k]
#                     #M_ = M_mix[k]
#                     gamma_[i,k] = (p_i[k]*p_mix[k])/normalise_
                
            #print("About to calculate mean")
            #print("p: ",p_mix)
            #print("gamma: ", gamma_)
            #print("M_ = ")
            #pdb.set_trace()
            # M-step
            #print(gamma_)
            nk = [np.sum(gamma_[:,k]) for k in range(K)]
            
            for k in range(K):
                # update mean of the kth component
                Ex_k = 0
                for i in range(N):
                    Ex_k = Ex_k+gamma_[i,k]*X[i]
                M_mix[k] = Ex_k/nk[k]
                
                # update covariance
                X_0 = [X_-M_mix[k] for X_ in X]
                _,Cov_mix[k],_=self.anormal(X_0,coef,var,pow_,constraint,normalised=True,
                                          for_mix_model=True,gamma_k=gamma_[:,k],Cov_Old=Cov_mix[k])
#                 for j in range(3): # Avoid sigularity
#                     Cov_mix[k][j] = Cov_mix[k][j] + 0.01*np.identity(Cov_mix[k][j].shape[0]) 
                #print("Cov_mix[k]:",Cov_mix[0])
                # update cluster assignment probabilities
                p_mix[k] = nk[k]/N
                #print("p_mix_",k, "=", p_mix[k])
                #print("Cov_mix_",k, "\n", Cov_mix[k])
                
            #pdb.set_trace()
            # Check for convergence:
            if (reps+1)%1==0:
                dummy = [np.diag(Cov_) for Cov_ in Cov_mix]
                
                all_cov_e = [self.kron(Cov_mix[k])-all_cov_o[k] for k in range(K)]
                all_mean_e = [M_mix[k]-all_mean_o[k] for k in range(K)]
                all_pi_e = [p_mix[k]-all_pi_o[k] for k in range(K)]
                err_cov = sum([norm(all_cov_e[k])  for k in range(K)]) 
                err_mean = sum([norm(all_mean_e[k])  for k in range(K)]) 
                err_pi = sum([norm(all_pi_e[k])  for k in range(K)]) 
                print("EM step ",1,"\n err_cov  : ", err_cov,"err_mean: ",err_mean,"err_pi: ",err_pi)
                if err_cov<eps and err_mean<eps and err_pi<eps:
                    print("EM converged in steps: ", reps)
                    break
                else:
                    X
                    all_cov_o = [self.kron(Cov_mix[k]) for k in range(K)]
                    all_mean_o = [M_mix[k] for k in range(K)]
                    all_pi_o = [p_mix[k] for k in range(K)]
                    
             
        return (M_mix, Cov_mix, p_mix)
        
        
        