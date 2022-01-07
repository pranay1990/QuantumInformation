"""
Created on Wed Nov  4 

@author: pranay barkataki
"""
import numpy as np
import math
import cmath
import scipy.linalg.lapack as la


class LinearAlgebra:
        
    def inverse_matrix(self,mat):
        """ Calculates the inverse of a matrix
            Attributes:
                mat : Inverse of the array or matrix to be calculated. 
            Return: inverse of matrix mat
        """
        assert np.linalg.det(mat) !=  0, "Determinant of the matrix is zero"
        return np.linalg.inv(mat)
    
    def power_smatrix(self,mat1,k,precision=10**(-10)):
        """
        It calculates the power of a real symmetric matrix.
        Attributes:
            mat1 : The matrix or array of which power is  to be calculated.
            k : value of the power
            precision: if the absolute eigenvalues below the precision 
                       value will be considered as zero 
        Return: k'th Power of symmetric matrix mat1
        """
        eigenvalues,eigenvectors,info=la.dsyev(mat1)
        flag=0
        for i in eigenvalues:
            if i < 0.0:
                flag=1
        if flag==0:
            diag=np.zeros([eigenvectors.shape[0],eigenvectors.shape[1]],\
                          dtype='float64')
        else:
            diag=np.zeros([eigenvectors.shape[0],eigenvectors.shape[1]],\
                          dtype='complex_')
        for i in range(0,eigenvectors.shape[0]):
            if abs(eigenvalues[i]) <= precision:
                diag[i,i]=0.0
                eigenvalues[i]=0.0
            if eigenvalues[i] < 0.0:
                diag[i,i]=pow(abs(eigenvalues[i]),k)*pow(complex(0,1),2*k)
            else:
                diag[i,i]=pow(eigenvalues[i],k)
        diag=np.matmul(np.matmul(eigenvectors,diag),np.transpose(eigenvectors))
        return diag
    
    def power_hmatrix(self,mat1,k,precision=10**(-10)):
        """
        It calculates the power of a Hermitian matrix.
        Attributes:
            mat1 : The matrix or array of which power is  to be calculated.
            k : value of the power
            precision: if the absolute eigenvalues below the precision 
                       value will be considered as zero 
        Return: k'th Power of Hermitian matrix mat1
        """
        eigenvalues,eigenvectors,info=la.zheev(mat1)
        flag=0
        for i in eigenvalues:
            if i < 0.0:
                flag=1
        if flag==0:
            diag=np.zeros([eigenvectors.shape[0],eigenvectors.shape[1]],\
                          dtype='float64')
        else:
            diag=np.zeros([eigenvectors.shape[0],eigenvectors.shape[1]],\
                          dtype='complex_')
        for i in range(0,eigenvectors.shape[0]):
            if abs(eigenvalues[i]) <= precision:
                diag[i,i]=0.0
                eigenvalues[i]=0.0
            if eigenvalues[i] < 0.0:
                diag[i,i]=pow(abs(eigenvalues[i]),k)*pow(complex(0,1),2*k)
            else:
                diag[i,i]=pow(eigenvalues[i],k)
        diag=np.matmul(np.matmul(eigenvectors,diag),np.conjugate(\
                       np.transpose(eigenvectors)))
        return diag  
    
    def power_gmatrix(self,mat1,k,precision=10**(-10)):
        """ 
        Calculates the power of a general non-Herimitian matrix
        Attributes:
            mat1 : The matrix or array of which power is  to be calculated.
            k : value of the power
            precision: if the absolute eigenvalues below the precision 
                       value will be considered as zero 
        Return: k'th Power of non-Hermitian matrix mat1
        """
        eigenvalues,vl,eigenvectors,info=np.linalg.eig(mat1)
        diag=np.zeros([eigenvectors.shape[0],eigenvectors.shape[1]],dtype=np.complex_)
        for i in range(0,eigenvectors.shape[0]):
            if abs(eigenvalues[i]) <= precision:
                diag[i,i]=complex(0.0,0.0)
            else:
                diag[i,i]=pow(eigenvalues[i],k)
        diag=np.matmul(np.matmul(eigenvectors,diag),np.linalg.inv(eigenvectors))
        return diag        
    
    def function_smatrix(self, mat1, mode="exp",log_base=2):
        """
        It calculates the function of a real symmetric matrix.  
        Attributes:
            mat1 : The symmetric matrix of which function is to be calculated.
            mode: Primarily calculates the following,
                mode='exp': Exponential of a matrix. It is the default mode.
                mode='sin': sine of a matrix.
                mode='cos': cosine of matrix.
                mode='tan': tan of matrix.
                mode='log': Logarithm of a matrix, by default log base 2.
            log_base: base of the log function
        Return: Function of symmetric matrix mat1
        """
        
        assert np.allclose(mat1, np.matrix.transpose(mat1))==True,\
            "The matrix entered is not a symmetric matrix"
        
        assert mat1.shape[0] == mat1.shape[1],\
            "Entered matrix is not a square matrix"
        
        if mode not in ["exp","sin","cos","tan","log"]:
            raise Exception(f"Sorry, the entered mode {mode} is not available")
        
        eigenvalues,eigenvectors,info=la.dsyev(mat1)
        
        if mode == 'exp':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=float)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = math.exp(eigenvalues[i])
                
        if mode == 'sin':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=float)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = math.sin(eigenvalues[i])                

        if mode == 'cos':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=float)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = math.cos(eigenvalues[i])            
    
        if mode == 'tan':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=float)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = math.tan(eigenvalues[i])        
        
        if mode == 'log':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=float)
            for i in range(0,diagonal.shape[0]):
                assert eigenvalues[i] > 0.0,\
                "eigenvalues of the matrix are negative or zero"
                diagonal[i,i] = math.log(eigenvalues[i],log_base)
        
        return np.matmul(np.matmul(eigenvectors,diagonal),\
                         np.matrix.transpose(eigenvectors))
        
    
    def function_hmatrix(self, mat1, mode="exp",log_base=2):
        """
        It calculates the function of hermitian matrix. 
        Attributes:
            mat1 : The Hermitian matrix of which function is to be calculated.
            mode: Primarily calculates the following,
                mode='exp': Exponential of a matrix. It is the default mode.
                mode='sin': sine of a matrix.
                mode='cos': cosine of matrix.
                mode='tan': tan of matrix.
                mode='log': Logarithm of a matrix, by default log base 2.
            log_base: base of the log function
        Return: Function of Hermitian matrix mat1
        """        

        assert np.allclose(mat1, np.transpose(np.conjugate(mat1)))==True \
                              ,"The matrix entered is not a hermitian matrix"
        
        assert mat1.shape[0] == mat1.shape[1],\
            "Entered matrix is not a square matrix"
        
        if mode not in ["exp","sin","cos","tan","log"]:
            raise Exception(f"Sorry, the entered mode {mode} is not available")
        
        eigenvalues,eigenvectors,info=la.zheev(mat1)
        
        if mode == 'exp':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=float)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = math.exp(eigenvalues[i])
                
        if mode == 'sin':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=float)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = math.sin(eigenvalues[i])                

        if mode == 'cos':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=float)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = math.cos(eigenvalues[i])            
    
        if mode == 'tan':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=float)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = math.tan(eigenvalues[i])        
        
        if mode == 'log':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=float)
            for i in range(0,diagonal.shape[0]):
                assert eigenvalues[i] > 0.0, "eigenvalues of the matrix are negative"
                diagonal[i,i] = math.log(eigenvalues[i],log_base)
        
        return np.matmul(np.matmul(eigenvectors,diagonal),\
                         np.transpose(np.conjugate(eigenvectors)))
        
    def function_gmatrix(self, mat1, mode="exp",log_base=2):
        """ 
        It calculates the function of general diagonalizable matrix. 
        Attributes:
            mat1: The general matrix of which function is to be calculated.
            mode: Primarily calculates the following,
                Primarily calculates the following,
                mode='exp': Exponential of a matrix.
                mode='sin': sine of a matrix.
                mode='cos': cosine of matrix.
                mode='tan': tan of matrix.
                mode='log': Logarithm of a matrix, by default log base 2.
        Return: Function of general matrix mat1
        """        
        
        assert mat1.shape[0] == mat1.shape[1],\
            "Entered matrix is not a square matrix"
        
        if mode not in ["exp","sin","cos","tan","log"]:
            raise Exception(f"Sorry, the entered mode {mode} is not available")
        
        eigenvalues,eigenvectors=np.linalg.eig(mat1)
        print(eigenvalues)
        
        if mode == 'exp':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=complex)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = cmath.exp(eigenvalues[i])
                
        if mode == 'sin':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=complex)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = cmath.sin(eigenvalues[i])                

        if mode == 'cos':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=complex)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = cmath.cos(eigenvalues[i])            
    
        if mode == 'tan':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=complex)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = cmath.tan(eigenvalues[i])        
        
        if mode == 'log':
            diagonal=np.zeros((mat1.shape[0],mat1.shape[1]),dtype=complex)
            for i in range(0,diagonal.shape[0]):
                diagonal[i,i] = cmath.log(eigenvalues[i],log_base)
        assert np.linalg.det(eigenvectors) != 0, "Determinant of eigenvectors \
                                           matrix is zero"
        
        return np.matmul(np.matmul(eigenvectors,diagonal),\
                         np.linalg.inv(eigenvectors))
    
    def trace_norm_rmatrix(self,mat1, precision=10**(-13)):
        """ 
        Calculates the trace norm of a real matrix
        Attributes:
            mat1 : The matrix or array of which trace norm is to be calculated.
            precision: the absolute value of the eigenvalues below precision
                       value will be considered as zero
        Return: 
            trace_norm: trace norm of matrix mat1
        """
        eigenvalues,eigenvectors,info=la.dsyev(np.matmul(np.transpose(mat1),\
                                                         mat1))
        trace_norm=0.0
        for i in range(len(eigenvalues)):
            if abs(eigenvalues[i]) < precision:
                eigenvalues[i]=0.0
            trace_norm=trace_norm+np.sqrt(eigenvalues[i])
        
        return trace_norm

    def trace_norm_cmatrix(self,mat1, precision=10**(-13)):
        """ 
        Calculates the trace norm of a complex matrix
        Attributes:
            mat1 : The matrix or array of which trace norm is to be calculated.
            precision: the absolute value of the eigenvalues below precision
                        value will be considered as zero.
        Return: 
            trace_norm: trace norm of matrix mat1
        """
        eigenvalues,eigenvectors,info=\
        la.zheev(np.matmul(np.conjugate(np.transpose(mat1)),mat1))
        trace_norm=0.0
        for i in range(len(eigenvalues)):
            if abs(eigenvalues[i]) < precision:
                eigenvalues[i]=0.0
            trace_norm=trace_norm+np.sqrt(eigenvalues[i])
        
        return trace_norm        
        
    def hilbert_schmidt_norm_rmatrix(self,mat1, precision=10**(-13)):
        """ 
        Calculates the Hilbert-Schmidt norm of matrix of a real matrix
        Attributes:
            mat1 : The matrix or array of which Hilbert-Schmidt norm
                      is to be calculated.
            precision: tolerance value, the magnitude of eigenvalues below 
                       precision is considered zero
        Return:
            htrace_norm: Hilbert-Schmidt norm of matrix mat1.
        """
        eigenvalues,eigenvectors,info=la.dsyev(np.matmul(np.transpose(mat1),\
                                                         mat1))
        htrace_norm=0.0
        for i in range(len(eigenvalues)):
            if abs(eigenvalues[i]) < precision:
                eigenvalues[i]=0.0
            htrace_norm=htrace_norm+eigenvalues[i]
        htrace_norm=np.sqrt(htrace_norm)
        return htrace_norm

    def hilbert_schmidt_norm_cmatrix(self,mat1, precision=10**(-13)):
        """ 
        Calculates the trace norm of a complex matrix
        Attributes:
            mat1 : The matrix or array of which Hilbert-Schmidt norm 
                      is to be calculated.
            precision: tolerance value, the magnitude of eigenvalues below 
                       precision is considered zero.
        Return:
            htrace_norm: Hilbert-Schmidt norm of matrix mat1.
        """
        eigenvalues,eigenvectors,info=\
        la.zheev(np.matmul(np.conjugate(np.transpose(mat1)),mat1))
        htrace_norm=0.0
        for i in range(len(eigenvalues)):
            if abs(eigenvalues[i]) < precision:
                eigenvalues[i]=0.0
            htrace_norm=htrace_norm+eigenvalues[i]
        htrace_norm=np.sqrt(htrace_norm)
        
        return htrace_norm                
        
    def absolute_value_rmatrix(self,mat1):
        """
        Calculates the absolute value of a real matrix
        Attributes:
            mat1 : The matrix of which absolute form has to calculated.
        Return:
            res_mat: Absoulte value of matrix mat1
        """
        res_mat=self.power_smatrix(np.matmul(np.transpose(mat1),\
                                             mat1),0.50)
                
        return res_mat             
        
    def absolute_value_cmatrix(self,mat1):
        """
        Calculates the absolute value of a complex matrix
        Attributes:
            mat1 : The matrix of which absolute form has to calculated.
        Return:
            res_mat: Absoulte value of matrix mat1
        """
        res_mat=self.power_hmatrix(np.matmul(np.conjugate(np.transpose(mat1)),\
                                             mat1),0.50)           
        return res_mat         

    def hilbert_schmidt_inner_product(self,A,B):
        """ 
        Calculates the Hilbert-Schmidt inner product between matrices.
        Attributes:
            A: It is a complex or real input matrix.
            B: It is a complex or real input matrix.
        Return: Hilbert-Schmidt inner product between A and B.
        """
        return np.trace(np.matmul(np.conjugate(np.transpose(A)),B))


    def gram_schmidt_ortho_rmatrix(self,vectors):
        """
        Orthornormal set of real vectors are calculated
        Attributes:
          vectors: A matrix whose columns are non-orthogonal set real vectors
        Return:
          orthonormal_vec: A matrix whose columns are orthonormal to each other
        """
        orthonormal_vec=np.zeros((vectors.shape[0],vectors.shape[1]),
                                    dtype='float64')
        for col in range(0,vectors.shape[1]):
            if col != 0:
                for col2 in range(0,col):
                    tr=0.0
                    for row2 in range(0,vectors.shape[0]):
                        tr=tr+(orthonormal_vec[row2,col2]*vectors[row2,col])
                    orthonormal_vec[:,col]=orthonormal_vec[:,col]+\
                                               (tr*orthonormal_vec[:,col2])
                orthonormal_vec[:,col]=vectors[:,col]-orthonormal_vec[:,col]
            if col == 0:
                orthonormal_vec[:,col]=vectors[:,col]
            tr=0.0
            for row in range(0,vectors.shape[0]):
                tr=tr+(orthonormal_vec[row,col]*orthonormal_vec[row,col])
            orthonormal_vec[:,col]=orthonormal_vec[:,col]/np.sqrt(tr)
            
        return orthonormal_vec


    def gram_schmidt_ortho_cmatrix(self,vectors):
        """
        Orthornormal set of complex vectors are calculated
        Attributes:
          vectors: A matrix whose columns are non-orthogonal set 
                   complex vectors
        Return:
          orthonormal_vec: A matrix whose columns are orthonormal to each other
        """
        orthonormal_vec=np.zeros((vectors.shape[0],vectors.shape[1]),\
                                 dtype=np.complex_)
        for col in range(0,vectors.shape[1]):
            if col != 0:
                orthonormal_vec[:,col]=vectors[:,col].copy()
                for col2 in range(0,col):
                    tr=complex(0.0,0.0)
                    for row2 in range(0,vectors.shape[0]):
                        tr=tr+(np.conjugate(orthonormal_vec[row2,col2])\
                               *vectors[row2,col])
                    orthonormal_vec[:,col]=orthonormal_vec[:,col]-\
                                               (tr*\
                                                orthonormal_vec[:,col2].copy())
            if col == 0:
                orthonormal_vec[:,col]=vectors[:,col].copy()
            tr=complex(0.0,0.0)
            for row in range(0,vectors.shape[0]):
                tr=tr+(np.conjugate(orthonormal_vec[row,col])*\
                       orthonormal_vec[row,col])
            orthonormal_vec[:,col]=orthonormal_vec[:,col]/np.sqrt(tr.real)
            
        return orthonormal_vec
        
    
    
    
    
    
