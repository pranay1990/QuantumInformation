"""
Created on Thu Aug 22 19:18:53 2019

@author: Dr. M. S. Ramkarthik and Dr. Pranay Barkataki
"""
#import scipy.linalg.lapack as la
import numpy as np
import re

class QRandom:
    def __init__(self):
        """It is a class dealing with partial trace and transpose
        it primarily intrinsi functions of uses numpy, math, cmath.
        """
        
    def random_gaussian_rvec(self,tup,mu=0,sigma=1):
        """
        Construct a real random state from a Gaussian distribution 
        Input:
            tup: tuple holds the dimension of the output matrix.
            mu: it is the average value of the gaussian distribution.
            sigma: it is the standard deviation of the gaussian distribution
        Output:
            gauss_state: it the gaussian distributed real state
        """
        gauss_state=np.random.normal(mu, sigma, tup)
        return gauss_state
    
    def random_gaussian_cvec(self,tup,mu=0,sigma=1):
        """
        Construct a complex random state from a Gaussian distribution
        Input:
            tup: tuple holds the dimension of the output matrix.
            mu: it is the average value of the gaussian distribution.
            sigma: it is the standard deviation of the gaussian distribution
        Output:
            gauss_state: it the gaussian distributed complex state
        """
        rpart=np.random.normal(mu, sigma, tup)
        cpart=np.random.normal(mu, sigma, tup)
        state_gauss=np.zeros(tup,dtype=np.complex_)
        state_gauss=rpart+(complex(0,1)*cpart)
        return state_gauss
    
    def random_unifrom_rvec(self,tup,low=0.0,high=1.0):
        """
        Construct a real random state from a uniform distribution
        Input:
            tup: tuple holds the dimension of the output matrix.
            low: it is the lower bound of the uniform distribution.
            high: it is the upper bound of the uniform distribution
        Output:
            uniform_state: it the uniform distributed real state
        """
        uniform_state=np.random.uniform(low=low, high=high, size=tup)
        return uniform_state
       
    def random_unifrom_cvec(self,tup,low=0.0,high=1.0):
        """
        Construct a complex random state from a uniform distribution
        Input:
            tup: tuple holds the dimension of the output matrix.
            low: it is the lower bound of the uniform distribution.
            high: it is the upper bound of the uniform distribution
        Output:
            uniform_state: it the uniform distributed complex state
        """
        uniform_rpart=np.random.uniform(low=low, high=high, size=tup)
        uniform_cpart=np.random.uniform(low=low, high=high, size=tup)
        uniform_state=uniform_rpart+(complex(0,1)*uniform_cpart)
        return uniform_state
    
    def random_symmetric_matrix(self,size,distribution="gaussian",mu=0,\
                                sigma=1,low=0.0,high=1.0):
        """
        Construct a random symmetric matrix
        Input:
            size: it is a tuple (a,b), where a in number of rows and b is 
                  number of columns of the symmetric matrix
            distribution: it is the type of distribution
        Output:
            smatrix: symmetric matrix 
        """
        assert re.findall("^gaussian|^uniform",distribution),\
        "Invalid distribution type"
        if distribution == 'gaussian':
            smatrix=np.random.normal(mu, sigma, size=size)
        if distribution == 'uniform':
            smatrix=np.random.uniform(low=low, high=high, size=size)
        smatrix=smatrix+np.matrix.transpose(smatrix)
        smatrix=0.5*smatrix
        return smatrix
    
    def random_hermitian_matrix(self,size,distribution="gaussian",mu=0,\
                                sigma=1,low=0.0,high=1.0):
        """
        Construct a random Hermitian matrix
        Input:
            size: it is a tuple (a,b), where a in number of rows and b is 
                  number of columns of the hermitian matrix
            distribution: it is the type of distribution
        Output:
            hmatrix: hermitian matrix 
        """
        assert re.findall("^gaussian|^uniform",distribution),\
        "Invalid distribution type"
        hmatrix=np.zeros([size[0],size[1]],dtype=np.complex_)
        if distribution=='gaussian':
            for i in range(0,hmatrix.shape[0]):
                for j in range(0,hmatrix.shape[1]):
                    hmatrix[i,j]=complex(np.random.normal(mu, sigma, size= None),\
                           np.random.normal(mu, sigma, size= None))
        else:
            for i in range(0,hmatrix.shape[0]):
                for j in range(0,hmatrix.shape[1]):
                    hmatrix[i,j]=complex(np.random.uniform(low=low, high=high,\
                           size=None),np.random.uniform(low=low, high=high,\
                                    size=None))
        hmatrix=hmatrix+np.matrix.conjugate(np.matrix.transpose(hmatrix))
        hmatrix=0.5*hmatrix
        return hmatrix
    
    def random_orthogonal_matrix(self,size,distribution="gaussian",mu=0,\
                              sigma=1,low=0.0,high=1.0):
        """
        Construct a random orthogonal matrix
        Input:
            size: it is a tuple (a,b), where a in number of rows and b is 
                  number of columns of the orthogonal matrix
            distribution: it is the type of distribution
        Output:
            omatrix: orthogonal matrix 
        """
        assert re.findall("^gaussian|^uniform",distribution),\
        "Invalid distribution type"
        
        if distribution == 'gaussian':
            omatrix=np.random.normal(mu, sigma, size= size)
        else:
            omatrix=np.random.uniform(low=low, high=high, size=size)
        omatrix,r=np.linalg.qr(omatrix,mode='complete')
        return omatrix
    
    def random_unitary_matrix(self,size,distribution="gaussian",mu=0,\
                              sigma=1,low=0.0,high=1.0):
        """
        Construct a random unitary matrix
        Input:
            size: it is a tuple (a,b), where a in number of rows and b is 
                  number of columns of the unitary matrix
            distribution: it is the type of distribution
        Output:
            umatrix: unitary matrix 
        """
        assert re.findall("^gaussian|^uniform",distribution),\
        "Invalid distribution type"
        umatrix=np.zeros([size[0],size[1]],dtype=np.complex_)
        
        if distribution == 'gaussian':
            for i in range(0,umatrix.shape[0]):
                for j in range(0,umatrix.shape[1]):
                    umatrix[i,j]=complex(np.random.normal(mu,sigma,size=None),\
                           np.random.normal(mu,sigma,size=None))
        else:
            for i in range(0,umatrix.shape[0]):
                for j in range(0,umatrix.shape[1]):
                    umatrix[i,j]=complex(\
                           np.random.uniform(low=low,high=high,size=None),\
                           np.random.uniform(low=low,high=high,size=None))
        umatrix,r=np.linalg.qr(umatrix,mode='complete')
        return umatrix
    
    def random_real_ginibre(self,N):
        """
        Construct a real Ginibre matrix
        Input:
            N: dimension of the NxN Ginibre matrix
        Output:
            real ginibre matrix
        """
        return np.random.normal(0.0, 1.0, size= (N,N))
    
    def random_complex_ginibre(self,N):
        """
        Construct a complex Ginibre matrix
        Input:
            N: dimension of the NxN complex Ginibre
        Output:
            cginibre: complex ginibre matrix
        """
        cginibre=np.zeros([N,N],dtype=np.complex_)
        for i in range(0,cginibre.shape[0]):
            for j in range(0,cginibre.shape[1]):
                cginibre[i,j]=complex(np.random.normal(0.0, 1.0, size= None),\
                        np.random.normal(0.0, 1.0, size= None))
        return cginibre
    
    def random_real_wishart(self,N):
        """
        Construct a real Wishart matrix
        Input:
            N: dimension of the NxN real Wishart matrix
        Output:
            rwishart: real Wishart matrix
        """
        g=self.random_real_ginibre(N)
        rwishart=np.matmul(g,np.matrix.transpose(g))
        return rwishart
    
    def random_complex_wishart(self,N):
        """
        Construct a complex Wishart matrix
        Input:
            N: dimension of the NxN complex Wishart matrix
        Output:
            cwishart: complex Wishart matrix
        """
        g=self.random_complex_ginibre(N)
        cwishart=np.matmul(g,np.matrix.conjugate(np.matrix.transpose(g)))
        return cwishart
    
    def random_probability_vec(self,N):
        """
        Constructs a random probability vector
        Input:
            N: dimension of the probability vector.
        Output:
            prob_vec: The probability vector
        """
        prob_vec=np.random.uniform(low=0,high=1.0,size=N)
        norm=prob_vec.sum()
        prob_vec=prob_vec/norm
        return prob_vec
    
    def random_qrstate(self,N,distribution='gaussian',\
                       mu=0,sigma=1,low=0.0,high=1.0):
        """
        Constructs a random real pure quantum state
        Input:
            N: number of qubits
            distribution: it is the type of distribution
        Output:
            qrstate: real quantum state
        """
        assert re.findall("^gaussian|^uniform",distribution),\
        "Invalid distribution type"
        if distribution=='gaussian':
            qrstate=self.random_gaussian_rvec(2**N,mu=0,sigma=1)
        else:
            qrstate=self.random_unifrom_rvec(2**N,low=0.0,high=1.0)
        norm=np.matmul(np.matrix.transpose(qrstate),qrstate)
        qrstate=qrstate/np.sqrt(norm)
        return qrstate
    
    def random_qcstate(self,N,distribution='gaussian',\
                       mu=0,sigma=1,low=0.0,high=1.0):
        """
        Constructs a random complex pure quantum state
        Input:
            N: number of qubits
            distribution: it is the type of distribution
        Output:
            qcstate: complex quantum state
        """
        assert re.findall("^gaussian|^uniform",distribution),\
        "Invalid distribution type"
        if distribution=='gaussian':
            qcstate=self.random_gaussian_cvec(2**N,mu=0,sigma=1)
        else:
            qcstate=self.random_unifrom_cvec(2**N,low=0.0,high=1.0)
        norm=abs(np.matmul(\
                       np.matrix.conjugate(\
                                           np.matrix.transpose(qcstate)),\
                                           qcstate))
        qcstate=qcstate/np.sqrt(norm)
        return qcstate
    
    def random_rden(self,N):
        """
        Constructs a random real pure density matrix
        Input:
            N: number of qubits
        Output:
            rden: real random density matrix
        """
        rden=self.random_real_wishart(2**N)
        rden=rden/np.trace(rden)
        return rden
    
    def random_cden(self,N):
        """
        Input:
            N: number of qubits
        Output:
            cden: complex random density matrix
        """
        cden=self.random_complex_wishart(2**N)
        cden=cden/np.trace(cden)
        return cden
        
    
        
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
    