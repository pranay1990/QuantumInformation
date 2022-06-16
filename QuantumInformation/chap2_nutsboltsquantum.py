#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 2021

@authors: M. S. Ramkarthik and Pranay Barkataki
"""

import numpy as np
import math

class QuantumMechanics:
    
    def __init__(self):
        """It is a quantum mechanics class, and it primarily intrinsic
            functions of uses numpy and scipy
        """
    
    def inner_product(self,vec1,vec2):
        """
        Here we compute the inner product 
        Attributes:
            vec1: it is a column vector.
            vec2: it is the second vector.
        Returns:
            inn: inner product between vec1 and vec2.
        """
        inn=complex(0.0,0.0)
        assert len(vec1) ==  len(vec2),\
        "Dimension of two vectors not equal to each other"
        for i in range(0,len(vec1)): 
            inn=inn+(np.conjugate(vec1[i])*vec2[i])
            
        return inn
    
    def norm_vec(self,vec):
        """
        Here we calculate norm of a vector
        Attributes:
            vec: column vector 
        Returns:
            norm: it contains the norm of the column vector vec
        """
        norm=0.0
        for i in range(0,len(vec)):
            norm=norm+abs(np.conjugate(vec[i])*vec[i])
        return np.sqrt(norm)
    
    def normalization_vec(self,vec):
        """
        Here normalize a given vector
        Attributes:
            vec: unnormalized column vector
        Returns:
            vec: normalized column vector
        """
        norm=0.0
        for i in range(0,len(vec)):
            norm=norm+abs(np.conjugate(vec[i])*vec[i])
        vec=vec/np.sqrt(norm)
        return vec
    
    def outer_product_rvec(self,vec1,vec2):
        """
        Here we calculate the outer product
        Attributes:
            vec1: it is a column real vector
            vec2: it is another real vector
        Returns:
            matrix: outer product of vec1 and vec2
        """
        matrix = np.zeros([len(vec1),len(vec2)],dtype='float64')
        for i in range(0,len(vec1)):
            for j in range(0,len(vec2)):
                matrix[i,j]=vec1[i]*vec2[j]
        return matrix
    
    def outer_product_cvec(self,vec1,vec2):
        """
        Here we calculate the outer product
        Attributes:
            vec1: it is a column complex vector
            vec2: it is another complex vector
        Returns:
            matrix: outer product of vec1 and vec2
        """
        matrix = np.zeros([len(vec1),len(vec2)],dtype=np.complex_)
        for i in range(0,len(vec1)):
            for j in range(0,len(vec2)):
                matrix[i,j]=vec1[i]*np.conjugate(vec2[j])
        return matrix    
    
    def tensor_product_matrix(self,A,B):
        """
        Here we calculate the tensor product of two matrix A and B
        Attributes:
            A: it is either a 1D or 2D array
            B: it is either a 1D or 2D array
        Returns: tensor product of A and B        
        """
        if len(A.shape)==1:
            A=A.reshape(A.shape[0],1)
        if len(B.shape)==1:
            B=B.reshape(B.shape[0],1)    
        return np.kron(A,B)                
    
    def commutation(self,A,B):
        """
        Here we calculate commutation between matrices A and B 
        Attributes:
            A: it is a square matrix
            B: it is another square matrix
        Returns: AB-BA matrix 
        """
        return np.matmul(A,B)-np.matmul(B,A)
    
    def anti_commutation(self,A,B):
        """
        Here we calculate anti commutation between matrices A and B 
        Attributes:
            A: input a square matrix
            B: input another square matrix
        Returns: AB+BA matrix
        """
        return np.matmul(A,B)+np.matmul(B,A)
    
    def rstate_shift(self,vec,shift=1,shift_direction='right'):
        """
        Shifting of a real quantum state
        Inputs:
            vec: real quantum state.
            shift: degree of the state.
            shift_direction: It shows the direction of the shift, 
                             by default it is right
        Return:
            state2: shifted state of vec
        """
        assert vec.shape[0]%2==0,"Not a qubit quantum state"
        N=int(math.log(vec.shape[0],2))
        basis=np.zeros([N],dtype=int)
        state2=np.zeros([2**N],dtype='float64')
        for i in range(0,vec.shape[0]):
            if vec[i] != 0.0:
                basis=self.decimal_binary(i,N)
                basis=self.binary_shift(basis,shift=shift,\
                                        shift_direction=shift_direction)
                j=int(self.binary_decimal(basis))
                state2[j]=vec[i]
        
        return state2
    
    def cstate_shift(self,vec,shift=1,shift_direction='right'):
        """
        Shifting of a complex quantum state
        Inputs:
            vec: complex quantum state.
            shift: degree of the state.
            shift_direction: It shows the direction of the shift, by default 
                            it is right
        Return:
            state2: shifted state of vec
        """
        assert vec.shape[0]%2==0,"Not a qubit quantum state"
        N=int(math.log(vec.shape[0],2))
        basis=np.zeros([N],dtype=int)
        state2=np.zeros([2**N],dtype=np.complex_)
        for i in range(0,vec.shape[0]):
            if vec[i] != complex(0.0,0.0):
                basis=self.decimal_binary(i,N)
                basis=self.binary_shift(basis,shift=shift,\
                                        shift_direction=shift_direction)
                j=self.binary_decimal(basis)
                state2[j]=vec[i]
        
        return state2
    
    def decimal_binary(self,i,N):
        """
        Decimal to binary conversion
        Inputs:
            i: input decimal number to changed to binary number
            N: number of the binary string
        Returns:
            bnum: binary number equivalent to i, it is column matrix.
        """
        bnum=np.zeros([N],dtype=int)
        for j in range(0,N):
            bnum[bnum.shape[0]-1-j]=i%2
            i=int(i/2)
        return bnum
            
    def binary_decimal(self,vec):
        """
        Binary to decimal conversion
        Input:
            vec:  array containing binary matrix elements i.e. 0 or 1
        Returns:
            dec: decimal equivalent number of binary array vec
        """
        dec=0
        for i in range(0,vec.shape[0]):
            dec=dec+int((2**(vec.shape[0]-1-i))*vec[i])
        dec=int(dec)
        return dec
    
    def binary_shift(self,vec,shift=1,shift_direction='right'):
        """
        Shifting of string of binary number
        Input:
            vec: array containing binary matrix elements i.e. 0 or 1
            shift: degree of the shift
            shift_direction: its value is either left or right
        Output:
            shift_vec: Shifted vec
        """
        assert shift_direction == 'left' or shift_direction == 'right',\
        'Not proper shift direction'
        assert shift >= 1 and shift < vec.shape[0],\
        "degree of shift is not proper"
        shift_vec=np.zeros([vec.shape[0]],dtype='int_')
        if shift_direction =='right':
            for i in range(0,vec.shape[0]):
                ishift=i+shift
                if ishift >= vec.shape[0]:
                    ishift=ishift-vec.shape[0]
                shift_vec[ishift]=vec[i]
        if shift_direction == 'left':
            for i in range(0,vec.shape[0]):
                ishift=i-shift
                if ishift < 0:
                    ishift=ishift+vec.shape[0]
                shift_vec[ishift]=vec[i]
        
        return shift_vec












    
