#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:35:07 2021

@author: pranay barkataki
"""

import numpy as np
import math
import cmath
from QuantumInformation import RecurNum
from QuantumInformation import QuantumMechanics as QM
from QuantumInformation import LinearAlgebra as LA
import re

class GatesTools:
    
    def __init__(self):
        """It is a class dealing with Gates, entropy, and
        it primarily intrinsi functions of uses numpy, math, cmath.
        """
        
    def sx(self,N=1):
        """
        Construct N-qubit Pauli spin matrix sigma_x
        Inputs:
            N: number of spins
        Output:
            sigmax: It stores the Pauli spin matrix sx
        """
        sigmax=np.zeros([2**N,2**N])
        j=(2**N)-1
        for i in range(0,2**N):
            sigmax[i,j]=1
            j=j-1
        return sigmax
    
    def sy(self,N=1):
        """
        Construct N-qubit Pauli spin matrix sigma_y
        Inputs:
            N: Number of spins
        Outputs:
            sigmay: It stores the Pauli spin matrix sy
        """
        sigmay2=np.array([[0,complex(0,-1)],[complex(0,1),0]])
        if N >1:
            for i in range(2,N+1):
                if i==2:
                    sigmay=np.kron(sigmay2, sigmay2)
                elif i > 2:
                    sigmay=np.kron(sigmay, sigmay2)
        else:
            sigmay=sigmay2
        
        return sigmay
    
    def sz(self,N=1):
        """
        Construct N-qubit Pauli spin matrix sigma_z
        Inputs:
            N: Number of spins
        Outputs:
            sigmaz: It stores the Pauli spin matrix sz
        """
        sigmaz2=np.array([[1,0],[0,-1]])
        if N >1:
            for i in range(2,N+1):
                if i==2:
                    sigmaz=np.kron(sigmaz2, sigmaz2)
                elif i > 2:
                    sigmaz=np.kron(sigmaz, sigmaz2)
        else:
            sigmaz=sigmaz2
        
        return sigmaz
    
    def hadamard_mat(self,N=1):
        """
        Construct N-qubit Hadamard matrix
        Inputs:
            N: Number of spins
        Outputs:
            hadamard: It stores the Hadamard matrix
        """
        hadamard2=np.array([[1/np.sqrt(2),1/np.sqrt(2)],\
                           [1/np.sqrt(2),-1/np.sqrt(2)]])
        if N >1:
            for i in range(2,N+1):
                if i==2:
                    hadamard=np.kron(hadamard2, hadamard2)
                elif i > 2:
                    hadamard=np.kron(hadamard, hadamard2)
        else:
            hadamard=hadamard2
        
        return hadamard
    
    def phase_gate(self,N=1):
        """
        Construct N-qubit phase gate matrix
        Inputs:
            N: Number of spins
        Outputs:
            phaseg: It stores the phase gate matrix
        """
        phaseg2=np.array([[1,0],\
                           [0,complex(0,1)]])
        if N >1:
            for i in range(2,N+1):
                if i==2:
                    phaseg=np.kron(phaseg2, phaseg2)
                elif i > 2:
                    phaseg=np.kron(phaseg, phaseg2)
        else:
            phaseg=phaseg2
        
        return phaseg
    
    def rotation_gate(self,k,N=1):
        """
        Input:
            k: is a positive number
            N: number of spins
        Returns:
            rotg: Rotation gate matrix
        """
        assert k > 0, "k is not positive number"  
        
        z=complex(0,(2*math.pi)/(2**k))
        rotg2=np.array([[1,0],[0,cmath.exp(z)]])
        if N >1:
            for i in range(2,N+1):
                if i==2:
                    rotg=np.kron(rotg2, rotg2)
                elif i > 2:
                    rotg=np.kron(rotg, rotg2)
        else:
            rotg=rotg2
        
        return rotg     
    
    def cx_gate(self):
        """
        It returns controlled NOT gate
        """
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    
    def cz_gate(self):
        """
        It returns controlled Z gate
        """
        
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
    
    def swap_gate(self):
        """
        It returns a swap gate
        """
        
        return np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    
    def toffoli_gate(self):
        """
        It returns a Toffoli gate
        """
        
        return np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],\
                         [0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],\
                         [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],\
                         [0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]])
    
    def fredkin_gate(self):
        """
        It returns a Fredkin gate
        """
        
        return np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],\
                         [0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],\
                         [0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0],\
                         [0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1]])
    

    # RN and LN state having plus sign
    def bell1(self,tot_spins=2,shift=0):
        """
        Construct N tensor products of the |bell1> or T|bell1> Bell state
        Input:
            tot_spins: The total number of spins
            shift: for value 0 we get |bell1> and for value 1 T|bell1>.
        Output:
            state: the result will be |bell1> or T|bell1> state.
        """
        assert tot_spins%2==0, "the total number of spins is not an even number"
        assert shift==0 or shift==1, "Invalid entry of the shift value"
        terms=int(tot_spins/2)
        row=np.zeros([terms,1])
        mylist=[]
        icount=-1
        RecurNum.RecurChainRL1(row,tot_spins,icount,mylist,shift)
        mylist=np.array(mylist)
        state=np.zeros([2**tot_spins])
        factor=1/math.sqrt(2)
        len_mylist=len(mylist)
        for x in range(0,len_mylist):
            state[mylist.item(x)]=factor**terms
        return(state)
    

    # RN and LN state constructed from the singlet state
    def bell2(self,tot_spins=2,shift=0):
        """
        Construct N tensor products of the |bell2> or T|bell2> Bell state
        Input:
            tot_spins: The total number of spins
            shift: for value 0 we get |bell2> and for value 1 T|bell2>.
        Output:
            state: the result will be |bell2> or T|bell2> state.
        """
        assert tot_spins%2==0, "the total number of spins is not an even number"
        assert shift==0 or shift==1, "Invalid entry of the shift value"
        terms=int(tot_spins/2)
        row=np.zeros([terms,1])
        mylist=[]
        icount=-1
        RecurNum.RecurChainRL2(row,tot_spins,icount,mylist,shift)
        mylist=np.array(mylist)
        state=np.zeros([2**tot_spins])
        factor=1/math.sqrt(2)
        len_mylist=len(mylist)
        for x in range(0,len_mylist):
            if mylist.item(x)<0:
                state[-mylist.item(x)]=-factor**terms
            elif mylist.item(x)>=0:
                state[mylist.item(x)]=factor**terms
        return(state)
        
    # 00 and 11 bell state having plus sign
    def bell3(self,tot_spins=2,shift=0):
        """
        Construct N tensor products of the |bell3> or T|bell3> Bell state
        Input:
            tot_spins: The total number of spins
            shift: for value 0 we get |bell3> and for value 1 T|bell3>.
        Output:
            state: the result will be |bell3> or T|bell3> state.
        """
        assert tot_spins%2==0, "the total number of spins is not an even number"
        assert shift==0 or shift==1, "Invalid entry of the shift value"
        terms=int(tot_spins/2)
        row=np.zeros([terms])
        mylist=[]
        icount=-1
        RecurNum.RecurChainRL3(row,tot_spins,icount,mylist,shift)
        mylist=np.array(mylist)
        state=np.zeros([2**tot_spins])
        factor=1/math.sqrt(2)
        len_mylist=len(mylist)
        for x in range(0,len_mylist):
            state[mylist.item(x)]=factor**terms
        return(state)
    
    # 00 and 11 bell state having negative sign
    def bell4(self,tot_spins=2,shift=0):
        """
        Construct N tensor products of the |bell4> or T|bell4> Bell state
        Input:
            tot_spins: The total number of spins
            shift: for value 0 we get |bell4> and for value 1 T|bell4>.
        Output:
            state: the result will be |bell4> or T|bell4> state.
        """
        assert tot_spins%2==0, "the total number of spins is not an even number"
        assert shift==0 or shift==1, "Invalid entry of the shift value"
        terms=int(tot_spins/2)
        row=np.zeros([terms,1])
        mylist=[]
        icount=-1
        RecurNum.RecurChainRL4(row,tot_spins,icount,mylist,shift)
        mylist=np.array(mylist)
        state=np.zeros([2**tot_spins])
        factor=1/math.sqrt(2)
        len_mylist=len(mylist)
        for x in range(0,len_mylist):
            if mylist.item(x)<0:
                state[-mylist.item(x)]=-factor**terms
            elif mylist.item(x)>=0:
                state[mylist.item(x)]=factor**terms
        return(state)
    
    def nGHZ(self,tot_spins=3):
        """
        Construct N-qubit GHZ state
        Input:
            tot_spins: it is the total number of spins, it should be equal to 
                       or greater than 3.
        Output:
            state: N-qubit GHZ state.
        """
        assert tot_spins >= 3, "Total number of spins are less than 3"
        state=np.zeros([2**tot_spins])
        state[0]=1/np.sqrt(2)
        state[(2**tot_spins)-1]=1/np.sqrt(2)
        return state
    
    def nW(self,tot_spins=3):
        """
        Construct N-qubit W state
        Input:
            tot_spins: it is the total number of spins, it should be equal to 
                       or greater than 3.
        Output:
            state: N-qubit W state.
        """        
        assert tot_spins >= 3, "Total number of spins are less than 3"
        state=np.zeros([2**tot_spins])
        for i in range(0,tot_spins):
            state[2**i]=1/np.sqrt(tot_spins)
        return state
        
    def nWerner(self,p,tot_spins=2):
        """
        Construct N-qubit Werner state
        Input:
            tot_spins: it is the total number of spins, it should be equal to
                        or greater than 2.
            p: it is the mixing probability
        Output:
            rho: N-qubit Werner state.
        """
        assert tot_spins >= 2, "Total number of spins are less than 2"
        qobj=QM()
        if tot_spins == 2:
            state=self.bell3()
        else:
            state=self.nGHZ(tot_spins=tot_spins)
        den=qobj.outer_product_rvec(state,state)
        identity=np.identity(2**tot_spins, dtype = 'float64')
        identity=identity*(1/(2**tot_spins))
        rho=(p*den)+((1-p)*identity)
        return rho
    
    def shannon_entropy(self,pvec):
        """
        Calculates the Shannon entropy
        Input:
            pvec: column vector which contains probabilities
        Output:
            se: it returns the Shannon entropy value
        """
        size=pvec.shape[0]
        se=0.0
        for i in range(0,size):
            assert pvec[i]<=1 and pvec[i] >=0, "probability values are incorrect"
            se=se-(pvec[i]*math.log2(pvec[i]))
        return se
    
    def linear_entropy(self, rho):
        """
        Calculates the Linear entropy
        Input:
            rho: it is the density matrix
        Output:
            le: linear entropy value
        """
        tr=np.trace(rho)
        assert np.allclose(abs(tr),1), "density matrix is not correct"
        tr2=np.trace(np.matmul(rho,rho))
        le=1.0-abs(tr2)
        return le
    
    def relative_entropy(self,rho,sigma):
        """
        Calculates relative entropy
        Input:
            rho: input density matrix
            sigma: input density matrix
        Output:
            rtent: the value of relative entropy
        """
        laobj=LA()
        typerho=str(rho.dtype)
        typesig=str(sigma.dtype)
        if re.findall('^float|^int',typerho):
            logrho=laobj.function_smatrix(rho, mode="log",\
                                          log_base=math.exp(1))
        elif re.findall("^complex",typerho):
            logrho=laobj.function_hmatrix(rho, mode="log",\
                                          log_base=math.exp(1))
        if re.findall('^float|^int',typesig):
            logsig=laobj.function_smatrix(sigma, mode="log",\
                                          log_base=math.exp(1))
        elif re.findall("^complex",typesig):
            logsig=laobj.function_hmatrix(sigma, mode="log",\
                                          log_base=math.exp(1))    
        rtent=np.trace(np.matmul(rho,logrho))-np.trace(np.matmul(rho,logsig))
        rtent=abs(rtent)
        return rtent
    
    def trace_distance(self,rho,sigma):
        """
        Calculates trace distance between two density matrices
        Input:
            rho: input density matrix
            sigma: input density matrix
        Output:
            trd: it stores trace distance
        """
        res=rho-sigma
        laobj=LA()
        typeres=str(res.dtype)
        if re.findall('^float|^int',typeres):
            trd=laobj.trace_norm_rmatrix(res)
            trd=trd/2
        elif re.findall("^complex",typeres):
            trd=laobj.trace_norm_cmatrix(res)
            trd=trd/2
        return trd
    
    def fidelity_den2(self,rho,sigma):
        """
        Calculates fidelity between two density matrices
        Input:
            rho: input density matrix
            sigma: input density matrix
        Output:
            fidelity: it stores the value of fidelity
        """
        laobj=LA()
        typerho=str(rho.dtype)
        typesig=str(sigma.dtype)
        flag=0
        if re.findall('^float|^int',typerho):
            rhosq=laobj.power_smatrix(rho,0.5)
        elif re.findall("^complex",typerho):
            rhosq=laobj.power_hmatrix(rho,0.5)
            flag=1
        if re.findall('^float|^int',typesig):
            sigsq=laobj.power_smatrix(sigma,0.5)
        elif re.findall("^complex",typesig):
            sigsq=laobj.power_hmatrix(sigma,0.5)
            flag=1
        if flag==0:
            fidelity=laobj.trace_norm_rmatrix(np.matmul(rhosq,sigsq))
            fidelity=fidelity**2
        else:
            fidelity=laobj.trace_norm_cmatrix(np.matmul(rhosq,sigsq))
            fidelity=fidelity**2
        return fidelity
        
    def fidelity_vec2(self,vecrho,vecsigma):
        """
        Calculates fidelity between two quantum states
        Input:
            vecrho: input pure state vector.
            vecsigma: input pure state vector.
        Output:
            fidelity: it stores the value of fidelity
        """
        typerho=str(vecrho.dtype)
        if re.findall('^complex',typerho):
            fidelity=np.matmul(np.matrix.\
                               conjugate(np.matrix.transpose(vecrho)),\
                               vecsigma)
        else:
            fidelity=np.matmul(np.matrix.transpose(vecrho), vecsigma)
        fidelity=abs(fidelity)**2
        return fidelity
    
    def fidelity_vecden(self,vec,sigma):
        """
        Calculates fidelity between a quantum state and a density matrix
        Input:
            vec: input pure state vector.
            sigma: input density matrix
        Output:
            fidelity: it stores the value of fidelity
        """
        typevec=str(vec.dtype)
        if re.findall('^complex',typevec):
            fidelity=np.matmul(np.matrix.\
                               conjugate(np.matrix.transpose(vec)),\
                               np.matmul(sigma,vec))
        else:
            fidelity=np.matmul(np.matrix.transpose(vec),\
                               np.matmul(sigma,vec))
        fidelity=abs(fidelity)
        return fidelity
        
    def super_fidelity(self,rho,sigma):
        """
        Calculates super fidelity between two density matrices
        Input:
            rho: input density matrix.
            sigma: input density matrix.
        output:
            sf: the value of the super fidelity
        """
        tr_rho2=np.trace(np.matmul(rho,rho))
        tr_sigma2=np.trace(np.matmul(sigma,sigma))
        tr_rhosigma=np.trace(np.matmul(rho,sigma))
        sf=tr_rhosigma+np.sqrt((1-tr_rho2)*(1-tr_sigma2))
        sf=abs(sf)        
        return sf
        
    def bures_distance_vec(self,rho,sigma):
        """
        Calculates Bures distance between two quantum state
        Input:
            rho: input state vector
            sigma: input state vector
        Output:
            bd: the value of the Bures distance
        """
        fid=self.fidelity_vec2(rho,sigma)
        bd=np.sqrt(2*(1-np.sqrt(fid)))
        
        return bd
    
    def bures_distance_den(self,rho,sigma):
        """
        Calculates Bures distance between two density matrix
        Input:
            rho: input density matrix
            sigma: input density matrix
        Output:
            bd: the value of the Bures distance
        """
        fid=self.fidelity_den2(rho,sigma)
        bd=np.sqrt(2*(1-np.sqrt(fid)))
        
        return bd
    
    def expectation_vec(self,vec,obs):
        """
        Expectation values of observable for a quantum state
        Input:
            vec: input state vector
            obs: observable operator
        Output:
            expc: the expectation value of the measurement operator
        """
        typevec=str(vec.dtype)
        if re.findall('^complex',typevec):
            expc=np.matmul(np.matmul(np.matrix.\
                               conjugate(np.matrix.transpose(vec)),\
                               obs),vec)
        else:
            expc=np.matmul(np.matmul(np.matrix.transpose(vec),obs),vec)
        return expc
        
    def expectation_den(self,rho,obs):
        """
        Expectation values of observable for a density matrix
        Input:
            rho: input density matrix
            obs: observable operator
        Output:
            expc: the expectation value of the observable operator
        """
        return np.trace(np.matmul(rho,obs))
    

        
    

        
        
        
        
        