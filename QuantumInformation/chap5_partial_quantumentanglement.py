"""
Created on Thu Aug 22 19:18:53 2019

@author: Pranay Barkataki 
"""

import numpy as np
import math
from QuantumInformation import RecurNum
from QuantumInformation import LinearAlgebra as LA
from QuantumInformation import QuantumMechanics as QM
import scipy.linalg.lapack as la
import re

qobj=QM()

class PartialTr:
    def __init__(self):
        """It is a class dealing with partial trace and transpose
        it primarily intrinsic functions of uses numpy, math, cmath.
        """
    # partial trace operation subroutine for a real pure state 
    #entry is in the column form
    def partial_trace_vec(self,state,sub_tr):
        """
        Partial trace operation on a quantum state
        Input:
            state: real state vector
            sub_tr: details of the subsystems not to be traced out
        Output:
            red_den: reduced density matrix
        """
        typestate=str(state.dtype)
        N=int(math.log2(state.shape[0]))
        length=len(sub_tr)
        # count=length, and count0= N-length
        assert set(sub_tr).issubset(set(np.arange(1,N+1))),\
        "Invalid subsystems to be traced out"
        if re.findall("^complex",typestate):
            red_den=np.zeros([2**(length),2**(length)],dtype=np.complex_)
            vec=np.zeros([(N-length),1])
            im=0
            for ii in range(1,N+1):
                if ii not in sub_tr:
                    vec[im]=2**(N-ii)
                    im=im+1
            mylist=[]
            icount=0
            sum2=0
            RecurNum.recur_comb_add(mylist,vec,icount,sum2)
            irow=np.zeros([N,1])
            icol=np.zeros([N,1])
            mylist=np.array(mylist)
            len_mylist=len(mylist)    
            for i1 in range(0,2**length):
                col1=self.__dectobin(i1,length)
                for i2 in range(0,2**length):
                    col2=self.__dectobin(i2,length)
                    i3=0
                    for k in range(0,N):
                        if k+1 not in sub_tr:
                            irow[k]=0
                        else:
                            irow[k]=col1[i3]
                            i3=i3+1 
                    ic=0
                    for k2 in range(0,N):
                        if k2+1 not in sub_tr:
                            icol[k2]=0
                        else:
                            icol[k2]=col2[ic]
                            ic=ic+1
                    icc=self.__bintodec(irow)
                    jcc=self.__bintodec(icol)
                    red_den[i1,i2]=red_den[i1,i2]+(state[icc]*\
                           np.conjugate(state[jcc]))
                    for jj in range(0,len_mylist):
                        icc2=icc+mylist[jj]
                        jcc2=jcc+mylist[jj]
                        red_den[i1,i2]=red_den[i1,i2]+(state[icc2]*\
                               np.conjugate(state[jcc2]))
        else:
            red_den=np.zeros([2**(length),2**(length)],dtype='float64')
            vec=np.zeros([(N-length),1])
            im=0
            for ii in range(1,N+1):
                if ii not in sub_tr:
                    vec[im]=2**(N-ii)
                    im=im+1
            mylist=[]
            icount=0
            sum2=0
            RecurNum.recur_comb_add(mylist,vec,icount,sum2)
            irow=np.zeros([N,1])
            icol=np.zeros([N,1])
            mylist=np.array(mylist)
            len_mylist=len(mylist)    
            for i1 in range(0,2**length):
                col1=self.__dectobin(i1,length)
                for i2 in range(0,2**length):
                    col2=self.__dectobin(i2,length)
                    i3=0
                    for k in range(0,N):
                        if k+1 not in sub_tr:
                            irow[k]=0
                        else:
                            irow[k]=col1[i3]
                            i3=i3+1 
                    ic=0
                    for k2 in range(0,N):
                        if k2+1 not in sub_tr:
                            icol[k2]=0
                        else:
                            icol[k2]=col2[ic]
                            ic=ic+1
                    icc=self.__bintodec(irow)
                    jcc=self.__bintodec(icol)
                    red_den[i1,i2]=red_den[i1,i2]+(state[icc]*state[jcc])
                    for jj in range(0,len_mylist):
                        icc2=icc+mylist[jj]
                        jcc2=jcc+mylist[jj]
                        red_den[i1,i2]=red_den[i1,i2]+(state[icc2]*state[jcc2])                   
        return(red_den)
    
    # partial trace operation for a real state density matrix    
    def partial_trace_den(self,state,sub_tr):
        """
        Partial trace operation on a density matrix
        Input:
            state: input real density matrix
            sub_tr: details of the subsystem not to be traced out
        Output:
            red_den: reduced density matrix
        """
        typestate=str(state.dtype)
        N=int(math.log2(state.shape[0]))
        length=len(sub_tr)
        # count=length, and count0= N-length
        assert set(sub_tr).issubset(set(np.arange(1,N+1))),\
        "Invalid subsystems to be traced out"
        if re.findall("^complex",typestate):
            red_den=np.zeros([2**(length),2**(length)],dtype=np.complex_)
            vec=np.zeros([(N-length),1])
            im=0
            for ii in range(1,N+1):
                if ii not in sub_tr:
                    vec[im]=2**(N-ii)
                    im=im+1
            mylist=[]
            icount=0
            sum2=0
            RecurNum.recur_comb_add(mylist,vec,icount,sum2)
            irow=np.zeros([N,1])
            icol=np.zeros([N,1])
            mylist=np.array(mylist)
            len_mylist=len(mylist)    
            for i1 in range(0,2**length):
                col1=self.__dectobin(i1,length)
                for i2 in range(0,2**length):
                    col2=self.__dectobin(i2,length)
                    i3=0
                    for k in range(0,N):
                        if k+1 not in sub_tr:
                            irow[k]=0
                        else:
                            irow[k]=col1[i3]
                            i3=i3+1 
                    ic=0
                    for k2 in range(0,N):
                        if k2+1 not in sub_tr:
                            icol[k2]=0
                        else:
                            icol[k2]=col2[ic]
                            ic=ic+1
                    icc=self.__bintodec(irow)
                    jcc=self.__bintodec(icol)
                    red_den[i1,i2]=red_den[i1,i2]+(state[icc,jcc])
                    for jj in range(0,len_mylist):
                        icc2=icc+mylist[jj]
                        jcc2=jcc+mylist[jj]
                        red_den[i1,i2]=red_den[i1,i2]+(state[icc2,jcc2])
        else:
            red_den=np.zeros([2**(length),2**(length)],dtype='float64')
            vec=np.zeros([(N-length),1])
            im=0
            for ii in range(1,N+1):
                if ii not in sub_tr:
                    vec[im]=2**(N-ii)
                    im=im+1
            mylist=[]
            icount=0
            sum2=0
            RecurNum.recur_comb_add(mylist,vec,icount,sum2)
            irow=np.zeros([N,1])
            icol=np.zeros([N,1])
            mylist=np.array(mylist)
            len_mylist=len(mylist)    
            for i1 in range(0,2**length):
                col1=self.__dectobin(i1,length)
                for i2 in range(0,2**length):
                    col2=self.__dectobin(i2,length)
                    i3=0
                    for k in range(0,N):
                        if k+1 not in sub_tr:
                            irow[k]=0
                        else:
                            irow[k]=col1[i3]
                            i3=i3+1 
                    ic=0
                    for k2 in range(0,N):
                        if k2+1 not in sub_tr:
                            icol[k2]=0
                        else:
                            icol[k2]=col2[ic]
                            ic=ic+1
                    icc=self.__bintodec(irow)
                    jcc=self.__bintodec(icol)
                    red_den[i1,i2]=red_den[i1,i2]+(state[icc,jcc])
                    for jj in range(0,len_mylist):
                        icc2=icc+mylist[jj]
                        jcc2=jcc+mylist[jj]
                        red_den[i1,i2]=red_den[i1,i2]+(state[icc2,jcc2])                   
        return(red_den)
        
    # Partial Transpose of real pure state
    def ptranspose_vec(self,state,sub_tr):
        """
        Partial transpose operation on a quantum state
        Parameters
            state : It is a real or complex state.
            sub_tr : List of number designating the subsystems
                     to be partially transposed.
        Returns
            denc2: It is partially transposed density matrix

        """
        N=int(math.log2(state.shape[0]))
        assert set(sub_tr).issubset(set(np.arange(1,N+1))),\
        "Invalid subsystems to be traced out"
        typestate=str(state.dtype)
        if re.findall("^complex",typestate):
            denc2=np.zeros([2**N,2**N],dtype=np.complex_)
            for i in range(state.shape[0]):
                vec_row=qobj.decimal_binary(i,N)
                for j in range(state.shape[0]):
                    vec_col=qobj.decimal_binary(j,N)
                    vec_row2=vec_row.copy()
                    for k in sub_tr:
                        temp=vec_row2[k-1]
                        vec_row2[k-1]=vec_col[k-1]
                        vec_col[k-1]=temp
                    row=qobj.binary_decimal(vec_row2)
                    col=qobj.binary_decimal(vec_col)
                    denc2[row,col]=state[i]*np.conjugate(state[j])
        else:            
            denc2=np.zeros([2**N,2**N],dtype='float64')
            for i in range(state.shape[0]):
                vec_row=qobj.decimal_binary(i,N)
                for j in range(state.shape[0]):
                    vec_col=qobj.decimal_binary(j,N)
                    vec_row2=vec_row.copy()
                    for k in sub_tr:
                        temp=vec_row2[k-1]
                        vec_row2[k-1]=vec_col[k-1]
                        vec_col[k-1]=temp
                    row=qobj.binary_decimal(vec_row2)
                    col=qobj.binary_decimal(vec_col)
                    denc2[row,col]=state[i]*state[j]
        return(denc2)
    
    # Partial Transpose of real density matrix
    def ptranspose_den(self,denc,sub_tr):
        """
        Partial transpose operation on density matrix
        Parameters
            denc : It is a real or complex density matrix.
            sub_tr : List of number designating the subsystems
                     to be partially transposed.
        Returns
            denc2: It is partially transposed density matrix
        """
        N=int(math.log2(denc.shape[0]))
        assert set(sub_tr).issubset(set(np.arange(1,N+1))),\
        "Invalid subsystems to be traced out"
        typestate=str(denc.dtype)
        if re.findall("^complex",typestate):
            denc2=np.zeros([2**N,2**N],dtype=np.complex_)
            for i in range(denc.shape[0]):
                vec_row=qobj.decimal_binary(i,N)
                for j in range(denc.shape[1]):
                    vec_col=qobj.decimal_binary(j,N)
                    vec_row2=vec_row.copy()
                    for k in sub_tr:
                        temp=vec_row2[k-1]
                        vec_row2[k-1]=vec_col[k-1]
                        vec_col[k-1]=temp
                    row=qobj.binary_decimal(vec_row2)
                    col=qobj.binary_decimal(vec_col)
                    denc2[row,col]=denc[i,j]
        else:
            denc2=np.zeros([2**N,2**N],dtype='float64')
            for i in range(denc.shape[0]):
                vec_row=qobj.decimal_binary(i,N)
                for j in range(denc.shape[1]):
                    vec_col=qobj.decimal_binary(j,N)
                    vec_row2=vec_row.copy()
                    for k in sub_tr:
                        temp=vec_row2[k-1]
                        vec_row2[k-1]=vec_col[k-1]
                        vec_col[k-1]=temp
                    row=qobj.binary_decimal(vec_row2)
                    col=qobj.binary_decimal(vec_col)
                    denc2[row,col]=denc[i,j]
        return(denc2)
        
    def __dectobin(self,n,l):
        """It converts decimal to binary.
        Attributes:
            n: entry of the decimal number
            l: length of the binary output
        Returns:
            dtb: a numpy array containing the binary equivalent of number n
        """
        import numpy as np
        p=n
        dtb=np.empty([l,1])
        for i in range(0,l):
            dtb[l-1-i]=int(p % 2)
            p=int(p/2)
        #print(dtb)    
        return(dtb)
        
    # Binary to decimal conversion
    def __bintodec(self,vec):
        """
        It converts biinary to decimal
        Attributes:
            vec: entry of 1D array of binary numbers {0,1}
        Returns:
            t: decimal equivalent of the vec
        """
        t=0
        for i in range(0,len(vec)):
            t=t+vec[len(vec)-1-i]*(2**i)
        #print(dtb)    
        return(int(t))


class Entanglement(PartialTr):
    
    # Concurrence calculation for a real pure state
    def concurrence_vec(self,state,i,j,eps=10**(-13)):
        """
        Calculation of concurrence for a quantum state
        Parameters
            state : Real or complex state
            i : It stores the place values of the qubits.
            j : It stores the place values of the qubits.
            eps : Below the eps value the eigenvalues will be considered zero.
                  The default is 10**(-13).

        Returns
            conc: concurrence value
        """
        sigmay=np.zeros([4,4],dtype='float64')
        typestate=str(state.dtype)
        if re.findall("^complex",typestate):
            sigmay[0,3]=-1
            sigmay[1,2]=1
            sigmay[2,1]=1
            sigmay[3,0]=-1
            sub_tr=[i,j]
            rdm= self.partial_trace_vec(state,sub_tr)
            rhot3=rdm@sigmay@np.conjugate(rdm)@sigmay
            w,vl,vr,info =la.zgeev(rhot3)
            wc=[]
            for i in range(0,4):
                if abs(w.item(i))<eps:
                    wc.append(0.000000000000000)
                else:
                    wc.append(abs(w.item(i)))
            wc.sort(reverse=True)
            wc=np.array(wc,dtype='float64')
            conc=math.sqrt(wc.item(0))-math.sqrt(wc.item(1))-\
            math.sqrt(wc.item(2))-math.sqrt(wc.item(3))
            if conc<0:
                conc=0
        else:    
            sigmay[0,3]=-1
            sigmay[1,2]=1
            sigmay[2,1]=1
            sigmay[3,0]=-1
            sub_tr=[i,j]
            rdm= self.partial_trace_vec(state,sub_tr)
            rhot3=rdm@sigmay@rdm@sigmay
            wr,wi,vl,vr,info =la.dgeev(rhot3)
            w=[]
            for i in range(0,4):
                if wr[i] < eps:
                    w.append(0.000000000000000)
                else:
                    w.append(np.float64(wr.item(i)))
            w.sort(reverse=True)
            w=np.array(w,dtype='float64')
            conc=math.sqrt(w.item(0))-math.sqrt(w.item(1))-\
            math.sqrt(w.item(2))-math.sqrt(w.item(3))
            if conc<0:
                conc=0.0
        return(np.float64(conc))
    
    # Concurrence calculation for real state density matrix 
    def concurrence_den(self,state,i,j,eps=10**(-13)):
        """
        Calculation of concurrence for a density matrix
        Parameters
            state : Real or complex density matrix
            i : It stores the place values of the qubits.
            j : It stores the place values of the qubits.
            eps : Below the eps value the eigenvalues will be considered zero.
                  The default is 10**(-13).

        Returns
            conc: concurrence value
        """
        sigmay=np.zeros([4,4],dtype='float64')
        typestate=str(state.dtype)
        if re.findall("^complex",typestate):
            sigmay[0,3]=-1
            sigmay[1,2]=1
            sigmay[2,1]=1
            sigmay[3,0]=-1
            sub_tr=[i,j]
            rdm= self.partial_trace_den(state,sub_tr)
            rhot3=rdm@sigmay@np.conjugate(rdm)@sigmay
            w,vl,vr,info =la.zgeev(rhot3)
            wc=[]
            for i in range(0,4):
                if abs(w.item(i))<eps:
                    wc.append(0.000000000000000)
                else:
                    wc.append(abs(w.item(i)))
            wc.sort(reverse=True)
            wc=np.array(wc,dtype='float64')
            conc=math.sqrt(wc.item(0))-math.sqrt(wc.item(1))-\
            math.sqrt(wc.item(2))-math.sqrt(wc.item(3))
            if conc<0:
                conc=0
        else:
            sigmay[0,3]=-1
            sigmay[1,2]=1
            sigmay[2,1]=1
            sigmay[3,0]=-1
            sub_tr=[i,j]
            rdm= self.partial_trace_den(state,sub_tr)
            rhot3=rdm@sigmay@rdm@sigmay
            wr,wi,vl,vr,info =la.dgeev(rhot3)
            w=[]
            for i in range(0,4):
                if wr[i] < eps:
                    w.append(0.000000000000000)
                else:
                    w.append(np.float64(wr.item(i)))
            w.sort(reverse=True)
            w=np.array(w,dtype='float64')
            conc=math.sqrt(w.item(0))-math.sqrt(w.item(1))-\
            math.sqrt(w.item(2))-math.sqrt(w.item(3))
            if conc<0:
                conc=0.0
        return(np.float64(conc))    
    
    # Block entropy for a pure real state
    def block_entropy_vec(self,state,sub_tr,eps=10**(-13)):
        """
        Calculation of block entropy for a quantum state
        Parameters
            state : Real or complex state
            sub_tr: List of numbers designating the particular subsystems
                    not to be traced out.
            eps : Below the eps value the eigenvalues will be considered zero.
                  The default is 10**(-13).

        Returns
            Bent: Block entropy value
        """
        typestate=str(state.dtype)
        rdm= self.partial_trace_vec(state,sub_tr)
        if re.findall("^complex",typestate):
            w,v,info=la.zheev(rdm)
        else:
            w,v,info=la.dsyev(rdm)
        wlen=len(w)
        Bent=0.0
        for x in range(0,wlen):
            if abs(w.item(x))<eps:
                w[x]=0.000000000000000
            else:
                assert w.item(x) > 0.0,\
                "The density matrix entered is not correct as the eigenvalues are negative"
                Bent=Bent-(w.item(x)*math.log(w.item(x),2))
        return(Bent)

    # Block entropy for a pure real density matrix
    def block_entropy_den(self,state,sub_tr,eps=10**(-13)):
        """
        Calculation of block entropy for a density matrix
        Parameters
            state : Real or complex density matrix
            sub_tr: List of numbers designating the particular subsystems
                    not to be traced out.
            eps : Below the eps value the eigenvalues will be considered zero.
                  The default is 10**(-13).

        Returns
            Bent: Block entropy value
        """
        typestate=str(state.dtype)
        rdm= self.partial_trace_den(state,sub_tr)
        if re.findall("^complex",typestate):
            w,v,info=la.zheev(rdm)
        else:
            w,v,info=la.dsyev(rdm)
        wlen=len(w)
        Bent=0.0
        for x in range(0,wlen):
            if abs(w.item(x))<eps:
                w[x]=0.000000000000000
            else:
                assert w.item(x) > 0.0,\
                "The density matrix entered is not correct as the eigenvalues are negative"
                Bent=Bent-(w.item(x)*math.log(w.item(x),2))
        return(Bent)
       
    # Q measure for pure real state
    def QMeasure_vec(self,state): 
        """
        Calculation of Q measure for a quantum state
        Parameters
            state : Real or complex state

        Returns
            Qmeas: Q measure value

        """
        NN=math.log2(state.shape[0])/math.log2(2)
        NN=int(NN)
        sub_tr=np.zeros([NN,1])
        sum3=0.0
        for x in range(0,NN):
            sub_tr=[]
            sub_tr.append(x+1)
            rho=self.partial_trace_vec(state,sub_tr)
            rho=np.matmul(rho,rho)
            tr2=np.trace(rho)
            sum3=sum3+tr2
        Qmeas=2*(1-(sum3/NN))
        return abs(Qmeas)
        
    # Q measure for real density matrix
    def QMeasure_den(self,den):
        """
        Calculation of Q measure for a density matrix
        Parameters
            den : Real or complex density matrix

        Returns
            Qmeas: Q measure value

        """
        NN=math.log2(den.shape[0])/math.log2(2)  
        NN=int(NN)
        sub_tr=np.zeros([NN,1])
        sum3=0.0
        for x in range(0,NN):
            sub_tr=[]
            sub_tr.append(x+1)
            rho=self.partial_trace_den(den,sub_tr)
            rho=np.matmul(rho,rho)
            tr2=np.trace(rho)
            sum3=sum3+tr2
        Qmeas=2*(1-(sum3/NN))
        return abs(Qmeas)

    # Negativity of real pure state
    def negativity_log_vec(self,state,sub_tr,eps=10**(-13)):
        """
        Calculation of negativity and logarithmic negativity for a quantum state
        Parameters
            state : Real or complex state
            sub_tr: List of numbers designating the particular subsystems
                    to be transposed.
            eps : Below the eps value the eigenvalues will be considered zero.
                  The default is 10**(-13).

        Returns
            negv,lognegv : negativity and log negativity values, respectively
        """
        laobj=LA()
        typestate=str(state.dtype)
        rhoa=self.ptranspose_vec(state,sub_tr)
        if re.findall("^complex",typestate):
            negv=laobj.trace_norm_cmatrix(rhoa,precision=eps)
        else:
            negv=laobj.trace_norm_rmatrix(rhoa,precision=eps)
        assert negv > 0.0,\
        "The density matrix entered is not correct as the negativity is negative"
        lognegv=math.log2(negv)
        negv=(negv-1)/2
        return(negv,lognegv)
    
    # Negativity of real pure state
    def negativity_log_den(self,den,sub_tr,eps=10**(-13)):
        """
        Calculation of negativity and logarithmic negativity for a density matrix
        Parameters
            state : Real or complex density matrix
            sub_tr: List of numbers designating the particular subsystems
                    to be transposed.
            eps : Below the eps value the eigenvalues will be considered zero.
                  The default is 10**(-13).

        Returns
            negv,lognegv : negativity and log negativity values, respectively
        """
        laobj=LA()
        typestate=str(den.dtype)
        rhoa=self.ptranspose_den(den,sub_tr)
        if re.findall("^complex",typestate):
            negv=laobj.trace_norm_cmatrix(rhoa,precision=eps)
        else:
            negv=laobj.trace_norm_rmatrix(rhoa,precision=eps)
        assert negv > 0.0,\
        "The density matrix entered is not correct as the negativity is negative"
        lognegv=math.log2(negv)
        negv=(negv-1)/2
        return(negv,lognegv)
        
    def renyi_entropy(self,rho,alpha):
        """
        Calculation of Renyi entropy
        Parameters
            rho : Real or complex density matrix
            alpha : It is the value of Renyi index

        Returns
            renyi : Renyi Entropy value

        """
        assert alpha != 1.0, "alpha should not be equal to 1"
        typerho=str(rho.dtype)
        laobj=LA()
        if re.findall('^complex',typerho):
            renyi=math.log(abs(np.trace(laobj.power_hmatrix(rho,alpha))))/(1-alpha)
        else:
            renyi=math.log(np.trace(laobj.power_smatrix(rho,alpha)))/(1-alpha)
        return renyi
    
    def entanglement_spectrum(self,rho):
        """
        Calculation of entanglement spectrum of a density matrix
        Parameters
            rho :  Real or complex density matrix

        Returns
            eigenvalues : List containing the eigenvalues of rho
            logeigenvalues : List containing the negative logarithmic
                            eigenvalues of rho
        """
        typerho=str(rho.dtype)
        if re.findall('^complex',typerho):
            eigenvalues,eigenvectors,info=la.zheev(rho)
        else:
            eigenvalues,eigenvectors,info=la.dsyev(rho)
        logeigenvalues=np.zeros([eigenvalues.shape[0]],dtype='float64')
        for i in range(0,eigenvalues.shape[0]):
            assert eigenvalues[i]>0.0,\
            "The eigenvalues of the matrix is coming less than equal to zero"
            logeigenvalues[i]=(-1)*math.log(eigenvalues[i])
        return (eigenvalues,logeigenvalues)
    
    def residual_entanglement_vec(self,state):
        """
        Calculation of residual entanglement for a three-qubit quantum state
        Parameters
            state : Real or complex 3-qubit state

        Returns
            res_tang : Residual entanglement value

        """
        assert state.shape[0]==8,"It is not a three qubit quantum system"
        det=np.linalg.det(self.partial_trace_vec(state,[1]))
        det=4*det
        res_tang=det-(self.concurrence_vec(state,1,2)**2)-\
        (self.concurrence_vec(state,1,3)**2)
        res_tang=abs(res_tang)
        return res_tang
    
    def residual_entanglement_den(self,den):
        """
        Calculation of residual entanglement for a three-qubit density matrix
        Parameters
            den : Real or complex 3-qubit density matrix

        Returns
            res_tang : Residual entanglement value
        """
        assert den.shape[0]==8,"It is not a three qubit quantum system"
        det=np.linalg.det(self.partial_trace_den(den,[1]))
        det=4*det
        res_tang=det-(self.concurrence_den(den,1,2)**2)-\
        (self.concurrence_den(den,1,3)**2)
        res_tang=abs(res_tang)
        return res_tang




