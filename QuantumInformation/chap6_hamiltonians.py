"""
Created on Sun Jul  7 17:42:28 2019

@authors: Pranay Barkataki 

Nth neighbor inhomogeneous spin interaction Hamiltonian with periodic boundary 
condition
"""
import numpy as np
from QuantumInformation import QuantumMechanics as QM
qobj=QM()

class Hamiltonian:
    
    def field_xham(self,N,mode='homogeneous',B=1):
        """
        Constructs Hamiltonian of external magnetic field in X direction
        Input
            N: number of spins
            mode: 'homogeneous' or 'inhomogeneous' magnetic field
            B: it list of value if mode='inhomogeneous', and constant if
               mode='homogeneous' 
        Output
            xham: Hamiltonian
        """
        assert mode == 'homogeneous' or mode == 'inhomogeneous',\
        "Entered mode is invalid"
        assert N >= 1, "number of spins entered is not correct"
        if mode == 'homogeneous':
            xham=np.zeros([2**N,2**N],dtype='float64')
            for i in range(0,2**N,1):
                for j in range(0,2**N,1):
                    bvec=qobj.decimal_binary(j,N)
                    sum1=0.0
                    for k in range(0,N):
                        bb=np.copy(bvec)
                        bb[k]=1-bb[k]
                        row=qobj.binary_decimal(bb)
                        if row == i:
                            sum1=sum1+B
                    xham[i,j]=sum1                             
        else:
            assert len(B)==N,\
            "The entered values of magnetic strengths are not equal to number of spins"
            xham=np.zeros([2**N,2**N],dtype='float64')
            for i in range(0,2**N,1):
                for j in range(0,2**N,1):
                    bvec=qobj.decimal_binary(j,N)
                    sum1=0.0
                    for k in range(0,N):
                        bb=np.copy(bvec)
                        bb[k]=1-bb[k]
                        row=qobj.binary_decimal(bb)
                        if row == i:
                            sum1=sum1+B[k]
                    xham[i,j]=sum1
        return xham
    
    def field_yham(self,N,mode='homogeneous',B=1):
        """
        Constructs Hamiltonian of external magentic field in Y direction
        Input
            N: number of spins
            mode: 'homogeneous' or 'inhomogeneous' magnetic field
            B: it list of value if mode='inhomogeneous', and constant if
               mode='homogeneous' 
        Output
            yham: Hamiltonian
        """
        assert mode == 'homogeneous' or mode == 'inhomogeneous',\
        "Entered mode is invalid"
        assert N >= 1, "number of spins entered is not correct"
        if mode == 'homogeneous':
            yham=np.zeros([2**N,2**N],dtype=np.complex_)
            for i in range(0,2**N,1):
                for j in range(0,2**N,1):
                    bvec=qobj.decimal_binary(j,N)
                    sum1=0.0
                    for k in range(0,N):
                        bb=np.copy(bvec)
                        bb[k]=1-bb[k]
                        row=qobj.binary_decimal(bb)
                        if row == i:
                            sum1=sum1+(B*complex(0,1)*((-1)**bvec[k]))
                    yham[i,j]=sum1                             
        else:
            assert len(B)==N,\
            "The entered values of magnetic strengths are not equal to number of spins"
            yham=np.zeros([2**N,2**N],dtype=np.complex_)
            for i in range(0,2**N,1):
                for j in range(0,2**N,1):
                    bvec=qobj.decimal_binary(j,N)
                    sum1=0.0
                    for k in range(0,N):
                        bb=np.copy(bvec)
                        bb[k]=1-bb[k]
                        row=qobj.binary_decimal(bb)
                        if row == i:
                            sum1=sum1+(B[k]*complex(0,1)*((-1)**bvec[k]))
                    yham[i,j]=sum1
        return yham
    
    def field_zham(self,N,mode='homogeneous',B=1):
        """
        Constructs Hamiltonian of external magentic field in Z direction
        Input
            N: number of spins
            mode: 'homogeneous' or 'inhomogeneous' magnetic field
            B: it list of value if mode='inhomogeneous', and constant if
               mode='homogeneous' 
        Output
            zham: Hamiltonian
        """
        assert mode == 'homogeneous' or mode == 'inhomogeneous',\
        "Entered mode is invalid"
        assert N >= 1, "number of spins entered is not correct"
        zham=np.zeros([2**N,2**N],dtype='float64')
        if mode == 'homogeneous':
            for i in range(0,2**N):
                sum1=0.0
                bvec=qobj.decimal_binary(i,N)
                for k in range(0,N):
                    sum1=sum1+(((-1)**bvec[k])*B)
                zham[i,i]=sum1
        else:
            assert len(B)==N,\
            "The entered values of magnetic strengths are not equal to number of spins"
            for i in range(0,2**N):
                sum1=0.0
                bvec=qobj.decimal_binary(i,N)
                for k in range(0,N):
                    sum1=sum1+(((-1)**bvec[k])*B[k])
                zham[i,i]=sum1
        return zham     

    def ham_xx(self,N,nn=1,mode='homogeneous',jx=1,condition='periodic'):
        """
        Constructs Hamiltonian of spin-spin interaction in X direction
        Input
            N: number of spins
            nn: specifying rth interaction
            mode: 'homogeneous' or 'inhomogeneous' magnetic field
            jx: it list of values if mode='inhomogeneous', and constant if
                 mode='homogeneous'
            condition:  defining boundary conditions of the Hamiltonian
        Output
            ham: Hamiltonian
        """
        assert condition == 'periodic' or condition == 'open'
        if condition == 'periodic':
            kn=N
        else:
            kn=N-nn
        assert mode == 'homogeneous' or mode == 'inhomogeneous',\
        "Entered mode is invalid"
        assert N >= 1, "number of spins entered is not correct"
        assert nn<=N-1 and nn>=1,"Not valid interaction"
        if mode == 'homogeneous':
            ham=np.zeros([2**N,2**N],dtype='float64')
            col2=np.zeros([N,1])
            for i in range(0,2**N):
                for j in range(0,2**N):
                    col=qobj.decimal_binary(j,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        col2=col.copy()
                        col2[k]=1-col2[k]
                        col2[k1]=1-col2[k1]
                        dec=qobj.binary_decimal(col2)
                        if dec==i:
                            inn=1
                        else:
                            inn=0
                        ham[i,j]=ham[i,j]+(inn*jx)
        else:
            if condition=='periodic':
                assert len(jx)==N,\
                "The entered values of magnetic strengths are not equal to number of spins"
            else:
                assert len(jx)==N-nn,\
                "The entered values of magnetic strengths are not equal to number of spins"
            ham=np.zeros([2**N,2**N],dtype='float64')
            col2=np.zeros([N,1])
            for i in range(0,2**N):
                for j in range(0,2**N):
                    col=qobj.decimal_binary(j,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        col2=col.copy()
                        col2[k]=1-col2[k]
                        col2[k1]=1-col2[k1]
                        dec=qobj.binary_decimal(col2)
                        if dec==i:
                            inn=1
                        else:
                            inn=0
                        ham[i,j]=ham[i,j]+(inn*jx[k])
        return ham               
    
    def ham_yy(self,N,nn=1,mode='homogeneous',jy=1,condition='periodic'):
        """
        Constructs Hamiltonian of spin-spin interaction in Y direction
        Input
            N: number of spins
            nn: specifying rth interaction
            mode: 'homogeneous' or 'inhomogeneous' magnetic field
            jy: it list of values if mode='inhomogeneous', and constant if
                mode='homogeneous'
            condition:  defining boundary conditions of the Hamiltonian
        Output
            ham: Hamiltonian
        
        """
        assert condition == 'periodic' or condition == 'open'
        if condition == 'periodic':
            kn=N
        else:
            kn=N-nn
        assert mode == 'homogeneous' or mode == 'inhomogeneous',\
        "Entered mode is invalid"
        assert N >= 1, "number of spins entered is not correct"
        assert nn<=N-1 and nn>=1,"Not valid interaction"
        ham=np.zeros([2**N,2**N],dtype='float64')
        col2=np.zeros([N,1])
        if mode == 'homogeneous':
            for i in range(0,2**N):
                for j in range(0,2**N):
                    col=qobj.decimal_binary(j,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        col2=col.copy()
                        col2[k]=1-col2[k]
                        col2[k1]=1-col2[k1]
                        if col2[k]==col2[k1]:
                            ind=-1
                        else:
                            ind=1
                        dec=qobj.binary_decimal(col2)
                        if dec==i:
                            inn=1
                        else:
                            inn=0
                        ham[i,j]=ham[i,j]+(inn*jy*ind)
        else:
            if condition=='periodic':
                assert len(jy)==N,\
                "The entered values of magnetic strengths are not equal to number of spins"
            else:
                assert len(jy)==N-nn,\
                "The entered values of magnetic strengths are not equal to number of spins"
            for i in range(0,2**N):
                for j in range(0,2**N):
                    col=qobj.decimal_binary(j,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        col2=col.copy()
                        col2[k]=1-col2[k]
                        col2[k1]=1-col2[k1]
                        if col2[k]==col2[k1]:
                            ind=-1
                        else:
                            ind=1
                        dec=qobj.binary_decimal(col2)
                        if dec==i:
                            inn=1
                        else:
                            inn=0
                        ham[i,j]=ham[i,j]+(inn*jy[k]*ind)
        return ham
        
    def ham_zz(self,N,nn=1,mode='homogeneous',jz=1,condition='periodic'):
        """
        Constructs Hamiltonian of spin-spin interaction in Z direction
        Input
            N: number of spins
            nn: specifying rth interaction
            mode: 'homogeneous' or 'inhomogeneous' magnetic field
            jz: it list of values if mode='inhomogeneous', and constant if
                mode='homogeneous'
            condition:  defining boundary conditions of the Hamiltonian
        Output
            ham: Hamiltonian
        """
        assert condition == 'periodic' or condition == 'open'
        if condition == 'periodic':
            kn=N
        else:
            kn=N-nn
        assert mode == 'homogeneous' or mode == 'inhomogeneous',\
        "Entered mode is invalid"
        assert N >= 1, "number of spins entered is not correct"
        assert nn<=N-1 and nn>=1,"Not valid interaction"
        ham=np.zeros([2**N,2**N],dtype='float64')
        if mode == 'homogeneous':
            for i in range(0,2**N):
                for j in range(0,2**N):
                    col=qobj.decimal_binary(j,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        egv1=1-2*col[k]
                        egv1=int(egv1)
                        egv2=1-2*col[k1]
                        egv2=int(egv2)
                        if i==j:
                            inn=1
                        else:
                            inn=0
                        ham[i,j]=ham[i,j]+(inn*jz*egv1*egv2)
        else:
            if condition=='periodic':
                assert len(jz)==N,\
                "The entered values of magnetic strengths are not equal to number of spins"
            else:
                assert len(jz)==N-nn,\
                "The entered values of magnetic strengths are not equal to number of spins"
            for i in range(0,2**N):
                for j in range(0,2**N):
                    col=qobj.decimal_binary(j,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        egv1=1-2*col[k]
                        egv1=int(egv1)
                        egv2=1-2*col[k1]
                        egv2=int(egv2)
                        if i==j:
                            inn=1
                        else:
                            inn=0
                        ham[i,j]=ham[i,j]+(inn*jz[k]*egv1*egv2)
        return ham
    
    def heisenberg_hamiltonian(self,N,nn=1,mode='homogeneous',j=1.0,\
                               condition='periodic'):
        """
        Construct Heisenberg interaction type Hamiltonian
        Input
            N: number of spins
            nn: specifying rth interaction
            mode: 'homogeneous' or 'inhomogeneous' magnetic field
            j: it list of values if mode='inhomogeneous', and constant if
                mode='homogeneous'
            condition:  defining boundary conditions of the Hamiltonian
        Output
            ham: Hamiltonian

        """
        assert condition == 'periodic' or condition == 'open'
        if condition == 'periodic':
            kn=N
        else:
            kn=N-nn
        assert mode == 'homogeneous' or mode == 'inhomogeneous',\
        "Entered mode is invalid"
        assert N >= 1, "number of spins entered is not correct"
        assert nn<=N-1 and nn>=1,"Not valid interaction"
        ham=np.zeros([2**N,2**N],dtype='float64')
        if mode == 'homogeneous':
            for i in range(0,2**N):
                for jj in range(0,2**N):
                    col=qobj.decimal_binary(jj,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        col2=col.copy()
                        temp=col2[k]
                        col2[k]=col2[k1]
                        col2[k1]=temp
                        dec=qobj.binary_decimal(col2)
                        if dec==i:
                            inn=1
                        else:
                            inn=0
                        ham[i,jj]=ham[i,jj]+(inn*j*2.0)
                        if i==jj:
                            ham[i,jj]=ham[i,jj]-j
        else:
            if condition=='periodic':
                assert len(j)==N,\
                "Entered list of values j are not equal to number of interaction in PBC"
            else:
                assert len(j)==N-nn,\
                "Entered list of values j are not equal to number of interaction in OBC"
            for i in range(0,2**N):
                for jj in range(0,2**N):
                    col=qobj.decimal_binary(jj,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        col2=col.copy()
                        temp=col2[k]
                        col2[k]=col2[k1]
                        col2[k1]=temp
                        dec=qobj.binary_decimal(col2)
                        if dec==i:
                            inn=1
                        else:
                            inn=0
                        ham[i,jj]=ham[i,jj]+(inn*j[k]*2.0)
                        if i==jj:
                            ham[i,jj]=ham[i,jj]-j[k]
        return ham
    
    def dm_xham(self,N,nn=1,mode='homogeneous',dx=1.0, condition='periodic'):
        """
        Construct DM Hamiltonian in X direction
        Input
            N: number of spins
            nn: specifying rth interaction
            mode: 'homogeneous' or 'inhomogeneous' magnetic field
            dx: it list of values if mode='inhomogeneous', and constant if
                mode='homogeneous'
            condition:  defining boundary conditions of the Hamiltonian
        Output
            ham: Hamiltonian

        """
        assert condition == 'periodic' or condition == 'open'
        if condition == 'periodic':
            kn=N
        else:
            kn=N-nn
        assert mode == 'homogeneous' or mode == 'inhomogeneous',\
        "Entered mode is invalid"
        assert N >= 1, "number of spins entered is not correct"
        assert nn<=N-1 and nn>=1,"Not valid interaction"
        ham=np.zeros([2**N,2**N],dtype=np.complex_)
        if mode == 'homogeneous':
            for i in range(0,2**N):
                for j in range(i,2**N):
                    col=qobj.decimal_binary(j,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        col2=col.copy()
                        col2[k]=1-col2[k]
                        dec=qobj.binary_decimal(col2)
                        inn1=complex(0.0,0.0)
                        if dec==i:                            
                            phase=complex(0,1)*(-1)**col[k]
                            szpart=(-1)**col[k1]
                            inn1=phase*szpart
                        col2=col.copy()
                        col2[k1]=1-col2[k1]
                        dec=qobj.binary_decimal(col2)
                        inn2=complex(0.0,0.0)
                        if dec==i:                            
                            phase=complex(0,1)*(-1)**col[k1]
                            szpart=(-1)**col[k]
                            inn2=phase*szpart
                        ham[i,j]=ham[i,j]+((inn1-inn2)*dx)
                    ham[j,i]=np.conjugate(ham[i,j])
        else:
            if condition=='periodic':
                assert len(dx)==N,\
                "The entered values of magnetic strengths are not equal to number of spins"
            else:
                assert len(dx)==N-nn,\
                "The entered values of magnetic strengths are not equal to number of spins"
            for i in range(0,2**N):
                for j in range(i,2**N):
                    col=qobj.decimal_binary(j,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        col2=col.copy()
                        col2[k]=1-col2[k]
                        dec=qobj.binary_decimal(col2)
                        inn1=0.0
                        if dec==i:                            
                            phase=complex(0,1)*(-1)**col[k]
                            szpart=(-1)**col[k1]
                            inn1=phase*szpart
                        col2=col.copy()
                        col2[k1]=1-col2[k1]
                        dec=qobj.binary_decimal(col2)
                        inn2=0.0
                        if dec==i:                            
                            phase=complex(0,1)*(-1)**col[k1]
                            szpart=(-1)**col[k]
                            inn2=phase*szpart
                        ham[i,j]=ham[i,j]+((inn1-inn2)*dx[k])
                    ham[j,i]=np.conjugate(ham[i,j])
        return ham
    
    def dm_yham(self,N,nn=1,mode='homogeneous',dy=1.0, condition='periodic'):
        """
        Construct DM Hamiltonian in Y direction
        Input
            N: number of spins
            nn: specifying rth interaction
            mode: 'homogeneous' or 'inhomogeneous' magnetic field
            dy: it list of values if mode='inhomogeneous', and constant if
                mode='homogeneous'
            condition:  defining boundary conditions of the Hamiltonian
        Output
            ham: Hamiltonian

        """
        assert condition == 'periodic' or condition == 'open'
        if condition == 'periodic':
            kn=N
        else:
            kn=N-nn
        assert mode == 'homogeneous' or mode == 'inhomogeneous',\
        "Entered mode is invalid"
        assert N >= 1, "number of spins entered is not correct"
        assert nn<=N-1 and nn>=1,"Not valid interaction"
        ham=np.zeros([2**N,2**N],dtype='float64')
        if mode == 'homogeneous':
            for i in range(0,2**N):
                for j in range(i,2**N):
                    col=qobj.decimal_binary(j,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        col2=col.copy()
                        col2[k1]=1-col2[k1]
                        dec=qobj.binary_decimal(col2)
                        inn1=0.0
                        if dec==i:                            
                            inn1=(-1)**col[k]
                        col2=col.copy()
                        col2[k]=1-col2[k]
                        dec=qobj.binary_decimal(col2)
                        inn2=0.0
                        if dec==i:                          
                            inn2=(-1)**col[k1]
                        ham[i,j]=ham[i,j]+((inn1-inn2)*dy)
                    ham[j,i]=ham[i,j]
        else:
            if condition=='periodic':
                assert len(dy)==N,\
                "The entered values of magnetic strengths are not equal to number of spins"
            else:
                assert len(dy)==N-nn,\
                "The entered values of magnetic strengths are not equal to number of spins"
            for i in range(0,2**N):
                for j in range(i,2**N):
                    col=qobj.decimal_binary(j,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        col2=col.copy()
                        col2[k1]=1-col2[k1]
                        dec=qobj.binary_decimal(col2)
                        inn1=0.0
                        if dec==i:                            
                            inn1=(-1)**col[k]
                        col2=col.copy()
                        col2[k]=1-col2[k]
                        dec=qobj.binary_decimal(col2)
                        inn2=0.0
                        if dec==i:                          
                            inn2=(-1)**col[k1]
                        ham[i,j]=ham[i,j]+((inn1-inn2)*dy[k])
                    ham[j,i]=ham[i,j]
        return ham
        
    def dm_zham(self,N,nn=1,mode='homogeneous',dz=1.0, condition='periodic'):
        """
        Construct DM Hamiltonian in Z direction
        Input
            N: number of spins
            nn: specifying rth interaction
            mode: 'homogeneous' or 'inhomogeneous' magnetic field
            dz: it list of values if mode='inhomogeneous', and constant if
                mode='homogeneous'
            condition:  defining boundary conditions of the Hamiltonian
        Output
            ham: Hamiltonian

        """
        assert condition == 'periodic' or condition == 'open'
        if condition == 'periodic':
            kn=N
        else:
            kn=N-nn
        assert mode == 'homogeneous' or mode == 'inhomogeneous',\
        "Entered mode is invalid"
        assert N >= 1, "number of spins entered is not correct"
        assert nn<=N-1 and nn>=1,"Not valid interaction"
        ham=np.zeros([2**N,2**N],dtype=np.complex_)
        if mode == 'homogeneous':
            for i in range(0,2**N):
                for j in range(i,2**N):
                    col=qobj.decimal_binary(j,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        col2=col.copy()
                        col2[k]=1-col2[k]
                        col2[k1]=1-col2[k1]
                        dec=qobj.binary_decimal(col2)
                        inn1=complex(0.0,0.0)
                        inn2=complex(0.0,0.0)
                        if dec==i:                            
                            inn1=complex(0,1)*(-1)**col[k1]
                            inn2=complex(0,1)*(-1)**col[k]
                        ham[i,j]=ham[i,j]+((inn1-inn2)*dz)
                    ham[j,i]=np.conjugate(ham[i,j])
        else:
            if condition=='periodic':
                assert len(dz)==N,\
                "The entered values of magnetic strengths are not equal to number of spins"
            else:
                assert len(dz)==N-nn,\
                "The entered values of magnetic strengths are not equal to number of spins"
            for i in range(0,2**N):
                for j in range(i,2**N):
                    col=qobj.decimal_binary(j,N)
                    for k in range(0,kn):
                        k1=k+nn
                        if k1>=N:
                            k1=k1-N
                        col2=col.copy()
                        col2[k]=1-col2[k]
                        col2[k1]=1-col2[k1]
                        dec=qobj.binary_decimal(col2)
                        inn1=complex(0.0,0.0)
                        inn2=complex(0.0,0.0)
                        if dec==i:                            
                            inn1=complex(0,1)*(-1)**col[k1]
                            inn2=complex(0,1)*(-1)**col[k]
                        ham[i,j]=ham[i,j]+((inn1-inn2)*dz[k])
                    ham[j,i]=np.conjugate(ham[i,j])
        return ham
        
        
        
        
    
    