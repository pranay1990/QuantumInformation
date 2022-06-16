"""
Created on Mon Jul 15 18:26:57 2019

@author: Dr. M. S. Ramkarthik and Dr. Pranay Barkataki
"""

def recur_comb_add(mylist,vec,icount,sum2):
    lenvec=len(vec)
    if icount<=lenvec-1:
        for j in range(icount,lenvec):
            sum3=sum2+vec[j]
            sum3=int(sum3)
            mylist.append(sum3)
            recur_comb_add(mylist,vec,j+1,sum3)
            if j==lenvec:
                return()
    if icount==lenvec:
        return()


def RecurChainRL1(row,tot_spins,icount,mylist,shift):
    len_row=len(row)
    if icount<len_row:
        for x in range(icount,len_row):
            row2=row.copy()
            if x>=0:
                row2[x]=1
            if shift==0:
                y=0
            if shift==1:
                y=1
            sumr=0
            if shift==0:
                for x1 in range(len_row-1,-1,-1):
                    if row2[x1]==0:
                        sumr=sumr+(2**y)
                    if row2[x1]==1:
                        sumr=sumr+(2**(y+1))
                    y=y+2
                mylist.append(sumr)
            if shift==1:
                for x1 in range(len_row-1,-1,-1):
                    if row2[x1]==0 and x1==len_row-1:
                        sumr=sumr+(2**(tot_spins-1))
                    if row2[x1]==1 and x1==len_row-1:
                        sumr=sumr+1
                    if row2[x1]==0 and x1!=len_row-1:
                        sumr=sumr+(2**y)
                        y=y+2
                    if row2[x1]==1 and x1!=len_row-1:
                        sumr=sumr+(2**(y+1))
                        y=y+2
                mylist.append(sumr)
            if x<len_row-1 and x!=-1:    
                RecurChainRL1(row2,tot_spins,x+1,mylist,shift)
            if x==len_row-1:
                return()
    if icount >= len_row:
        return()


def RecurChainRL2(row,tot_spins,icount,mylist,shift):
    len_row=len(row)
    if icount<len_row:
        for x in range(icount,len_row):
            row2=row.copy()
            if x>=0:
                row2[x]=1
            if shift==0:
                y=0
            if shift==1:
                y=1
            sumr=0
            cntr=0
            if shift==0:
                for x1 in range(len_row-1,-1,-1):
                    if row2[x1]==0:
                        sumr=sumr+(2**y)
                        cntr=cntr+0
                    if row2[x1]==1:
                        sumr=sumr+(2**(y+1))
                        cntr=cntr+1
                    y=y+2
                if cntr%2==0:
                    mylist.append(sumr)
                if cntr%2==1:
                    mylist.append(-sumr)
            if shift==1:
                for x1 in range(len_row-1,-1,-1):
                    if row2[x1]==0 and x1==len_row-1:
                        sumr=sumr+(2**(tot_spins-1))
                    if row2[x1]==1 and x1==len_row-1:
                        sumr=sumr+1
                        cntr=cntr+1
                    if row2[x1]==0 and x1!=len_row-1:
                        sumr=sumr+(2**y)
                        y=y+2
                    if row2[x1]==1 and x1!=len_row-1:
                        sumr=sumr+(2**(y+1))
                        y=y+2
                        cntr=cntr+1
                if cntr%2==0:
                    mylist.append(sumr)
                    #print(sumr)
                if cntr%2!=0:
                    mylist.append(-sumr)
                    #print(sumr)
            if x<len_row-1 and x!=-1:    
                RecurChainRL2(row2,tot_spins,x+1,mylist,shift)
            if x==len_row-1:
                return()
    if icount >= len_row:
        return()
        

def RecurChainRL3(row,tot_spins,icount,mylist,shift):
    len_row=len(row)
    if icount<len_row:
        for x in range(icount,len_row):
            row2=row.copy()
            if x>=0:
                row2[x]=1
            if shift==0:
                y=0
            if shift==1:
                y=1
            sumr=0
            if shift==0:
                for x1 in range(len_row-1,-1,-1):
                    if row2[x1]==0:
                        sumr=sumr+0
                    if row2[x1]==1:
                        sumr=sumr+(2**y)+(2**(y+1))
                    y=y+2
                mylist.append(sumr)
            if shift==1:
                for x1 in range(len_row-1,-1,-1):
                    if row2[x1]==0 and x1==len_row-1:
                        sumr=sumr+0
                    if row2[x1]==1 and x1==len_row-1:
                        sumr=sumr+1+(2**(tot_spins-1))
                    if row2[x1]==0 and x1!=len_row-1:
                        sumr=sumr+0
                        y=y+2
                    if row2[x1]==1 and x1!=len_row-1:
                        sumr=sumr+(2**(y+1))+(2**y)
                        y=y+2
                mylist.append(sumr)
            if x<len_row-1 and x!=-1:    
                RecurChainRL3(row2,tot_spins,x+1,mylist,shift)
            if x==len_row-1:
                return()
    if icount >= len_row:
        return()


def RecurChainRL4(row,tot_spins,icount,mylist,shift):
    len_row=len(row)
    if icount<len_row:
        for x in range(icount,len_row):
            row2=row.copy()
            if x>=0:
                row2[x]=1
            if shift==0:
                y=0
            if shift==1:
                y=1
            sumr=0
            cntr=0
            if shift==0:
                for x1 in range(len_row-1,-1,-1):
                    if row2[x1]==0:
                        sumr=sumr+0
                        cntr=cntr+0
                    if row2[x1]==1:
                        sumr=sumr+(2**y)+(2**(y+1))
                        cntr=cntr+1
                    y=y+2
                if cntr%2==0:
                    mylist.append(sumr)
                if cntr%2==1:
                    mylist.append(-sumr)
            if shift==1:
                for x1 in range(len_row-1,-1,-1):
                    if row2[x1]==0 and x1==len_row-1:
                        sumr=sumr+0
                    if row2[x1]==1 and x1==len_row-1:
                        sumr=sumr+1+(2**(tot_spins-1))
                        cntr=cntr+1
                    if row2[x1]==0 and x1!=len_row-1:
                        sumr=sumr+0
                        y=y+2
                    if row2[x1]==1 and x1!=len_row-1:
                        sumr=sumr+(2**(y+1))+(2**y)
                        y=y+2
                        cntr=cntr+1
                if cntr%2==0:
                    mylist.append(sumr)
                    #print(sumr)
                if cntr%2!=0:
                    mylist.append(-sumr)
                    #print(sumr)
            if x<len_row-1 and x!=-1:    
                RecurChainRL4(row2,tot_spins,x+1,mylist,shift)
            if x==len_row-1:
                return()
    if icount >= len_row:
        return()