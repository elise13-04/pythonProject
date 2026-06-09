doc= open('GPS_data.txt')
import numpy as np
def rdoc(n):
    L=[]
    for k in range(1,n+1):
        l=doc.readline()
        l=l.split(" ")
        L.append(l)
    return L

def nbr(L):
    n=len(L)
    for k in range(n):
        L[k]=float(L[k])
    return L

def rdocnbr(n):
    M=rdoc(n)
    for N in M:
        N=nbr(N)
    return np.array(M)
tab=rdocnbr(5121)

doc2=open('fichier.txt','w')
for k in range(len(tab)):
    s=''
    for i in range(len(tab[k])):
        s=s+' '+str(tab[k][i])
    s=s+"\n"
    doc2.write(s)




