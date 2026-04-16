# Hyper-chaotic 6D System for Key Generation

import math


def hyper6d_step(x1,x2,x3,x4,x5,x6,a,b,c,s1,s2,h):

    dx1 = a*(x2 - x1)
    dx2 = b*x1 - x2 - x1*x3 + s1*(x4 - x5)
    dx3 = x1*x2 - c*x3

    dx4 = a*(x5 - x4)
    dx5 = b*x4 - x5 - x4*x6 + s2*(x1 - x2)
    dx6 = x4*x5 - c*x6

    x1 += h*dx1
    x2 += h*dx2
    x3 += h*dx3
    x4 += h*dx4
    x5 += h*dx5
    x6 += h*dx6

    return x1,x2,x3,x4,x5,x6

def generate_chaos_6d(x1,x2,x3,x4,x5,x6,L,N0):

    a = 10
    b = 28
    c = 8/3
    s1 = 0.05
    s2 = 0.05

    h = 0.001

    seq1,seq2,seq3,seq4,seq5,seq6 = [],[],[],[],[],[]

    for i in range(L + N0):

        x1,x2,x3,x4,x5,x6 = hyper6d_step(
            x1,x2,x3,x4,x5,x6,a,b,c,s1,s2,h
        )

        if i >= N0:
            seq1.append(x1)
            seq2.append(x2)
            seq3.append(x3)
            seq4.append(x4)
            seq5.append(x5)
            seq6.append(x6)

    return seq1,seq2,seq3,seq4,seq5,seq6

def generate_keys_6d(x1,x2,x3,x4,x5,x6,L,N0,m,n):

    s1,s2,s3,s4,s5,s6 = generate_chaos_6d(
        x1,x2,x3,x4,x5,x6,L,N0
    )

    K5=[]
    K6=[]
    K7=[]
    K8=[]
    K9=[]
    K10=[]

    for a,b,c,d,e,f in zip(s1,s2,s3,s4,s5,s6):

        k5 = math.floor(d * 1e13) % 8 + 1
        k6 = math.floor(e * 1e13) % n + 1
        k7 = math.floor(f * 1e13) % m + 1
        k8 = math.floor(a * 1e13) % 4
        k9 = math.floor(b * 1e13) % 4
        k10 = math.floor(c * 1e13) % 4
        
        K5.append(k5)
        K6.append(k6)
        K7.append(k7)
        K8.append(k8)
        K9.append(k9)
        K10.append(k10)

    return K5,K6,K7,K8,K9,K10