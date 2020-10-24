import phe
from phe import paillier
import numpy as np
import random
from numba import jit
import sympy
import math

class ThresholdPaillier(object):
    def __init__(self,size_of_n):
        #size_of_n = 1024
        pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
        self.p1 = priv.p
        self.q1 = priv.q

        while sympy.primetest.isprime(2*self.p1 +1)!= True:
            pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
            self.p1 = priv.p
        while sympy.primetest.isprime(2*self.q1 +1)!= True:
            pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
            self.q1 = priv.q

        self.p = (2*self.p1) + 1
        self.q = (2*self.q1) + 1
        print(sympy.primetest.isprime(self.p),sympy.primetest.isprime(self.q),sympy.primetest.isprime(self.p1),sympy.primetest.isprime(self.q1))
        self.n = self.p * self.q
        self.s = 1
        self.ns = pow(self.n, self.s)
        self.nSPlusOne = pow(self.n,self.s+1)
        self.nPlusOne = self.n + 1
        self.nSquare = self.n*self.n

        self.m = self.p1 * self.q1
        self.nm = self.n*self.m
        self.l = 5 # Number of shares of private key
        self.w = 2 # The minimum of decryption servers needed to make a correct decryption.
        self.delta = self.factorial(self.l)
        self.rnd = random.randint(1,1e50)
        self.combineSharesConstant = sympy.mod_inverse((4*self.delta*self.delta)%self.n, self.n)
        self.d = self.m * sympy.mod_inverse(self.m, self.n)

        self.ais = [self.d]
        for i in range(1, self.w):
            self.ais.append(random.randint(0,self.nm-1))

        self.r = random.randint(1,self. p) ## Need to change upper limit from p to one in paper
        while math.gcd(self.r,self.n) != 1:
            self.r = random.randint(0, self.p)
        self.v = (self.r*self.r) % self.nSquare

        self.si = [0] * self.l
        self.viarray = [0] * self.l

        for i in range(self.l):
            self.si[i] = 0
            X = i + 1
            for j in range(self.w):
                self.si[i] += self.ais[j] * pow(X, j)
            self.si[i] = self.si[i] % self.nm
            self.viarray[i] = pow(self.v, self.si[i] * self.delta, self.nSquare)

        self.priv_keys = []
        for i in range(self.l):
            self.priv_keys.append(ThresholdPaillierPrivateKey(self.n, self.l, self.combineSharesConstant, self.w, \
                                            self.v, self.viarray, self.si[i], i+1, self.r, self.delta, self.nSPlusOne))
        self.pub_key = ThresholdPaillierPublicKey(self.n, self.nSPlusOne, self.r, self.ns, self.w,\
                                                 self.delta, self.combineSharesConstant)

    def factorial(self, n):
        fact = 1
        for i in range(1,n+1):
            fact *= i
        return fact

    def computeGCD(self, x, y):
       while(y):
           x, y = y, x % y
       return x

class PartialShare(object):
    def __init__(self, share, server_id):
        self.share = share
        self.server_id =server_id

class ThresholdPaillierPrivateKey(object):
    def __init__(self,n, l,combineSharesConstant, w, v, viarray, si, server_id, r, delta, nSPlusOne):
        self.n = n
        self.l = l
        self.combineSharesConstant = combineSharesConstant
        self.w = w
        self.v = v
        self.viarray = viarray
        self.si = si
        self.server_id = server_id
        self.r = r
        self.delta = delta
        self.nSPlusOne = nSPlusOne

    def partialDecrypt(self, c):
        return PartialShare(pow(c.c, self.si*2*self.delta, self.nSPlusOne), self.server_id)

class ThresholdPaillierPublicKey(object):
    def __init__(self,n, nSPlusOne, r, ns, w, delta, combineSharesConstant):
        self.n = n
        self.nSPlusOne = nSPlusOne
        self.r = r
        self.ns =ns
        self.w = w
        self.delta = delta
        self.combineSharesConstant = combineSharesConstant

    def encrypt(self, msg):
        msg = msg % self.nSPlusOne if msg < 0 else msg
        c = (pow(self.n+1, msg, self.nSPlusOne) * pow(self.r, self.ns, self.nSPlusOne)) % self.nSPlusOne
        return EncryptedNumber(c, self.nSPlusOne, self.n)

class EncryptedNumber(object):
    def __init__(self, c, nSPlusOne, n):
        self.c = c
        self.nSPlusOne = nSPlusOne
        self.n = n

    def __mul__(self, cons):
        if cons < 0:
            return EncryptedNumber(pow(sympy.mod_inverse(self.c, self.nSPlusOne), -cons, self.nSPlusOne), self.nSPlusOne, self.n)
        else:
            return EncryptedNumber(pow(self.c, cons, self.nSPlusOne), self.nSPlusOne, self.n)

    def __add__(self, c2):
        return EncryptedNumber((self.c * c2.c) % self.nSPlusOne, self.nSPlusOne, self.n)

def combineShares(shrs, w, delta, combineSharesConstant, nSPlusOne, n, ns):
        cprime = 1
        for i in range(w):
            ld = delta
            for iprime in range(w):
                if i != iprime:
                    if shrs[i].server_id != shrs[iprime].server_id:
                        ld = (ld * -shrs[iprime].server_id) // (shrs[i].server_id - shrs[iprime].server_id)
            #print(ld)
            shr = sympy.mod_inverse(shrs[i].share, nSPlusOne) if ld < 0 else shrs[i].share
            ld = -1*ld if ld <1 else ld
            temp = pow(shr, 2 * ld, nSPlusOne)
            cprime = (cprime * temp) % nSPlusOne
        L = (cprime - 1) // n
        result = (L * combineSharesConstant) % n
        return result - ns if result > (ns // 2) else result

tp = ThresholdPaillier(2048)
priv_keys = tp.priv_keys
pub_key = tp.pub_key
print(priv_keys)