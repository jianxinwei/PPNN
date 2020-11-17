  
#! ~/anaconda3/python3
# -*- coding: utf-8 -*-

import copy
import torch

import phe
from phe import paillier
import numpy as np
import random
from numba import jit
import sympy
import math

from multipledispatch import dispatch
import torch
import ipdb

from contextlib import contextmanager
import time

BASE_NUM = 10e100
POWER = 100


class ThresholdPaillier(object):
    def __init__(self, size_of_n, l, w):
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

        '''
        print(sympy.primetest.isprime(self.p), 
            sympy.primetest.isprime(self.q), 
            sympy.primetest.isprime(self.p1), 
            sympy.primetest.isprime(self.q1)
        )
        '''
        
        self.n = self.p * self.q
        self.s = 1
        self.ns = pow(self.n, self.s)
        self.nSPlusOne = pow(self.n,self.s+1)
        self.nPlusOne = self.n + 1
        self.nSquare = self.n*self.n

        self.m = self.p1 * self.q1
        self.nm = self.n*self.m
        self.l = l # Number of shares of private key
        self.w = w # The minimum of decryption servers needed to make a correct decryption.
        self.delta = self.factorial(self.l)
        self.rnd = random.randint(1,1e50)
        self.combineSharesConstant = sympy.mod_inverse((4*self.delta*self.delta)%self.n, self.n)
        self.d = self.m * sympy.mod_inverse(self.m, self.n)

        self.ais = [self.d]
        for i in range(1, self.w):
            self.ais.append(random.randint(0,self.nm-1))

        self.r = random.randint(1, self.p) ## Need to change upper limit from p to one in paper
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
            self.priv_keys.append(ThresholdPaillierPrivateKey(self.si[i], i+1, self.delta, self.nSPlusOne))
        self.pub_key = ThresholdPaillierPublicKey(self.n, self.nSPlusOne, self.r, self.ns)

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
    def __init__(self, si, server_id, delta, nSPlusOne):
        self.si = si
        self.server_id = server_id
        self.delta = delta
        self.nSPlusOne = nSPlusOne

    # @dispatch(int)
    def partialDecrypt(self, c):
        return PartialShare(pow(c.c, self.si*2*self.delta, self.nSPlusOne), self.server_id)

class ThresholdPaillierPublicKey(object):
    def __init__(self, n, nSPlusOne, r, ns):
        self.n = n
        self.nSPlusOne = nSPlusOne
        self.r = r
        self.ns =ns

    # @dispatch(int)
    def encrypt(self, msg):
        msg = msg % self.nSPlusOne if msg < 0 else msg
        c = (pow(self.n+1, msg, self.nSPlusOne)*pow(self.r, self.ns, self.nSPlusOne)) % self.nSPlusOne
        return EncryptedNumber(c, self.nSPlusOne, self.n)

class EncryptedNumber(object):
    def __init__(self, c, nSPlusOne, n):
        self.c = c
        self.nSPlusOne = nSPlusOne
        self.n = n

    def __add__(self, c2):
        return EncryptedNumber((self.c * c2.c) % self.nSPlusOne, self.nSPlusOne, self.n)
    '''
    def __mul__(self, cons):
        if cons < 0:
            return EncryptedNumber(pow(sympy.mod_inverse(self.c, self.nSPlusOne), -cons, self.nSPlusOne), self.nSPlusOne, self.n)
        else:
            return EncryptedNumber(pow(self.c, cons, self.nSPlusOne), self.nSPlusOne, self.n)
    '''

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


def encrypt_torch_state_dict(pub_key, state_dict):
    encrypted_state_dict = {}
    state_shape = {}
    for k in state_dict.keys():
        encrypted_state_dict[k] = state_dict[k].numpy()
        state_shape[k] = encrypted_state_dict[k].shape
        if len(encrypted_state_dict[k].shape) <= 1:
            encrypted_state_dict[k] = encrypt_vector(pub_key, encrypted_state_dict[k])
        else:
            # ravel the high dimension nd array
            encrypted_state_dict[k] = encrypt_vector(pub_key, encrypted_state_dict[k].ravel())
    return encrypted_state_dict


def decrypt_torch_state_dict(encrypted_state_dict, partial_priv_keys, w, delta, combineSharesConstant, nSPlusOne, n, ns, total_num, state_dict):
    intermediate_state_dict = copy.deepcopy(encrypted_state_dict)
    state_shape = {}
    for k in encrypted_state_dict.keys():
        state_shape[k] = state_dict[k].shape
        shares = [decrypt_vector(partial_key, encrypted_state_dict[k]) for partial_key in partial_priv_keys]
        intermediate_state_dict[k] = torch.from_numpy(np.array(batch_decrypt(shares, w, delta, combineSharesConstant, nSPlusOne, n, ns)).reshape(state_shape[k])/total_num)
    return intermediate_state_dict

def encrypt_vector(public_key, x):
    return [public_key.encrypt(int(i*BASE_NUM)) for i in x]

def decrypt_vector(private_key, x):
    # return np.array([private_key.decrypt(i) for i in x])
    return [private_key.partialDecrypt(i) for i in x]

def batch_add(nums1, nums2):
    result = []
    for num1, num2 in zip(nums1, nums2):
        result.append(num1 + num2)
    return result

def batch_mul(nums1, nums2):
    result = []
    for num1, num2 in zip(nums1, nums2):
        result.append(num1 * num2)
    return result

def batch_decrypt(shares, w, delta, combineSharesConstant, nSPlusOne, n, ns):
    result = []
    share_num = len(shares)
    for idx, _ in enumerate(shares[0]):
        result.append(combineShares([share[idx] for share in shares], w, delta, combineSharesConstant, nSPlusOne, n, ns)/BASE_NUM)
    return result

'''
if __name__ == '__main__':
    @contextmanager
    def timer():
        """Helper for measuring runtime"""
        time0 = time.perf_counter()
        yield
        print('[elapsed time: %.7f s]' % (time.perf_counter() - time0))

    tp = ThresholdPaillier(1024, 5, 2)
    priv_keys = tp.priv_keys
    pub_key = tp.pub_key
    # ipdb.set_trace()
    m1 = 0.1141232316371
    m2 = 0.4128491416247126

    c1 = pub_key.encrypt(int(m1*BASE_NUM))
    c2 = pub_key.encrypt(int(m2*BASE_NUM))
    c3 = c1 + c2
    share1 = priv_keys[2].partialDecrypt(c3)
    share2 = priv_keys[3].partialDecrypt(c3)

    with timer() as t:  
        dec_c3 = combineShares([share1, share2],
            tp.w, tp.delta, tp.combineSharesConstant, 
            pub_key.nSPlusOne, pub_key.n, pub_key.ns)/BASE_NUM

    math_cal = np.around(m1+m2, POWER)
    try:
        assert math_cal == np.around(dec_c3, POWER)
    except Exception as e:
        print(math_cal, dec_c3)
        ipdb.set_trace()
        raise e

    int1 = 133
    int2 = 841423
    c_int1 = pub_key.encrypt(int1)
    c_int2 = pub_key.encrypt(int2)
    c_int3 = c_int1 + c_int2
    share1 = priv_keys[4].partialDecrypt(c_int3)
    share2 = priv_keys[3].partialDecrypt(c_int3)

    with timer() as t:
        dec_c_int3 = combineShares([share1, share2],
            tp.w, tp.delta, tp.combineSharesConstant, 
            pub_key.nSPlusOne, pub_key.n, pub_key.ns)

    print(int1+int2, dec_c_int3)
'''