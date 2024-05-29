from finitefield.finitefield import FiniteField
from welchberlekamp import makeEncoderDecoder
from finitefield.polynomial import polynomialsOver

import galois
import random
import copy
import time
import numpy as np

class Snip:

    def __init__(self, P, LCC_para, alphas, betas):
        self.p, self.q, self.r, self.g, self.T, self.N, self.K = LCC_para
        self.K = 1 # overwrite
        self.P = P
        self.GF = galois.GF(self.q)
        self.F = FiniteField(p=self.q)
        self.alphas = alphas
        self.betas = betas
        self.betas = list(range(1 + self.N, self.N + self.K + self.T + 1)) # overwrite
        self.rr = random.randint(1, self.q - 1)

        # beaver triplet
        a, b, c = 3, 5, self.F(3 * 5)
        self.a_shares, _, _ = generate_shares(self.N, self.T, self.K, [a]*self.K, self.p, self.q, self.r, self.g, self.alphas, self.betas)
        self.b_shares, _, _ = generate_shares(self.N, self.T, self.K, [b]*self.K, self.p, self.q, self.r, self.g, self.alphas, self.betas)
        self.c_shares, _, _ = generate_shares(self.N, self.T, self.K, [c]*self.K, self.p, self.q, self.r, self.g, self.alphas, self.betas)

        # generate suitable circuit for a snip run
        self.us = [random.randint(1, self.q - 1) for _ in range(self.P)]  # upper input wires
        # vs = [random.randint(1,q-1) for i in range(P)] # lower input wires
        self.vs = copy.deepcopy(self.us)

        self.p_idx = list(range(1, 1 + P)) # specify the topology order of gates

        self.user_wb = {_id: {'h': [], 'x1': [], 'x2': []} for _id in range(0, self.N)}
        xs1 = self.us
        for x in xs1:
            secrets = [int(x)]  # K=1
            shares, _, _ = generate_shares(self.N, self.T, self.K, secrets, self.p, self.q, self.r, self.g, self.alphas, self.betas)
            for share in shares:
                idx, value = share
                self.user_wb[idx - 1]['x1'].append(value)
        xs2 = self.vs
        for x in xs2:
            secrets = [int(x)]  # K=1
            shares, _, _ = generate_shares(self.N, self.T, self.K, secrets, self.p, self.q, self.r, self.g, self.alphas, self.betas)
            for share in shares:
                idx, value = share
                self.user_wb[idx - 1]['x2'].append(value)

    def prove(self):
        st = time.time()

        f_u = galois.lagrange_poly(self.GF(self.p_idx), self.GF(self.us))
        f_v = galois.lagrange_poly(self.GF(self.p_idx), self.GF(self.vs))
        f_h = f_u * f_v
        # for pi, pp in enumerate(p_idx):
        #     assert f_h(pp) == us[pi] * vs[pi] % q
        print('polyr', f_h(self.rr))

        h_coefs = f_h.coeffs

        for hc in h_coefs:
            secrets = [int(hc)]  # K=1
            shares, _, _ = generate_shares(self.N, self.T, self.K, secrets, self.p, self.q, self.r, self.g, self.alphas, self.betas)
            for share in shares:
                idx, value = share
                self.user_wb[idx - 1]['h'].append(value)

        return time.time() - st

    def verify(self):
        # self.rr = rr

        tl = []
        for _id in range(0, self.N):  # for a prover _id
            st = time.time()
            # us, vs = [], []  # recon us,vs
            us, vs = self.user_wb[_id]['x1'], self.user_wb[_id]['x2']
            f_u = galois.lagrange_poly(self.GF(self.p_idx), self.GF(self.us))
            f_v = galois.lagrange_poly(self.GF(self.p_idx), self.GF(vs))
            y, z = int(f_u(self.rr)), int(f_v(self.rr))
            #     y = _lagrange_interpolate(rr, p_idx, us, q)
            #     z = _lagrange_interpolate(rr, p_idx, vs, q)
            self.user_wb[_id]['y'], self.user_wb[_id]['z'] = y, z

            f_h = galois.Poly(self.GF(self.user_wb[_id]['h']))
            yz = int(f_h(self.rr))
            self.user_wb[_id]['yz'] = yz

            tl.append(time.time() - st)

        return np.mean(tl)

    def beaver(self):
        ds, es = [], []
        accum_time = [0 for _ in range(0, self.N)]
        for _id in range(0, self.N):
            st = time.time()
            ds.append((_id + 1, self.F(self.user_wb[_id]['y'] - share_ith(self.a_shares, _id + 1))))
            es.append((_id + 1, self.F(self.user_wb[_id]['z'] - share_ith(self.b_shares, _id + 1))))
            accum_time[_id] += time.time() - st
        st = time.time()
        # d = RSGao_reconstruct_secret(ds, F, T, K, N, betas)
        # e = RSGao_reconstruct_secret(es, F, T, K, N, betas)
        d = reconstruct_secret(random.sample(ds, self.T + self.K), self.q, self.betas, self.K)
        e = reconstruct_secret(random.sample(es, self.T + self.K), self.q, self.betas, self.K)
        accum_time[0] += time.time() - st
        d = d[0]
        e = e[0]
        shares = []
        for _id in range(0, self.N):
            st = time.time()
            value = self.F(d*e + d*share_ith(self.b_shares,_id+1) + e*share_ith(self.a_shares,_id+1) +share_ith(self.c_shares,_id+1))
            shares.append(( _id+1 ,  value))
            accum_time[_id] += time.time()-st
        return shares, np.mean(accum_time)

    def server_check(self, lhs_shares):
        st = time.time()
        lhs = reconstruct_secret(random.sample(lhs_shares, self.T + self.K), self.q, self.betas, self.K)
        shares = []
        for _id in range(0, self.N):
            # st = time.time()
            shares.append((_id + 1, self.F(self.user_wb[_id]['yz'])))
            # vfr_time[_id] += time.time() - st
        # st = time.time()
        # rhs = RSGao_reconstruct_secret(shares, F, T, K, N, betas)
        rhs = reconstruct_secret(random.sample(shares, self.T + self.K), self.q, self.betas, self.K)
        print('lhs', lhs, 'rhs', rhs)
        assert lhs == rhs
        return time.time()-st



class Mul_Gate:

    def __init__(self, input1, input2, idx):
        self.u = input1
        self.v = input2
        self.h = input1 * input2
        self.idx = idx

    def get_output(self):
        return self.h

    def get_inputs(self):
        return self.u, self.v

    def set_index(self, idx):
        self.idx = idx

def isprime(n):
    if n == 2:
        return True
    if n == 1 or n % 2 == 0:
        return False
    i = 3
    while i <= math.sqrt(n):
        if n % i == 0:
            return False
        i = i + 2
    return True

def share_ith(shares, i):
    for share in shares:
        if share[0] == i:
            return share[1]
    return None

def initial(Z_lower=100):
    # generate q bigger than z_lower
    q = Z_lower
    while True:
        if isprime(q):
            break
        else:
            q = q + 1
    print("q = " + str(q))
    print("\nq is prime\n")

    # Find p and r
    r = 1
    while True:
        p = r * q + 1
        if isprime(p):
            print("r = " + str(r))
            print("p = " + str(p))
            print("\np is prime\n")
            break
        r = r + 1

    # Compute elements of Z_p*
    Z_p_star = []
    for i in range(0, p):
        if (math.gcd(i, p) == 1):
            Z_p_star.append(i)
        if len(Z_p_star) > 10:
            break

    # print("Z_p* = ")
    # print(Z_p_star) # , len(Z_p_star) same length, i.e. range(p)

    # Compute elements of G = {h^r mod p | h in Z_p*}
    G = []
    for i in Z_p_star:
        G.append(i ** r % p)

    G = list(set(G))
    G.sort()
    # print("\nG = ")
    # print(G)
    # print("Order of G is " + str(len(G)) + ". This must be equal to q.")

    # Since the order of G is prime, any element of G except 1 is a generator
    g = random.choice(list(filter(lambda g: g != 1, G)))
    print("\ng = " + str(g) + "\n")

    return p, q, r, g


def generate_shares(N, T, K, secrets, p, q, r, g, alphas, betas):
    secrets_check = []
    for secret in secrets:
        if secret == 0:
            secrets_check.append(1)  # "0": one exception after quantization
        else:
            secrets_check.append(int(secret))
    secrets = secrets_check
    for secret in secrets:
        assert secret >= 1 and secret <= q, "secret not in range"

    FIELD_SIZE = q
    noises = [random.randrange(0, FIELD_SIZE) for _ in range(T)]

    shares = []
    for alpha in alphas:
        y = _lagrange_interpolate(alpha, betas, secrets + noises, q)
        shares.append((alpha, y))

    commitments, verifications = [0, ], [0, ]
    return shares, commitments, verifications


def share_ith(shares, i):
    for share in shares:
        if share[0] == i:
            return share[1]
    return None


def quick_pow(a, b, q):  # compute a^b mod q, in a faster way？
    temp = 1
    for i in range(1, b + 1):
        temp = (temp * a) % q
    return temp % q


def commitment(paras, g, p):
    commitments = []
    for para in paras:
        # c = g ** coefficient_value % p
        c = quick_pow(g, para, p)
        commitments.append(c)
    return commitments


def verification(commitments, alpha, betas, p, q):
    # v_pos, v_neg = 1, 1
    v = 1
    for i, c in enumerate(commitments):
        num, den = 1, 1
        for k, _ in enumerate(commitments):
            if k != i:
                num *= alpha - betas[k]
                den *= betas[i] - betas[k]
            else:
                pass
        # if num / den > 0:
        #     v_pos = v_pos * quick_pow(c, int(num / den) % q, p) # c ** int(num / den) % p
        # else:
        #     v_neg = v_neg * quick_pow(c, int(- num / den) % q, p) # c ** int(-num / den) % p

        # v = (v * quick_pow(c, int(num / den) % q, p)) % p
        v = (v * quick_pow(c, _divmod(num, den, q) % q, p)) % p
    # v = _divmod(v_pos, v_neg, p)
    return v


def reconstruct_secret(pool, q, betas, K):
    start = time.time()
    out = []
    x_s, y_s = [], []
    for share in pool:
        x_s.append(int(share[0]))
        y_s.append(int(share[1]))
    for k in range(K):
        beta = betas[k]
        # out.append(f_rec(beta,pool,q))
        out.append(_lagrange_interpolate(beta, x_s, y_s, q))
    #     print("reconstruct_secret time takes:", time.time() - start)
    return out


def _lagrange_interpolate(x, x_s, y_s, q):
    """
    Find the y-value for the given x, given n (x, y) points;
    k points will define a polynomial of up to kth order.
    """
    k = len(x_s)
    assert k == len(set(x_s)), "points must be distinct"

    def PI(vals):  # upper-case PI -- product of inputs
        accum = 1
        for v in vals:
            accum *= v
        return accum

    nums = []  # avoid inexact division
    dens = []
    L = 0
    for i in range(k):
        others = list(x_s)
        cur = others.pop(i)
        nums.append(PI(x - o for o in others))
        dens.append(PI(cur - o for o in others))

        L += _divmod(y_s[i] * nums[i], dens[i], q)
        # den = PI(dens)
    # num = sum([_divmod(nums[i] * den * y_s[i] % q, dens[i], q) for i in range(k)])

    # L = sum( [_divmod(y_s[i] * nums[i], dens[i], q) for i in range(k)] )

    return L % q
    # return _divmod(num, den, q) % q


def _extended_gcd(a, b):
    """
    Division in integers modulus p means finding the inverse of the
    denominator modulo p and then multiplying the numerator by this
    inverse (Note: inverse of A is B such that A*B % p == 1) this can
    be computed via extended Euclidean algorithm
    http://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Computation
    """
    x = 0
    last_x = 1
    y = 1
    last_y = 0
    while b != 0:
        quot = a // b
        a, b = b, a % b
        x, last_x = last_x - quot * x, x
        y, last_y = last_y - quot * y, y
    return last_x, last_y


def _divmod(num, den, p):
    """Compute num / den modulo prime p

    To explain what this means, the return value will be such that
    the following is true: den * _divmod(num, den, p) % p == num
    """
    inv, _ = _extended_gcd(den, p)
    return num * inv


def BWRS_reconstruct_secret(encoded_shares, Field, solveSystem, betas, K):
    encoded_msg = []
    for share in encoded_shares:
        x, y = share[0], share[1]
        encoded_msg.append([Field(x), Field(y)])
    Q, E = solveSystem(encoded_msg)
    P, remainder = (Q.__divmod__(E))

    recon_secrets = []
    for i in range(K):
        recon_secrets.append(int(P(betas[i])))
    return recon_secrets


def lagrange2(x, w, P, pts):
    # modify from source: https://github.com/scipy/scipy/blob/v1.8.1/scipy/interpolate/_interpolate.py#L25-L91
    M = len(x)
    p = P(0)
    for j in range(M):
        pt = P(w[j])
        pt *= pts[j]
        p += pt
    return p


def compute_pts(x, F, P):
    M = len(x)
    pts = []
    for j in range(M):
        pt = F(1)
        for k in range(M):
            if k == j:
                continue
            fac = x[j] - x[k]
            pt *= P([-x[k], 1]) / fac
        pts.append(pt)
    return pts


def extendedEuclideanAlgorithm(a, b, nkdiv2):
    if abs(b) > abs(a):
        (x, y, d) = extendedEuclideanAlgorithm(b, a)
        return (y, x, d)

    if abs(b) == 0:
        return (1, 0, a)

    x1, x2, y1, y2 = 0, 1, 1, 0
    while abs(b) > 0:
        q, r = divmod(a, b)
        x = x2 - q * x1
        y = y2 - q * y1
        a, b, x2, x1, y2, y1 = b, r, x1, x, y1, y

        if r.degree() < nkdiv2:
            return (y1, y2, a, b)


def RSGao_reconstruct_secret(shares, F, T, K, N, betas):
    P = polynomialsOver(F)

    x = []
    y = []
    for share in shares:
        xx, yy = share[0], share[1]
        x.append(F(xx))
        y.append(F(yy))

    y[0] += 10  # corrupt data
    pts = compute_pts(x, F, P)  # pre-compute
    g1 = lagrange2(x, y, P, pts)
    g0 = P([-x[0], 1])
    for i in range(1, len(x)):
        g0 *= P([-x[i], 1])
    v, _, _, g = extendedEuclideanAlgorithm(g0, g1, (T + K + N) / 2)
    f1, r = divmod(g, v)

    if r == 0 and f1.degree() < T + K:
        # print("decode succussful \n polynomial", f1)
        recon_secrets = []
        for i in range(K):
            recon_secrets.append(int(f1(betas[i])))
        return recon_secrets
    else:
        # print("decode fail")
        return None