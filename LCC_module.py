from finitefield.finitefield import FiniteField
from finitefield.polynomial import polynomialsOver
from welchberlekamp import makeEncoderDecoder
from snip_module import Snip
import random
import math
import time
from tqdm import tqdm, trange
import pickle

import torch
import models
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np
from copy import deepcopy
from torch.nn import functional as F

def secure_aggregation(global_model, agent_updates_dict, LCC_para, args):
    print("===== secure aggregation starts for this round =====")
    p, q, r, g, T, N, K = LCC_para
    alphas = list(range(1, 1 + N))
    betas = list(range(1 + N, N + K + T + 1))
    mul_scale = 10 ** 4
    F = FiniteField(p=q)
    # enc, dec, solveSystem = makeEncoderDecoder(N, T+K, q)
    print("parameters:", LCC_para)
    print("finite field is ", q, "; amplify scale is ", mul_scale)

    agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
    concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
    print(concat_col_vectors.shape, torch.max(concat_col_vectors), torch.min(concat_col_vectors))
    # assert torch.max(concat_col_vectors)<1 and torch.min(concat_col_vectors)>-1

    # quantization and padding
    quant_concat_col_vectors = torch.round(concat_col_vectors * mul_scale) % q
    pad_width = concat_col_vectors.shape[1]
    pad_length = (K - quant_concat_col_vectors.shape[0] % K) % K
    quant_concat_col_vectors = torch.cat([quant_concat_col_vectors, torch.ones(pad_length, pad_width).to(args.device)], 0)

    # generate shares
    user_wb = {_id:{} for _id in range(0, N)}
    server_time, user_time = {}, {}
    user_tl = [0 for _ in range(0, N)]
    for _id in trange(0, N): # sender
        st = time.time()
        update_chunk = quant_concat_col_vectors[:, _id].chunk(quant_concat_col_vectors.shape[0]//K)
        # break
        for chunk_k, chunk in enumerate(update_chunk):
            secrets = chunk.cpu().detach().tolist()
            shares, _, _ = generate_shares(N, T, K, secrets, p, q, r, g, alphas, betas)
            for share in shares:
                holder_id, value = share # holder (who receives the share)
                user_wb[_id][(holder_id, chunk_k)] = value
        user_tl[_id] += time.time() - st
    user_time['Encoding'] = (np.mean(user_tl), np.max(user_tl))

    with open('./user_wb.pkl', 'wb') as f:  # open a text file
        pickle.dump(user_wb, f)  # serialize the list
    with open('./user_wb.pkl', 'rb') as f:
        user_wb = pickle.load(f)  # deserialize using load()

    P = 10 # the number of mul gates
    user_tl = [0 for _ in range(0, N)]
    svr_tl = 0
    for i_id in trange(0, N):
        snip = Snip(P, LCC_para, alphas, betas)  # initial a snip run
        prover_time = snip.prove()
        verifier_time = snip.verify()
        lhs_shares, verifier_time2 = snip.beaver()
        server_check_time = snip.server_check(lhs_shares)
        # print(prover_time, verifier_time, server_check_time)

        for j_id in range(0, N):
            if i_id == j_id:
                user_tl[i_id] += prover_time
            else:
                user_tl[j_id] += verifier_time
        svr_tl += server_check_time
    user_time['Snip'] = (np.mean(user_tl) * (quant_concat_col_vectors.shape[0] // K), np.max(user_tl) * (quant_concat_col_vectors.shape[0] // K))
    server_time['Snip'] = svr_tl * (quant_concat_col_vectors.shape[0] // K)

    print("snip finished")

    # user computation
    server_agg = {_id: None for _id in range(0, N)}
    user_tl = [0 for _ in range(0, N)]
    svr_tl = 0
    for i_id in trange(0, N): # holder
        st = time.time()
        update_agg = []
        for chunk_k, _ in enumerate(update_chunk):
            share_value = 0
            for j_id in range(0, N): # sender
                share_value = (share_value + user_wb[j_id][(i_id+1, chunk_k)]) % q
            share_value = int(share_value) % q
            update_agg.append(share_value)
        server_agg[i_id] = update_agg # server collects the holders share of aggregation
        user_tl[i_id] += time.time() - st
    user_time['secure compute'] = (np.mean(user_tl), np.max(user_tl))

    # server decodes
    update_recon = []
    # countf, counts = 0,0
    st = time.time()
    for chunk_k, _ in enumerate(update_chunk):
        shares = []
        for _id in range(0, N):
            value = server_agg[_id][chunk_k]
            shares.append((_id + 1, value))
        pool = random.sample(shares, T + K)
        secret_reconstructed1 = reconstruct_secret(pool, q, betas, K)

        # secret_reconstructed2 = BWRS_reconstruct_secret(shares, F, solveSystem, betas, K)
        # print("\n RSBW_reconstruct_secret:", secret_reconstructed2)
        #
        # secret_reconstructed3 = RSGao_reconstruct_secret(shares, F, T, K, N, betas)
        #
        # if not secret_reconstructed3 == secret_reconstructed1:
        #     # print("decoding fail")
        #     countf += 1
        # else:
        #     # print("decoding success:", secret_reconstructed3)
        #     counts += 1

        update_recon.append(secret_reconstructed1)
    svr_tl += time.time() - st
    server_time['Decoding'] = svr_tl
    # print('success rate: ', countf, counts, counts/(countf + counts))
    # with open('./decode_status.pkl', 'wb') as f:  # open a text file
    #     pickle.dump((countf, counts, counts/(countf + counts)), f)  # serialize the list
    update_recon = torch.tensor(update_recon).view(-1).to(args.device)
    update_recon = update_recon[0:concat_col_vectors.shape[0]] # prune the padding

    print(server_time, user_time)
    with open('./time_{}.pkl'.format(time.time()), 'wb') as f:  # open a text file
        pickle.dump((server_time, user_time), f)  # serialize the list

    # de - quantization and averaging
    neg_part = update_recon > q/2
    neg_part = neg_part.long()
    update_recon += neg_part * (-q) # map to -q/2 ~ q/2
    update_recon = (1/args.num_agents) * update_recon/mul_scale

    n_params = len(parameters_to_vector(global_model.parameters()))
    lr_vector = torch.Tensor([args.server_lr] * n_params).to(args.device)
    cur_global_params = parameters_to_vector(global_model.parameters())
    new_global_params = (cur_global_params + lr_vector * update_recon).float()
    vector_to_parameters(new_global_params, global_model.parameters())

    print("===== secure aggregation finishes for this round =====")

    return


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

    #     commitments = commitment(secrets + noises, g, p)
    #     start = time.time()

    #     verifications = []
    #     for alpha in alphas:
    #         # check1 = g ** shares[i-1][1] % p
    #         # check1 = g ** share_ith(shares, i) % p
    #         check1 = quick_pow(g, share_ith(shares, alpha), p)
    #         check2 = verification(commitments, alpha, betas, p, q)
    #         verifications.append(check2)
    #         if (check1 % p) == (check2 % p):
    #             pass
    #         else:
    #             print("checking fails with:", check1, check2)
    #             1/0
    # #         print(alpha, "-th user ============= tag at time ", time.time()-start,"seconds =============")
    #         start = time.time()
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
