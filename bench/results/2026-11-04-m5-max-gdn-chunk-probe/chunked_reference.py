# Chunked gated delta rule vs the sequential recurrence of gdn_step_par_tok (per head):
#   S *= g_t; kv = S k_t; delta = (v_t - kv) * beta_t; S += delta k_t^T; o_t = S q_t
import numpy as np
rng = np.random.default_rng(1)
T, D, C = 256, 128, 64
q = rng.standard_normal((T, D)); k = rng.standard_normal((T, D))
k /= np.linalg.norm(k, axis=1, keepdims=True); q /= np.linalg.norm(q, axis=1, keepdims=True); q *= D ** -0.5
v = rng.standard_normal((T, D)); g = np.exp(-rng.uniform(0, 0.5, T)); beta = 1 / (1 + np.exp(-rng.standard_normal(T)))
S0 = rng.standard_normal((D, D)) * 0.1   # S[dv][dk]
# sequential
S = S0.copy(); o_seq = np.zeros((T, D))
for t in range(T):
    S *= g[t]; kv = S @ k[t]; delta = (v[t] - kv) * beta[t]; S += np.outer(delta, k[t]); o_seq[t] = S @ q[t]
S_seq = S
# chunked
S = S0.copy(); o_ch = np.zeros((T, D))
for c0 in range(0, T, C):
    K, Q, V, G, B = k[c0:c0+C], q[c0:c0+C], v[c0:c0+C], g[c0:c0+C], beta[c0:c0+C]
    lg = np.cumsum(np.log(G)); Gam = np.exp(lg)
    dec = np.exp(lg[:, None] - lg[None, :])            # Gam_i / Gam_j
    KK = K @ K.T
    A = np.tril(B[:, None] * dec * KK, -1)               # j < i
    # forward substitution: W_i = B_i Gam_i k_i - sum_{j<i} A_ij W_j ; Uv_i = B_i v_i - sum_{j<i} A_ij Uv_j
    W = np.zeros((C, D)); Uv = np.zeros((C, D))
    for i in range(C):
        W[i] = B[i] * Gam[i] * K[i] - A[i, :i] @ W[:i]
        Uv[i] = B[i] * V[i] - A[i, :i] @ Uv[:i]
    U = Uv - W @ S.T                                     # C x dv
    P = np.tril(Q @ K.T * dec)                           # j <= i
    o_ch[c0:c0+C] = (Gam[:, None] * Q) @ S.T + P @ U
    S = Gam[-1] * S + U.T @ ((Gam[-1] / Gam)[:, None] * K)
print("max |o| diff", np.abs(o_ch - o_seq).max(), "rel", np.abs(o_ch - o_seq).max() / np.abs(o_seq).max())
print("max |S| diff", np.abs(S - S_seq).max())
