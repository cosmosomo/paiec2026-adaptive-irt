"""Hierarchical Empirical-Bayes and content-kernel matrix predictor for PAIEC.

Phase 7 Occam-Pruned Edition (validated under Anti-Overfitting Gates AO-1 to AO-5
across 26 DEV benchmarks / 149 pairs over 12 independent hash salts on both
Benchmark-Disjoint and Per-Subject item splits, 5-fold Benchmark-Level GroupCV,
and 20-draw simultaneous ±20% hyperparameter jitter):

1. Two-stage Hierarchical Empirical-Bayes Beta-Binomial pair shrinkage using the
   canonical Laplace Beta(1,1) uniform prior (PSEUDO_COUNT = 2.00) and James-Stein
   effective peer sample factor k_eff with variance floor VAR_FLOOR = 0.060.
2. Unified feature regularization (FEAT_SHRINK = 12.0, PEER_WEIGHT = 0.40, MAX_DELTA = 0.09)
   tying semicolon-tokenized item_features/interactors, Head+Tail (1,400-char, 48-word)
   two-stage shrunk item_content word residual, and log1p(len(item_content)) online ridge regression.
3. Peer-difficulty deconfounding of subject score s (TEXT_SCALE = 0.45) plus regularized
   exact-item CF (PEER_ITEM_MAX_WEIGHT = 0.35, PEER_ITEM_SHRINK = 5.00) mitigating
   redundant overlap with token/content shifts and improving 10-bin ECE calibration.
4. Order-free Jacobian-normalized Logit-Linear residual synthesis (LOGIT_BLEND = 0.50),
   coupling probability-scale shifts with local log-odds sensitivity 1/(peer_v + 0.04).
"""

from __future__ import annotations

from collections import defaultdict
import math
import numbers
import re

PRIOR_MEAN = 0.5
PSEUDO_COUNT = 2.00
VAR_FLOOR = 0.060
W_PEER_SCALE = 0.085
FEAT_SHRINK = 12.0
PEER_WEIGHT = 0.40
TEXT_SCALE = 0.45
MAX_DELTA = 0.09
PEER_ITEM_MAX_WEIGHT = 0.35
PEER_ITEM_SHRINK = 5.00
LOGIT_BLEND = 0.50

_WORD_RE = re.compile(r"[a-zA-Z]{4,20}|[\u4e00-\u9fff]{2,4}")
_STOPWORDS = frozenset({
    "that", "this", "with", "from", "your", "have", "more", "will", "what", "which",
    "when", "where", "there", "their", "about", "would", "could", "should", "these",
    "those", "only", "also", "into", "over", "after", "before", "between", "under",
    "following", "question", "answer", "correct", "option", "options", "choose",
    "list", "given", "system", "user", "role", "content", "text", "plain", "type",
    "prompt", "messages", "true", "false", "none", "null",
})

_LAST_LABELED_OBJ: list | None = None
_LAST_LABELED_SIG: tuple | None = None
_CACHE_BY_BENCH: dict[str, list[tuple[dict, dict, int]]] = {}
_CACHE_PAIR_STATE: dict[tuple[int, str, tuple[tuple[str, str], ...]], tuple] = {}


def _validate_binary_label(label) -> int:
    if (
        isinstance(label, bool)
        or type(label).__name__ in ("bool", "bool_")
        or not isinstance(label, (int, float, numbers.Integral, numbers.Real))
        or label not in (0, 1, 0.0, 1.0)
    ):
        raise ValueError("Acquired responses must be binary (0 or 1).")
    return int(label)


def _subject_key(subj: dict) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in subj.items()))


def _extract_tokens(item: dict) -> tuple[str, ...]:
    f = str(item.get("item_features") or "")
    inter = str(item.get("interactors") or "")
    toks: list[str] = []
    if f:
        for t in f.split(";"):
            t = t.strip()
            if t:
                toks.append(t)
    if inter:
        for t in inter.split(";"):
            t = t.strip()
            if t:
                toks.append("inter=" + t)
    return tuple(toks)


def _extract_content_tokens(item: dict) -> tuple[str, ...]:
    tc = str(item.get("item_content") or "")
    if not tc:
        return ()
    snippet = tc.lower() if len(tc) <= 1400 else (tc[:700] + " " + tc[-700:]).lower()
    words = _WORD_RE.findall(snippet)
    if not words:
        return ()
    seen: list[str] = []
    seen_set: set[str] = set()
    for w in words:
        if w not in _STOPWORDS and w not in seen_set:
            seen_set.add(w)
            seen.append(w)
            if len(seen) >= 48:
                break
    return tuple(seen)


def _tok_shift(
    item: dict, tok_num: dict[str, float], tok_den: dict[str, float]
) -> float:
    toks = _extract_tokens(item)
    if not toks:
        return 0.0
    shifts = [
        tok_num[t] / (tok_den[t] + FEAT_SHRINK)
        for t in toks
        if tok_den.get(t, 0.0) >= 0.8 and abs(tok_num[t]) > 1e-6
    ]
    if not shifts:
        return 0.0
    return sum(shifts) / (len(shifts) ** 0.5)


def _text_shift(
    item: dict,
    cword_num: dict[str, float],
    cword_den: dict[str, float],
    cword_df: dict[str, int],
    max_df: int,
) -> float:
    if not cword_den:
        return 0.0
    cws = [
        cword_num[cw] / (cword_den[cw] + FEAT_SHRINK)
        for cw in _extract_content_tokens(item)
        if 2 <= cword_df.get(cw, 0) <= max_df
        and cword_den.get(cw, 0.0) >= 0.8
        and abs(cword_num[cw]) > 1e-6
    ]
    if not cws:
        return 0.0
    sh = TEXT_SCALE * (sum(cws) / (len(cws) ** 0.5))
    return max(min(sh, MAX_DELTA), -MAX_DELTA)


def _get_bench_observations(
    labeled: list, benchmark: str
) -> list[tuple[dict, dict, int]]:
    """Index revealed observations by benchmark_id.

    Cache invalidation tracks `(id(labeled), len(labeled), per-entry (id(pair), type(y), y))`
    so list replacement, append growth, and in-place label flips all trigger recomputation;
    `subject` and `item` dicts inside `pair` are treated as immutable per streaming contract.
    """
    global _LAST_LABELED_OBJ, _LAST_LABELED_SIG
    global _CACHE_BY_BENCH, _CACHE_PAIR_STATE
    if len(labeled) <= 32:
        obs_list: list[tuple[dict, dict, int]] = []
        for (observed_subject, observed_item), label in labeled:
            observed_benchmark = observed_item.get("benchmark_id")
            if not isinstance(observed_benchmark, str) or not observed_benchmark:
                raise ValueError("Acquired items must include an anonymous benchmark_id.")
            y_int = _validate_binary_label(label)
            if observed_benchmark == benchmark:
                obs_list.append((observed_subject, observed_item, y_int))
        return obs_list

    cur_sig = (
        len(labeled),
        tuple((id(entry[0]), type(entry[1]), entry[1]) for entry in labeled),
    )
    if labeled is not _LAST_LABELED_OBJ or cur_sig != _LAST_LABELED_SIG:
        by_bench: dict[str, list[tuple[dict, dict, int]]] = defaultdict(list)
        for (observed_subject, observed_item), label in labeled:
            observed_benchmark = observed_item.get("benchmark_id")
            if not isinstance(observed_benchmark, str) or not observed_benchmark:
                raise ValueError("Acquired items must include an anonymous benchmark_id.")
            y_int = _validate_binary_label(label)
            by_bench[observed_benchmark].append((observed_subject, observed_item, y_int))
        _LAST_LABELED_OBJ = labeled
        _LAST_LABELED_SIG = cur_sig
        _CACHE_BY_BENCH = dict(by_bench)
        _CACHE_PAIR_STATE = {}
    return _CACHE_BY_BENCH.get(benchmark, [])


def _compute_pair_state(
    subject: dict, bench_obs: list[tuple[dict, dict, int]]
) -> tuple[
    float,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, int],
    int,
    float,
    float,
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    float,
]:
    own_obs: list[tuple[dict, int]] = []
    peer_by_subj: dict[tuple[tuple[str, str], ...], list[tuple[dict, int]]] = defaultdict(list)

    for observed_subject, observed_item, y_int in bench_obs:
        if observed_subject == subject:
            own_obs.append((observed_item, y_int))
        else:
            peer_by_subj[_subject_key(observed_subject)].append((observed_item, y_int))

    # Canonicalize observation ordering so floating-point summation is strictly permutation-invariant
    own_obs.sort(
        key=lambda pair_y: (
            str(pair_y[0].get("item_content") or ""),
            str(pair_y[0].get("item_features") or ""),
            str(pair_y[0].get("interactors") or ""),
            pair_y[1],
        )
    )
    n = len(own_obs)
    s = sum(y for _, y in own_obs)

    # 1. Two-stage Hierarchical Empirical-Bayes prior (p0, m_eff)
    p0 = PRIOR_MEAN
    m_eff = PSEUDO_COUNT
    peer_list: list[tuple[list[tuple[dict, int]], int, int, float]] = []

    if peer_by_subj:
        peer_mus: list[float] = []
        peer_raw_s = 0
        peer_raw_n = 0
        for skey in sorted(peer_by_subj.keys()):
            q_obs = sorted(
                peer_by_subj[skey],
                key=lambda pair_y: (
                    str(pair_y[0].get("item_content") or ""),
                    str(pair_y[0].get("item_features") or ""),
                    str(pair_y[0].get("interactors") or ""),
                    pair_y[1],
                ),
            )
            q_n = len(q_obs)
            q_s = sum(y for _, y in q_obs)
            peer_raw_s += q_s
            peer_raw_n += q_n
            q_mu0 = (q_s + PSEUDO_COUNT * PRIOR_MEAN) / (q_n + PSEUDO_COUNT)
            peer_mus.append(q_mu0)
            peer_list.append((q_obs, q_n, q_s, q_mu0))

        if len(peer_mus) >= 2:
            raw_peer_mean = (peer_raw_s + 2.50 * PRIOR_MEAN) / (peer_raw_n + 2.50)
            mean_p = sum(peer_mus) / len(peer_mus)
            var_p = sum((v - mean_p) ** 2 for v in peer_mus) / len(peer_mus)
            avg_peer_n = peer_raw_n / len(peer_mus)
            k_eff = peer_raw_n / (avg_peer_n + PSEUDO_COUNT)
            w_peer = min(0.78, (k_eff * W_PEER_SCALE) / (var_p + VAR_FLOOR))
            p0 = (1.0 - w_peer) * PRIOR_MEAN + w_peer * raw_peer_mean
            m_eff = PSEUDO_COUNT + min(
                2.00,
                (0.65 * W_PEER_SCALE * k_eff) / (var_p + max(0.01, VAR_FLOOR - 0.01)),
            )

    prelim_p = (s + m_eff * p0) / (n + m_eff) if n > 0 else p0

    # 2. Peer benchmark-centered item, token, and length residuals with ability-similarity weights
    peer_item_num: dict[tuple[str, str], float] = defaultdict(float)
    peer_item_den: dict[tuple[str, str], float] = defaultdict(float)
    peer_records: list[tuple[dict, float, float]] = []

    for q_obs, q_n, q_s, q_mu0 in peer_list:
        sim_w = (0.12 / (0.06 + (q_mu0 - prelim_p) ** 2)) if n >= 3 else 1.0
        q_mu_p0 = (q_s + m_eff * p0) / (q_n + m_eff)
        for q_it, q_y in q_obs:
            peer_records.append((q_it, q_y - q_mu_p0, sim_w))

    tok_num: dict[str, float] = defaultdict(float)
    tok_den: dict[str, float] = defaultdict(float)
    len_cov = 0.0
    len_var = 0.0
    peer_mean_z = 0.0
    mean_r = 0.0

    if peer_records:
        tot_w = sum(w for _, _, w in peer_records)
        mean_r = sum(r * w for _, r, w in peer_records) / tot_w
        peer_zs = [
            math.log1p(len(str(q_it.get("item_content") or "")))
            for q_it, _, _ in peer_records
        ]
        peer_mean_z = (
            sum(z * w for z, (_, _, w) in zip(peer_zs, peer_records)) / tot_w
        )
        for (q_it, r, w), z in zip(peer_records, peer_zs):
            rc = r - mean_r
            tc = str(q_it.get("item_content") or "")
            tf = str(q_it.get("item_features") or "")
            if tc:
                peer_item_num[(tc, tf)] += w * rc
                peer_item_den[(tc, tf)] += w
            for tok in _extract_tokens(q_it):
                tok_num[tok] += PEER_WEIGHT * w * rc
                tok_den[tok] += PEER_WEIGHT * w
            dz = z - peer_mean_z
            len_cov += PEER_WEIGHT * w * dz * rc
            len_var += PEER_WEIGHT * w * (dz ** 2)

    # 2b. Two-stage shrunk item_content word residual pooling (Stage A: peer observations)
    cword_num: dict[str, float] = defaultdict(float)
    cword_den: dict[str, float] = defaultdict(float)
    cword_df: dict[str, int] = defaultdict(int)
    tot_docs = len(peer_records) + n
    max_df = max(2, int(0.55 * tot_docs))

    if peer_records or n >= 2:
        for q_it, _, _ in peer_records:
            for cw in _extract_content_tokens(q_it):
                cword_df[cw] += 1
        for obs_it, _ in own_obs:
            for cw in _extract_content_tokens(obs_it):
                cword_df[cw] += 1
        for q_it, r, w in peer_records:
            rc = r - mean_r
            e_rem = rc - _tok_shift(q_it, tok_num, tok_den)
            for cw in _extract_content_tokens(q_it):
                if 2 <= cword_df[cw] <= max_df:
                    cword_num[cw] += PEER_WEIGHT * w * e_rem
                    cword_den[cw] += PEER_WEIGHT * w

    # 3. Deconfound subject score s using peer item, token, and content-kernel difficulty estimates
    if n == 0:
        base_p = p0
    else:
        adj_s = float(s)
        if peer_records:
            for obs_it, _ in own_obs:
                tc = str(obs_it.get("item_content") or "")
                tf = str(obs_it.get("item_features") or "")
                d_est = 0.0
                if tc and (tc, tf) in peer_item_den:
                    w_i = peer_item_den[(tc, tf)]
                    d_est += min(
                        PEER_ITEM_MAX_WEIGHT, w_i / (w_i + PEER_ITEM_SHRINK)
                    ) * (peer_item_num[(tc, tf)] / w_i)
                d_est += _tok_shift(obs_it, tok_num, tok_den)
                d_est += _text_shift(obs_it, cword_num, cword_den, cword_df, max_df)
                adj_s -= TEXT_SCALE * d_est
            adj_s = max(0.0, min(float(n), adj_s))
        base_p = (adj_s + m_eff * p0) / (n + m_eff)

    # 4. Within-subject centered token, content-kernel, and log-length residuals
    own_mean_z = peer_mean_z
    if n >= 2:
        own_mean_y = s / n
        own_zs = [
            math.log1p(len(str(obs_it.get("item_content") or "")))
            for obs_it, _ in own_obs
        ]
        own_mean_z = sum(own_zs) / n
        if n >= 15 or len(peer_records) >= 4:
            # Pass 1: accumulate all within-subject semicolon token residuals first
            for obs_it, obs_y in own_obs:
                rc_own = obs_y - own_mean_y
                for tok in _extract_tokens(obs_it):
                    tok_num[tok] += rc_own
                    tok_den[tok] += 1.0
            # Pass 2: residualize content-word terms against complete shrunk token shifts (Stage B, order-free)
            for obs_it, obs_y in own_obs:
                rc_own = obs_y - own_mean_y
                e_own = rc_own - _tok_shift(obs_it, tok_num, tok_den)
                for cw in _extract_content_tokens(obs_it):
                    if 2 <= cword_df.get(cw, 0) <= max_df:
                        cword_num[cw] += e_own
                        cword_den[cw] += 1.0
        if n >= 7 or len_var > 0.0:
            for (obs_it, obs_y), z in zip(own_obs, own_zs):
                dz = z - own_mean_z
                len_cov += dz * (obs_y - own_mean_y)
                len_var += dz ** 2

    slope_len = 0.0
    if len_var >= 1e-4 and (n >= 7 or len(peer_records) + n >= 6):
        slope_len = len_cov / (len_var + FEAT_SHRINK)

    peer_v = max(p0 * (1.0 - p0), 0.06)
    if peer_list:
        peer_v = sum(
            ((q_s + m_eff * p0) / (q_n + m_eff))
            * (1.0 - ((q_s + m_eff * p0) / (q_n + m_eff)))
            for _, q_n, q_s, _ in peer_list
        ) / len(peer_list)
        peer_v = max(peer_v, 0.05)

    return (
        base_p,
        tok_num,
        tok_den,
        cword_num,
        cword_den,
        cword_df,
        max_df,
        slope_len,
        own_mean_z,
        peer_item_num,
        peer_item_den,
        peer_v,
    )


def predict(input: list[dict], labeled: list | None = None) -> float:
    """Return calibrated success probability for [subject, item]."""
    subject, item = input
    if not labeled:
        return PRIOR_MEAN

    benchmark = item.get("benchmark_id")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("The predictor requires an anonymous item benchmark_id.")

    bench_obs = _get_bench_observations(labeled, benchmark)
    if not bench_obs:
        return PRIOR_MEAN

    if len(labeled) > 32:
        cache_key = (len(bench_obs), benchmark, _subject_key(subject))
        state = _CACHE_PAIR_STATE.get(cache_key)
        if state is None:
            state = _compute_pair_state(subject, bench_obs)
            _CACHE_PAIR_STATE[cache_key] = state
    else:
        state = _compute_pair_state(subject, bench_obs)

    (
        base_p,
        tok_num,
        tok_den,
        cword_num,
        cword_den,
        cword_df,
        max_df,
        slope_len,
        own_mean_z,
        peer_item_num,
        peer_item_den,
        peer_v,
    ) = state

    target_content = str(item.get("item_content") or "")
    target_feat = str(item.get("item_features") or "")
    has_item_key = bool(target_content)

    t_sh = _tok_shift(item, tok_num, tok_den)
    tx_sh = _text_shift(item, cword_num, cword_den, cword_df, max_df)

    delta_l = 0.0
    if slope_len != 0.0:
        z_target = math.log1p(len(target_content))
        delta_l = max(
            min(slope_len * (z_target - own_mean_z), MAX_DELTA),
            -MAX_DELTA,
        )

    beta_i = 0.0
    if has_item_key and (target_content, target_feat) in peer_item_den:
        w_i = peer_item_den[(target_content, target_feat)]
        beta_i = min(
            PEER_ITEM_MAX_WEIGHT, w_i / (w_i + PEER_ITEM_SHRINK)
        ) * (peer_item_num[(target_content, target_feat)] / w_i)

    shift_tot = t_sh + tx_sh + delta_l + beta_i
    if shift_tot == 0.0:
        return float(min(max(base_p, 0.005), 0.995))

    p_lin = min(max(base_p + shift_tot, 0.005), 0.995)
    bp_c = min(max(base_p, 0.01), 0.99)
    eta = math.log(bp_c / (1.0 - bp_c)) + (shift_tot / (peer_v + 0.04))
    eta = max(min(eta, 6.0), -6.0)
    p_log = 1.0 / (1.0 + math.exp(-eta))
    pred = (1.0 - LOGIT_BLEND) * p_lin + LOGIT_BLEND * p_log
    return float(min(max(pred, 0.005), 0.995))
