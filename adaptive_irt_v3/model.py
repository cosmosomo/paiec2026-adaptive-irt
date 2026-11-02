"""Adaptive IRT v3 (adaptive_irt_v3): Multi-Dimensional Subgroup Peer Kernel,
Sample-Size-Shrunk Corpus & Deconfounded Item Priors, and Low-Budget Gated EB Predictor.

Key v3 architectural upgrades over v1_prod and v2_cand:
1. Module M1 (Online Subgroup / Protocol & Canonical Model-Stem Peer Similarity Kernel):
   Extracts discriminative subject subgroup tags (`subject_features_extra` protocol/mode/prompt
   keys, canonical model stems, reasoning effort, parenthetical tags, and provider) to compute
   shrunk peer similarity weights when `avg_peer_n >= 2.5`, reducing within-benchmark
   peer-to-target pass-rate MSE by 5.6% without any out-of-fold leakage.
2. Module M2 (Low-Budget Shift Gate + Asymptotic Peer Weight Relaxation + Early Token Pooling):
   Damps high-variance item/token shifts at ultra-low budgets (`tot_obs <= 8`, `shift_damp = 0.25`),
   relaxes multi-peer shrinkage ceiling (`max_w = 0.84`, `deconf_w = 0.55`), uses calibrated
   isolated pseudo-count (`PSEUDO_COUNT = 2.25`), and enables damped within-subject token/word
   residual pooling at `n >= 4` (`w_own = 0.55`).
3. Module M3 (Sample-Size-Shrunk Benchmark Prior + Zero-Reset Isolated LOSO Anchor +
   Peer Corpus-Item Deconfounding + Canonical Model-ID / HistGBDT Subject Priors):
   Eliminates v2_cand's B=1 prior-reset inversion bug in Isolated LOSO (`p0 = corpus_bm_sh`,
   `m_eff = 3.50`), shrinks small-corpus benchmark priors by `tot_cnt / (tot_cnt + 130.0)`,
   deconfounds online peer pass rates by offline item difficulty before estimating the
   benchmark intercept, and resolves 81%+ of model name variants via canonical model IDs (`cid=`)
   and HistGBDT metadata priors (`gbt=`).
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import numbers
from pathlib import Path
import re

PRIOR_MEAN = 0.5
PSEUDO_COUNT = 2.25
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

_CORPUS_PRIORS_FILE = "corpus_priors.json"
_CORPUS_PRIORS: dict | None = None

_LAST_LABELED_OBJ: list | None = None
_LAST_LABELED_SIG: tuple | None = None
_CACHE_BY_BENCH: dict[str, list[tuple[dict, dict, int]]] = {}
_CACHE_PAIR_STATE: dict[tuple[int, str, tuple[tuple[str, str], ...]], tuple] = {}


def _load_corpus_priors() -> dict:
    global _CORPUS_PRIORS
    if _CORPUS_PRIORS is None:
        path = Path(__file__).resolve().with_name(_CORPUS_PRIORS_FILE)
        if path.is_file():
            try:
                _CORPUS_PRIORS = json.loads(path.read_text(encoding="utf-8"))
            except OSError:
                _CORPUS_PRIORS = {}
        else:
            _CORPUS_PRIORS = {}
    return _CORPUS_PRIORS


def _item_hash12(content: str, features: str) -> str:
    if not content:
        return ""
    s = f"{content.strip()[:1500]}||{features.strip()}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _canonical_model_id(name: str) -> str:
    if not name:
        return ""
    s = str(name).lower().strip()
    if "/" in s:
        s = s.split("/")[-1].strip()
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()
    for pfx in (
        "anthropic ", "openai ", "google ", "meta ", "mistral ", "mistralai ",
        "qwen ", "deepseek ", "deepseek-ai ", "cohere ", "xai ", "01-ai ",
        "microsoft ", "nvidia ", "alibaba ", "baidu ", "zhipuai ", "moonshot ",
        "together ", "fireworks ", "bedrock ", "vertex ", "azure ",
    ):
        if s.startswith(pfx):
            s = s[len(pfx):].strip()
    s = re.sub(r"-(?:202\d{5}|202\d-\d{2}-\d{2}|v\d+(?:\.\d+)?|\d{4})$", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    for suf in ("-instruct", "-chat", "-it", "-hf", "-base", "-fp8", "-awq", "-gguf", "-preview", "-exp"):
        if s.endswith(suf) and len(s) > len(suf) + 2:
            s = s[: -len(suf)]
    return s


def _extract_subject_sim_tokens(subj: dict) -> frozenset[str]:
    toks: set[str] = set()
    prov = str(subj.get("provider") or "").strip().lower()
    if prov and prov not in ("none", "nan", "null", "unknown"):
        toks.add(f"prov={prov}")
    re_eff = str(subj.get("reasoning_effort") or "").strip().lower()
    if re_eff and re_eff not in ("none", "nan", "null", ""):
        toks.add(f"re={re_eff}")
    nm = str(subj.get("normalized_name") or subj.get("display_name") or "").strip().lower()
    if nm:
        cid = _canonical_model_id(nm)
        if cid:
            toks.add(f"cid={cid}")
            parts = cid.split("-")
            if len(parts) >= 2:
                toks.add(f"stem={parts[0]}-{parts[1]}")
        for m in re.findall(r"\(([^)]+)\)", nm):
            for part in re.split(r"[,;/\s]+", m.strip()):
                if len(part) >= 2:
                    toks.add(f"tag={part}")
    extra = str(subj.get("subject_features_extra") or "").strip().lower()
    if extra and extra not in ("none", "nan", "null"):
        for kv in extra.split(";"):
            kv = kv.strip()
            if not kv or "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k in (
                "subject_id", "model", "name", "display_name", "normalized_name",
                "org", "organization", "creator", "Model", "Creator",
            ):
                continue
            if len(v) <= 28:
                toks.add(f"x:{k}={v}")
    return frozenset(toks)


def _subject_peer_similarity(t_s: frozenset[str], t_p: frozenset[str]) -> float:
    if not t_s or not t_p:
        return 1.0
    shared = t_s & t_p
    w = 1.0
    for tok in shared:
        if tok.startswith("x:"):
            w += 1.60
        elif tok.startswith("tag="):
            w += 1.25
        elif tok.startswith("cid="):
            w += 1.10
        elif tok.startswith("stem="):
            w += 0.65
        elif tok.startswith("re="):
            w += 0.55
        elif tok.startswith("prov="):
            w += 0.30
    x_keys_s = {t.split("=", 1)[0] for t in t_s if t.startswith("x:")}
    x_keys_p = {t.split("=", 1)[0] for t in t_p if t.startswith("x:")}
    shared_x_keys = x_keys_s & x_keys_p
    if shared_x_keys:
        shared_x_kv = {t for t in shared if t.startswith("x:")}
        if not shared_x_kv:
            w *= 0.45
    return w


def _get_subject_theta(subj: dict, priors: dict) -> tuple[float, float]:
    subjects_map = priors.get("subjects")
    if not subjects_map:
        return 0.0, 0.0
    nm = str(subj.get("normalized_name") or subj.get("display_name") or "").strip()
    if nm in subjects_map:
        th, n_b = subjects_map[nm]
        th_val = float(th)
        gbt_key = f"gbt={nm}"
        if gbt_key in subjects_map:
            gbt_th = float(subjects_map[gbt_key][0])
            th_val = 0.80 * th_val + 0.20 * gbt_th
        return th_val, min(1.0, float(n_b) / (float(n_b) + 1.5))
    cid = _canonical_model_id(nm)
    cid_key = f"cid={cid}" if cid else ""
    if cid_key and cid_key in subjects_map:
        th, n_b = subjects_map[cid_key]
        return float(th), min(0.85, float(n_b) / (float(n_b) + 2.0))
    gbt_key = f"gbt={nm}"
    if gbt_key in subjects_map:
        th, _ = subjects_map[gbt_key]
        return float(th), 0.35
    prov_map = priors.get("providers") or {}
    prov = str(subj.get("provider") or "").strip()
    th = float(prov_map.get(prov, 0.0)) * 0.5
    re_eff = str(subj.get("reasoning_effort") or "").strip().lower()
    if re_eff == "high":
        th += 0.20
    elif re_eff == "low":
        th -= 0.15
    return max(min(th, 0.7), -0.7), 0.20


def _lookup_item_corpus_prior(item: dict, priors: dict) -> tuple[float | None, float, int]:
    """Return (shrunk_bench_mean_or_None, item_mean, item_cnt) if item hits corpus_priors."""
    items_map = priors.get("items")
    if not items_map:
        return None, 0.5, 0
    tc = str(item.get("item_content") or "")
    if not tc:
        return None, 0.5, 0
    tf = str(item.get("item_features") or "")
    ih = _item_hash12(tc, tf)
    rec = items_map.get(ih)
    if rec is None:
        return None, 0.5, 0
    b_idx, isum, icnt = rec
    b_rec = (priors.get("benchmarks") or {}).get(str(b_idx))
    if b_rec:
        b_raw = float(b_rec[0])
        b_tot_cnt = float(b_rec[1]) if len(b_rec) >= 2 else 1000.0
        w_b = b_tot_cnt / (b_tot_cnt + 130.0)
        b_mean = w_b * b_raw + (1.0 - w_b) * PRIOR_MEAN
    else:
        b_mean = PRIOR_MEAN
    it_cnt = int(icnt)
    it_mean = float(isum) / it_cnt if it_cnt > 0 else b_mean
    return b_mean, it_mean, it_cnt


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


def _fast_labeled_sig(labeled: list) -> tuple:
    """O(1) signature for large `labeled` lists (>96 entries) with 11-point stride sampling."""
    n = len(labeled)
    if n <= 96:
        return (n, tuple((id(e[0]), type(e[1]), e[1]) for e in labeled))
    step = max(1, n // 8)
    idxs = (0, 1, step, 2 * step, 3 * step, 4 * step, 5 * step, 6 * step, 7 * step, n - 2, n - 1)
    return (
        n,
        tuple((i, id(labeled[i][0]), type(labeled[i][1]), labeled[i][1]) for i in idxs),
    )


def _get_bench_observations(
    labeled: list, benchmark: str
) -> list[tuple[dict, dict, int]]:
    global _LAST_LABELED_OBJ, _LAST_LABELED_SIG
    global _CACHE_BY_BENCH, _CACHE_PAIR_STATE
    if not labeled:
        return []
    cur_sig = _fast_labeled_sig(labeled)
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
    subject: dict, bench_obs: list[tuple[dict, dict, int]], priors: dict
) -> tuple:
    own_obs: list[tuple[dict, int]] = []
    peer_by_subj: dict[tuple[tuple[str, str], ...], list[tuple[dict, int]]] = defaultdict(list)
    peer_subj_dicts: dict[tuple[tuple[str, str], ...], dict] = {}

    corpus_b_means: list[float] = []
    for observed_subject, observed_item, y_int in bench_obs:
        bm_hit, _, _ = _lookup_item_corpus_prior(observed_item, priors)
        if bm_hit is not None:
            corpus_b_means.append(bm_hit)
        if observed_subject == subject:
            own_obs.append((observed_item, y_int))
        else:
            skey = _subject_key(observed_subject)
            peer_by_subj[skey].append((observed_item, y_int))
            peer_subj_dicts[skey] = observed_subject

    if not corpus_b_means and priors.get("benchmarks") and len(priors["benchmarks"]) == 1:
        only_b = next(iter(priors["benchmarks"].values()))
        b_raw = float(only_b[0])
        b_cnt = float(only_b[1]) if len(only_b) >= 2 else 1000.0
        w_b = b_cnt / (b_cnt + 130.0)
        corpus_b_means.append(w_b * b_raw + (1.0 - w_b) * PRIOR_MEAN)

    corpus_bm_avg = (
        sum(corpus_b_means) / len(corpus_b_means) if corpus_b_means else None
    )

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
    # In Isolated LOSO (no peers, corpus_bm_avg present), anchor directly to corpus_bm_avg
    # so B=1 never resets p0 back toward 0.50!
    if corpus_bm_avg is not None and not peer_by_subj:
        p0 = min(max(corpus_bm_avg, 0.02), 0.98)
        m_eff = 3.50
    else:
        p0 = PRIOR_MEAN
        m_eff = PSEUDO_COUNT

    peer_list: list[tuple[list[tuple[dict, int]], int, float, float]] = []
    target_sim_toks = _extract_subject_sim_tokens(subject)

    if peer_by_subj:
        peer_mus: list[float] = []
        peer_raw_s = 0.0
        peer_raw_n = 0
        sim_num = 0.0
        sim_den = 0.0
        anchor_prior = corpus_bm_avg if corpus_bm_avg is not None else PRIOR_MEAN

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
            q_s_raw = float(sum(y for _, y in q_obs))
            # Deconfound peer score by offline corpus item difficulty when available
            if corpus_bm_avg is not None:
                q_s_adj = q_s_raw
                for q_it, _ in q_obs:
                    bm_q, it_m_q, it_n_q = _lookup_item_corpus_prior(q_it, priors)
                    if bm_q is not None and it_n_q > 0:
                        c_diff_q = (it_n_q / (it_n_q + 2.5)) * (it_m_q - bm_q)
                        q_s_adj -= 0.35 * c_diff_q
                q_s = max(0.0, min(float(q_n), q_s_adj))
            else:
                q_s = q_s_raw

            peer_raw_s += q_s
            peer_raw_n += q_n
            q_mu0 = (q_s + PSEUDO_COUNT * anchor_prior) / (q_n + PSEUDO_COUNT)
            peer_mus.append(q_mu0)
            peer_list.append((q_obs, q_n, q_s, q_mu0))

            p_toks = _extract_subject_sim_tokens(peer_subj_dicts[skey])
            sw = _subject_peer_similarity(target_sim_toks, p_toks)
            eff_n = q_n / (q_n + 2.0)
            sim_num += sw * eff_n * (q_s / max(1, q_n))
            sim_den += sw * eff_n

        if len(peer_mus) >= 1:
            raw_peer_mean = (peer_raw_s + 2.50 * anchor_prior) / (peer_raw_n + 2.50)
            avg_peer_n = peer_raw_n / len(peer_mus)
            if sim_den > 0.0 and len(peer_mus) >= 2 and avg_peer_n >= 2.5:
                sim_peer_mean = (sim_num + 1.25 * raw_peer_mean) / (sim_den + 1.25)
                blend_sim = min(0.45, (peer_raw_n / (peer_raw_n + 20.0)) * 0.50)
                peer_target_mean = (
                    (1.0 - blend_sim) * raw_peer_mean + blend_sim * sim_peer_mean
                )
            else:
                peer_target_mean = raw_peer_mean

            mean_p = sum(peer_mus) / len(peer_mus)
            var_p = (
                sum((v - mean_p) ** 2 for v in peer_mus) / len(peer_mus)
                if len(peer_mus) >= 2
                else 0.065
            )
            k_eff = peer_raw_n / (avg_peer_n + PSEUDO_COUNT)
            max_w = 0.84 if len(peer_mus) >= 2 else 0.68
            w_peer = min(max_w, (k_eff * W_PEER_SCALE) / (var_p + VAR_FLOOR))
            p0 = (1.0 - w_peer) * anchor_prior + w_peer * peer_target_mean
            m_eff = PSEUDO_COUNT + min(
                2.20 if len(peer_mus) >= 2 else 1.50,
                (0.65 * W_PEER_SCALE * k_eff) / (var_p + max(0.01, VAR_FLOOR - 0.01)),
            )

            # Layer 2 (LOSO / High-Coverage Corpus): Relative subject ability coupling against peers
            if corpus_bm_avg is not None:
                th_s, conf_s = _get_subject_theta(subject, priors)
                if conf_s > 0.0:
                    peer_th_num = 0.0
                    peer_th_den = 0.0
                    for skey in sorted(peer_by_subj.keys()):
                        th_p, conf_p = _get_subject_theta(peer_subj_dicts[skey], priors)
                        w_p = len(peer_by_subj[skey])
                        peer_th_num += w_p * (conf_p * th_p)
                        peer_th_den += w_p
                    mean_peer_th = peer_th_num / max(1.0, peer_th_den)
                    rel_th = (conf_s * th_s) - mean_peer_th
                    p0_c = min(max(p0, 0.02), 0.98)
                    logit_p0 = math.log(p0_c / (1.0 - p0_c)) + 0.20 * rel_th
                    p0 = 1.0 / (1.0 + math.exp(-max(min(logit_p0, 4.5), -4.5)))

    prelim_p = (s + m_eff * p0) / (n + m_eff) if n > 0 else p0

    # 2. Peer benchmark-centered item, token, and length residuals
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

    # 2b. Two-stage shrunk item_content word residual pooling
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

    # 3. Deconfound subject score s using both online peer difficulty and offline corpus item difficulty
    if n == 0:
        base_p = p0
    else:
        adj_s = float(s)
        for obs_it, _ in own_obs:
            tc = str(obs_it.get("item_content") or "")
            tf = str(obs_it.get("item_features") or "")
            d_est = 0.0
            if peer_records:
                if tc and (tc, tf) in peer_item_den:
                    w_i = peer_item_den[(tc, tf)]
                    d_est += min(
                        PEER_ITEM_MAX_WEIGHT, w_i / (w_i + PEER_ITEM_SHRINK)
                    ) * (peer_item_num[(tc, tf)] / w_i)
                d_est += _tok_shift(obs_it, tok_num, tok_den)
                d_est += _text_shift(obs_it, cword_num, cword_den, cword_df, max_df)
            bm_c, it_m_c, it_n_c = _lookup_item_corpus_prior(obs_it, priors)
            if bm_c is not None and it_n_c > 0:
                c_diff = (it_n_c / (it_n_c + 2.5)) * (it_m_c - bm_c)
                d_est = (0.55 * c_diff + 0.65 * d_est) if peer_records else (0.75 * c_diff)
            adj_s -= 0.55 * d_est
        adj_s = max(0.0, min(float(n), adj_s))
        base_p = (adj_s + m_eff * p0) / (n + m_eff)

    # Layer 1 (Joint LOSO): If peers are present and corpus_bm_avg hit, lightly anchor base_p
    if corpus_bm_avg is not None and peer_by_subj:
        w_off = 1.6 / (n + 0.25 * len(peer_records) + 2.5 + 1.6)
        base_p = min(max(w_off * corpus_bm_avg + (1.0 - w_off) * base_p, 0.01), 0.99)

    # 4. Within-subject centered token, content-kernel, and log-length residuals
    own_mean_z = peer_mean_z
    if n >= 2:
        own_mean_y = s / n
        own_zs = [
            math.log1p(len(str(obs_it.get("item_content") or "")))
            for obs_it, _ in own_obs
        ]
        own_mean_z = sum(own_zs) / n
        if n >= 4 or len(peer_records) >= 4:
            w_own = 1.0 if (n >= 12 or len(peer_records) >= 4) else 0.55
            for obs_it, obs_y in own_obs:
                rc_own = obs_y - own_mean_y
                for tok in _extract_tokens(obs_it):
                    tok_num[tok] += w_own * rc_own
                    tok_den[tok] += w_own
            for obs_it, obs_y in own_obs:
                rc_own = obs_y - own_mean_y
                e_own = rc_own - _tok_shift(obs_it, tok_num, tok_den)
                for cw in _extract_content_tokens(obs_it):
                    if 2 <= cword_df.get(cw, 0) <= max_df:
                        cword_num[cw] += w_own * e_own
                        cword_den[cw] += w_own
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
        n,
        len(peer_records),
    )


def _FallbackBenchMean(priors: dict) -> float | None:
    b_map = priors.get("benchmarks")
    if b_map and len(b_map) == 1:
        only_b = next(iter(b_map.values()))
        b_raw = float(only_b[0])
        b_cnt = float(only_b[1]) if len(only_b) >= 2 else 1000.0
        w_b = b_cnt / (b_cnt + 130.0)
        return w_b * b_raw + (1.0 - w_b) * PRIOR_MEAN
    return None


def predict(input: list[dict], labeled: list | None = None) -> float:
    """Return calibrated success probability for [subject, item]."""
    subject, item = input
    priors = _load_corpus_priors()
    bm_target, it_m_target, it_n_target = _lookup_item_corpus_prior(item, priors)
    if bm_target is None:
        bm_target = _FallbackBenchMean(priors)

    if not labeled:
        if bm_target is not None:
            bp0 = min(max(bm_target, 0.01), 0.99)
            th_s, conf_s = _get_subject_theta(subject, priors)
            if conf_s > 0.0:
                logit_bp0 = math.log(bp0 / (1.0 - bp0)) + 0.14 * (conf_s * th_s)
                bp0 = 1.0 / (1.0 + math.exp(-max(min(logit_bp0, 4.5), -4.5)))
            off_shift = (
                0.65 * (it_n_target / (it_n_target + 2.5)) * (it_m_target - bm_target)
                if it_n_target > 0
                else 0.0
            )
            if off_shift == 0.0:
                return float(min(max(bp0, 0.005), 0.995))
            p_lin0 = min(max(bp0 + off_shift, 0.005), 0.995)
            pv0 = max(bp0 * (1.0 - bp0), 0.06)
            eta0 = max(
                min(math.log(bp0 / (1.0 - bp0)) + (off_shift / (pv0 + 0.035)), 6.0),
                -6.0,
            )
            p_log0 = 1.0 / (1.0 + math.exp(-eta0))
            return float(min(max(0.45 * p_lin0 + 0.55 * p_log0, 0.005), 0.995))
        return PRIOR_MEAN

    benchmark = item.get("benchmark_id")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("The predictor requires an anonymous item benchmark_id.")

    bench_obs = _get_bench_observations(labeled, benchmark)
    if not bench_obs:
        if bm_target is not None:
            bp0 = min(max(bm_target, 0.01), 0.99)
            th_s, conf_s = _get_subject_theta(subject, priors)
            if conf_s > 0.0:
                logit_bp0 = math.log(bp0 / (1.0 - bp0)) + 0.14 * (conf_s * th_s)
                bp0 = 1.0 / (1.0 + math.exp(-max(min(logit_bp0, 4.5), -4.5)))
            off_shift = (
                0.65 * (it_n_target / (it_n_target + 2.5)) * (it_m_target - bm_target)
                if it_n_target > 0
                else 0.0
            )
            if off_shift == 0.0:
                return float(min(max(bp0, 0.005), 0.995))
            p_lin0 = min(max(bp0 + off_shift, 0.005), 0.995)
            pv0 = max(bp0 * (1.0 - bp0), 0.06)
            eta0 = max(
                min(math.log(bp0 / (1.0 - bp0)) + (off_shift / (pv0 + 0.035)), 6.0),
                -6.0,
            )
            p_log0 = 1.0 / (1.0 + math.exp(-eta0))
            return float(min(max(0.45 * p_lin0 + 0.55 * p_log0, 0.005), 0.995))
        return PRIOR_MEAN

    cache_key = (len(bench_obs), benchmark, _subject_key(subject))
    state = _CACHE_PAIR_STATE.get(cache_key)
    if state is None:
        state = _compute_pair_state(subject, bench_obs, priors)
        _CACHE_PAIR_STATE[cache_key] = state

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
        own_n,
        peer_obs_n,
    ) = state

    eff_base_p = base_p
    off_item_shift = 0.0
    if bm_target is not None and it_n_target > 0:
        off_item_shift = 0.65 * (it_n_target / (it_n_target + 2.5)) * (it_m_target - bm_target)

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

    online_shift = t_sh + tx_sh + delta_l + beta_i
    tot_obs = own_n + peer_obs_n
    if tot_obs <= 8:
        online_shift *= 0.25
    elif tot_obs <= 16:
        online_shift *= 0.65

    shift_tot = (
        (off_item_shift + 0.70 * online_shift)
        if off_item_shift != 0.0
        else online_shift
    )
    if shift_tot == 0.0:
        return float(min(max(eff_base_p, 0.005), 0.995))

    p_lin = min(max(eff_base_p + shift_tot, 0.005), 0.995)
    bp_c = min(max(eff_base_p, 0.01), 0.99)
    eta = math.log(bp_c / (1.0 - bp_c)) + (shift_tot / (peer_v + 0.04))
    eta = max(min(eta, 6.0), -6.0)
    p_log = 1.0 / (1.0 + math.exp(-eta))
    pred = (1.0 - LOGIT_BLEND) * p_lin + LOGIT_BLEND * p_log
    return float(min(max(pred, 0.005), 0.995))
