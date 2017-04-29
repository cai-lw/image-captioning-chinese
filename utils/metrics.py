from collections import Counter
from math import log, sqrt, exp

def bleu(cand, refs, n):
    len_cand = len(cand) - n + 1
    if len_cand <= 0:
        return 0
    cand_count = Counter(make_ngram(cand, n))
    refs_count = [Counter(make_ngram(ref, n)) for ref in refs]
    score = 0
    for ngram, count in cand_count.items():
        m_max = max(ref_count.get(ngram, 0) for ref_count in refs_count)
        score += min(count, m_max) / len_cand
    return score

def make_ngram(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]

def rouge_l(cand, refs):
    ref_lcs = [lcs(cand, ref) for ref in refs]
    ref_len = [len(ref) for ref in refs]
    recall = max(x / y for x, y in zip(ref_lcs, ref_len))
    precision = max(ref_lcs) / len(cand)
    beta = 1.2
    if recall == 0 and precision == 0:
        return 0
    else:
        return (1 + beta ** 2) * recall * precision / (recall + beta ** 2 * precision)

def lcs(a, b):
    dp = [0] * len(b)
    for i in range(len(a)):
        dp_new = [0] * len(b)
        if b[0] == a[i]:
            dp_new[0] = 1
        else:
            dp_new[0] = dp[0]
        for j in range(1, len(b)):
            if b[j] == a[i]:
                dp_new[j] = dp[j - 1] + 1
            else:
                dp_new[j] = max(dp_new[j - 1], dp[j])
        dp = dp_new
    return max(dp)

def build_idf(doc_dict):
    doc_size = len(doc_dict)
    idf = {}
    for n in range(1, 5):
        ref_ngrams = [[ngram for ref in refs for ngram in make_ngram(ref, n)] for refs in doc_dict.values()]
        df = {}
        for ngrams in ref_ngrams:
            for ngram in set(ngrams):
                df.setdefault(ngram, 0)
                df[ngram] += 1
        idf_n = dict((ngram, log(doc_size / f)) for ngram, f in df.items())
        idf.update(idf_n)
    return idf

def cider_g(seq, idf, n_img, n):
    ngram_count = Counter(make_ngram(seq, n))
    len_seq = len(seq) - n + 1
    return dict((ngram, f / len_seq * idf.get(ngram, log(n_img))) for ngram, f in ngram_count.items())

def dict_min(d1, d2):
    all_keys = set(d1.keys()).intersection(d2.keys())
    return dict((k, min(d1[k], d2[k])) for k in all_keys)

def dict_inner(d1, d2):
    return sum(v * d2.get(k, 0) for k, v in d1.items())

def dict_norm(d):
    return sqrt(sum(x ** 2 for x in d.values()))

def cider_d_n(cand, refs, idf, n_img, n):
    cand_g = cider_g(cand, idf, n_img, n)
    sigma = 6
    accum = 0
    for ref in refs:
        ref_g = cider_g(ref, idf, n_img, n)
        coeff = exp(-(len(cand) - len(ref)) ** 2 / (2 * sigma ** 2))
        if dict_norm(cand_g) == 0 or dict_norm(ref_g) == 0:
            cider = 0
        else:
            cider = dict_inner(dict_min(cand_g, ref_g), ref_g) / (dict_norm(cand_g) * dict_norm(ref_g))
        accum += coeff * cider
    return 10 / len(refs) * accum

def cider_d(cand, refs, idf, n_img):
    return sum(cider_d_n(cand, refs, idf, n_img, n) for n in range(1, 5)) / 4