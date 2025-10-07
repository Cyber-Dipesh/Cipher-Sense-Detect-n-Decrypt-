#!/usr/bin/env python3

import sys
import argparse
import base64
import binascii
import string
import math
import random
import itertools
import gzip
import os
from collections import Counter, defaultdict

#  Utilities & Scoring 
ENGLISH_FREQ = {
    'E': 12.02,'T':9.10,'A':8.12,'O':7.68,'I':7.31,'N':6.95,'S':6.28,'R':6.02,'H':5.92,
    'D':4.32,'L':3.98,'U':2.88,'C':2.71,'M':2.61,'F':2.30,'Y':2.11,'W':2.09,'G':2.03,
    'P':1.82,'B':1.49,'V':1.11,'K':0.69,'X':0.17,'Q':0.11,'J':0.10,'Z':0.07
}
COMMON_WORDS = set(['the','be','to','of','and','a','in','that','have','I','it','for','not','on','with','he','as','you','do','at'])

QUADGRAMS = {}
TOTAL_QUADGRAMS_LOG = 0.0


def load_quadgrams(path='quadgrams.txt'):
    """Load quadgram log-probabilities from file into QUADGRAMS dict (global)."""
    global QUADGRAMS, TOTAL_QUADGRAMS_LOG
    if not os.path.exists(path):
        # If missing, fall back to a tiny built-in set (still better than nothing)
        QUADGRAMS.clear()
        QUADGRAMS.update({
            'TION': -3.2, 'THER': -3.3, 'HERE': -3.4, 'THAT': -3.5, 'OF_T': -4.0
        })
        TOTAL_QUADGRAMS_LOG = -1000.0
        return
    q = {}
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            gram, count = parts
            q[gram.upper()] = int(count)
            total += int(count)

    # convert counts to log probabilities
    QUADGRAMS.clear()
    for gram, cnt in q.items():
        QUADGRAMS[gram] = math.log10(float(cnt) / total)
    TOTAL_QUADGRAMS_LOG = math.log10(1.0/total) if total > 0 else -1000.0


def quadgram_score(text):
    """Score text using quadgram log probabilities. Higher is better."""
    if not QUADGRAMS:
        return 0.0
    text = ''.join([c for c in text.upper() if c.isalpha()])
    if len(text) < 4:
        return -1e6
    score = 0.0
    for i in range(len(text)-3):
        gram = text[i:i+4]
        score += QUADGRAMS.get(gram, TOTAL_QUADGRAMS_LOG)
    return score


def score_english(text):
    """Combine quadgram score and word-match bonuses; return higher = better."""
    qscore = quadgram_score(text)
    # letter-frequency chi2 fallback
    text_u = ''.join([c for c in text.upper() if c.isalpha()])
    if not text_u:
        return -1e9
    counter = Counter(text_u)
    total = len(text_u)
    chi2 = 0.0
    for ch, exp in ENGLISH_FREQ.items():
        obs = counter.get(ch, 0)
        expected = total * (exp / 100.0)
        chi2 += (obs - expected)**2 / (expected + 0.0001)
    words = text.split()
    common = sum(1 for w in words if w.lower().strip(string.punctuation) in COMMON_WORDS)
    return qscore * 15.0 - chi2 * 0.1 + common * 6.0


def is_printable_ratio(s, threshold=0.80):
    if not s:
        return False
    printable = sum(1 for c in s if c in string.printable)
    return (printable / max(1, len(s))) >= threshold


# Encodings + gzip detection 
def try_hex(s):
    try:
        b = binascii.unhexlify(s.strip())
        try:
            txt = b.decode('utf-8', errors='replace')
        except:
            txt = str(b)
        return txt
    except Exception:
        return None


def try_base64(s):
    t = s.strip()
    padding = len(t) % 4
    if padding:
        t += '=' * (4 - padding)
    try:
        b = base64.b64decode(t, validate=False)
        return b.decode('utf-8', errors='replace')
    except Exception:
        return None


def try_rot13(s):
    return s.translate(str.maketrans(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"
    ))


def try_gzip_base64(s):
    t = s.strip()
    padding = len(t) % 4
    if padding:
        t += '=' * (4 - padding)
    try:
        b = base64.b64decode(t, validate=False)
    except Exception:
        return None
    try:
        decompressed = gzip.decompress(b)
        return decompressed.decode('utf-8', errors='replace')
    except Exception:
        return None



# Caesar (shift) 

def caesar_shift(s, shift):
    res = []
    for ch in s:
        if 'a' <= ch <= 'z':
            res.append(chr((ord(ch)-97 + shift) % 26 + 97))
        elif 'A' <= ch <= 'Z':
            res.append(chr((ord(ch)-65 + shift) % 26 + 65))
        else:
            res.append(ch)
    return ''.join(res)


def try_caesar(s, top=5):
    candidates = []
    for k in range(26):
        p = caesar_shift(s, k)
        sc = score_english(p)
        candidates.append((sc, k, p))
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[:top]



#  Vigenere 

def index_of_coincidence(s):
    s_alpha = [c.upper() for c in s if c.isalpha()]
    n = len(s_alpha)
    if n <= 1:
        return 0.0
    freqs = Counter(s_alpha)
    ic = sum(v*(v-1) for v in freqs.values()) / (n*(n-1))
    return ic


def kasiski_examination(s, min_len=3, max_len=5):
    s_clean = ''.join([c for c in s if c.isalpha()])
    repeats = defaultdict(list)
    for L in range(min_len, max_len+1):
        for i in range(len(s_clean)-L):
            sub = s_clean[i:i+L]
            j = i+L
            while True:
                idx = s_clean.find(sub, j)
                if idx == -1:
                    break
                repeats[sub].append(idx - i)
                j = idx+1
    distances = []
    for sub, ds in repeats.items():
        for d in ds:
            distances.append(d)
    if not distances:
        return []
    def gcd(a,b):
        while b:
            a,b = b, a%b
        return a
    gcd_counts = Counter()
    for a,b in itertools.combinations(distances, 2):
        g = gcd(a, b)
        if g > 1 and g <= 40:
            gcd_counts[g] += 1
    likely = [k for k,_ in gcd_counts.most_common(6)]
    return likely


def likely_vigenere_keylengths(s, max_k=16):
    scores = []
    for k in range(1, max_k+1):
        chunks = [''.join(s[i::k]) for i in range(k)]
        ics = [index_of_coincidence(chunk) for chunk in chunks if len(chunk) > 1]
        avg_ic = sum(ics)/len(ics) if ics else 0.0
        scores.append((avg_ic, k))
    scores.sort(reverse=True)
    return [k for _, k in scores[:6]]


def vigenere_decrypt_with_key(s, key):
    res = []
    key_up = [ord(c.upper()) - 65 for c in key]
    ki = 0
    for ch in s:
        if ch.isalpha():
            base = 65 if ch.isupper() else 97
            k = key_up[ki % len(key)]
            res.append(chr((ord(ch) - base - k) % 26 + base))
            ki += 1
        else:
            res.append(ch)
    return ''.join(res)


def guess_vigenere_key_via_chi2(s, keylen):
    key = ''
    for i in range(keylen):
        col = ''.join([c for idx,c in enumerate(s) if idx % keylen == i and c.isalpha()])
        best_shift = 0
        best_score = 1e9
        for shift in range(26):
            dec = caesar_shift(col, shift)
            counter = Counter(dec.upper())
            total = max(1, len(dec))
            chi2 = 0.0
            for ch, exp in ENGLISH_FREQ.items():
                obs = counter.get(ch, 0)
                expected = total * (exp / 100.0)
                chi2 += (obs - expected)**2 / (expected + 0.0001)
            if chi2 < best_score:
                best_score = chi2
                best_shift = shift
        key += chr((26 - best_shift) % 26 + 65)
    return key


def try_vigenere(s, max_keylen=12, top=4):
    candidates = []
    ks_kasiski = kasiski_examination(s, 3, 5)
    ks_ic = likely_vigenere_keylengths(s, max_k=max_keylen)
    keylens = []
    for k in ks_kasiski:
        if k not in keylens:
            keylens.append(k)
    for k in ks_ic:
        if k not in keylens:
            keylens.append(k)

    for k in keylens[:12]:
        key = guess_vigenere_key_via_chi2(s, k)
        p = vigenere_decrypt_with_key(s, key)
        candidates.append((score_english(p), key, p))
    candidates.sort(reverse=True)
    return candidates[:top]


#  XOR (single-byte / short key) 

def xor_bytes(data, key_bytes):
    return bytes([b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(data)])


def try_xor_short(s, max_key_len=3, top=5):
    results = []
    try:
        raw = s.encode('latin1') if isinstance(s, str) else s
    except Exception:
        raw = s
    candidates_input_bytes = [raw]
    try:
        hb = binascii.unhexlify(s.strip())
        candidates_input_bytes.append(hb)
    except Exception:
        pass
    try:
        bb = base64.b64decode(s + '=' * (4 - (len(s) % 4)), validate=False)
        candidates_input_bytes.append(bb)
    except Exception:
        pass
    seen = set()
    for data in candidates_input_bytes:
        for keylen in range(1, max_key_len+1):
            for key in itertools.product(range(256), repeat=keylen):
                keyb = bytes(key)
                pt = xor_bytes(data, keyb)
                try:
                    pt_text = pt.decode('utf-8', errors='replace')
                except:
                    pt_text = str(pt)
                sc = score_english(pt_text)
                key_repr = keyb.hex()
                if (keylen, key_repr) in seen:
                    continue
                seen.add((keylen, key_repr))
                results.append((sc, key_repr, pt_text))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top]



# Substitution cipher solver (tuned) 

def substitution_score(plaintext):
    # Favor quadgram fitness primarily, with word-match bonus
    q = quadgram_score(plaintext)
    if q < -1e5:
        return -1e9
    words = plaintext.split()
    common = sum(1 for w in words if w.lower().strip(string.punctuation) in COMMON_WORDS)
    return q * 10.0 + common * 6.0


def random_key_alphabet():
    letters = list(string.ascii_uppercase)
    random.shuffle(letters)
    return ''.join(letters)


def decrypt_substitution(ct, key_map):
    res = []
    for ch in ct:
        if ch.isalpha():
            is_upper = ch.isupper()
            mapped = key_map[ch.upper()]
            res.append(mapped if is_upper else mapped.lower())
        else:
            res.append(ch)
    return ''.join(res)


def key_map_from_string(kstr):
    m = {}
    for i, c in enumerate(kstr):
        m[chr(ord('A')+i)] = c
    return m


def swap_chars_in_key(kstr, i, j):
    lst = list(kstr)
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)


def solve_substitution_simulated_annealing(ct, max_iters=5000, restarts=6):
    """Simulated annealing with an exponential cooling schedule and adaptive moves."""
    ct = ct
    best_global = (None, -1e9)
    # Seed with frequency mapping
    freq_order = ''.join([p[0] for p in Counter([c.upper() for c in ct if c.isalpha()]).most_common()])
    english_order = ''.join(sorted(ENGLISH_FREQ.keys(), key=lambda k: -ENGLISH_FREQ[k]))
    seed_map = list('?'*26)
    for i,ch in enumerate(freq_order):
        if i < 26:
            seed_map[ord(ch)-65] = english_order[i]
    remaining = [c for c in english_order if c not in seed_map]
    for i in range(26):
        if seed_map[i] == '?':
            seed_map[i] = remaining.pop(0) if remaining else random.choice(list(string.ascii_uppercase))
    seed_key = ''.join(seed_map)

    for restart in range(restarts):
        if restart == 0:
            current_key = seed_key
        else:
            current_key = random_key_alphabet()
        current_map = key_map_from_string(current_key)
        current_plain = decrypt_substitution(ct, current_map)
        current_score = substitution_score(current_plain)
        best_local = (current_key, current_score)
        # faster exponential cooling
        T_start = 1.0
        T_end = 0.001
        for it in range(1, max_iters+1):
            T = T_start * (T_end / T_start) ** (it / max(1, max_iters))
            # choose between swap and 3-swap for larger exploration early
            if random.random() < 0.85:
                i,j = random.sample(range(26), 2)
                new_key = swap_chars_in_key(current_key, i, j)
            else:
                # triple swap
                a,b,c = random.sample(range(26), 3)
                tmp = swap_chars_in_key(current_key, a, b)
                new_key = swap_chars_in_key(tmp, a, c)
            new_map = key_map_from_string(new_key)
            new_plain = decrypt_substitution(ct, new_map)
            new_score = substitution_score(new_plain)
            delta = new_score - current_score
            if delta > 0 or math.exp(delta / max(1e-9, T)) > random.random():
                current_key = new_key
                current_map = new_map
                current_plain = new_plain
                current_score = new_score
                if current_score > best_local[1]:
                    best_local = (current_key, current_score)
            # little random restart inside loop
            if it % 1200 == 0 and random.random() < 0.25:
                current_key = random_key_alphabet()
                current_map = key_map_from_string(current_key)
                current_plain = decrypt_substitution(ct, current_map)
                current_score = substitution_score(current_plain)
        if best_local[1] > best_global[1]:
            best_global = best_local
    return decrypt_substitution(ct, key_map_from_string(best_global[0])), best_global[0], best_global[1]





# ----------------- Auto-explore mode -----------------

def try_decode_bytes_variants(s: str):
    """Return list of (label, decoded_text) for sensible decode attempts."""
    out = []
    # try base64
    b64 = None
    try:
        t = s.strip()
        if len(t) % 4:
            t += '=' * (4 - (len(t) % 4))
        b = base64.b64decode(t, validate=False)
        try:
            btxt = b.decode('utf-8', errors='replace')
        except:
            btxt = None
        out.append(('base64', btxt if btxt is not None else b))
    except Exception:
        pass
    # try hex
    try:
        hb = binascii.unhexlify(s.strip())
        try:
            htxt = hb.decode('utf-8', errors='replace')
        except:
            htxt = None
        out.append(('hex', htxt if htxt is not None else hb))
    except Exception:
        pass
    # try rot13
    r13 = try_rot13(s)
    if r13 != s:
        out.append(('rot13', r13))
    return out


def is_likely_plaintext(s: str, min_common=2):
    # heuristic: printable high ratio and at least some common words
    if not s or not isinstance(s, str):
        return False
    pr = sum(1 for c in s if c in string.printable) / max(1, len(s))
    if pr < 0.7:
        return False
    common = sum(1 for w in s.split() if w.lower().strip(string.punctuation) in COMMON_WORDS)
    return common >= min_common or score_english(s) > 0


def auto_explore(s: str, max_depth=4, verbose=False):
    """Recursively attempt suggested transforms from detection, return list of successful decryptions.
    Stops when no new readable plaintext is produced or depth reached.
    Returns list of dicts with keys: depth, method_chain, plaintext, evidence_score
    """
    seen = set()
    results = []

    def _explore(curr_text, depth, chain):
        if depth > max_depth:
            return
        key = (curr_text.strip(), tuple(chain))
        if key in seen:
            return
        seen.add(key)
        # run detection to get suggestions
        candidates = detect_and_try_all(curr_text, mode=None)
        # also include direct decodes
        decs = try_decode_bytes_variants(curr_text)
        # process decodes first
        for label, decoded in decs:
            if decoded is None:
                continue
            if isinstance(decoded, bytes):
                try:
                    decoded_s = decoded.decode('utf-8', errors='replace')
                except:
                    decoded_s = str(decoded)
            else:
                decoded_s = decoded
            if decoded_s.strip() == '' or decoded_s == curr_text:
                continue
            if is_likely_plaintext(decoded_s):
                score = score_english(decoded_s)
                results.append({'depth': depth, 'method_chain': chain + [label], 'plaintext': decoded_s, 'score': score})
                if verbose:
                    print(f"[auto] depth={depth} got plaintext via {label} (score={score:.2f})")
            # always recurse to find chained encodings (e.g., base64->gzip->vigenere)
            _explore(decoded_s, depth+1, chain + [label])

        # process candidate attacks from detect_and_try_all
        for kind, p, sc in candidates:
            kl = kind.lower()
            if 'base64' in kl or 'hex' in kl:
                # decoded variant already handled above, but also add if p is different
                if p and p != curr_text:
                    _explore(p, depth+1, chain + [kind])
            elif 'xor' in kl and 'keyhex' in kl:
                # try to extract keyhex and decrypt (we already have plaintext p)
                if p and is_likely_plaintext(p):
                    results.append({'depth': depth, 'method_chain': chain + [kind], 'plaintext': p, 'score': sc})
            elif 'caesar' in kl:
                # take the best caesar candidate(s)
                if p and is_likely_plaintext(p):
                    results.append({'depth': depth, 'method_chain': chain + [kind], 'plaintext': p, 'score': sc})
                else:
                    # try brute force caesar shifts
                    caes = try_caesar(curr_text, top=6)
                    for sc2, k, p2 in caes:
                        if is_likely_plaintext(p2):
                            results.append({'depth': depth, 'method_chain': chain + [f'caesar({k})'], 'plaintext': p2, 'score': sc2})
                            _explore(p2, depth+1, chain + [f'caesar({k})'])
            elif 'vigenere' in kl:
                # try vigenere top candidates
                vigs = try_vigenere(curr_text, max_keylen=12, top=4)
                for sc2, key, p2 in vigs:
                    if is_likely_plaintext(p2):
                        results.append({'depth': depth, 'method_chain': chain + [f'vigenere(key={key})'], 'plaintext': p2, 'score': sc2})
                        _explore(p2, depth+1, chain + [f'vigenere(key={key})'])
            elif 'substitution' in kl:
                # run substitute solver (expensive)
                try:
                    sub_plain, sub_key, sub_score = solve_substitution_simulated_annealing(curr_text, max_iters=3000, restarts=3)
                    if is_likely_plaintext(sub_plain):
                        results.append({'depth': depth, 'method_chain': chain + [f'substitution(key={sub_key})'], 'plaintext': sub_plain, 'score': sub_score})
                        _explore(sub_plain, depth+1, chain + [f'substitution(key={sub_key})'])
                except Exception:
                    pass
            else:
                # other heuristics already included
                if p and is_likely_plaintext(p):
                    results.append({'depth': depth, 'method_chain': chain + [kind], 'plaintext': p, 'score': sc})
                    _explore(p, depth+1, chain + [kind])

    _explore(s, 1, [])
    # dedupe by plaintext
    seen_plain = set()
    unique = []
    for r in sorted(results, key=lambda x: (-x['score'], x['depth'])):
        pt = r['plaintext'].strip()
        if pt in seen_plain:
            continue
        seen_plain.add(pt)
        unique.append(r)
    return unique



#Top-level detector

def detect_and_try_all(s, mode=None):
    s_stripped = s.strip()
    outputs = []

    modes_allowed = set()
    if mode is None:
        modes_allowed = {'all'}
    else:
        modes_allowed = set(mode)

    # gzip+base64
    if 'all' in modes_allowed or 'encodings' in modes_allowed:
        gz_try = try_gzip_base64(s_stripped)
        if gz_try and is_printable_ratio(gz_try):
            outputs.append(('gzip+base64', gz_try, score_english(gz_try)))

    # hex & base64
    if 'all' in modes_allowed or 'encodings' in modes_allowed:
        hex_try = try_hex(s_stripped)
        if hex_try and is_printable_ratio(hex_try):
            outputs.append(('hex', hex_try, score_english(hex_try)))
        b64_try = try_base64(s_stripped)
        if b64_try and is_printable_ratio(b64_try):
            outputs.append(('base64', b64_try, score_english(b64_try)))

    # rot13
    if 'all' in modes_allowed:
        r13 = try_rot13(s)
        if r13 != s and is_printable_ratio(r13):
            outputs.append(('rot13', r13, score_english(r13)))

    # caesar
    if 'all' in modes_allowed or 'caesar' in modes_allowed:
        caes = try_caesar(s, top=6)
        for sc, k, p in caes:
            outputs.append((f'caesar shift={k}', p, sc))

    # vigenere
    if 'all' in modes_allowed or 'vigenere' in modes_allowed:
        vigs = try_vigenere(s, max_keylen=12, top=4)
        for sc, key, p in vigs:
            outputs.append((f'vigenere key={key}', p, sc))

    # xor short
    if 'all' in modes_allowed or 'xor' in modes_allowed:
        try:
            xors = try_xor_short(s, max_key_len=2, top=5)
            for sc, keyhex, p in xors:
                outputs.append((f'xor keyhex={keyhex}', p, sc))
        except Exception:
            pass

    # substitution (try only if length is sufficiently large and contains many letters)
    if 'all' in modes_allowed or 'substitution' in modes_allowed:
        letter_count = sum(1 for c in s if c.isalpha())
        if letter_count >= 40 and len(set(c.lower() for c in s if c.isalpha())) > 8:
            try:
                sub_plain, sub_key, sub_score = solve_substitution_simulated_annealing(s, max_iters=5000, restarts=5)
                outputs.append((f'substitution key={sub_key}', sub_plain, sub_score))
            except Exception:
                pass

    outputs_sorted = sorted(outputs, key=lambda x: x[2] if x[2] is not None else -1e9, reverse=True)
    seenp = set()
    final = []
    for kind, p, sc in outputs_sorted:
        p_short = p.strip()
        if p_short in seenp:
            continue
        seenp.add(p_short)
        final.append((kind, p, sc))
        if len(final) >= 10:
            break
    return final



# CLI

def main():
    parser = argparse.ArgumentParser(description="Detect and attempt to decrypt common encodings/ciphers (CTF/learning only).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str, help='Input text to analyze (use quotes).')
    group.add_argument('--input-file', type=str, help='Path to file containing the ciphertext.')
    parser.add_argument('--mode', type=str, nargs='+', help='Restrict attacks. Options: vigenere substitution caesar xor encodings all')
    parser.add_argument('--show-all', action='store_true', help='Show all candidates instead of top ones')
    parser.add_argument('--auto', action='store_true', help='Automatically run suggested attacks and attempt to decrypt/chain them')
    args = parser.parse_args()

    load_quadgrams()

    if args.input_file:
        with open(args.input_file, 'rb') as f:
            raw = f.read()
        try:
            s = raw.decode('utf-8', errors='replace')
        except:
            s = raw.decode('latin1', errors='replace')
    else:
        s = args.text

    print("Input (first 200 chars):")
    print(s[:200])
    print("Attempting detection & decryption...")

    if args.auto:
        print('[*] Auto mode enabled: attempting suggested attacks and chaining results...')
        successes = auto_explore(s, max_depth=4, verbose=True)
        if not successes:
            print('[auto] no readable decryptions found by auto mode')
        else:
            for i, r in enumerate(successes, 1):
                print(f"[auto] Result {i} (depth {r['depth']}, score {r['score']:.2f}): chain={' -> '.join(r['method_chain'])}")
                print(r['plaintext'][:2000])
                print()
        # still run regular detection afterwards
    results = detect_and_try_all(s, mode=args.mode)
    if not results:
        print("No promising candidates found with this tool.")
        return

    for i, (kind, p, sc) in enumerate(results, 1):
        print(f"--- Candidate {i} | method: {kind} | score: {sc:.2f} ---")
        print(p[:2000])
        print()

if __name__ == '__main__':
    main()
