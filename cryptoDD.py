#!/usr/bin/env python3

import sys
import base64
import binascii
import string
import math
import random
import time
import webbrowser
import subprocess
import zlib
from collections import Counter

# --- COLORS & FORMATTING ---
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# --- DEPENDENCY HANDLING ---
def check_dependencies():
    missing = []
    try:
        from google import genai
    except ImportError:
        missing.append("google-genai")
    try:
        from Crypto.Cipher import AES
    except ImportError:
        missing.append("pycryptodome")
    return missing

def install_dependencies():
    print(f"\n{YELLOW}[*] Checking for missing dependencies...{RESET}")
    missing = check_dependencies()
    if not missing:
        print(f"{GREEN}[+] All dependencies are already installed!{RESET}")
        time.sleep(2)
        return

    print(f"{RED}[!] Missing packages: {', '.join(missing)}{RESET}")
    print(f"{YELLOW}[*] Installing via pip...{RESET}")
    cmd = [sys.executable, "-m", "pip", "install"] + missing
    try:
        subprocess.check_call(cmd)
        print(f"\n{GREEN}[+] Installation complete! Restart the script.{RESET}")
        sys.exit(0)
    except subprocess.CalledProcessError:
        print(f"\n{RED}[!] Standard install failed.{RESET}")
        print(f"{YELLOW}[*] Retrying with '--break-system-packages' (Kali Fix)...{RESET}")
        cmd.append("--break-system-packages")
        try:
            subprocess.check_call(cmd)
            print(f"\n{GREEN}[+] Installation success! Restart the script.{RESET}")
            sys.exit(0)
        except subprocess.CalledProcessError:
            print(f"\n{RED}[!] Fatal Error. Install manually:{RESET}")
            print(f"    sudo apt install python3-google-genai python3-pycryptodome")
            input("[Press Enter]")

# --- SCORING CONFIGURATION ---
ENGLISH_FREQ = {
    'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 7.31, 'N': 6.95, 'S': 6.28, 'R': 6.02,
    'H': 5.92, 'D': 4.32, 'L': 3.98, 'U': 2.88, 'C': 2.71, 'M': 2.61, 'F': 2.30, 'Y': 2.11,
    'W': 2.09, 'G': 2.03, 'P': 1.82, 'B': 1.49, 'V': 1.11, 'K': 0.69, 'X': 0.17, 'Q': 0.11,
    'J': 0.10, 'Z': 0.07
}

# Massive Update: Top English Words + CTF Jargon
COMMON_WORDS = {
    # --- BASIC ENGLISH (Top 100) ---
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it', 'for', 'not', 'on', 'with', 'he', 'as',
    'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an',
    'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who',
    'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
    'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our',
    'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most',
    'us', 'is', 'are', 'was', 'were', 'has', 'had', 'may', 'should', 'very', 'am', 'been',
    
    # --- CTF & HACKING JARGON ---
    'flag', 'key', 'password', 'secret', 'admin', 'root', 'user', 'login', 'access', 'token', 'auth', 
    'pass', 'code', 'decode', 'encode', 'encrypt', 'decrypt', 'cipher', 'text', 'string', 'value', 'hash',
    'md5', 'sha1', 'sha256', 'base64', 'hex', 'binary', 'ascii', 'byte', 'bit', 'buffer', 'overflow',
    'stack', 'heap', 'memory', 'pointer', 'address', 'shell', 'bash', 'system', 'linux', 'windows',
    'hack', 'pico', 'ctf', 'htb', 'thm', 'challenge', 'hidden', 'stego', 'image', 'file', 'data',
    'network', 'packet', 'ip', 'port', 'tcp', 'udp', 'http', 'https', 'ssh', 'ftp', 'sql', 'injection',
    'exploit', 'vuln', 'vulnerability', 'attack', 'secure', 'security', 'crypto', 'cryptography',
    'rsa', 'aes', 'des', 'xor', 'rot13', 'caesar', 'vigenere', 'substitution', 'brute', 'force',
    'crack', 'payload', 'malware', 'virus', 'trojan', 'worm', 'bot', 'botnet', 'server', 'client',
    'web', 'site', 'html', 'php', 'js', 'javascript', 'python', 'script', 'programming', 'language',
    'error', 'fail', 'success', 'true', 'false', 'null', 'undefined', 'nan', 'debug', 'test',
    'example', 'sample', 'demo', 'flag{', 'picoctf', 'hackthebox', 'tryhackme', 'easy', 'hard',
    'medium', 'score', 'points', 'rank', 'leaderboard', 'team', 'player', 'welcome', 'hello', 'world',
    'congrats', 'congratulations', 'solved', 'solution', 'writeup', 'hint', 'clue', 'level', 'stage'
}

# --- SCORING ENGINE ---
def score_text(text):
    if not text: return -1e12
    # Printable Check
    printable = sum(1 for c in text if c in string.printable)
    length = len(text)
    if length == 0: return -1e12
    if (printable / length) < 0.95: return -1e12  # Strict trash filter
    
    text_u = ''.join([c for c in text.upper() if c.isalpha()])
    if not text_u: return 0  # Neutral score for numbers/symbols

    # Frequency Analysis (Chi-Squared)
    counter = Counter(text_u)
    total = len(text_u)
    chi2 = 0.0
    for ch, exp in ENGLISH_FREQ.items():
        obs = counter.get(ch, 0)
        expected = total * (exp / 100.0)
        chi2 += (obs - expected)**2 / (expected + 0.0001)

    # Word Bonus
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    # Boost if words are found in our expanded dictionary
    word_bonus = sum(500 for w in words if w in COMMON_WORDS)
    
    # Check for "flag{" pattern specifically
    if "flag{" in text.lower() or "ctf{" in text.lower():
        word_bonus += 5000

    return word_bonus - (chi2 * 0.5)

# --- DECRYPTION HELPERS ---
def fix_padding(s):
    return s + '=' * (-len(s) % 4)

def fix_padding_b32(s):
    return s + '=' * (-len(s) % 8)

def try_bases(text):
    results = []
    s = text.strip().replace(" ", "")
    
    def add_if_valid(name, decoded_bytes):
        try:
            pt = decoded_bytes.decode('utf-8')
            if all(c in string.printable and c not in '\x0b\x0c' for c in pt):
                results.append((10000 + score_text(pt), name, pt)) 
        except: pass

    try: add_if_valid("Base64", base64.b64decode(fix_padding(s), validate=False))
    except: pass
    try: add_if_valid("Base32", base64.b32decode(fix_padding_b32(s), validate=False))
    except: pass
    try: add_if_valid("Base85", base64.b85decode(s))
    except: pass
    try: add_if_valid("Hex", binascii.unhexlify(s))
    except: pass
    try:
        if all(c in '01' for c in s) and len(s) % 8 == 0:
            byte_arr = int(s, 2).to_bytes(len(s) // 8, byteorder='big')
            add_if_valid("Binary", byte_arr)
    except: pass
    try:
        decomp = zlib.decompress(s.encode('latin1'), 16+zlib.MAX_WBITS)
        add_if_valid("Gzip (Raw)", decomp)
    except: pass
    try:
        b64_bytes = base64.b64decode(fix_padding(s))
        decomp = zlib.decompress(b64_bytes, 16+zlib.MAX_WBITS)
        add_if_valid("Gzip-Base64", decomp)
    except: pass
    return results

def solve_caesar(text):
    results = []
    for s in range(26):
        dec = ""
        for c in text:
            if c.isalpha():
                base = 65 if c.isupper() else 97
                dec += chr((ord(c) - base - s) % 26 + base)
            else: dec += c
        results.append((score_text(dec), f"Caesar (Shift {s})", dec))
    return results

def vigenere_decrypt(text, key):
    res, ki = [], 0
    key = key.upper()
    if not key: return text
    for ch in text:
        if ch.isalpha():
            base = 65 if ch.isupper() else 97
            shift = ord(key[ki % len(key)]) - 65
            res.append(chr((ord(ch) - base - shift) % 26 + base))
            ki += 1
        else: res.append(ch)
    return "".join(res)

def solve_vigenere(text, known_key=None):
    results = []
    if known_key:
         pt = vigenere_decrypt(text, known_key)
         results.append((score_text(pt), f"Vigenere (User Key: {known_key})", pt))

    keys = ["KEY", "FLAG", "PASSWORD", "SECRET", "CRYPTO", "ADMIN", "ABC", "123456", "VIGENERE"]
    for k in keys:
        pt = vigenere_decrypt(text, k)
        results.append((score_text(pt), f"Vigenere (Dict Key: {k})", pt))
    
    for klen in range(2, 8): 
        best_key = ""
        for i in range(klen):
            col = "".join([c for idx, c in enumerate(text) if idx % klen == i and c.isalpha()])
            if not col: continue
            best_s, best_sc = 0, -1e9
            for s in range(26):
                dec_col = "".join([chr((ord(c.upper()) - 65 - s) % 26 + 65) for c in col])
                sc = score_text(dec_col)
                if sc > best_sc: best_sc, best_s = sc, s
            best_key += chr(best_s + 65)
        pt = vigenere_decrypt(text, best_key)
        results.append((score_text(pt), f"Vigenere (Auto Key: {best_key})", pt))
    return results

def solve_xor(text, known_key=None):
    results = []
    try: data = text.encode('utf-8')
    except: data = text.encode('latin1')
    
    for k in range(256):
        try:
            pt = bytes([b ^ k for b in data]).decode('utf-8')
            if all(c in string.printable for c in pt):
                results.append((score_text(pt), f"XOR (Byte: {hex(k)})", pt))
        except: pass
    
    keys = [b"key", b"flag", b"123", b"password", b"secret"]
    if known_key: keys.insert(0, known_key.encode())
    
    for k in keys:
        try:
            pt_bytes = bytes([b ^ k[i % len(k)] for i, b in enumerate(data)])
            pt = pt_bytes.decode('utf-8')
            results.append((score_text(pt), f"XOR (Repeating Key: {k.decode()})", pt))
        except: pass
    return results

def solve_substitution(text, iterations=1500):
    if len(text) < 15: return []
    def decrypt(t, m): return "".join([m.get(c.upper(), c) if c.isupper() else m.get(c.upper(), c).lower() if c.islower() else c for c in t])
    alpha = list(string.ascii_uppercase)
    random.shuffle(alpha)
    best_map = dict(zip(string.ascii_uppercase, alpha))
    best_text = decrypt(text, best_map)
    best_score = score_text(best_text)
    
    parent_list = list(best_map.values())
    for _ in range(iterations):
        a, b = random.sample(range(26), 2)
        child = parent_list[:]
        child[a], child[b] = child[b], child[a]
        c_map = dict(zip(string.ascii_uppercase, child))
        c_text = decrypt(text, c_map)
        c_score = score_text(c_text)
        if c_score > best_score:
            best_score, best_text, parent_list = c_score, c_text, child
            best_map = c_map
    return [(best_score, "Substitution (Annealing)", best_text)]

def solve_aes(text, known_key=None):
    missing = check_dependencies()
    if 'pycryptodome' in missing: return []
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad
    
    results = []
    try: raw = base64.b64decode(fix_padding(text.strip()))
    except: 
        try: raw = binascii.unhexlify(text.strip())
        except: return []

    keys = ["password12345678", "secretkey1234567", "0000000000000000", "aesencryptionkey", "1234567890123456"]
    if known_key: keys.insert(0, known_key)

    for k in keys:
        k_bytes = k.encode('utf-8')
        if len(k_bytes) not in [16, 24, 32]:
            k_bytes = k_bytes.ljust(32, b'\0')[:32]
        try:
            cipher = AES.new(k_bytes, AES.MODE_ECB)
            pt = unpad(cipher.decrypt(raw), 16).decode('utf-8')
            results.append((10000 + score_text(pt), f"AES-ECB (Key: {k})", pt))
        except: pass
        try:
            iv = b'\x00' * 16
            cipher = AES.new(k_bytes, AES.MODE_CBC, iv)
            pt = unpad(cipher.decrypt(raw), 16).decode('utf-8')
            results.append((10000 + score_text(pt), f"AES-CBC (Key: {k})", pt))
        except: pass
    return results

# --- INTERFACES ---
def run_simple_analyze():
    while True:
        print(f"\n{GREEN}" + "="*40)
        print(" [1] SIMPLE ANALYZE (Multi-Decoder)")
        print("="*40 + f"{RESET}")
        text = input(f"Enter Ciphertext (or type {BOLD}back{RESET} to return): ").strip()
        if text.lower() == 'back': break
        if not text: continue
        key_hint = input(f"Enter Key/Hint (AES/Vig/XOR) [Press Enter to skip]: ").strip()
        
        print(f"\n{GREEN}[*] Analyzing...{RESET}")
        candidates = []
        candidates.extend(try_bases(text)) 
        candidates.extend(solve_caesar(text))
        candidates.extend(solve_vigenere(text, known_key=key_hint))
        candidates.extend(solve_xor(text, known_key=key_hint))
        candidates.extend(solve_aes(text, known_key=key_hint))
        candidates.extend(solve_substitution(text))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        print(f"\n{GREEN}--- TOP CANDIDATES ---{RESET}")
        count = 0
        for score, method, pt in candidates:
            if count >= 5: break
            if score < -100: continue 
            res_color = GREEN if score > 500 else CYAN 
            print(f"[{count+1}] Method: {method} | Score: {score:.1f}")
            print(f"    Result: {res_color}{pt[:100]}...{RESET}")
            count += 1
            
        if count == 0:
            print(f"{RED}[-] No readable text found.{RESET}")
        input(f"\n[Press Enter]")

def run_ai_analyze():
    missing = check_dependencies()
    if 'google-genai' in missing:
        print(f"\n{RED}[!] Error: 'google-genai' missing.{RESET}")
        print(f"{YELLOW}[*] Go to Option 0 to install.{RESET}")
        time.sleep(3)
        return
    from google import genai
    print(f"\n{GREEN}" + "="*40)
    print(" [2] NEURAL CIPHER AI (Gemini)")
    print("="*40 + f"{RESET}")
    api_key = input(f"Enter Gemini API Key (or type {BOLD}back{RESET}): ").strip()
    if api_key.lower() == 'back': return
    print(f"{GREEN}[*] Verifying...{RESET}")
    try:
        client = genai.Client(api_key=api_key)
        client.models.generate_content(model="gemini-1.5-flash", contents="ping")
        print(f"{GREEN}[+] Verified!{RESET}")
    except Exception as e:
        print(f"{RED}[!] Error: {e}{RESET}")
        input("[Press Enter]")
        return
    while True:
        text = input(f"\nEnter Ciphertext (or {BOLD}back{RESET}): ").strip()
        if text.lower() == 'back': break
        try:
            print(f"{GREEN}[*] Thinking...{RESET}")
            resp = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"Identify cipher and decrypt: {text}"
            )
            print(f"\n{GREEN}--- AI RESPONSE ---{RESET}")
            print(resp.text)
        except Exception as e:
            print(f"{RED}[!] Error: {e}{RESET}")

def main_menu():
    while True:
        print("\033[H\033[J", end="") 
        print(f"{GREEN}========================================")
        print("     ADVANCED CRYPTO ANALYZER v7.0")
        print("========================================")
        print("0. Install Dependencies (Kali/Debian Fix)")
        print("1. Simple Analyze (Base64/AES/Classic)")
        print("2. Neural Cipher AI (Gemini API)")
        print("3. CLAASP (Browser)")
        print(f"4. Exit{RESET}")
        print(f"{GREEN}========================================{RESET}")
        c = input("\nOption [0-4]: ").strip()
        if c=='0': install_dependencies()
        elif c=='1': run_simple_analyze()
        elif c=='2': run_ai_analyze()
        elif c=='3': webbrowser.open("https://app.claasp-tii.com/"); input("[Enter]")
        elif c=='4': sys.exit(0)

if __name__ == "__main__":
    main_menu()
