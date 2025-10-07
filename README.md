Crypto Detect & Decrypt (cryptoDD) is a smart, automated tool designed to help cybersecurity enthusiasts, CTF players, and researchers analyze and decrypt ciphertexts safely and efficiently.

🚀 Overview
cryptoDD is an intelligent ciphertext detector and auto-decryptor built for:
- 🕵️‍♂️ CTF players
- 🧑‍💻 Cybersecurity learners
- 🛡️ Ethical hackers
It identifies and handles a wide range of encoding and encryption schemes, including:
- Base64 / Base32 / Base85
- Hex / Binary / Gzip-Base64
- Caesar / Substitution / Vigenère
- XOR (single-byte and repeating key)
- AES (with known key or bruteforce pattern)
- More formats coming soon...

✨ Key Features
- 🔍 Intelligent cipher detection using entropy, Index of Coincidence (IC), and frequency analysis
- 🔄 Auto-decrypt mode that recursively explores multi-layered encryption
- 🧠 Classical cipher solvers: Caesar, ROT13, Vigenère, Substitution, XOR
- 🛡️ Safety-first design: works only on data you are authorized to analyze
- 🐳 Docker-ready for reproducible deployments and isolated environments
- 📊 Plaintext scoring using English heuristics and quadgram statistics
- 🧩 CLI options to run specific attacks or full auto-detection suite

🎯 Use Cases
- 🏁 Capture-the-Flag (CTF) competitions
- 📚 Educational tool for learning classical cryptography
- 🔐 Ethical penetration testing on authorized datasets

⚙️ Installation
Clone the repository
git clone https://github.com/Cyber-Dipesh/Cipher-Sense-Detect-n-Decrypt-.git
cd cryptoDD


Install dependencies
pip install -r requirements.txt


Make executable (Linux)
chmod +x cryptoDD.py



🐳 Docker Setup
Build the image
docker build -t crypto-dd .


Run the container
docker run --rm crypto-dd --text "aGVsbG8gd29ybGQ=" --auto



🧩 Usage Examples
Detect any ciphertext
./cryptoDD.py --text "aGVsbG8gd29ybGQ="


Run a specific attack
./cryptoDD.py --text "GIEWIVrGMTLIVrHIQS" --only-vigenere


Try all modes automatically
./cryptoDD.py --file samples/sample_xor.txt --auto

🧠 Supported Modes
Cipher Type	          Detection	Auto Decrypt
Base64 / Base32	            ✅	✅
Gzip+Base64	                ✅	✅
Caesar / ROT	              ✅	✅
Substitution (AI-tuned)	    ✅	✅
Vigenère	                  ✅	✅
Single-byte XOR	            ✅	✅
AES (Known key)	            ⚙️	⚙️
Other (binary/hex)	        ✅	✅



📦 requirements.txt
numpy
pytest
tqdm




⚠️ Limitations of Crypto Detect & Decrypt (cryptoDD)

1.Modern strong encryption not breakable
  The tool cannot brute-force AES, RSA, or other modern strong ciphers without the key. It only detects patterns or weak usage (e.g., ECB repeated blocks) and classical crypto.

2.Heuristic-based detection
  Detection and “auto mode” rely on heuristics like entropy, Index of Coincidence, or quadgram frequency. This can lead to false positives or missed detections in some ciphertexts.

3.Limited keyspace attacks
  Substitution solver, Vigenère attacks, and XOR crackers work for short keys or reasonable key lengths. Very long keys may be impractical due to computational limits.

4.Performance considerations
  Auto mode that chains multiple decodings and attacks can be computationally expensive, especially on large files. Defaults are conservative, but deep recursive analysis may take time.

5.Partial readability only
  Output plaintexts are filtered based on heuristics (printable ratio, English quadgram score). Non-English or binary plaintext may not be recognized as valid.

6.File size / binary data limitations
  The tool is designed primarily for text-based ciphertexts or small encrypted files. Large binary blobs may be skipped unless forced.

7.Security disclaimer
  This tool is intended for educational purposes, CTFs, and authorized security testing only. Using it against unauthorized data is illegal and outside the scope of this project.

8.Limited scope for multi-layered modern encryption
  While chaining simple encodings works (e.g., Base64 → Gzip → Vigenère), complex layered modern encryptions (AES + RSA + compression + custom obfuscation) may not be fully solvable.



⚖️ License
This project is licensed under the MIT License.


👤 Author
Dipesh Patel
GitHub: Cyber-Dipesh

⚠️ Disclaimer
This tool is intended for educational and authorized use only.
Do not use it to decrypt or access data you do not own or have explicit permission to test.
