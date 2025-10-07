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
git clone https://github.com/Cyber-Dipesh/cryptoDD.git
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



📦 requirements.txt
numpy
pytest
tqdm



🐳 Dockerfile
FROM python:3.10-slim

LABEL maintainer="Dipesh Patel <your.email@example.com>"
LABEL description="Crypto Detect & Decrypt - CTF helper tool"

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "cryptoDD.py"]
CMD ["--help"]



⚖️ License
This project is licensed under the MIT License.

👤 Author
Dipesh Patel
GitHub: Cyber-Dipesh

⚠️ Disclaimer
This tool is intended for educational and authorized use only.
Do not use it to decrypt or access data you do not own or have explicit permission to test.
