import hashlib
from cryptography.fernet import Fernet
import os
from pathlib import Path

class SecurityManager:
    def __init__(self, key_path="secret.key"):
        self.key_path = Path(key_path)
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)

    def _load_or_generate_key(self):
        if self.key_path.exists():
            with open(self.key_path, "rb") as key_file:
                return key_file.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_path, "wb") as key_file:
                key_file.write(key)
            return key

    def generate_file_hash(self, file_path):
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except FileNotFoundError:
            return None

    def encrypt_data(self, data):
        if isinstance(data, str):
            data = data.encode()
        return self.cipher.encrypt(data)

    def decrypt_data(self, encrypted_data):
        return self.cipher.decrypt(encrypted_data).decode()

    def encrypt_file(self, input_path, output_path):
        with open(input_path, "rb") as f:
            data = f.read()
        encrypted = self.cipher.encrypt(data)
        with open(output_path, "wb") as f:
            f.write(encrypted)

    def decrypt_file(self, input_path, output_path):
        with open(input_path, "rb") as f:
            encrypted = f.read()
        decrypted = self.cipher.decrypt(encrypted)
        with open(output_path, "wb") as f:
            f.write(decrypted)
