"""
Encrypted Storage — AES-256-GCM encryption for data at rest.

All sensitive artifacts (video clips, entity crops, CSVs) can be
encrypted before storage. Keys derived from PRIVACY_ENCRYPTION_KEY
environment variable via PBKDF2.

Edge-level: no cloud key management. For production, replace with
TPM/HSM-backed key storage.
"""

import hashlib
import logging
import os
import struct
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_NONCE_SIZE = 12       # 96-bit nonce for GCM
_TAG_SIZE = 16         # 128-bit auth tag
_SALT_SIZE = 16        # 128-bit salt for PBKDF2
_KEY_SIZE = 32         # 256-bit key
_MAGIC = b"ENC1"       # File magic bytes for identification
_HEADER_SIZE = len(_MAGIC) + _SALT_SIZE + _NONCE_SIZE  # 32 bytes


def _get_encryption_key() -> bytes | None:
    """
    Retrieve the encryption key from environment.
    Returns None if not configured.
    """
    key_str = os.environ.get("PRIVACY_ENCRYPTION_KEY")
    if not key_str:
        return None
    return key_str.encode("utf-8")


def _derive_key(password: bytes, salt: bytes) -> bytes:
    """Derive a 256-bit key from password + salt using PBKDF2-HMAC-SHA256."""
    return hashlib.pbkdf2_hmac(
        "sha256", password, salt, iterations=100_000, dklen=_KEY_SIZE
    )


def is_encryption_available() -> bool:
    """Check if encryption is enabled and a key is configured."""
    if not settings.encryption.enabled:
        return False
    key = _get_encryption_key()
    if key is None:
        logger.warning("Encryption enabled but PRIVACY_ENCRYPTION_KEY not set")
        return False
    return True


def encrypt_bytes(data: bytes) -> bytes:
    """
    Encrypt bytes using AES-256-GCM.
    Returns: MAGIC + salt + nonce + ciphertext + tag

    Uses the `cryptography` library for AEAD encryption.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        logger.error("cryptography package not installed — cannot encrypt")
        raise RuntimeError("cryptography package required for encryption")

    master_key = _get_encryption_key()
    if master_key is None:
        raise RuntimeError("PRIVACY_ENCRYPTION_KEY environment variable not set")

    salt = os.urandom(_SALT_SIZE)
    derived_key = _derive_key(master_key, salt)
    nonce = os.urandom(_NONCE_SIZE)

    aesgcm = AESGCM(derived_key)
    ciphertext = aesgcm.encrypt(nonce, data, None)  # ciphertext + tag appended

    return _MAGIC + salt + nonce + ciphertext


def decrypt_bytes(encrypted_data: bytes) -> bytes:
    """
    Decrypt AES-256-GCM encrypted bytes.
    Expects format: MAGIC + salt + nonce + ciphertext + tag
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise RuntimeError("cryptography package required for decryption")

    if not encrypted_data.startswith(_MAGIC):
        raise ValueError("Invalid encrypted data: missing magic bytes")

    offset = len(_MAGIC)
    salt = encrypted_data[offset : offset + _SALT_SIZE]
    offset += _SALT_SIZE
    nonce = encrypted_data[offset : offset + _NONCE_SIZE]
    offset += _NONCE_SIZE
    ciphertext_with_tag = encrypted_data[offset:]

    master_key = _get_encryption_key()
    if master_key is None:
        raise RuntimeError("PRIVACY_ENCRYPTION_KEY environment variable not set")

    derived_key = _derive_key(master_key, salt)
    aesgcm = AESGCM(derived_key)

    return aesgcm.decrypt(nonce, ciphertext_with_tag, None)


def encrypt_file(file_path: str | Path) -> Path:
    """
    Encrypt a file at rest. Writes an `.enc` copy and removes the original.
    Returns the path to the encrypted file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = path.read_bytes()
    encrypted = encrypt_bytes(data)

    enc_path = path.with_suffix(path.suffix + ".enc")
    enc_path.write_bytes(encrypted)

    # Remove original plaintext file
    path.unlink()

    logger.info("Encrypted %s → %s (%d bytes)", path.name, enc_path.name, len(encrypted))
    return enc_path


def decrypt_file(encrypted_path: str | Path) -> bytes:
    """
    Decrypt an encrypted file and return its contents as bytes.
    Does NOT write to disk — caller decides what to do with plaintext.
    """
    path = Path(encrypted_path)
    if not path.exists():
        raise FileNotFoundError(f"Encrypted file not found: {path}")

    encrypted_data = path.read_bytes()
    plaintext = decrypt_bytes(encrypted_data)

    logger.info("Decrypted %s (%d bytes plaintext)", path.name, len(plaintext))
    return plaintext


def compute_checksum(data: bytes) -> str:
    """Compute SHA-256 checksum for integrity verification."""
    return hashlib.sha256(data).hexdigest()
