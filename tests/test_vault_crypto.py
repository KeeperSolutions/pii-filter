"""Unit tests for the Task 11 vault encryption primitives.

Covers ``VaultCipher`` (AES-256-GCM ENC1 envelope), ``BlindIndex`` (keyed
HMAC dedup token), ``KeyManager`` (key loading / fail-closed validation), the
``Pipeline._build_vault_crypto`` startup wiring (spec §6), and
``ThreadVault._decrypt_stored_value`` (the never-raise read-path fallback,
spec D3 / §7.2 / §7.3).

These are pure in-memory tests — no Postgres process and no spaCy model are
required, so they run on every host regardless of ``pg_ctl`` availability.
"""

from __future__ import annotations

import base64

import pytest
from cryptography.exceptions import InvalidTag

from pii_filter import BlindIndex, KeyManager, Pipeline, ThreadVault, VaultCipher

# Two independent, deterministic 32-byte test keys (E2: enc key != blind key).
_KEY_A = bytes(range(32))  # 0x00..0x1f
_KEY_B = bytes(range(32, 64))  # 0x20..0x3f
_KEY_A_B64 = base64.b64encode(_KEY_A).decode("ascii")
_KEY_B_B64 = base64.b64encode(_KEY_B).decode("ascii")

_ENC1 = "ENC1:"


def _envelope_bytes(envelope: str) -> bytes:
    """Decode the raw packed bytes inside an ENC1 envelope string."""
    return base64.b64decode(envelope[len(_ENC1) :], validate=True)


# ---------------------------------------------------------------------------
# VaultCipher
# ---------------------------------------------------------------------------


def test_vault_cipher_round_trip() -> None:
    """encrypt → decrypt returns the original plaintext; output is an ENC1 blob."""
    cipher = VaultCipher(_KEY_A, key_id=1)
    plaintext = "Ivan Horvat — OIB 12345678903"
    envelope = cipher.encrypt(plaintext)

    assert VaultCipher.is_encrypted(envelope)
    assert envelope.startswith(_ENC1)
    assert cipher.decrypt(envelope) == plaintext


def test_vault_cipher_envelope_parses_version_keyid_nonce_tag() -> None:
    """The envelope packs version=1, the configured key_id, a 12-byte nonce,
    and a 16-byte GCM tag; the nonce is fresh per call."""
    cipher = VaultCipher(_KEY_A, key_id=0x01020304)
    e1 = cipher.encrypt("hello")
    e2 = cipher.encrypt("hello")

    # Random nonce → two encryptions of the same plaintext differ.
    assert e1 != e2

    raw = _envelope_bytes(e1)
    assert raw[0] == 1  # version
    assert int.from_bytes(raw[1:5], "big") == 0x01020304  # key_id round-trips
    # 1B version + 4B key_id + 12B nonce + (>=0B ct) + 16B tag.
    assert len(raw) >= 1 + 4 + 12 + 16

    # Both still decrypt despite differing nonces.
    assert cipher.decrypt(e1) == "hello"
    assert cipher.decrypt(e2) == "hello"


def test_vault_cipher_is_encrypted_false_for_plaintext() -> None:
    assert VaultCipher.is_encrypted("Ivan Horvat") is False
    assert VaultCipher.is_encrypted("") is False


def test_vault_cipher_wrong_key_raises_invalid_tag() -> None:
    envelope = VaultCipher(_KEY_A).encrypt("secret")
    with pytest.raises(InvalidTag):
        VaultCipher(_KEY_B).decrypt(envelope)


def test_vault_cipher_tampered_ciphertext_raises_invalid_tag() -> None:
    cipher = VaultCipher(_KEY_A)
    envelope = cipher.encrypt("secret")
    raw = bytearray(_envelope_bytes(envelope))
    raw[-1] ^= 0x01  # flip a tag bit
    tampered = _ENC1 + base64.b64encode(bytes(raw)).decode("ascii")
    with pytest.raises(InvalidTag):
        cipher.decrypt(tampered)


def test_vault_cipher_rejects_bad_key_length() -> None:
    with pytest.raises(ValueError):
        VaultCipher(b"\x00" * 31)


def test_vault_cipher_decrypt_rejects_non_envelope() -> None:
    with pytest.raises(ValueError):
        VaultCipher(_KEY_A).decrypt("not an envelope")


# ---------------------------------------------------------------------------
# BlindIndex
# ---------------------------------------------------------------------------


def test_blind_index_deterministic_32_bytes() -> None:
    blind = BlindIndex(_KEY_B)
    h1 = blind.compute("chatA", "PERSON", "Ivan Horvat")
    h2 = blind.compute("chatA", "PERSON", "Ivan Horvat")
    assert h1 == h2
    assert isinstance(h1, bytes)
    assert len(h1) == 32


def test_blind_index_cross_thread_isolation() -> None:
    """Same PII in chatA vs chatB hashes differently (chat_id is framed in)."""
    blind = BlindIndex(_KEY_B)
    ha = blind.compute("chatA", "PERSON", "Ivan Horvat")
    hb = blind.compute("chatB", "PERSON", "Ivan Horvat")
    assert ha != hb


def test_blind_index_framing_collision_resistant() -> None:
    """('a','bc',v) vs ('ab','c',v) must not collide (length-prefixed framing)."""
    blind = BlindIndex(_KEY_B)
    h1 = blind.compute("a", "bc", "value")
    h2 = blind.compute("ab", "c", "value")
    assert h1 != h2


def test_blind_index_distinct_key_distinct_hash() -> None:
    ha = BlindIndex(_KEY_A).compute("chatA", "PERSON", "Ivan")
    hb = BlindIndex(_KEY_B).compute("chatA", "PERSON", "Ivan")
    assert ha != hb


def test_blind_index_rejects_bad_key_length() -> None:
    with pytest.raises(ValueError):
        BlindIndex(b"\x00" * 16)


# ---------------------------------------------------------------------------
# KeyManager (local backend)
# ---------------------------------------------------------------------------


def test_key_manager_local_loads_keys() -> None:
    manager = KeyManager(
        backend="local",
        encryption_key_b64=_KEY_A_B64,
        blind_index_key_b64=_KEY_B_B64,
    )
    assert manager.load_encryption_key() == _KEY_A
    assert manager.load_blind_index_key() == _KEY_B


def test_key_manager_empty_key_raises() -> None:
    manager = KeyManager(backend="local", blind_index_key_b64="")
    with pytest.raises(RuntimeError):
        manager.load_blind_index_key()


def test_key_manager_short_key_raises() -> None:
    short = base64.b64encode(b"\x00" * 16).decode("ascii")
    manager = KeyManager(backend="local", blind_index_key_b64=short)
    with pytest.raises(RuntimeError):
        manager.load_blind_index_key()


def test_key_manager_bad_base64_raises() -> None:
    manager = KeyManager(backend="local", blind_index_key_b64="!!! not base64 !!!")
    with pytest.raises(RuntimeError):
        manager.load_blind_index_key()


def test_key_manager_unknown_backend_raises() -> None:
    manager = KeyManager(backend="redis_kms", blind_index_key_b64=_KEY_B_B64)
    with pytest.raises(RuntimeError):
        manager.load_blind_index_key()


def test_key_manager_gcp_empty_secret_names_real_valve_field() -> None:
    """The fail-closed message for an empty gcp secret names the actual valve
    field (`vault_gcp_enc_secret` / `vault_gcp_blind_secret`), not a
    `vault_gcp_<label>_secret` placeholder — so an operator is pointed at the
    right env var. The empty-name guard runs before the lazy google.cloud
    import, so no package is required to exercise it.
    """
    manager = KeyManager(backend="gcp_kms", gcp_enc_secret="", gcp_blind_secret="")
    with pytest.raises(RuntimeError, match="vault_gcp_enc_secret"):
        manager.load_encryption_key()
    with pytest.raises(RuntimeError, match="vault_gcp_blind_secret"):
        manager.load_blind_index_key()


# ---------------------------------------------------------------------------
# Pipeline._build_vault_crypto — on_startup fail-closed validation (§6)
# ---------------------------------------------------------------------------


def test_build_vault_crypto_blind_key_always_required(pipeline: Pipeline) -> None:
    """Even with encryption disabled, the blind-index key is required (D1)."""
    pipeline.valves.vault_encryption_enabled = False
    pipeline.valves.vault_blind_index_key = ""
    with pytest.raises(RuntimeError):
        pipeline._build_vault_crypto()


def test_build_vault_crypto_enc_key_required_when_enabled(pipeline: Pipeline) -> None:
    pipeline.valves.vault_encryption_enabled = True
    pipeline.valves.vault_blind_index_key = _KEY_B_B64
    pipeline.valves.vault_encryption_key = ""  # missing
    with pytest.raises(RuntimeError):
        pipeline._build_vault_crypto()


def test_build_vault_crypto_short_enc_key_raises(pipeline: Pipeline) -> None:
    pipeline.valves.vault_encryption_enabled = True
    pipeline.valves.vault_blind_index_key = _KEY_B_B64
    pipeline.valves.vault_encryption_key = base64.b64encode(b"\x00" * 8).decode("ascii")
    with pytest.raises(RuntimeError):
        pipeline._build_vault_crypto()


def test_build_vault_crypto_success_encryption_on(pipeline: Pipeline) -> None:
    pipeline.valves.vault_encryption_enabled = True
    pipeline.valves.vault_encryption_strict = True
    pipeline.valves.vault_blind_index_key = _KEY_B_B64
    pipeline.valves.vault_encryption_key = _KEY_A_B64
    cipher, blind_index = pipeline._build_vault_crypto()
    assert cipher is not None
    assert isinstance(blind_index, BlindIndex)
    assert cipher.decrypt(cipher.encrypt("x")) == "x"


def test_build_vault_crypto_success_encryption_off(pipeline: Pipeline) -> None:
    pipeline.valves.vault_encryption_enabled = False
    pipeline.valves.vault_blind_index_key = _KEY_B_B64
    cipher, blind_index = pipeline._build_vault_crypto()
    assert cipher is None
    assert isinstance(blind_index, BlindIndex)


# ---------------------------------------------------------------------------
# ThreadVault._decrypt_stored_value — never-raise read-path fallback
# (spec D3 / §7.2 / §7.3). Constructed without a pool — no Postgres needed.
# ---------------------------------------------------------------------------


def _vault(*, encryption: bool, strict: bool) -> ThreadVault:
    cipher = VaultCipher(_KEY_A) if encryption else None
    return ThreadVault(
        dsn="postgresql://unused@127.0.0.1:1/none",
        cipher=cipher,
        blind_index=BlindIndex(_KEY_B),
        encryption_strict=strict,
    )


def test_decrypt_stored_value_encrypted_round_trip() -> None:
    vault = _vault(encryption=True, strict=False)
    stored = VaultCipher(_KEY_A).encrypt("Ivan Horvat")
    assert vault._decrypt_stored_value(stored, "chatA", "[PERSON_1]") == "Ivan Horvat"


def test_decrypt_stored_value_wrong_key_returns_none() -> None:
    """A row encrypted under a different key fails the GCM tag → skipped (None),
    not raised — preserves the never-crash outlet contract."""
    vault = _vault(encryption=True, strict=False)
    foreign = VaultCipher(_KEY_B).encrypt("Ivan Horvat")  # encrypted with the wrong key
    assert vault._decrypt_stored_value(foreign, "chatA", "[PERSON_1]") is None


def test_decrypt_stored_value_tampered_returns_none() -> None:
    vault = _vault(encryption=True, strict=False)
    raw = bytearray(_envelope_bytes(VaultCipher(_KEY_A).encrypt("Ivan Horvat")))
    raw[-1] ^= 0x01
    tampered = _ENC1 + base64.b64encode(bytes(raw)).decode("ascii")
    assert vault._decrypt_stored_value(tampered, "chatA", "[PERSON_1]") is None


def test_decrypt_stored_value_malformed_envelope_returns_none() -> None:
    """A row carrying the ENC1 prefix but a structurally broken body is skipped
    (None), never raised. ``base64.b64decode(validate=True)`` raises
    ``binascii.Error`` which is a ``ValueError`` subclass, so it is caught by
    the read path's ``(InvalidTag, ValueError)`` handler — this locks that in
    (regression guard for a code-review false positive).
    """
    vault = _vault(encryption=True, strict=False)
    # Body is not valid base64 at all → binascii.Error (a ValueError subclass).
    assert vault._decrypt_stored_value("ENC1:@@@not-base64@@@", "chatA", "[PERSON_1]") is None
    # Valid base64 but too short to hold version + key_id + nonce + tag.
    too_short = _ENC1 + base64.b64encode(b"short").decode("ascii")
    assert vault._decrypt_stored_value(too_short, "chatA", "[PERSON_1]") is None


def test_decrypt_stored_value_plaintext_non_strict_returns_asis() -> None:
    """Encryption enabled but a legacy plaintext row exists; non-strict serves it."""
    vault = _vault(encryption=True, strict=False)
    assert vault._decrypt_stored_value("Ivan Horvat", "chatA", "[PERSON_1]") == "Ivan Horvat"


def test_decrypt_stored_value_plaintext_strict_returns_none() -> None:
    """Strict mode refuses an unexpected plaintext row → skipped (None)."""
    vault = _vault(encryption=True, strict=True)
    assert vault._decrypt_stored_value("Ivan Horvat", "chatA", "[PERSON_1]") is None


def test_decrypt_stored_value_plaintext_no_cipher_returns_asis() -> None:
    """Encryption disabled → plaintext is expected and returned as-is even if
    strict is (nonsensically) set, because plaintext is not 'unexpected' here."""
    vault = _vault(encryption=False, strict=True)
    assert vault._decrypt_stored_value("Ivan Horvat", "chatA", "[PERSON_1]") == "Ivan Horvat"
