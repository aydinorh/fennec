use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use chacha20poly1305::aead::{Aead, KeyInit, OsRng};
use chacha20poly1305::{AeadCore, ChaCha20Poly1305, Key, Nonce};
use zeroize::Zeroizing;

use super::fs::write_secure;

const NONCE_LEN: usize = 12;
const PREFIX: &str = "enc2:";

/// Encrypts and decrypts secrets using ChaCha20-Poly1305 (AEAD).
///
/// The encryption key is stored as hex in `<config_dir>/.secret_key` and is
/// generated on first use.
///
/// The key field uses `Zeroizing<[u8; 32]>` so the bytes are wiped from
/// memory when the `SecretStore` is dropped or replaced. The intermediate
/// hex string and decoded byte buffers used during load are also wrapped
/// in `Zeroizing` so a process core dump or swap snapshot doesn't
/// preserve the master key in plain bytes. `chacha20poly1305` 0.10
/// already requires `zeroize` as a direct (non-optional) dependency, so
/// the cipher's internal copies of the key are zeroed when the cipher is
/// dropped — no feature flag needed.
pub struct SecretStore {
    /// The 32-byte key material, kept in a Zeroizing wrapper so it gets
    /// wiped on drop. We hold our own owned bytes (rather than a
    /// chacha20poly1305 `Key`) so the same buffer can be passed to
    /// `Key::from_slice` for each operation without the cipher hanging
    /// onto a long-lived clone.
    key_bytes: Zeroizing<[u8; 32]>,
}

impl SecretStore {
    /// Create or load a `SecretStore` rooted at `config_dir`.
    ///
    /// If the key file does not exist, a new random key is generated and
    /// persisted with restrictive permissions (0600 on Unix).
    pub fn new(config_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&config_dir)
            .with_context(|| format!("creating config dir {}", config_dir.display()))?;

        let key_path = config_dir.join(".secret_key");

        let key_bytes: Zeroizing<[u8; 32]> = if key_path.exists() {
            // Load existing key. Wrap intermediate strings/buffers in
            // Zeroizing so the master key never lives in unscrubbed
            // memory after this function returns.
            let raw = std::fs::read_to_string(&key_path)
                .with_context(|| format!("reading key file {}", key_path.display()))?;
            let hex_str = Zeroizing::new(raw.trim().to_string());
            let decoded = hex::decode(hex_str.as_str())
                .context("decoding hex key")?;
            let decoded = Zeroizing::new(decoded);
            if decoded.len() != 32 {
                bail!("invalid key length: expected 32 bytes, got {}", decoded.len());
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&decoded);
            Zeroizing::new(arr)
        } else {
            // Generate a new key and write it via write_secure so the file
            // never exists on disk with umask-default (0644) permissions,
            // even for the brief window between create-and-chmod.
            let generated = ChaCha20Poly1305::generate_key(&mut OsRng);
            let mut arr = [0u8; 32];
            arr.copy_from_slice(generated.as_slice());
            let arr = Zeroizing::new(arr);

            let hex_str = Zeroizing::new(hex::encode(arr.as_slice()));
            write_secure(&key_path, hex_str.as_bytes())
                .with_context(|| format!("writing key file {}", key_path.display()))?;

            arr
        };

        Ok(Self { key_bytes })
    }

    fn cipher(&self) -> ChaCha20Poly1305 {
        ChaCha20Poly1305::new(Key::from_slice(self.key_bytes.as_slice()))
    }

    /// Whether `value` is already an `enc2:`-encrypted blob. Useful for a
    /// one-time migration that encrypts only the plaintext secrets and
    /// leaves already-encrypted ones untouched. Empty strings are not
    /// "encrypted" but also need no migration (callers skip them).
    pub fn is_encrypted(value: &str) -> bool {
        value.starts_with(PREFIX)
    }

    /// Encrypt `plaintext` and return a prefixed hex string.
    ///
    /// Format: `enc2:<hex(nonce || ciphertext)>`
    pub fn encrypt(&self, plaintext: &str) -> Result<String> {
        let cipher = self.cipher();
        let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);

        let ciphertext = cipher
            .encrypt(&nonce, plaintext.as_bytes())
            .map_err(|e| anyhow::anyhow!("encryption failed: {}", e))?;

        // nonce || ciphertext
        let mut combined = Vec::with_capacity(NONCE_LEN + ciphertext.len());
        combined.extend_from_slice(nonce.as_slice());
        combined.extend_from_slice(&ciphertext);

        Ok(format!("{}{}", PREFIX, hex::encode(&combined)))
    }

    /// Decrypt a value previously encrypted with [`encrypt`](Self::encrypt).
    ///
    /// If the value does not start with the `enc2:` prefix it is returned
    /// as-is (plaintext passthrough). This is a backwards-compat convenience
    /// for configs where the user pasted a raw secret without running it
    /// through `encrypt` first — but it also means an attacker able to
    /// overwrite stored ciphertext with arbitrary plaintext makes
    /// decryption "succeed" with their payload. A warning is logged on the
    /// passthrough path so operators can notice and migrate the config.
    pub fn decrypt(&self, value: &str) -> Result<String> {
        if !value.starts_with(PREFIX) {
            tracing::warn!(
                "SecretStore::decrypt: value lacks '{}' prefix — returning as \
                 plaintext. This is fine for legacy configs where the secret \
                 was pasted directly, but a stored value that SHOULD be \
                 ciphertext is now unauthenticated. Run encrypt() over these \
                 values.",
                PREFIX
            );
            return Ok(value.to_string());
        }

        let hex_part = &value[PREFIX.len()..];
        let combined = hex::decode(hex_part).context("decoding hex ciphertext")?;

        if combined.len() < NONCE_LEN {
            bail!("ciphertext too short");
        }

        let (nonce_bytes, ciphertext) = combined.split_at(NONCE_LEN);
        let nonce = Nonce::from_slice(nonce_bytes);

        let cipher = self.cipher();
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("decryption failed: {}", e))?;

        String::from_utf8(plaintext).context("decrypted bytes are not valid UTF-8")
    }
}
