# Encryption at rest — design and operations

Status: shipped as an opt-in capability, default off. This document describes
what is implemented in `src/keystore.rs`, `src/memory/storage.rs` and
`src/bin/shodh_keyctl.rs`; the deferred parts are listed at the end so their
absence is a stated decision rather than an omission.

## 1. Threat model

- **T1 — at-rest / cold-disk theft.** The attacker has the RocksDB files (and
  `keystore.json`), but not the passphrase. Covered for the primary `Memory`
  record. Not covered for the secondary index, facts, graph, vector index and
  sibling column families (§6).
- **T2 — a runtime observer of process I/O and access patterns.** Not
  addressed. Content is encrypted; which records are touched, and when, leaks.
- **T3 — a hosted multi-tenant service hiding queries from the server.** Out
  of scope.

The unseal secret must live somewhere the T1 attacker does not also get. A
passphrase in an env file on the data disk gives them both.

## 2. Key hierarchy (envelope)

```
passphrase ──Argon2id(salt,m,t,p)──▶ unseal key ──┐
recovery code ──SHA-256──▶ recovery key ───────────┼─▶ AES-256-GCM unwrap ─▶ master key (KEK)
                                                                                   │
                                                                     wraps ▶ DEK_epoch_N
                                                                                   │
                                                        XChaCha20-Poly1305 record encryption
```

- The **KEK** is never stored raw: only wrapped, once per enabled unseal
  provider (`kek_wraps`: `passphrase`, optionally `recovery`). Any one wrap
  unseals — so the KEK is as strong as the weakest enabled provider.
- **DEKs** are per epoch, wrapped by the KEK with the epoch bound in as AAD so
  a DEK cannot be substituted across epochs. The active epoch encrypts new
  writes; retired epochs stay in the keystore so their records stay readable.
- Every wrap is AES-256-GCM with a domain-separating AAD.

## 3. Keystore file

`<data-dir>/storage/keystore.json`, JSON, holding no plaintext key material:

| field | purpose |
|---|---|
| `crypto_version`, `schema_version` | format gates |
| `kdf` | Argon2id `m_cost` (KiB), `t_cost`, `p_cost`, base64 salt |
| `kek_wraps[]` | `{provider, nonce, ciphertext}` per unseal provider |
| `deks[]` | `{epoch, wrapped, state: active\|retired}` |
| `active_epoch` | epoch new writes use |
| `kek_fingerprint` | `SHA-256(KEK)[..4]`, base64 — unseal-to-wrong-key tripwire |
| `generation` | monotonic, bumped on every mutation — rollback guard (§5) |
| `mac` | HMAC-SHA256 over the file, keyed by a KEK-derived key — in-place tamper guard |

Argon2id parameters are read from the file, so they are bounded on both sides
before the KDF runs: a ceiling (4 GiB, 64 passes, 64 lanes) so a tampered
file cannot force an OOM before any key check, and a floor (8 MiB) so it
cannot downgrade the KDF to make the passphrase wrap cheap to brute-force.
Production creation uses 256 MiB, 3 passes, 1 lane.

Writes are atomic: temp file created owner-only (`0600` on unix), fsync,
rename over the target, fsync the directory; the previous file is kept as
`keystore.json.bak`. On Windows the file inherits the directory's ACL —
restrict the data directory.

## 4. Record envelope

Serialize first (the existing SHO/postcard envelope, CRC and all), then seal:

```
ENC\0 | crypto_version(1) | epoch(4 LE) | nonce(24) | XChaCha20-Poly1305 ct+tag
```

- The AEAD associated data is `shodh:record:v1:epoch:<epoch>` + `0x1f` + the
  record's RocksDB key (its memory id). A ciphertext copied onto another key
  fails to decrypt; so does a tampered epoch byte.
- XChaCha20 (192-bit random nonce) rather than AES-GCM for the high-volume
  path, so there is no nonce-collision ceiling to rotate against.
- `ENC\0` cannot collide with the `SHO` envelope or any legacy bincode
  record, so a mixed store (records from before the keystore existed) is
  unambiguous byte by byte.

## 5. Storage integration and fail-loud rules

`MemoryStorage::new` decides once, at open:

| `keystore.json` | `SHODH_MASTER_PASSPHRASE` | result |
|---|---|---|
| absent | unset | plaintext store; nothing written, nothing changed |
| absent | set | keystore created and persisted, then encryption on |
| present | set | unsealed; wrong passphrase, tampered file, or rollback → error |
| present | unset | **error** — encryption was requested and is unavailable |

Then, with a keystore active:

- **One encoder.** `encode_memory` is the only producer of a stored memory
  record; every write path uses it — `store`, `update`/`modify`, the
  access-metadata rewrite on recall (`persist_access_updates`), forgetting,
  lazy migration, bulk migration, and `store_with_vectors`. The branch this
  was ported from had one path (`persist_access_updates`) serializing with
  `encode_sho` directly, so every recalled record was rewritten in plaintext;
  `tests/encryption_round_trip.rs` reads the bytes after a recall to pin
  that this cannot recur.
- **One decoder.** `deserialize_memory_checked` unwraps the envelope under
  the record's key before the SHO envelope is read. A record this process
  cannot decrypt (no key for its epoch, failed tag) is an error, never a
  fabrication, and never deleted by the corruption cleanup.
- **The tripwire.** A plaintext record read under an active keystore is
  counted (`plaintext_reads_under_keystore()`), logged at WARN, and — when
  reached through `get` — rewritten encrypted on the spot. With
  `SHODH_REQUIRE_ENCRYPTED_READS=1` (read per call) it is refused with an
  error instead. Tests: `encryption_plaintext_tripwire_warn.rs`,
  `encryption_plaintext_tripwire_strict.rs`.
- **Rollback guard.** The index CF holds `meta:keystore_generation`, the
  highest generation this database has seen. A keystore file with a lower
  generation is refused; `SHODH_ALLOW_KEYSTORE_ROLLBACK=true` accepts it once
  (for a deliberate restore from `.bak`) and resets the sentinel.
- **One keystore per process.** The record crypto is process-global, like
  the store. A second store opened with a different keystore is refused
  rather than silently sharing keys. The same keystore reopened at a newer
  generation replaces the cryptor set.

### Turning encryption on over an existing store

Set the passphrase and restart. Old records are plaintext until touched:
`get` re-encrypts what it reads; `POST /api/storage/migrate`
(`migrate_legacy`) re-encrypts everything in one pass and counts already
encrypted records as current. Until that pass runs, the tripwire will fire
for every old record read — that is the signal it exists to give.

## 6. Scope — what is and is not encrypted

Encrypted: the primary `Memory` record in the default column family, in full.

Plaintext, deliberately and documented:

- **Facts, the knowledge graph, and vector-index embeddings** — separate
  stores and files with their own encoders.
- **The secondary index column family** — tag, entity, episode, robot,
  mission, action, content-hash, external-id, parent, date, type, importance
  and geohash keys. An on-disk reader learns which terms exist and which ids
  carry them, without a record. HMAC blinding of the exact-match keys is
  designed (equal terms → equal tokens; range keys stay clear) and deferred.
- **Oplog** and the feedback / files / prospective / todos column families.

## 7. Operations — `shodh-keyctl`

All secrets come from the environment; nothing secret is accepted on argv.
Stop the server first; every command persists atomically before returning.

| command | reads | effect |
|---|---|---|
| `status` | — | versions, generation, epochs, providers, KDF params |
| `rotate-passphrase` | `SHODH_MASTER_PASSPHRASE`, `SHODH_NEW_MASTER_PASSPHRASE` | re-wraps the KEK; records untouched |
| `rotate-dek` | `SHODH_MASTER_PASSPHRASE` | new active epoch; old epochs remain readable |
| `add-recovery-code` | `SHODH_MASTER_PASSPHRASE` | prints a one-time 48-hex code; stores only its wrap |
| `recover` | `SHODH_RECOVERY_CODE`, `SHODH_NEW_MASTER_PASSPHRASE` | installs a new passphrase; prints a fresh code |

```
shodh-keyctl --keystore data/storage/keystore.json rotate-dek
```

Rotating the DEK does not re-encrypt existing records; they stay on their
epoch and remain readable. Re-keying old epochs is a future explicit
migration, not a read side effect.

## 8. Deferred (not in this change)

- **KMS unseal providers** (`SHODH_KMS_WRAP_KEY`-style local wrap, cloud KMS):
  the `kek_wraps` list already admits another provider id.
- **Index blinding** of exact-match secondary keys (§6).
- **HKDF-derived subkeys**, **Zeroizing return types** throughout, and an
  **authenticated rollback sentinel**.
- **Re-encrypt-to-current-epoch migration** after `rotate-dek`.
- **Oblivious access / PIR** — T2/T3 above; needs its own design discussion.
- **Windows owner-only ACL** on `keystore.json`.

## 9. Tests

| contract | test |
|---|---|
| store → recall → modify → reopen: still `ENC\0`, plaintext absent from bytes | `tests/encryption_round_trip.rs` |
| wrong passphrase is a hard error at open | `tests/encryption_wrong_passphrase.rs` |
| keystore present, passphrase absent: refuses to open, bytes untouched | `tests/encryption_unavailable_fails_loud.rs` |
| second store with a different keystore is refused | `tests/encryption_keystore_guard.rs` |
| ciphertext moved to another key fails to decrypt | `tests/encryption_relocation.rs` |
| plaintext under keystore: counted, WARN, re-encrypted on read | `tests/encryption_plaintext_tripwire_warn.rs` |
| `SHODH_REQUIRE_ENCRYPTED_READS=1`: refused, bytes untouched | `tests/encryption_plaintext_tripwire_strict.rs` |
| no keystore: bytes identical to `encode_sho`, no side files, no sentinel | `tests/encryption_default_off.rs` |
| DEK rotation, multi-epoch reads, rollback guard and its override | `tests/encryption_rotation.rs` |
| keystore primitives (wrap AAD, KDF bounds, MAC, recovery, atomic save) | `src/keystore.rs` unit tests |

Each integration test is its own binary because the record crypto is
process-global.
