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
                                              HMAC("dek-wrap:v1") ▶ wrap key ▶ DEK_epoch_N
                                                                                   │
                                                        XChaCha20-Poly1305 record encryption
```

- The **KEK** is never stored raw: only wrapped, once per enabled unseal
  provider (`kek_wraps`: `passphrase`, optionally `recovery`). Any one wrap
  unseals — so the KEK is as strong as the weakest enabled provider.
- **DEKs** are per epoch, wrapped under a key derived from the KEK
  (`HMAC-SHA256(KEK, "shodh:keystore:dek-wrap:v1")`) with the epoch bound in as
  AAD so a DEK cannot be substituted across epochs. The KEK itself only keys
  derivations: this wrap key, the integrity MAC key and the database binding.
  Keystores written before the wrap key was derived keep raw-KEK wraps, which
  carry no `wrap` field so their MAC still verifies; the first mutation
  (rotation, passphrase change, recovery code) re-wraps them. The active epoch encrypts new
  writes; retired epochs stay in the keystore so their records stay readable.
- Every wrap is AES-256-GCM with a domain-separating AAD.

## 3. Keystore file

One keystore per process. A standalone store keeps it at
`<data-dir>/storage/keystore.json`; the server keeps one at its data root,
`<data-root>/keystore.json`, and every tenant's store binds to it (§5).
`SHODH_KEYSTORE_DIR` overrides both. The file is JSON and holds no plaintext key
material:

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
file cannot force an OOM before any key check, and a floor so it cannot
downgrade the KDF to make the passphrase wrap cheap to brute-force: at least
19 MiB, with 2 passes below 46 MiB (OWASP's Argon2id minimums).
Production creation uses 256 MiB, 3 passes, 1 lane.

Writes are atomic: temp file created owner-only (`0600` on unix), fsync,
rename over the target, fsync the directory; the previous file is kept as
`keystore.json.bak`. On Windows the file inherits the directory's ACL —
restrict the data directory. `/api/backup` does not include the keystore:
back it up separately, or a restored database cannot be opened.

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

| binding | `keystore.json` | `SHODH_MASTER_PASSPHRASE` | result |
|---|---|---|---|
| none, no encrypted record | absent | unset | plaintext store; nothing written, nothing changed |
| none, no encrypted record | absent | set | keystore created and persisted, database bound, then encryption on |
| none | present | set | unsealed; bound only if an encrypted record here decrypts under it |
| present | present | set | unsealed; must be the bound keystore, at the bound generation or newer |
| present, or an encrypted record | absent | any | **error** — never a fresh keystore over encrypted data, never a plaintext open |
| any | present | unset | **error** — encryption was requested and is unavailable |

A database with no binding is scanned for encrypted records only when a
keystore is about to be created or adopted, never on a plaintext open.

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
  fabrication, and never deleted by the corruption cleanup: AEAD cannot tell
  a wrong key from damaged bytes, so cleanup keeps every encrypted record and
  counts the ones it cannot read.
- **The tripwire.** A plaintext record read under an active keystore is
  counted (`plaintext_reads_under_keystore()`), logged at WARN, and — when
  reached through `get` — rewritten encrypted on the spot. With
  `SHODH_REQUIRE_ENCRYPTED_READS=1` (read per call) it is refused with an
  error instead. Tests: `encryption_plaintext_tripwire_warn.rs`,
  `encryption_plaintext_tripwire_strict.rs`.
- **Binding.** The index CF holds `meta:keystore_binding`:
  `SKB1 | keystore id | generation | tag`. The id and the tag are HMACs keyed
  by a KEK-derived key, so DEK rotation and passphrase changes keep the id,
  another keystore cannot produce it, and the generation cannot be edited
  without the KEK. It is written (synced) before the first encrypted record
  and never removed. A database written before bindings existed is bound on
  its next keyed open, after one of its encrypted records decrypts under the
  keystore; the old unauthenticated `meta:keystore_generation` sentinel is
  read once as the rollback floor (a wrong-length value is an error) and
  removed.
- **Rollback guard.** A keystore whose generation is below the bound one is
  refused; `SHODH_ALLOW_KEYSTORE_ROLLBACK=true` accepts it once (a deliberate
  restore from `.bak`), resets the binding, and replaces the running
  process's cryptors so no record is written under an epoch the restored file
  does not hold.
- **One keystore per process.** The record crypto is process-global, like the
  passphrase. A second store opened with a different keystore is refused
  rather than silently sharing keys. The same keystore at a newer generation
  replaces the cryptor set; at an older one it is refused unless the rollback
  override accepted it.
- **Tenants.** The server sets the keystore root to its data directory, so
  every tenant binds to the one keystore its passphrase unseals. The first
  time a tenant opens under the root, a keystore beside its data moves to the
  root if its records are encrypted under it and no root keystore exists yet,
  is renamed `keystore.json.unused` if it encrypted nothing, and is refused if
  its records are encrypted under it and a root keystore already exists.

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
- **The BM25 index** (`<user>/bm25_index`, tantivy) is NOT written under a
  keystore: it stores `content` and token positions, so on disk it would be a
  plaintext copy of every memory. With a keystore active it is held in memory,
  filled at start by the backfill from the decrypted records, and an index left
  on disk from before encryption was enabled is deleted. Without a keystore it
  is on disk as before.
- **The audit log's `content_preview`.**
- **Vector embeddings can be inverted** to approximate the text they encode, so
  a plaintext vector index leaks content, not only similarity.
- **Old plaintext after turning encryption on.** Records written before the
  keystore stay in older SST files and in earlier backups until compaction and
  backup rotation retire them.

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
- **Zeroizing return types** throughout.
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
| lost keystore: refused with or without the passphrase, never recreated; restore reopens | `tests/encryption_binding_lost_keystore.rs` |
| cleanup keeps unreadable ciphertext; a keystore other than the bound one is refused | `tests/encryption_binding_foreign_keystore.rs` |
| tenants share the root keystore; per-store keystores move or are set aside | `tests/encryption_keystore_root_tenants.rs` |
| accepted rollback writes under the restored keystore's epochs; binding reset | `tests/encryption_rollback_follows_disk.rs` |
| keystore primitives (wrap AAD, KDF bounds, MAC, recovery, atomic save, binding) | `src/keystore.rs` unit tests |

Each integration test is its own binary because the record crypto is
process-global.
