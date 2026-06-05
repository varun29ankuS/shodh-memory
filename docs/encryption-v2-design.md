# Encryption-at-rest v2 — design

Status: DRAFT for review (supersedes the field/record encryption in PR #285).
Applies to: shodh-memory (Apache-2.0 base) and veld (BUSL-1.1 sibling). Hosted
PIR path is near-term for veld/substrate, parked for shodh.

## 0. Goals (from review + product direction)

1. Revert the ad-hoc content/record encryption; replace with a proper subsystem.
2. Envelope encryption: master key (KEK) wraps data keys (DEK).
3. Keystore with pluggable **unseal providers**: passphrase (Argon2id),
   env/file, OS keychain, external KMS.
4. Key **rotation** (cheap KEK rotation; epoched DEK rotation).
5. **Recovery**: keystore backup + an independent one-time recovery code.
6. **Full index blinding**: HMAC for exact-match; OPE/ORE for ranges.
7. **Oblivious-access seam** now (StorageAccess trait + cheap T2 mitigations);
   Path-ORAM as a future feature-flagged backend.
8. **PIR** exact-key-fetch path spec for the hosted (T3) veld/substrate case.

## 1. Threat model (make explicit in SECURITY.md)

- **T1 — at-rest / cold-disk theft.** Attacker has the RocksDB files. Covered by
  envelope encryption + index blinding. PIR/ORAM add nothing here.
- **T2 — runtime host/co-tenant observing process I/O & access patterns.**
  Content covered; **access patterns leak**. Addressed (partially) by the
  oblivious seam + cheap mitigations now; fully by Path-ORAM later.
- **T3 — hosted/multi-tenant service; client hides queries from the server.**
  PIR territory. Near-term for veld/substrate; parked for shodh.

What v2 does NOT hide (documented residuals): access patterns without ORAM;
index equality/frequency under HMAC blinding; approximate order/distribution
under OPE/ORE; query timing/volume without padding.

## 2. Key hierarchy (envelope)

```
passphrase ──Argon2id(salt,m,t,p)──▶ unseal key ─┐
OS keychain  ───────────────────────────────────┤
env/file     ───────────────────────────────────┼─▶ unwrap ─▶ MASTER KEY (KEK, 256-bit)
KMS (wrap/unwrap) ───────────────────────────────┘                  │
                                                                     ▼
                                                    wraps ▶ DEK_epoch_N (256-bit)
                                                                     │
                                            AES-256-GCM record encryption
```

- **KEK** never stored raw. Stored only as ciphertext, wrapped by one or more
  unseal providers (multi-wrap: same KEK wrapped by passphrase AND keychain AND
  KMS AND recovery code — any one unseals).
- **DEK**: active per epoch; wrapped by KEK in the keystore. Records tag the
  `epoch`/`key_id` they were written under so epochs coexist.

## 3. Keystore

File `keystore.json` (or `CF_KEYSTORE`), versioned, holding:

```
version, kdf{alg=argon2id, salt, m_cost, t_cost, p_cost},
wraps: [ {provider: passphrase|keychain|kms|recovery, wrapped_kek, nonce, aad} ],
active_epoch, deks: [ {epoch, wrapped_dek, nonce, created_at, state: active|retired} ],
index_hmac: {epoch, wrapped_key},     # blinding key, rotated with index rebuild
verify_tag                            # detects keystore tamper/rollback
```

- **Backup** is mandatory (loss = unrecoverable). Documented.
- **Rollback protection**: `verify_tag` over the keystore + monotonically
  increasing `active_epoch`; on open, refuse an epoch lower than the
  fingerprint-sentinel recorded in the data store (see §6).

## 4. Unseal providers (trait)

```rust
trait UnsealProvider {
    fn id(&self) -> &'static str;            // "passphrase" | "keychain" | "kms" | "recovery"
    fn unwrap_kek(&self, wrapped: &Wrapped) -> Result<Zeroizing<[u8;32]>>;
    fn wrap_kek(&self, kek: &[u8;32]) -> Result<Wrapped>;   // for (re)wrap/rotation
}
```

- **passphrase**: Argon2id(passphrase, salt, params) → AES-256-GCM key-wrap of KEK.
  Interactive (CLI/recovery). Params tunable + upgradeable.
- **env/file**: passphrase sourced from `*_MASTER_PASSPHRASE` env or a file path
  for unattended servers (keystore stays Argon2id-wrapped at rest).
- **keychain**: macOS Keychain / libsecret stores the (already KEK-derived) wrap
  key or a passphrase; boot-time unseal when the keychain is unlocked.
- **kms**: KEK wrapped by a cloud KMS CMK (AWS/GCP/Vault); server holds IAM/creds
  to call Decrypt. Built now (near-term hosted need).
- **recovery**: one-time printed recovery code (high-entropy) wraps the KEK
  independently → survives passphrase loss if the keystore survives.

Resolution order configurable; first successful unseal wins. At least one
non-interactive provider required for headless deploys.

## 5. Record encryption

- Record-level, encrypt-after-serialize. Wire:
  `ENC\0 | version | epoch(key_id) | nonce(12) | ct+tag`.
- **AAD binds `epoch` (+ record id)** so a record cannot be silently decrypted
  under the wrong DEK or substituted across records.
- 96-bit random nonce per encryption (OsRng). Single active DEK → safe to ~2^32
  records/epoch; rotate epoch well before.
- shodh: replaces the PR #285 field/record code. veld: replaces field-level
  base64 (offline migration, §7).

## 6. Index blinding

- **Exact-match** (`tag`,`entity`,`external`,`content_hash`,`parent`,`robot`,
  `mission`,`episode`): key = `HMAC-SHA256(index_hmac_key, term)` → blinded
  prefix; query HMACs the term and prefix-scans. Leaks equality/frequency only.
- **Range — hybrid (decided).** Coarse/low-cardinality fields (`date` at
  day-granularity, `importance` buckets, `geo` geohash, `episode_seq`) use
  **bucket+enumerate**: HMAC the bucket token; at query time enumerate the
  buckets covering the range and point-look each. No order leak beyond bucket
  granularity, no OPE dependency. Reserve **OPE/ORE** for genuinely
  high-cardinality range fields where bucketing is too coarse to be useful (none
  in the current schema; decide per-field if one is added). ORE preferred over
  OPE where a vetted Rust impl exists.
- **Sentinel** (`meta:encryption_key_fingerprint`, already shipped in pass-1)
  generalised to `{kek_fp, active_epoch}` — detects key loss/mismatch/rollback
  at open; hard error.
- Rotating `index_hmac_key`/OPE key ⇒ **offline index rebuild** from records.

## 7. Rotation & migration

- **KEK rotation** (passphrase/keychain/KMS/recovery change): re-wrap KEK across
  providers. O(1); records untouched. ← envelope payoff.
- **DEK rotation**: new active epoch; new writes use it; old epochs readable;
  lazy re-encrypt on write or offline bulk pass. `key_id` disambiguates.
- **Index key rotation**: offline rebuild (blinding is deterministic per key).
- **Migration commands** (offline, idempotent, resumable):
  - `migrate-encrypt`: plaintext → envelope (shodh & veld).
  - `migrate-veld-field-to-record`: decrypt field-base64 → re-encrypt record.
  - `rebuild-index`: re-derive `memory_index` under current blinding key.
  - All gated by the sentinel; crash-safe via per-record `key_id`.

## 8. Oblivious-access seam (T2)

- All record + index access goes through a `StorageAccess` trait:
  `get_block/put_block/scan_prefix`. Default impl = direct RocksDB.
- **Cheap mitigations now** (sub-ORAM, behind `oblivious-lite`): fixed-size
  result **padding** + **dummy queries**, **write-back re-encryption** of touched
  blocks (breaks read-linkability), batching/caching to blur timing.
- **Future** (`oblivious` feature): Path-ORAM backend — padded blocks, recursive
  position map, bounded stash, full-path read+evict (O(log N) amplification),
  plus oblivious index map. Semantic/vector search stays non-oblivious or
  degrades (documented); not default.

## 9. PIR (T3, hosted — veld/substrate near-term)

- **Scope: exact-key private fetch only.** Client retrieves memory by (blinded)
  key without the server learning which. cPIR (Spiral/FrodoPIR-class) single
  server; server processes O(bucket) per query; DB partitioned into PIR buckets;
  updates re-encode the touched bucket.
- **Explicitly out of scope / research-grade:** private *semantic/vector* recall
  (private nearest-neighbor). Hosted recall stays non-private unless/until a
  practical private-ANN scheme is adopted; documented as a known limitation.
- itPIR (≥2 non-colluding replicas) noted as an alternative if the deployment can
  guarantee non-collusion.

## 10. Phased implementation (each phase build- + test-gated, new PR)

- **P1 Crypto core**: envelope primitives, keystore, key-wrap, Argon2id,
  fingerprint/epoch sentinel, key types + tests. No wiring.
- **P2 Record encryption**: replace shodh ad-hoc + veld field-level via the
  keystore; AAD binding; `migrate-encrypt` + veld field→record migration.
- **P3 Index blinding**: exact-match HMAC; then range OPE/ORE; `rebuild-index`.
- **P4 Rotation + recovery**: KEK re-wrap, DEK epochs, recovery code, CLI.
- **P5 Unseal providers**: passphrase/env/file/keychain/**KMS**.
- **P6 Oblivious seam**: `StorageAccess` trait + `oblivious-lite` mitigations;
  Path-ORAM stub behind `oblivious`.
- **P7 PIR (veld/substrate)**: cPIR exact-key bucket service.

Open decisions deferred to their phase: ORE library choice (P3), KMS provider
(P5), PIR scheme + bucketing (P7).

## 11. Adversarial review — findings folded in (bifocal / overloop / breakers)

**B-1 Multi-wrap is weakest-link (headline).** Wrapping one KEK under passphrase
OR keychain OR KMS OR recovery = OR semantics → the KEK is only as strong as the
*weakest* enabled provider. A weak passphrase-wrapped copy defeats the
KMS-wrapped copy. → keystore carries a **policy**: enforce min-strength per
provider, and allow an **AND-of-providers / KMS-only** mode for high-security
deploys (disable the passphrase copy entirely). Argon2id strength enforced +
upgradeable. **CONFIRMED as designed.**

**B-2 env/file unseal breaks T1 if co-located.** An unattended passphrase in env
or a file on the *same disk* as the data means cold-disk theft yields both →
encryption defeated for T1. → document loudly; recommend keychain/KMS over
env/file when T1 matters; the unseal secret must live on different media/KMS.

**B-3 OPE/ORE on already-bucketed fields is leakage-theater.** For low-cardinality
fields (`importance` buckets, `geo` prefixes) order-preserving ciphertext ≈
reveals the value, at real complexity/dependency cost. → **revise §6**: default
to **bucket+enumerate** (HMAC the bucket, enumerate buckets in range at query
time — no order leak beyond bucket granularity); reserve OPE/ORE for genuinely
high-cardinality ranges where it earns its leakage. **RESOLVED → hybrid adopted
(see §6): bucket+enumerate for the coarse fields, OPE/ORE only for any future
high-cardinality range field.**

**B-4 KMS = availability coupling.** KMS unreachable at boot ⇒ store can't unseal
⇒ DoS. → optional cached wrapped-KEK to ride transient outages (tradeoff noted),
or fail-closed by explicit policy.

**B-5 Cross-store atomicity / rollback.** active_epoch lives in the keystore AND
a sentinel in the data store; a crash between updates = false rollback alarm or
missed rollback. → single **schema/crypto version vector** + a 2-phase
(write-intent → commit) update; migrations gated by the version vector, not just
the key fingerprint.

**B-6 "Oblivious mode" must not over-claim.** Path-ORAM hides block access, but
semantic/vector recall stays non-oblivious. Silently shipping `oblivious` while
recall leaks repeats the original PR's over-claim trap. → in `oblivious` mode,
either route recall through the oblivious layer (degraded) or **FAIL LOUD** that
recall is non-oblivious; document either way.

**B-7 Epoch rotation by count, not just time.** Random 96-bit nonces get risky
near 2^32 encryptions/DEK. → rotate the active epoch on a **record-count**
threshold as well as time.

**B-8 Recovery code is another weakest-link if stored digitally.** High-entropy,
one-time, print-only; document handling; never persist it in the keystore.

