# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in shodh-memory, please report it privately:

1. **Email**: Send details to 29.varuns@gmail.com
2. **GitHub**: Use [Security Advisories](https://github.com/varun29ankuS/shodh-memory/security/advisories/new) to report privately

**What to include:**
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes

**Response timeline:**
- Initial response within 48 hours
- Status update within 7 days
- Fix timeline depends on severity

**Do not:**
- Open public issues for security vulnerabilities
- Disclose publicly before a fix is available

We appreciate responsible disclosure.

---

# Encryption-at-rest & embedder egress — threat model

This section states precisely what the optional at-rest encryption and the HTTP
embedder do and do **not** protect, so the guarantees are not over-read from a
feature name.

## Encryption at rest (envelope keystore)

Record-level encryption is enabled when a keystore (`<storage>/keystore.json`)
is present and an unseal secret is available: `SHODH_MASTER_PASSPHRASE` (an
Argon2id-derived key) and/or `SHODH_KMS_WRAP_KEY` (a 32-byte AES-256-GCM wrap
key). On first run with a passphrase and no keystore, one is created. With no
keystore and no passphrase, records are stored as plaintext (backward
compatible); a keystore present with **neither** a passphrase nor a KMS key is a
**hard error** — the process refuses to start rather than serve ciphertext as
plaintext. (The legacy single-key `SHODH_ENCRYPTION_KEY` / AES-256-GCM path has
been removed.)

Keys form an envelope hierarchy: a master key (KEK) is wrapped by the
passphrase- and/or KMS-derived key (multi-wrap), and the KEK protects per-epoch
data keys (DEKs). Each `Memory` record is encrypted with **XChaCha20-Poly1305**
(192-bit random nonce) after serialization and stored as an `ENC\0` marker, a
crypto-version byte, the DEK epoch, the 24-byte nonce, and the AEAD
ciphertext+tag, in the primary column family.

### Covered

- The full serialized `Memory` record in the primary CF: content, summary, tags,
  entity refs, embeddings, and every other serialized field is opaque on disk.
- Ciphertext authentication (Poly1305 tag): tampering/corruption surfaces as a
  decryption **error**, not silent garbage.
- **Exact-match secondary-index terms are blinded.** Lookup keys for tags,
  entities, episodes, robots, missions, action types, content hashes, external
  IDs, and parents are stored as keyed `HMAC-SHA256` tokens (key derived from the
  KEK), not as cleartext. Equal terms map to equal tokens so point lookups still
  work, but an on-disk scan can no longer read the term. This closes the v1
  "index is plaintext" gap for exact-match terms.

### NOT covered (known, deliberate gaps)

- **Ordered / range index keys remain plaintext.** Date, type, importance, and
  geo keys are stored in the clear so range and ordered scans keep working —
  HMAC blinding destroys ordering and so cannot be applied to them. An on-disk
  scan can still infer "a memory from date Z / of type T / above importance N
  exists" without touching ciphertext.
- **Sibling column families** (feedback, files, prospective, todos) and any
  separately-keyed embedding blobs are not routed through the record encryptor
  and are stored plaintext.
- **Memory-resident plaintext.** Decrypted content lives in process memory while
  in use; this is at-rest protection only.

### Key-loss / tamper / rollback protection

- The keystore carries an **integrity MAC** (HMAC keyed by a KEK-derived key)
  verified on every open; a tampered `keystore.json` is a hard error.
- A non-secret **KEK fingerprint** pins the active keystore. shodh's encryption
  is process-global (one keystore per process); opening a second store with a
  *different* keystore fails loudly rather than silently reusing the first
  store's keys (cross-keystore data confusion).
- A monotonic **generation** sentinel is recorded in the DB; a `keystore.json`
  whose generation is older than last-seen is rejected (rollback guard).
- A malformed keystore or a wrong passphrase is a **hard failure**, never a
  silent fallback to plaintext.

### Key rotation — partial

The keystore supports multi-wrap unseal (passphrase + KMS) and epoched DEKs, so
the unseal secret can be re-wrapped without re-encrypting data. **Online
data-key rotation is not yet wired end-to-end:** the storage read path serves a
single active epoch and rejects records written under a different epoch
(`record epoch differs from active epoch`). Rotating the DEK therefore still
requires an offline re-encrypt pass; an online rotation path is deferred.

## HTTP embedder egress (`http-embedder` feature, off by default)

The HTTP embedder compiles only under the `http-embedder` Cargo feature. When
enabled, `SHODH_EMBEDDING_API_URL` is validated before use by a self-contained
guard in the embedder module (it does not depend on the integration-URL helpers
owned by PR #284):

- plain-HTTP to a **non-localhost** host is rejected by default (falls back to
  the localhost default), because the embedder POSTs user text to that URL;
- opt back in with `SHODH_ALLOW_INSECURE_REMOTE_EMBEDDER=true`.

This blocks the obvious env-injection exfiltration / SSRF vector
(`http://169.254.169.254/...`). It does **not** cover `https://` URLs to
internal/link-local hosts, or DNS rebinding (a hostname resolving to an internal
address), because no name resolution or address-range check is performed. Treat
the embedder URL as trusted configuration. Availability checks use `HEAD` (no
billable embedding request) and are cached for 30s.

---

## Secure Defaults & Hardening

shodh-memory ships secure-by-default. Each behavior below is enforced unless an
operator explicitly opts out via an environment variable.

| Area | Secure default | Override (use only if you understand the risk) |
|------|----------------|-------------------------------------------------|
| **Webhooks** | `/webhook/*` requests are rejected (HTTP 503) unless the matching `LINEAR_WEBHOOK_SECRET` / `GITHUB_WEBHOOK_SECRET` is configured, so every webhook is HMAC-verified. | `SHODH_ALLOW_UNSIGNED_WEBHOOKS=true` processes unsigned webhooks. |
| **Rate limiting** | Public routes (webhooks, context status, graph viewer) are rate-limited. Health probe routes (`/health*`) are never rate-limited. | `SHODH_PUBLIC_RATE_LIMIT=false` exempts public routes; `SHODH_RATE_LIMIT=0` disables rate limiting entirely. |
| **Metrics** | `/metrics` requires API-key authentication (`X-API-Key` or `Authorization: Bearer <key>`). | `SHODH_METRICS_PUBLIC=true` exposes `/metrics` without auth. |
| **Integration API URLs** | An insecure `http://` override of `GITHUB_API_URL` / `LINEAR_API_URL` for a non-localhost host is warned about. | `SHODH_ENFORCE_HTTPS=true` rejects such overrides and uses the secure default. |
| **Error responses** | In production (`SHODH_ENV=production`) 5xx responses return a generic message; full detail is logged server-side only. | — |
| **API authentication** | All `/api/*` routes require an API key. Production refuses authenticated requests when no key is configured. | — |
| **Transport** | Security headers (`X-Frame-Options`, CSP, `X-Content-Type-Options`, and HSTS in production) are always set. TLS is strongly recommended for any non-localhost deployment. | — |

See the configuration documentation for the full list of variables.
