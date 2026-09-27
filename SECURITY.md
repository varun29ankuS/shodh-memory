# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

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

## Encryption at rest (opt-in, default off)

This section states precisely what the optional at-rest encryption does and
does **not** protect, so the guarantee is not over-read from the feature name.
Design and operations: [docs/encryption-v2-design.md](docs/encryption-v2-design.md).

**Enabling it.** Set `SHODH_MASTER_PASSPHRASE` before starting the server. On
first start with no keystore, `<data-dir>/storage/keystore.json` is created
(owner-only permissions on unix; back it up — without it every record written
from then on is unrecoverable). With no keystore and no passphrase, nothing
changes: records are stored exactly as before, byte for byte.

**Fail-loud rules.** A keystore present without the passphrase, a wrong
passphrase, a tampered keystore, or a keystore file older than the one this
database last saw (rollback) is a **hard error at open** — the store never
opens in plaintext mode beside ciphertext. A record that fails to decrypt is
an error on read, never a fabricated memory. Reading a plaintext record while
a keystore is active is counted, logged at WARN, and rewritten encrypted by
`get`; with `SHODH_REQUIRE_ENCRYPTED_READS=1` it is refused instead.

### Covered

- The **primary `Memory` record** — every serialized field (content, tags,
  entities, metadata, embeddings stored inside the record) — is opaque on
  disk: `ENC\0` marker, crypto version, DEK epoch, 24-byte random nonce, then
  the XChaCha20-Poly1305 ciphertext and tag. Every write path goes through the
  one encoder, including the access-metadata rewrite on the recall hot path.
- **Authentication.** Tampered or corrupted ciphertext is a decrypt error.
  The record's key (its memory id) is bound in as associated data, so a
  ciphertext moved to another key fails to decrypt (anti-swap).
- **Key hierarchy.** An Argon2id-derived key (with floor and ceiling on the
  stored parameters, so a tampered `keystore.json` can neither downgrade the
  KDF nor trigger a multi-GB allocation) wraps a master key; the master key
  wraps per-epoch data keys. Rotating the passphrase re-wraps the master key
  in O(1); rotating the data key starts a new epoch and old records stay
  readable under theirs. An optional recovery code wraps the master key
  independently of the passphrase.

### NOT covered — plaintext on disk

- **Facts, the knowledge graph, and vector-index embeddings** (separately
  stored records and the Vamana index files).
- **The secondary index column family**: tag, entity, episode, robot,
  mission, action, content-hash, external-id, parent, date, type, importance
  and geohash keys are stored in the clear. An on-disk reader can enumerate
  which tags, entities and dates exist without touching a record. Blinding
  the exact-match keys is designed but not in this change.
- **The oplog, and the feedback/files/prospective/todos column families.**
- **Memory-resident plaintext**: decrypted content lives in process memory
  while in use. This is at-rest protection only.
- **The unseal secret itself.** A passphrase in the environment of a host
  whose disk is stolen along with the data defeats the point; keep it off the
  data disk (a secret store, or a KMS provider — the latter is a follow-up).

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
| **HTTP transport** | Security headers (`X-Frame-Options`, CSP, `X-Content-Type-Options`, and HSTS in production) are always set. TLS is strongly recommended for any non-localhost deployment. | — |
| **Local IPC** | Enabled by default. Clients authenticate the server before sending HMAC-bound requests; the reusable key never crosses IPC. Unix checks peer UID and owner-only permissions. Windows uses identification-only SQOS, peer-account checks, and a protected per-user pipe. | `SHODH_IPC_ENABLED=false` disables the listener. `SHODH_IPC_REQUIRED=true` makes bind/probe failure fatal instead of allowing HTTP fallback. `SHODH_IPC_ENDPOINT` overrides the platform default. |

See the configuration documentation for the full list of variables.
