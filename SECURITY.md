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

## Encryption at rest (`SHODH_ENCRYPTION_KEY`)

When `SHODH_ENCRYPTION_KEY` is set (64-char hex or 44-char base64 → 32 bytes),
each `Memory` record is encrypted with AES-256-GCM **after** serialization and
stored as `[ENC\0 marker][12-byte nonce][ciphertext+tag]` in the primary column
family. When unset, records are stored as plaintext (backward compatible);
pre-existing plaintext records remain readable and upgrade to ciphertext on next
write.

### Covered

- The full serialized `Memory` record in the primary CF: content, summary, tags,
  entity refs, embeddings, and all other serialized fields are opaque on disk.
- Ciphertext authentication (GCM tag): tampering/corruption surfaces as a
  decryption **error**, not silent garbage.

### NOT covered (known, deliberate gaps)

- **Secondary index (`memory_index` CF).** Derived lookup keys are stored in
  **plaintext**: `tag:<tag>:<id>`, `entity:<name>:<id>`, and
  date/type/importance/geo/action keys. An on-disk index scan can confirm "a
  memory tagged X (or referencing entity Y, or from date Z) exists" **without
  touching ciphertext**. The *values* live in the encrypted record; the
  *existence and key terms* do not. Blinding the index is tracked as separate,
  deferred work.
- **Sibling column families** (feedback, files, prospective, todos) and any
  separately-keyed embedding blobs are not routed through the encryptor and are
  stored plaintext.
- **Memory-resident plaintext.** Decrypted content lives in process memory while
  in use; this is at-rest protection only.

### Key-loss / key-mismatch protection

On first encrypted open, a non-secret 4-byte fingerprint (`SHA-256(key)[..4]`)
is written to a sentinel entry in `memory_index`. On every subsequent open:

- stored fingerprint present + configured key **mismatches** → hard error,
  refuse to serve (prevents serving ciphertext-as-plaintext after a key swap);
- stored fingerprint present + **no** key configured → hard error;
- no stored fingerprint + key configured → fingerprint is recorded.

A malformed `SHODH_ENCRYPTION_KEY` is a **hard failure** (panic at first use),
not a silent fallback to plaintext.

> The fingerprint is a tripwire, not an authenticator: non-secret, only 4 bytes.
> It defends against accidental key loss/mismatch, **not** against an adversary
> who can already write to the database (game-over regardless).

### Key rotation — UNSUPPORTED in-process

The process-global encryptor is initialised once (`OnceLock`) and cannot change
without a restart. There is **no online key-rotation path**. To rotate: stop the
service, run an offline re-encrypt (decrypt-with-old → encrypt-with-new over
every record, reset the sentinel), then restart with the new key. An offline
`rotate-key` subcommand is not yet implemented.

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
