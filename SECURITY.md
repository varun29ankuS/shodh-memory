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
