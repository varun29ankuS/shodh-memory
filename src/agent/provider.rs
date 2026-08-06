//! Env-configured model provider wiring for the agent turn loop.
//!
//! `rig-core` 0.41.0 does **not** ship an `Agent`/runtime type — its own module docs say so
//! plainly: "This crate owns the provider-agnostic model, message, tool, and storage
//! contracts. The sibling `rig-agent` crate provides the classic builder and run-loop API."
//! (`rig_core::lib` doc comment). This crate depends on `rig-core` only (see `Cargo.toml`'s
//! dependency comment — deliberately not the `rig` facade, and not `rig-agent`), so the turn
//! loop in [`crate::agent::turn_loop`] is built directly on `rig-core`'s low-level
//! [`rig_core::completion::CompletionModel`] trait, and this module's job is constructing one.
//!
//! Two providers are supported, both selected and configured entirely from environment
//! variables so the whole stack can run fully offline with no hardcoded cloud endpoint and no
//! API key requirement for local servers:
//!
//! - `SHODH_AGENT_PROVIDER` — `ollama` (default) or `openai`. `openai` here means "any
//!   OpenAI-compatible HTTP endpoint" (vLLM, llama.cpp's server, LM Studio, text-generation-
//!   webui, an actual OpenAI-compatible proxy, ...), reached through the Chat Completions API
//!   (`/v1/chat/completions`) rather than OpenAI's newer Responses API — Chat Completions is
//!   the de facto standard self-hosted servers implement; Responses is not.
//! - `SHODH_AGENT_MODEL` — model name/tag passed to the provider. Required; there is no safe
//!   universal default.
//! - `SHODH_AGENT_BASE_URL` — overrides the provider's base URL. For `ollama` this is optional
//!   (falls back to rig-core's own built-in default, `http://localhost:11434`). For `openai`
//!   this is **required** — there is deliberately no `api.openai.com` fallback baked in here;
//!   the caller must point it at whatever OpenAI-compatible server they are running.
//! - `SHODH_AGENT_API_KEY` — optional. Omit for unauthenticated local servers. `ollama`'s
//!   client accepts no key at all in that case (`Nothing`, matching rig-core's own default-
//!   Ollama example); an OpenAI-compatible client always sends *some* bearer token (that
//!   client type has no "no auth" mode), so a harmless placeholder is used when unset — local
//!   OpenAI-compatible servers that don't check auth simply ignore it.
//! - `SHODH_AGENT_MAX_TURNS` — bounds the turn loop's model-call count (see
//!   [`crate::agent::turn_loop`]). Defaults to [`DEFAULT_MAX_TURNS`].

use anyhow::{bail, Context, Result};

use rig_core::client::{CompletionClient, Nothing};
use rig_core::providers::{ollama, openai};

/// Default turn-loop bound when `SHODH_AGENT_MAX_TURNS` is unset or invalid.
pub const DEFAULT_MAX_TURNS: usize = 8;

/// Placeholder bearer token sent to OpenAI-compatible servers when no
/// `SHODH_AGENT_API_KEY` is configured. The OpenAI client type always attaches a bearer
/// header (unlike Ollama's client, it has no "no auth" input type); local servers that
/// don't check authentication simply ignore this value.
const OPENAI_COMPAT_NO_KEY_PLACEHOLDER: &str = "not-required";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Ollama,
    OpenAiCompatible,
}

/// A constructed, ready-to-call completion model for whichever provider was selected.
///
/// This is a closed enum, not a boxed trait object, because
/// [`rig_core::completion::CompletionModel`] is not object-safe (it has an `async fn` in the
/// trait and provider-specific associated `Response`/`StreamingResponse` types) — see the
/// trait definition in `rig-core`'s `completion/request.rs`. The turn loop itself
/// ([`crate::agent::turn_loop::run`]) is generic over `M: CompletionModel` rather than over
/// this enum, so tests exercise that generic function directly with a fake model instead of
/// going through provider construction at all; this enum exists purely so the HTTP handler can
/// hold "whichever concrete model env config selected" as one value.
pub enum AgentModel {
    Ollama(ollama::CompletionModel<reqwest::Client>),
    OpenAiCompatible(openai::completion::CompletionModel<reqwest::Client>),
}

/// Env-derived provider configuration. See module docs for each variable.
pub struct AgentProviderConfig {
    pub kind: ProviderKind,
    pub model: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub max_turns: usize,
}

impl AgentProviderConfig {
    /// Read configuration from the process environment. Returns an error describing exactly
    /// what is missing/invalid rather than silently falling back to a cloud default.
    pub fn from_env() -> Result<Self> {
        let kind = match std::env::var("SHODH_AGENT_PROVIDER")
            .unwrap_or_else(|_| "ollama".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "ollama" => ProviderKind::Ollama,
            "openai" => ProviderKind::OpenAiCompatible,
            other => bail!(
                "unknown SHODH_AGENT_PROVIDER '{other}' (expected 'ollama' or 'openai')"
            ),
        };

        let model = std::env::var("SHODH_AGENT_MODEL").context(
            "SHODH_AGENT_MODEL is not set — the agent turn loop needs a model name/tag \
             (e.g. 'llama3.1' for ollama, or the model id your OpenAI-compatible server serves)",
        )?;
        if model.trim().is_empty() {
            bail!("SHODH_AGENT_MODEL is set but empty");
        }

        let base_url = std::env::var("SHODH_AGENT_BASE_URL")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        if kind == ProviderKind::OpenAiCompatible && base_url.is_none() {
            bail!(
                "SHODH_AGENT_BASE_URL is required when SHODH_AGENT_PROVIDER=openai — there is \
                 no hardcoded cloud endpoint here; point it at your OpenAI-compatible server \
                 (e.g. http://localhost:8000/v1)"
            );
        }

        let api_key = std::env::var("SHODH_AGENT_API_KEY")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let max_turns = std::env::var("SHODH_AGENT_MAX_TURNS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_MAX_TURNS);

        Ok(Self {
            kind,
            model,
            base_url,
            api_key,
            max_turns,
        })
    }

    /// Construct the concrete completion model this configuration selects.
    pub fn build_model(&self) -> Result<AgentModel> {
        match self.kind {
            ProviderKind::Ollama => {
                let mut builder = ollama::Client::builder();
                if let Some(base) = &self.base_url {
                    builder = builder.base_url(base);
                }
                let client = match &self.api_key {
                    Some(key) => builder.api_key(key.as_str()).build(),
                    None => builder.api_key(Nothing).build(),
                }
                .context("failed to build ollama client")?;
                Ok(AgentModel::Ollama(client.completion_model(self.model.clone())))
            }
            ProviderKind::OpenAiCompatible => {
                let base = self
                    .base_url
                    .as_deref()
                    .expect("validated non-None in from_env for the openai provider");
                let key = self
                    .api_key
                    .clone()
                    .unwrap_or_else(|| OPENAI_COMPAT_NO_KEY_PLACEHOLDER.to_string());
                let client = openai::Client::builder()
                    .api_key(key.as_str())
                    .base_url(base)
                    .build()
                    .context("failed to build OpenAI-compatible client")?
                    // Chat Completions (`/v1/chat/completions`), not the Responses API — see
                    // module docs for why this is the compatible choice for self-hosted servers.
                    .completions_api();
                Ok(AgentModel::OpenAiCompatible(
                    client.completion_model(self.model.clone()),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes all env-var mutation in this module's tests against every other test in the
    /// crate that touches process environment (matches the convention in `src/auth.rs`).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn clear_env() {
        for key in [
            "SHODH_AGENT_PROVIDER",
            "SHODH_AGENT_MODEL",
            "SHODH_AGENT_BASE_URL",
            "SHODH_AGENT_API_KEY",
            "SHODH_AGENT_MAX_TURNS",
        ] {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn defaults_to_ollama_with_no_hardcoded_base_url_requirement() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        std::env::set_var("SHODH_AGENT_MODEL", "llama3.1");

        let config = AgentProviderConfig::from_env().expect("ollama needs no base url");
        assert_eq!(config.kind, ProviderKind::Ollama);
        assert_eq!(config.max_turns, DEFAULT_MAX_TURNS);
        assert!(config.base_url.is_none());

        clear_env();
    }

    #[test]
    fn openai_provider_requires_explicit_base_url() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        std::env::set_var("SHODH_AGENT_PROVIDER", "openai");
        std::env::set_var("SHODH_AGENT_MODEL", "some-local-model");

        let err = AgentProviderConfig::from_env()
            .expect_err("openai provider without SHODH_AGENT_BASE_URL must be rejected");
        assert!(err.to_string().contains("SHODH_AGENT_BASE_URL"));

        clear_env();
    }

    #[test]
    fn missing_model_is_a_clear_error_not_a_silent_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();

        let err = AgentProviderConfig::from_env().expect_err("model is required");
        assert!(err.to_string().contains("SHODH_AGENT_MODEL"));

        clear_env();
    }

    #[test]
    fn unknown_provider_name_is_rejected() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        std::env::set_var("SHODH_AGENT_PROVIDER", "bedrock");
        std::env::set_var("SHODH_AGENT_MODEL", "whatever");

        let err = AgentProviderConfig::from_env().expect_err("unknown provider must be rejected");
        assert!(err.to_string().contains("bedrock"));

        clear_env();
    }

    #[test]
    fn max_turns_env_override_is_respected_and_zero_falls_back_to_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        std::env::set_var("SHODH_AGENT_MODEL", "llama3.1");
        std::env::set_var("SHODH_AGENT_MAX_TURNS", "3");
        assert_eq!(AgentProviderConfig::from_env().unwrap().max_turns, 3);

        std::env::set_var("SHODH_AGENT_MAX_TURNS", "0");
        assert_eq!(
            AgentProviderConfig::from_env().unwrap().max_turns,
            DEFAULT_MAX_TURNS,
            "a zero turn bound is nonsensical and must fall back to the default"
        );

        clear_env();
    }

    #[test]
    fn ollama_model_builds_without_any_key_configured() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        std::env::set_var("SHODH_AGENT_MODEL", "llama3.1");

        let config = AgentProviderConfig::from_env().unwrap();
        let model = config.build_model().expect("ollama client build must not require a key");
        assert!(matches!(model, AgentModel::Ollama(_)));

        clear_env();
    }
}
