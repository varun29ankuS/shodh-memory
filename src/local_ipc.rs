//! Authenticated local IPC transport for the canonical Shodh HTTP route table.
//!
//! The wire protocol is deliberately small: at most one newline-delimited JSON
//! request is dispatched and one response is written before the connection
//! closes. Requests carry an HTTP method, origin-form path, and JSON body so the
//! existing Axum router remains the sole operation catalog shared by HTTP, MCP,
//! and local integrations.

use std::{
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use axum::{
    body::{to_bytes, Body},
    http::{Method, Request, StatusCode, Uri},
    response::{IntoResponse, Response},
    Router,
};
use hmac::{Hmac, Mac};
use rand::{rngs::OsRng, RngCore};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, value::RawValue, Value};
use sha2::Sha256;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    sync::{OwnedSemaphorePermit, RwLock, Semaphore},
    task::JoinSet,
};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;
use uuid::Uuid;

use crate::auth::{self, AuthError};

pub const PROTOCOL_VERSION: u16 = 2;
pub const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;
const MAX_BODY_BYTES: usize = MAX_FRAME_BYTES - 64 * 1024;
const NONCE_BYTES: usize = 32;
const MAX_PRE_AUTH_CONNECTIONS: usize = 16;
const PRE_AUTH_READ_TIMEOUT: Duration = Duration::from_secs(2);
/// Read (GET) deadline. Reads are idempotent, so a short deadline is safe.
const DEFAULT_CLIENT_TIMEOUT: Duration = Duration::from_secs(10);
/// Write (POST/PUT/PATCH/DELETE) deadline. Must exceed the server's own route
/// deadline (`request_timeout`, 60s by default) — otherwise a slow-but-successful
/// write is reported to the MCP caller as a failure *after* the server has already
/// committed it, and a non-idempotent operation cannot be safely retried.
const DEFAULT_WRITE_TIMEOUT: Duration = Duration::from_secs(120);

type HmacSha256 = Hmac<Sha256>;
const PROBE_RESPONSE_DOMAIN: &[u8] = b"shodh-ipc-v2/server-proof";
const REQUEST_DOMAIN: &[u8] = b"shodh-ipc-v2/request";
const RESPONSE_DOMAIN: &[u8] = b"shodh-ipc-v2/response";

const STREAMING_PATHS: &[&str] = &[
    "/api/stream",
    "/api/context/monitor",
    "/api/events",
    "/api/events/sse",
    "/api/context/sse",
];

const PUBLIC_API_PATHS: &[&str] = &["/api/context/status", "/api/context_status"];

/// Stable endpoint owned by Shodh rather than by the storage-path fallback.
///
/// The Windows default folds the current user's SID into the pipe name. The DACL
/// scopes access to the current user + LocalSystem, but the pipe namespace is
/// machine-global: a fixed name lets only the first user on a multi-user host bind
/// it. A per-user name removes that cross-user collision. The name remains
/// discoverable, so clients also verify the peer account and the keyed endpoint
/// proof before trusting it.
pub fn default_endpoint() -> PathBuf {
    #[cfg(windows)]
    {
        platform::default_endpoint()
    }

    #[cfg(unix)]
    {
        dirs::data_local_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("shodh")
            .join("shodh-memory.sock")
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestEnvelope {
    v: u16,
    id: String,
    auth: String,
    challenge: String,
    method: String,
    path: String,
    #[serde(default = "raw_null")]
    body: Box<RawValue>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponseEnvelope {
    v: u16,
    id: String,
    auth: String,
    status: u16,
    body: Value,
}

fn raw_null() -> Box<RawValue> {
    RawValue::from_string("null".to_string()).expect("null is valid JSON")
}

#[derive(Clone)]
struct ServerAuth {
    nonce: [u8; NONCE_BYTES],
}

impl ServerAuth {
    fn new() -> Self {
        let mut nonce = [0_u8; NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce);
        Self { nonce }
    }

    #[cfg(test)]
    fn fixed() -> Self {
        Self {
            nonce: [0x5a; NONCE_BYTES],
        }
    }
}

/// Reusable client for the local Shodh process.
#[derive(Clone)]
pub struct IpcClient {
    endpoint: PathBuf,
    api_key: String,
    server_nonce: Arc<RwLock<Option<[u8; NONCE_BYTES]>>>,
}

impl IpcClient {
    pub fn new(endpoint: PathBuf, api_key: String) -> Self {
        Self {
            endpoint,
            api_key,
            server_nonce: Arc::new(RwLock::new(None)),
        }
    }

    pub fn endpoint(&self) -> &Path {
        &self.endpoint
    }

    pub async fn get<R: DeserializeOwned>(&self, path: &str) -> Result<R, String> {
        self.request("GET", path, Value::Null, DEFAULT_CLIENT_TIMEOUT)
            .await
    }

    pub async fn post<T: Serialize, R: DeserializeOwned>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<R, String> {
        let body = serde_json::to_value(body)
            .map_err(|error| format!("failed to encode IPC request body: {error}"))?;
        self.request("POST", path, body, DEFAULT_WRITE_TIMEOUT)
            .await
    }

    pub async fn request<R: DeserializeOwned>(
        &self,
        method: &str,
        path: &str,
        body: Value,
        timeout: Duration,
    ) -> Result<R, String> {
        tokio::time::timeout(timeout, async {
            let response = if method == "GET" && path == "/health" {
                self.probe().await?
            } else {
                let nonce = match *self.server_nonce.read().await {
                    Some(nonce) => nonce,
                    None => {
                        self.probe().await?;
                        self.server_nonce.read().await.ok_or_else(|| {
                            "IPC server proof did not establish a nonce".to_string()
                        })?
                    }
                };
                self.signed_request(method, path, body, nonce).await?
            };
            serde_json::from_value(response)
                .map_err(|error| format!("failed to decode IPC response body: {error}"))
        })
        .await
        .map_err(|_| {
            format!(
                "Shodh IPC request timed out after {} ms",
                timeout.as_millis()
            )
        })?
    }

    async fn probe(&self) -> Result<Value, String> {
        let id = Uuid::new_v4().to_string();
        let challenge = random_nonce_hex();
        let request = RequestEnvelope {
            v: PROTOCOL_VERSION,
            id: id.clone(),
            auth: String::new(),
            challenge: challenge.clone(),
            method: "GET".to_string(),
            path: "/health".to_string(),
            body: raw_null(),
        };
        let encoded = encode_request(&request)?;
        let frame = exchange_frame(&self.endpoint, &encoded).await?;
        let response = decode_response(&frame, &id)?;
        let nonce = verify_probe_response(&self.api_key, &challenge, &response)?;
        *self.server_nonce.write().await = Some(nonce);
        response_body(response)
    }

    async fn signed_request(
        &self,
        method: &str,
        path: &str,
        body: Value,
        nonce: [u8; NONCE_BYTES],
    ) -> Result<Value, String> {
        let id = Uuid::new_v4().to_string();
        let raw_body = serde_json::value::to_raw_value(&body)
            .map_err(|error| format!("failed to encode IPC request body: {error}"))?;
        let auth = request_auth(
            &self.api_key,
            &nonce,
            PROTOCOL_VERSION,
            &id,
            method,
            path,
            raw_body.get().as_bytes(),
        );
        let request = RequestEnvelope {
            v: PROTOCOL_VERSION,
            id: id.clone(),
            auth: auth.clone(),
            challenge: String::new(),
            method: method.to_owned(),
            path: path.to_owned(),
            body: raw_body,
        };
        let encoded = encode_request(&request)?;
        let frame = exchange_frame(&self.endpoint, &encoded).await?;
        let response = decode_response(&frame, &id)?;
        verify_response(&self.api_key, &nonce, &auth, &response)?;
        response_body(response)
    }
}

fn encode_request(request: &RequestEnvelope) -> Result<Vec<u8>, String> {
    let mut encoded = serde_json::to_vec(request)
        .map_err(|error| format!("failed to encode IPC request: {error}"))?;
    if encoded.len() + 1 > MAX_FRAME_BYTES {
        return Err(format!(
            "IPC request exceeds the {MAX_FRAME_BYTES}-byte frame limit"
        ));
    }
    encoded.push(b'\n');
    Ok(encoded)
}

/// Exchange one already-encoded frame through the platform-safe Rust transport.
///
/// The bundled server binary exposes this for the TypeScript client on Windows,
/// where Node cannot set named-pipe SQOS flags itself. The frame is still decoded
/// and authenticated by the TypeScript caller.
pub async fn exchange_encoded(
    endpoint: &Path,
    encoded: &[u8],
    timeout: Duration,
) -> Result<Vec<u8>, String> {
    if encoded.len() > MAX_FRAME_BYTES
        || encoded.last() != Some(&b'\n')
        || encoded[..encoded.len().saturating_sub(1)].contains(&b'\n')
    {
        return Err("IPC helper requires exactly one bounded newline-delimited frame".to_string());
    }
    tokio::time::timeout(timeout, exchange_frame(endpoint, encoded))
        .await
        .map_err(|_| {
            format!(
                "Shodh IPC request timed out after {} ms",
                timeout.as_millis()
            )
        })?
}

async fn exchange_frame(endpoint: &Path, encoded: &[u8]) -> Result<Vec<u8>, String> {
    let mut stream = platform::connect(endpoint).await?;
    stream
        .write_all(encoded)
        .await
        .map_err(|error| format!("failed to write IPC request: {error}"))?;
    stream
        .flush()
        .await
        .map_err(|error| format!("failed to flush IPC request: {error}"))?;
    let frame = read_frame(&mut stream, FrameSide::Response)
        .await
        .map_err(|error| error.message)?;
    require_response_close(&mut stream).await?;
    Ok(frame)
}

fn decode_response(frame: &[u8], expected_id: &str) -> Result<ResponseEnvelope, String> {
    let response: ResponseEnvelope =
        serde_json::from_slice(frame).map_err(|error| format!("invalid IPC response: {error}"))?;
    if response.v != PROTOCOL_VERSION {
        return Err(format!(
            "unsupported IPC response version {}; expected {PROTOCOL_VERSION}",
            response.v
        ));
    }
    if !(100..=599).contains(&response.status) {
        return Err("IPC response status was invalid".to_string());
    }
    if !response.id.is_empty() && response.id != expected_id {
        return Err("IPC response request ID did not match".to_string());
    }
    Ok(response)
}

fn response_body(response: ResponseEnvelope) -> Result<Value, String> {
    if !(200..300).contains(&response.status) {
        return Err(format!(
            "Shodh IPC error {}: {}",
            response.status,
            compact_json(&response.body)
        ));
    }
    Ok(response.body)
}

fn random_nonce_hex() -> String {
    let mut nonce = [0_u8; NONCE_BYTES];
    OsRng.fill_bytes(&mut nonce);
    hex::encode(nonce)
}

fn mac_hex(key: &str, domain: &[u8], fields: &[&[u8]]) -> String {
    let mut mac = HmacSha256::new_from_slice(key.as_bytes()).expect("HMAC accepts any key size");
    update_mac_field(&mut mac, domain);
    for field in fields {
        update_mac_field(&mut mac, field);
    }
    hex::encode(mac.finalize().into_bytes())
}

fn verify_mac(key: &str, domain: &[u8], fields: &[&[u8]], encoded: &str) -> bool {
    let Ok(expected) = hex::decode(encoded) else {
        return false;
    };
    let mut mac = HmacSha256::new_from_slice(key.as_bytes()).expect("HMAC accepts any key size");
    update_mac_field(&mut mac, domain);
    for field in fields {
        update_mac_field(&mut mac, field);
    }
    mac.verify_slice(&expected).is_ok()
}

fn update_mac_field(mac: &mut HmacSha256, field: &[u8]) {
    mac.update(&(field.len() as u64).to_be_bytes());
    mac.update(field);
}

fn request_auth(
    key: &str,
    nonce: &[u8; NONCE_BYTES],
    version: u16,
    id: &str,
    method: &str,
    path: &str,
    body: &[u8],
) -> String {
    let version = version.to_be_bytes();
    let proof = mac_hex(
        key,
        REQUEST_DOMAIN,
        &[
            nonce,
            &version,
            id.as_bytes(),
            method.as_bytes(),
            path.as_bytes(),
            body,
        ],
    );
    format!("request:{}:{proof}", hex::encode(nonce))
}

fn verify_probe_response(
    key: &str,
    challenge: &str,
    response: &ResponseEnvelope,
) -> Result<[u8; NONCE_BYTES], String> {
    let mut parts = response.auth.splitn(3, ':');
    if parts.next() != Some("probe") {
        return Err("IPC health response did not authenticate the server".to_string());
    }
    let nonce_hex = parts
        .next()
        .ok_or_else(|| "IPC health response omitted the server nonce".to_string())?;
    let nonce = decode_nonce(nonce_hex)
        .ok_or_else(|| "IPC health response carried an invalid server nonce".to_string())?;
    let status = response.status.to_be_bytes();
    let expected = mac_hex(
        key,
        PROBE_RESPONSE_DOMAIN,
        &[
            response.id.as_bytes(),
            challenge.as_bytes(),
            &nonce,
            &status,
        ],
    );
    let proofs = parts
        .next()
        .ok_or_else(|| "IPC health response omitted its server proof".to_string())?;
    if !proofs
        .split(',')
        .any(|proof| auth::constant_time_compare(proof, &expected))
    {
        return Err("IPC health response was not signed by the configured server".to_string());
    }
    Ok(nonce)
}

fn verify_response(
    key: &str,
    nonce: &[u8; NONCE_BYTES],
    request_auth: &str,
    response: &ResponseEnvelope,
) -> Result<(), String> {
    let proof = response
        .auth
        .strip_prefix("response:")
        .ok_or_else(|| "IPC response did not authenticate the server".to_string())?;
    let status = response.status.to_be_bytes();
    if !verify_mac(
        key,
        RESPONSE_DOMAIN,
        &[
            nonce,
            request_auth.as_bytes(),
            response.id.as_bytes(),
            &status,
        ],
        proof,
    ) {
        return Err("IPC response server proof was invalid".to_string());
    }
    Ok(())
}

fn decode_nonce(encoded: &str) -> Option<[u8; NONCE_BYTES]> {
    let bytes = hex::decode(encoded).ok()?;
    bytes.try_into().ok()
}

impl fmt::Debug for IpcClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IpcClient")
            .field("endpoint", &self.endpoint)
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

/// Bound local listener. Binding validates endpoint ownership and permissions.
pub struct LocalIpcServer {
    listener: platform::Listener,
    endpoint: PathBuf,
    auth: Arc<ServerAuth>,
}

impl LocalIpcServer {
    pub async fn bind(endpoint: PathBuf) -> Result<Self, String> {
        let listener = platform::bind(&endpoint).await?;
        Ok(Self {
            listener,
            endpoint,
            auth: Arc::new(ServerAuth::new()),
        })
    }

    pub fn endpoint(&self) -> &Path {
        &self.endpoint
    }

    pub async fn serve(
        self,
        router: Router,
        shutdown: CancellationToken,
        max_concurrent: usize,
        request_timeout: Duration,
    ) -> Result<(), String> {
        platform::serve(
            self.listener,
            router,
            shutdown,
            self.auth,
            Arc::new(Semaphore::new(max_concurrent.max(1))),
            Arc::new(Semaphore::new(
                max_concurrent.clamp(1, MAX_PRE_AUTH_CONNECTIONS),
            )),
            request_timeout,
        )
        .await
    }
}

async fn handle_connection<S>(
    mut stream: S,
    router: Router,
    server_auth: Arc<ServerAuth>,
    dispatch_semaphore: Arc<Semaphore>,
    pre_auth_permit: OwnedSemaphorePermit,
    request_timeout: Duration,
) where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let read_timeout = PRE_AUTH_READ_TIMEOUT.min(request_timeout);
    let frame =
        match tokio::time::timeout(read_timeout, read_frame(&mut stream, FrameSide::Request)).await
        {
            Ok(Ok(frame)) => frame,
            Ok(Err(error)) => {
                let response = protocol_error("", error.status, error.code, error.message);
                write_response(&mut stream, response, request_timeout).await;
                return;
            }
            Err(_) => {
                let response = protocol_error(
                    "",
                    StatusCode::REQUEST_TIMEOUT,
                    "READ_TIMEOUT",
                    "IPC pre-auth request read timed out",
                );
                write_response(&mut stream, response, request_timeout).await;
                return;
            }
        };

    let prepared = prepare_frame(&frame, &server_auth).await;
    let prepared = match prepared {
        PreparedFrame::Respond(response) => {
            write_response(&mut stream, response, request_timeout).await;
            return;
        }
        PreparedFrame::Dispatch(prepared) => *prepared,
    };
    let response_id = prepared.id.clone();
    let signer = prepared.signer.clone();
    let dispatch_permit = match tokio::time::timeout(
        request_timeout,
        Arc::clone(&dispatch_semaphore).acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        Ok(Err(_)) => {
            let response = signed_protocol_error(
                &response_id,
                StatusCode::SERVICE_UNAVAILABLE,
                "DISPATCH_UNAVAILABLE",
                "IPC dispatch limiter closed",
                &signer,
                &server_auth,
            );
            write_response(&mut stream, response, request_timeout).await;
            return;
        }
        Err(_) => {
            let response = signed_protocol_error(
                &response_id,
                StatusCode::REQUEST_TIMEOUT,
                "DISPATCH_QUEUE_TIMEOUT",
                "IPC request timed out waiting for a dispatch slot",
                &signer,
                &server_auth,
            );
            write_response(&mut stream, response, request_timeout).await;
            return;
        }
    };
    drop(pre_auth_permit);

    let dispatch_auth = Arc::clone(&server_auth);
    let mut dispatch = tokio::spawn(async move {
        let _permit = dispatch_permit;
        dispatch_prepared(prepared, router, &dispatch_auth).await
    });
    let response = match tokio::time::timeout(request_timeout, &mut dispatch).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => signed_protocol_error(
            &response_id,
            StatusCode::INTERNAL_SERVER_ERROR,
            "DISPATCH_FAILED",
            format!("internal route dispatch task failed: {error}"),
            &signer,
            &server_auth,
        ),
        Err(_) => {
            let response = signed_protocol_error(
                &response_id,
                StatusCode::REQUEST_TIMEOUT,
                "DISPATCH_TIMEOUT",
                "IPC route dispatch timed out",
                &signer,
                &server_auth,
            );
            write_response(&mut stream, response, request_timeout).await;
            drop(stream);
            // Tokio blocking jobs cannot be aborted. Retain ownership of the
            // route task so server shutdown cannot flush storage concurrently
            // with work that outlived the client deadline.
            if let Err(error) = dispatch.await {
                tracing::warn!("Timed-out Shodh IPC dispatch task failed: {error}");
            }
            return;
        }
    };
    write_response(&mut stream, response, request_timeout).await;
}

async fn write_response<S>(stream: &mut S, response: ResponseEnvelope, write_timeout: Duration)
where
    S: AsyncWrite + Unpin,
{
    let write = async {
        let mut encoded = match serde_json::to_vec(&response) {
            Ok(encoded) if encoded.len() < MAX_FRAME_BYTES => encoded,
            Ok(_) => {
                // The encoded envelope (body plus JSON-escaping and framing overhead)
                // overflowed the frame limit even though the body passed its own
                // MAX_BODY_BYTES check. Report a small 413 rather than dropping the
                // response and leaving the client to interpret a bare EOF.
                tracing::error!("Shodh IPC response exceeded its frame limit; reporting 413");
                let replacement = protocol_error(
                    &response.id,
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "RESPONSE_TOO_LARGE",
                    "route response exceeded the IPC frame limit",
                );
                match serde_json::to_vec(&replacement) {
                    Ok(encoded) => encoded,
                    Err(error) => {
                        tracing::error!("Failed to serialize Shodh IPC 413 response: {error}");
                        return;
                    }
                }
            }
            Err(error) => {
                tracing::error!("Failed to serialize Shodh IPC response: {error}");
                return;
            }
        };
        encoded.push(b'\n');
        if let Err(error) = stream.write_all(&encoded).await {
            tracing::debug!("Failed to write Shodh IPC response: {error}");
            return;
        }
        let _ = stream.flush().await;
        let _ = stream.shutdown().await;
    };
    if tokio::time::timeout(write_timeout, write).await.is_err() {
        tracing::warn!("Shodh IPC response write timed out");
    }
}

/// Best-effort recovery of the correlation id from a frame whose envelope may have
/// failed to deserialize, so a protocol-error response can still echo it. Returns
/// "" only when the frame is not even parseable as JSON carrying a string `id`.
fn recover_frame_id(frame: &[u8]) -> String {
    #[derive(Deserialize)]
    struct RecoverableId {
        id: Option<String>,
    }

    serde_json::from_slice::<RecoverableId>(frame)
        .ok()
        .and_then(|value| value.id)
        .unwrap_or_default()
}

#[derive(Clone)]
enum ResponseSigner {
    None,
    Probe { challenge: String },
    Request { key: String, request_auth: String },
}

struct PreparedDispatch {
    id: String,
    request: Request<Body>,
    signer: ResponseSigner,
}

enum PreparedFrame {
    Respond(ResponseEnvelope),
    Dispatch(Box<PreparedDispatch>),
}

async fn prepare_frame(frame: &[u8], server_auth: &ServerAuth) -> PreparedFrame {
    let request: RequestEnvelope = match serde_json::from_slice(frame) {
        Ok(request) => request,
        Err(error) => {
            return PreparedFrame::Respond(protocol_error(
                &recover_frame_id(frame),
                StatusCode::BAD_REQUEST,
                "INVALID_REQUEST",
                format!("invalid IPC request: {error}"),
            ));
        }
    };

    let id = request.id.clone();
    if request.v != PROTOCOL_VERSION {
        return PreparedFrame::Respond(protocol_error(
            &id,
            StatusCode::BAD_REQUEST,
            "UNSUPPORTED_VERSION",
            format!(
                "unsupported IPC protocol version {}; expected {PROTOCOL_VERSION}",
                request.v
            ),
        ));
    }
    if request.id.is_empty() || request.id.len() > 128 {
        return PreparedFrame::Respond(protocol_error(
            "",
            StatusCode::BAD_REQUEST,
            "INVALID_REQUEST_ID",
            "request ID must contain between 1 and 128 bytes",
        ));
    }

    let is_health_probe = request.method == "GET" && request.path == "/health";
    let signer = if is_health_probe {
        if !request.auth.is_empty() {
            return PreparedFrame::Respond(protocol_error(
                &id,
                StatusCode::BAD_REQUEST,
                "HEALTH_AUTH_NOT_EMPTY",
                "IPC health probes must not carry authentication material",
            ));
        }
        if request.challenge.is_empty() {
            ResponseSigner::None
        } else if decode_nonce(&request.challenge).is_some() {
            ResponseSigner::Probe {
                challenge: request.challenge.clone(),
            }
        } else {
            return PreparedFrame::Respond(protocol_error(
                &id,
                StatusCode::BAD_REQUEST,
                "INVALID_CHALLENGE",
                "IPC health challenge must be a 32-byte hexadecimal nonce",
            ));
        }
    } else {
        if request.auth.is_empty() {
            return PreparedFrame::Respond(protocol_error(
                &id,
                StatusCode::UNAUTHORIZED,
                "MISSING_API_KEY",
                "missing IPC request proof",
            ));
        }
        if !request.challenge.is_empty() {
            return PreparedFrame::Respond(protocol_error(
                &id,
                StatusCode::BAD_REQUEST,
                "UNEXPECTED_CHALLENGE",
                "ordinary IPC requests must not carry a health challenge",
            ));
        }
        let keys = match auth::configured_api_keys() {
            Ok(keys) => keys,
            Err(error) => {
                return PreparedFrame::Respond(
                    envelope_from_response(id, error.into_response()).await,
                )
            }
        };
        let key = keys
            .into_iter()
            .find(|key| verify_request_auth(key, server_auth, &request));
        match key {
            Some(key) => ResponseSigner::Request {
                key,
                request_auth: request.auth.clone(),
            },
            None => {
                return PreparedFrame::Respond(
                    envelope_from_response(id, AuthError::InvalidApiKey.into_response()).await,
                )
            }
        }
    };

    let uri: Uri = match request.path.parse::<Uri>() {
        Ok(uri) if uri.scheme().is_none() && uri.authority().is_none() => uri,
        _ => {
            return PreparedFrame::Respond(signed_protocol_error(
                &id,
                StatusCode::BAD_REQUEST,
                "INVALID_PATH",
                "path must be an origin-form URI",
                &signer,
                server_auth,
            ));
        }
    };
    let path = uri.path();
    if !is_health_probe && !path.starts_with("/api/") {
        return PreparedFrame::Respond(signed_protocol_error(
            &id,
            StatusCode::FORBIDDEN,
            "PATH_NOT_ALLOWED",
            "IPC exposes only /health and ordinary /api routes",
            &signer,
            server_auth,
        ));
    }

    let method: Method = match request.method.parse() {
        Ok(method)
            if matches!(
                method,
                Method::GET | Method::POST | Method::PUT | Method::PATCH | Method::DELETE
            ) =>
        {
            method
        }
        _ => {
            return PreparedFrame::Respond(signed_protocol_error(
                &id,
                StatusCode::METHOD_NOT_ALLOWED,
                "METHOD_NOT_ALLOWED",
                "IPC supports GET, POST, PUT, PATCH, and DELETE",
                &signer,
                server_auth,
            ));
        }
    };

    if PUBLIC_API_PATHS.contains(&path) {
        return PreparedFrame::Respond(signed_protocol_error(
            &id,
            StatusCode::FORBIDDEN,
            "PATH_NOT_ALLOWED",
            "public HTTP routes are not exposed over local IPC",
            &signer,
            server_auth,
        ));
    }

    if STREAMING_PATHS.contains(&path) {
        return PreparedFrame::Respond(signed_protocol_error(
            &id,
            StatusCode::NOT_IMPLEMENTED,
            "STREAMING_NOT_SUPPORTED",
            "streaming routes require HTTP WebSocket or server-sent events",
            &signer,
            server_auth,
        ));
    }

    let raw_body = request.body.get();
    let body = if raw_body.trim() == "null" {
        Body::empty()
    } else if raw_body.len() <= MAX_BODY_BYTES {
        Body::from(raw_body.as_bytes().to_vec())
    } else {
        return PreparedFrame::Respond(signed_protocol_error(
            &id,
            StatusCode::PAYLOAD_TOO_LARGE,
            "BODY_TOO_LARGE",
            "request body exceeds the IPC body limit",
            &signer,
            server_auth,
        ));
    };

    let internal_request = match Request::builder()
        .method(method)
        .uri(uri)
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(body)
    {
        Ok(request) => request,
        Err(error) => {
            return PreparedFrame::Respond(signed_protocol_error(
                &id,
                StatusCode::BAD_REQUEST,
                "INVALID_REQUEST",
                format!("failed to construct internal request: {error}"),
                &signer,
                server_auth,
            ));
        }
    };

    PreparedFrame::Dispatch(Box::new(PreparedDispatch {
        id,
        request: internal_request,
        signer,
    }))
}

fn verify_request_auth(key: &str, server_auth: &ServerAuth, request: &RequestEnvelope) -> bool {
    let mut parts = request.auth.splitn(3, ':');
    if parts.next() != Some("request") {
        return false;
    }
    let Some(nonce_hex) = parts.next() else {
        return false;
    };
    let Some(proof) = parts.next() else {
        return false;
    };
    let Some(nonce) = decode_nonce(nonce_hex) else {
        return false;
    };
    if nonce != server_auth.nonce {
        return false;
    }
    let version = request.v.to_be_bytes();
    verify_mac(
        key,
        REQUEST_DOMAIN,
        &[
            &nonce,
            &version,
            request.id.as_bytes(),
            request.method.as_bytes(),
            request.path.as_bytes(),
            request.body.get().as_bytes(),
        ],
        proof,
    )
}

async fn dispatch_prepared(
    prepared: PreparedDispatch,
    router: Router,
    server_auth: &ServerAuth,
) -> ResponseEnvelope {
    let mut response = match router.oneshot(prepared.request).await {
        Ok(response) => envelope_from_response(prepared.id, response).await,
        Err(error) => protocol_error(
            &prepared.id,
            StatusCode::INTERNAL_SERVER_ERROR,
            "DISPATCH_FAILED",
            format!("internal route dispatch failed: {error}"),
        ),
    };
    sign_response(&mut response, &prepared.signer, server_auth);
    response
}

#[cfg(test)]
async fn process_frame(frame: &[u8], router: Router) -> ResponseEnvelope {
    let server_auth = ServerAuth::fixed();
    match prepare_frame(frame, &server_auth).await {
        PreparedFrame::Respond(response) => response,
        PreparedFrame::Dispatch(prepared) => {
            dispatch_prepared(*prepared, router, &server_auth).await
        }
    }
}

async fn envelope_from_response(id: String, response: Response) -> ResponseEnvelope {
    let status = response.status();
    let body = match to_bytes(response.into_body(), MAX_BODY_BYTES).await {
        Ok(bytes) if bytes.is_empty() => Value::Null,
        Ok(bytes) => serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned())),
        Err(error) => {
            return protocol_error(
                &id,
                StatusCode::PAYLOAD_TOO_LARGE,
                "RESPONSE_TOO_LARGE",
                format!("route response exceeded the IPC body limit: {error}"),
            );
        }
    };

    ResponseEnvelope {
        v: PROTOCOL_VERSION,
        id,
        auth: String::new(),
        status: status.as_u16(),
        body,
    }
}

fn signed_protocol_error(
    id: &str,
    status: StatusCode,
    code: &'static str,
    message: impl Into<String>,
    signer: &ResponseSigner,
    server_auth: &ServerAuth,
) -> ResponseEnvelope {
    let mut response = protocol_error(id, status, code, message);
    sign_response(&mut response, signer, server_auth);
    response
}

fn sign_response(
    response: &mut ResponseEnvelope,
    signer: &ResponseSigner,
    server_auth: &ServerAuth,
) {
    let status = response.status.to_be_bytes();
    match signer {
        ResponseSigner::None => {}
        ResponseSigner::Probe { challenge } => {
            let Ok(keys) = auth::configured_api_keys() else {
                return;
            };
            let proofs = keys
                .iter()
                .map(|key| {
                    mac_hex(
                        key,
                        PROBE_RESPONSE_DOMAIN,
                        &[
                            response.id.as_bytes(),
                            challenge.as_bytes(),
                            &server_auth.nonce,
                            &status,
                        ],
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            response.auth = format!("probe:{}:{proofs}", hex::encode(server_auth.nonce));
        }
        ResponseSigner::Request { key, request_auth } => {
            let proof = mac_hex(
                key,
                RESPONSE_DOMAIN,
                &[
                    &server_auth.nonce,
                    request_auth.as_bytes(),
                    response.id.as_bytes(),
                    &status,
                ],
            );
            response.auth = format!("response:{proof}");
        }
    }
}

fn protocol_error(
    id: &str,
    status: StatusCode,
    code: &'static str,
    message: impl Into<String>,
) -> ResponseEnvelope {
    ResponseEnvelope {
        v: PROTOCOL_VERSION,
        id: id.to_owned(),
        auth: String::new(),
        status: status.as_u16(),
        body: json!({
            "code": code,
            "message": message.into(),
        }),
    }
}

struct FrameError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

/// Which side's frame is being read, so error messages name the right noun. The
/// same `read_frame` reads requests on the server and responses on the client; a
/// bare "request" wording would be wrong when surfaced to the caller for a
/// malformed *response*.
#[derive(Clone, Copy)]
enum FrameSide {
    Request,
    Response,
}

impl FrameSide {
    fn noun(self) -> &'static str {
        match self {
            FrameSide::Request => "request",
            FrameSide::Response => "response",
        }
    }
}

async fn read_frame<R: AsyncRead + Unpin>(
    reader: &mut R,
    side: FrameSide,
) -> Result<Vec<u8>, FrameError> {
    let mut frame = Vec::with_capacity(4096);
    let mut buffer = [0_u8; 8192];

    loop {
        let count = reader.read(&mut buffer).await.map_err(|error| FrameError {
            status: StatusCode::BAD_REQUEST,
            code: "READ_FAILED",
            message: format!("failed to read IPC frame: {error}"),
        })?;
        if count == 0 {
            return Err(FrameError {
                status: StatusCode::BAD_REQUEST,
                code: "INCOMPLETE_FRAME",
                message: "IPC connection closed before a newline-delimited frame arrived"
                    .to_string(),
            });
        }

        if let Some(newline) = buffer[..count].iter().position(|byte| *byte == b'\n') {
            if frame.len() + newline + 1 > MAX_FRAME_BYTES {
                return Err(frame_too_large());
            }
            if newline + 1 != count {
                return Err(FrameError {
                    status: StatusCode::BAD_REQUEST,
                    code: "MULTIPLE_FRAMES",
                    message: format!(
                        "IPC accepts exactly one newline-delimited {} per connection",
                        side.noun()
                    ),
                });
            }
            frame.extend_from_slice(&buffer[..newline]);
            return Ok(frame);
        }

        if frame.len() + count >= MAX_FRAME_BYTES {
            return Err(frame_too_large());
        }
        frame.extend_from_slice(&buffer[..count]);
    }
}

async fn require_response_close<R: AsyncRead + Unpin>(reader: &mut R) -> Result<(), String> {
    let mut trailing = [0_u8; 1];
    match reader.read(&mut trailing).await {
        Ok(0) => Ok(()),
        Ok(_) => Err("Shodh IPC response contained data after its newline delimiter".to_string()),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
            ) =>
        {
            Ok(())
        }
        Err(error) => Err(format!("failed while closing Shodh IPC response: {error}")),
    }
}

fn frame_too_large() -> FrameError {
    FrameError {
        status: StatusCode::PAYLOAD_TOO_LARGE,
        code: "FRAME_TOO_LARGE",
        message: format!("IPC frame exceeds the {MAX_FRAME_BYTES}-byte limit"),
    }
}

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "unreadable error response".to_string())
}

#[cfg(unix)]
mod platform {
    use super::*;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
    use tokio::net::{UnixListener, UnixStream};

    const MAX_SOCKET_PATH_BYTES: usize = 103;

    pub struct Listener {
        inner: UnixListener,
        guard: EndpointGuard,
    }

    struct EndpointGuard {
        path: PathBuf,
        device: u64,
        inode: u64,
    }

    impl Drop for EndpointGuard {
        fn drop(&mut self) {
            if let Ok(metadata) = std::fs::symlink_metadata(&self.path) {
                if metadata.file_type().is_socket()
                    && metadata.dev() == self.device
                    && metadata.ino() == self.inode
                {
                    let _ = std::fs::remove_file(&self.path);
                }
            }
        }
    }

    pub async fn bind(path: &Path) -> Result<Listener, String> {
        if path.as_os_str().as_bytes().len() > MAX_SOCKET_PATH_BYTES {
            return Err(format!(
                "IPC socket path exceeds the portable {MAX_SOCKET_PATH_BYTES}-byte limit"
            ));
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .ok_or_else(|| "IPC socket path must have a parent directory".to_string())?;
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder
            .create(parent)
            .map_err(|error| format!("failed to create IPC directory: {error}"))?;
        let parent_handle = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(parent)
            .map_err(|error| format!("failed to open IPC directory safely: {error}"))?;
        let metadata = parent_handle
            .metadata()
            .map_err(|error| format!("failed to inspect IPC directory: {error}"))?;
        // SAFETY: geteuid has no preconditions and does not access Rust memory.
        let current_uid = unsafe { libc::geteuid() };
        if !metadata.file_type().is_dir() || metadata.uid() != current_uid {
            return Err(
                "IPC directory must be a non-symlink directory owned by the current user"
                    .to_string(),
            );
        }
        // The directory may pre-exist at a looser mode than DirBuilder creates with
        // — most notably `shodh init` makes the data dir 0755, and on macOS that is
        // the *same* directory as this endpoint's parent. We own it and it is a real
        // directory, so tighten it to 0700 rather than refusing to boot the daemon.
        if metadata.permissions().mode() & 0o777 != 0o700 {
            parent_handle
                .set_permissions(std::fs::Permissions::from_mode(0o700))
                .map_err(|error| format!("failed to secure IPC directory: {error}"))?;
        }

        if let Ok(metadata) = std::fs::symlink_metadata(path) {
            if !metadata.file_type().is_socket() {
                return Err("refusing to replace a non-socket IPC endpoint".to_string());
            }
            // SAFETY: geteuid has no preconditions and does not access Rust memory.
            if metadata.uid() != unsafe { libc::geteuid() } {
                return Err("refusing to replace an IPC socket owned by another user".to_string());
            }
            match tokio::time::timeout(Duration::from_millis(250), UnixStream::connect(path)).await
            {
                Ok(Ok(_)) => return Err("Shodh IPC endpoint is already serving".to_string()),
                Ok(Err(error))
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
                    ) => {}
                Ok(Err(error)) => {
                    return Err(format!("failed to probe existing IPC socket: {error}"));
                }
                Err(_) => {
                    return Err("existing IPC socket did not answer the ownership probe".to_string())
                }
            }
            let current = std::fs::symlink_metadata(path)
                .map_err(|error| format!("failed to recheck stale IPC socket: {error}"))?;
            if !current.file_type().is_socket()
                || current.dev() != metadata.dev()
                || current.ino() != metadata.ino()
            {
                return Err("IPC endpoint changed during the stale-socket check".to_string());
            }
            std::fs::remove_file(path)
                .map_err(|error| format!("failed to remove stale IPC socket: {error}"))?;
        }

        let inner = UnixListener::bind(path)
            .map_err(|error| format!("failed to bind IPC socket: {error}"))?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("failed to protect IPC socket: {error}"))?;
        let metadata = std::fs::symlink_metadata(path)
            .map_err(|error| format!("failed to inspect bound IPC socket: {error}"))?;
        Ok(Listener {
            inner,
            guard: EndpointGuard {
                path: path.to_owned(),
                device: metadata.dev(),
                inode: metadata.ino(),
            },
        })
    }

    pub async fn connect(path: &Path) -> Result<UnixStream, String> {
        let stream = UnixStream::connect(path)
            .await
            .map_err(|error| format!("failed to connect to Shodh IPC socket: {error}"))?;
        verify_peer(&stream, "server")?;
        Ok(stream)
    }

    fn verify_peer(stream: &UnixStream, role: &str) -> Result<(), String> {
        let credentials = stream
            .peer_cred()
            .map_err(|error| format!("failed to inspect IPC {role} credentials: {error}"))?;
        // SAFETY: geteuid has no preconditions and does not access Rust memory.
        let current_uid = unsafe { libc::geteuid() };
        if credentials.uid() != current_uid {
            return Err(format!(
                "refusing IPC {role} owned by uid {}; expected {current_uid}",
                credentials.uid()
            ));
        }
        Ok(())
    }

    pub async fn serve(
        listener: Listener,
        router: Router,
        shutdown: CancellationToken,
        server_auth: Arc<ServerAuth>,
        dispatch_semaphore: Arc<Semaphore>,
        pre_auth_semaphore: Arc<Semaphore>,
        request_timeout: Duration,
    ) -> Result<(), String> {
        let Listener { inner, guard } = listener;
        let _guard = guard;
        let mut connections = JoinSet::new();

        let terminal_error = loop {
            let pre_auth_permit = tokio::select! {
                _ = shutdown.cancelled() => break None,
                permit = Arc::clone(&pre_auth_semaphore).acquire_owned() => match permit {
                    Ok(permit) => permit,
                    Err(_) => break Some("IPC pre-auth limiter closed".to_string()),
                },
            };
            tokio::select! {
                _ = shutdown.cancelled() => break None,
                accepted = inner.accept() => {
                    let (stream, _) = match accepted {
                        Ok(connection) => connection,
                        Err(error) => {
                            // A transient accept() error (fd pressure such as
                            // EMFILE/ENFILE, or a peer that reset between readiness
                            // and accept) must NOT tear down the shared process —
                            // the HTTP server lives in the same task tree. Log, free
                            // the permit, back off briefly to avoid a hot spin, and
                            // keep serving.
                            tracing::warn!("Shodh IPC accept failed, continuing: {error}");
                            drop(pre_auth_permit);
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            continue;
                        }
                    };
                    if let Err(error) = verify_peer(&stream, "client") {
                        tracing::warn!("Rejected Shodh IPC client: {error}");
                        drop(pre_auth_permit);
                        continue;
                    }
                    let connection_router = router.clone();
                    let connection_auth = Arc::clone(&server_auth);
                    let connection_dispatch = Arc::clone(&dispatch_semaphore);
                    connections.spawn(async move {
                        handle_connection(
                            stream,
                            connection_router,
                            connection_auth,
                            connection_dispatch,
                            pre_auth_permit,
                            request_timeout,
                        )
                        .await;
                    });
                }
            }

            while let Some(result) = connections.try_join_next() {
                if let Err(error) = result {
                    tracing::warn!("Shodh IPC connection task failed: {error}");
                }
            }
        };

        while let Some(result) = connections.join_next().await {
            if let Err(error) = result {
                tracing::warn!("Shodh IPC connection task failed while draining: {error}");
            }
        }
        match terminal_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::{ffi::c_void, mem::size_of, ptr};
    use tokio::net::windows::named_pipe::{
        ClientOptions, NamedPipeClient, NamedPipeServer, ServerOptions,
    };
    use windows_sys::{
        core::PWSTR,
        Win32::{
            Foundation::{CloseHandle, GetLastError, LocalFree, ERROR_INSUFFICIENT_BUFFER, HANDLE},
            Security::{
                Authorization::{
                    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
                    SDDL_REVISION_1,
                },
                GetTokenInformation, TokenUser, PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES,
                TOKEN_QUERY, TOKEN_USER,
            },
            Storage::FileSystem::SECURITY_IDENTIFICATION,
            System::{
                Pipes::{GetNamedPipeClientProcessId, GetNamedPipeServerProcessId},
                Threading::{
                    GetCurrentProcess, OpenProcess, OpenProcessToken,
                    PROCESS_QUERY_LIMITED_INFORMATION,
                },
            },
        },
    };

    #[cfg(test)]
    use windows_sys::Win32::Security::{
        Authorization::{
            ConvertSecurityDescriptorToStringSecurityDescriptorW, GetSecurityInfo, SE_KERNEL_OBJECT,
        },
        GetSecurityDescriptorControl, DACL_SECURITY_INFORMATION, SE_DACL_PROTECTED,
    };

    const ERROR_PIPE_BUSY_CODE: i32 = 231;

    /// Per-user default pipe name. Falls back to the bare name only if the SID
    /// lookup fails (it must stay infallible for clap's `default_value_os_t`).
    pub(super) fn default_endpoint() -> PathBuf {
        match current_user_sid_string() {
            Ok(sid) => PathBuf::from(format!(r"\\.\pipe\shodh-memory-{sid}")),
            Err(_) => PathBuf::from(r"\\.\pipe\shodh-memory"),
        }
    }

    pub struct Listener {
        endpoint: PathBuf,
        next: Option<NamedPipeServer>,
    }

    pub async fn bind(path: &Path) -> Result<Listener, String> {
        validate_pipe_name(path)?;
        let next = create_server(path, true)?;
        Ok(Listener {
            endpoint: path.to_owned(),
            next: Some(next),
        })
    }

    pub async fn connect(path: &Path) -> Result<NamedPipeClient, String> {
        loop {
            let mut options = ClientOptions::new();
            options.security_qos_flags(SECURITY_IDENTIFICATION);
            match options.open(path) {
                Ok(client) => {
                    verify_pipe_server(&client)?;
                    return Ok(client);
                }
                Err(error) if error.raw_os_error() == Some(ERROR_PIPE_BUSY_CODE) => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(error) => {
                    return Err(format!("failed to connect to Shodh IPC pipe: {error}"));
                }
            }
        }
    }

    pub async fn serve(
        listener: Listener,
        router: Router,
        shutdown: CancellationToken,
        server_auth: Arc<ServerAuth>,
        dispatch_semaphore: Arc<Semaphore>,
        pre_auth_semaphore: Arc<Semaphore>,
        request_timeout: Duration,
    ) -> Result<(), String> {
        let endpoint = listener.endpoint;
        let mut next = listener.next.expect("listener always has a pipe instance");
        let mut connections = JoinSet::new();

        let terminal_error = 'serve: loop {
            let pre_auth_permit = tokio::select! {
                _ = shutdown.cancelled() => break None,
                permit = Arc::clone(&pre_auth_semaphore).acquire_owned() => match permit {
                    Ok(permit) => permit,
                    Err(_) => break Some("IPC pre-auth limiter closed".to_string()),
                },
            };
            tokio::select! {
                _ = shutdown.cancelled() => break None,
                result = next.connect() => {
                    if let Err(error) = result {
                        // Transient accept failure must not tear down the shared
                        // process (the HTTP server shares this task tree). Log, free
                        // the permit, back off, and keep serving on this instance.
                        tracing::warn!("Shodh IPC pipe accept failed, continuing: {error}");
                        drop(pre_auth_permit);
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                    let connected = next;
                    if let Err(error) = verify_pipe_client(&connected) {
                        tracing::warn!("Rejected Shodh IPC client: {error}");
                        drop(pre_auth_permit);
                        next = create_server(&endpoint, false)?;
                        continue;
                    }
                    let connection_router = router.clone();
                    let connection_auth = Arc::clone(&server_auth);
                    let connection_dispatch = Arc::clone(&dispatch_semaphore);
                    connections.spawn(async move {
                        handle_connection(
                            connected,
                            connection_router,
                            connection_auth,
                            connection_dispatch,
                            pre_auth_permit,
                            request_timeout,
                        )
                        .await;
                    });
                    // Re-arm a fresh pending instance. Without one we cannot accept
                    // further clients, so retry transient failures with backoff
                    // rather than tearing down the whole daemon.
                    next = loop {
                        match create_server(&endpoint, false) {
                            Ok(server) => break server,
                            Err(error) => {
                                tracing::warn!(
                                    "Shodh IPC failed to re-arm pipe, retrying: {error}"
                                );
                                tokio::select! {
                                    _ = shutdown.cancelled() => break 'serve None,
                                    _ = tokio::time::sleep(Duration::from_millis(100)) => {}
                                }
                            }
                        }
                    };
                }
            }

            while let Some(result) = connections.try_join_next() {
                if let Err(error) = result {
                    tracing::warn!("Shodh IPC connection task failed: {error}");
                }
            }
        };

        while let Some(result) = connections.join_next().await {
            if let Err(error) = result {
                tracing::warn!("Shodh IPC connection task failed while draining: {error}");
            }
        }
        match terminal_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn validate_pipe_name(path: &Path) -> Result<(), String> {
        let name = path.to_string_lossy();
        if !name.starts_with(r"\\.\pipe\") || name.len() <= r"\\.\pipe\".len() {
            return Err(r"Windows IPC endpoint must use the \\.\pipe\name form".to_string());
        }
        Ok(())
    }

    fn verify_pipe_server(pipe: &NamedPipeClient) -> Result<(), String> {
        use std::os::windows::io::AsRawHandle;

        let mut pid = 0_u32;
        // SAFETY: the Tokio client owns a connected named-pipe handle and pid
        // points to initialized writable storage.
        if unsafe { GetNamedPipeServerProcessId(pipe.as_raw_handle().cast(), &mut pid) } == 0 {
            return Err(last_windows_error(
                "failed to identify Shodh IPC pipe server",
            ));
        }
        verify_process_account(pid, "server")
    }

    fn verify_pipe_client(pipe: &NamedPipeServer) -> Result<(), String> {
        use std::os::windows::io::AsRawHandle;

        let mut pid = 0_u32;
        // SAFETY: the Tokio server owns a connected named-pipe handle and pid
        // points to initialized writable storage.
        if unsafe { GetNamedPipeClientProcessId(pipe.as_raw_handle().cast(), &mut pid) } == 0 {
            return Err(last_windows_error(
                "failed to identify Shodh IPC pipe client",
            ));
        }
        verify_process_account(pid, "client")
    }

    fn verify_process_account(pid: u32, role: &str) -> Result<(), String> {
        // SAFETY: OpenProcess performs all validation; no borrowed pointers cross
        // the call and the returned handle is owned by ProcessHandle.
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if process.is_null() {
            return Err(last_windows_error(&format!(
                "failed to open IPC {role} process {pid}"
            )));
        }
        let process = ProcessHandle(process);
        let actual = process_user_sid_string(process.0)?;
        let expected = current_user_sid_string()?;
        if actual != expected && actual != "S-1-5-18" {
            return Err(format!(
                "refusing IPC {role} process {pid} owned by {actual}; expected {expected} or LocalSystem"
            ));
        }
        Ok(())
    }

    fn create_server(path: &Path, first: bool) -> Result<NamedPipeServer, String> {
        let security = SecurityDescriptor::for_current_user()?;
        let mut attributes = SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: security.descriptor.cast(),
            bInheritHandle: 0,
        };
        let mut options = ServerOptions::new();
        options
            .first_pipe_instance(first)
            .reject_remote_clients(true);

        // SAFETY: attributes and its descriptor remain alive for the complete
        // CreateNamedPipeW call; the descriptor is valid self-relative SDDL,
        // and bInheritHandle is false.
        unsafe {
            options
                .create_with_security_attributes_raw(
                    path,
                    (&mut attributes as *mut SECURITY_ATTRIBUTES).cast::<c_void>(),
                )
                .map_err(|error| format!("failed to create protected IPC pipe: {error}"))
        }
    }

    #[cfg(test)]
    pub(super) fn inspect_security(listener: &Listener) -> Result<(String, String, bool), String> {
        use std::os::windows::io::AsRawHandle;

        let pipe = listener
            .next
            .as_ref()
            .ok_or_else(|| "listener has no pending named-pipe instance".to_string())?;
        let mut descriptor: PSECURITY_DESCRIPTOR = ptr::null_mut();
        // SAFETY: the pending server owns a valid kernel handle; all optional
        // outputs are null and descriptor points to writable storage.
        let result = unsafe {
            GetSecurityInfo(
                pipe.as_raw_handle().cast(),
                SE_KERNEL_OBJECT,
                DACL_SECURITY_INFORMATION,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                &mut descriptor,
            )
        };
        if result != 0 {
            return Err(format!(
                "failed to inspect named-pipe security descriptor: {}",
                std::io::Error::from_raw_os_error(result as i32)
            ));
        }
        let descriptor = SecurityDescriptor { descriptor };

        let mut control = 0_u16;
        let mut revision = 0_u32;
        // SAFETY: descriptor is a valid self-relative security descriptor and
        // both output pointers refer to initialized writable storage.
        if unsafe {
            GetSecurityDescriptorControl(descriptor.descriptor, &mut control, &mut revision)
        } == 0
        {
            return Err(last_windows_error(
                "failed to read named-pipe security descriptor control flags",
            ));
        }

        let mut text: PWSTR = ptr::null_mut();
        // SAFETY: descriptor is valid, text is an output pointer, and Windows
        // allocates the returned NUL-terminated SDDL string with LocalAlloc.
        if unsafe {
            ConvertSecurityDescriptorToStringSecurityDescriptorW(
                descriptor.descriptor,
                SDDL_REVISION_1,
                DACL_SECURITY_INFORMATION,
                &mut text,
                ptr::null_mut(),
            )
        } == 0
        {
            return Err(last_windows_error(
                "failed to format named-pipe security descriptor",
            ));
        }
        let text = LocalWideString(text);
        Ok((
            local_wide_to_string(&text)?,
            current_user_sid_string()?,
            control & SE_DACL_PROTECTED != 0,
        ))
    }

    struct SecurityDescriptor {
        descriptor: PSECURITY_DESCRIPTOR,
    }

    impl SecurityDescriptor {
        fn for_current_user() -> Result<Self, String> {
            let sid = current_user_sid_string()?;
            let sddl = format!("D:P(A;;GA;;;SY)(A;;GA;;;{sid})");
            let wide: Vec<u16> = sddl.encode_utf16().chain(std::iter::once(0)).collect();
            let mut descriptor: PSECURITY_DESCRIPTOR = ptr::null_mut();
            // SAFETY: wide is NUL-terminated and the output pointer is valid.
            let success = unsafe {
                ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    wide.as_ptr(),
                    SDDL_REVISION_1,
                    &mut descriptor,
                    ptr::null_mut(),
                )
            };
            if success == 0 {
                return Err(last_windows_error(
                    "failed to create IPC security descriptor",
                ));
            }
            Ok(Self { descriptor })
        }
    }

    impl Drop for SecurityDescriptor {
        fn drop(&mut self) {
            // SAFETY: ConvertStringSecurityDescriptor allocated this descriptor
            // with LocalAlloc and ownership remains with this wrapper.
            unsafe {
                let _ = LocalFree(self.descriptor.cast());
            }
        }
    }

    fn current_user_sid_string() -> Result<String, String> {
        // SAFETY: GetCurrentProcess has no preconditions and returns a pseudo-handle.
        process_user_sid_string(unsafe { GetCurrentProcess() })
    }

    fn process_user_sid_string(process: HANDLE) -> Result<String, String> {
        let mut token: HANDLE = ptr::null_mut();
        // SAFETY: process is either the current-process pseudo-handle or a live
        // handle opened with query rights; token points to writable storage.
        if unsafe { OpenProcessToken(process, TOKEN_QUERY, &mut token) } == 0 {
            return Err(last_windows_error("failed to open the process token"));
        }
        let token = TokenHandle(token);

        let mut required = 0_u32;
        // SAFETY: the null probe is the documented way to obtain buffer size.
        let probe =
            unsafe { GetTokenInformation(token.0, TokenUser, ptr::null_mut(), 0, &mut required) };
        if probe != 0 || unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER {
            return Err(last_windows_error(
                "failed to size process-token user information",
            ));
        }
        let mut buffer = vec![0_u8; required as usize];
        // SAFETY: buffer has the exact size requested by the preceding probe.
        if unsafe {
            GetTokenInformation(
                token.0,
                TokenUser,
                buffer.as_mut_ptr().cast(),
                required,
                &mut required,
            )
        } == 0
        {
            return Err(last_windows_error(
                "failed to read process-token user information",
            ));
        }
        // SAFETY: GetTokenInformation initialized a TOKEN_USER at the start of
        // the byte buffer. read_unaligned avoids assuming Vec<u8> alignment.
        let token_user = unsafe { ptr::read_unaligned(buffer.as_ptr().cast::<TOKEN_USER>()) };
        let mut sid_text: PWSTR = ptr::null_mut();
        // SAFETY: token_user.User.Sid points into buffer, which remains alive;
        // sid_text is a valid output pointer.
        if unsafe { ConvertSidToStringSidW(token_user.User.Sid, &mut sid_text) } == 0 {
            return Err(last_windows_error(
                "failed to format the process-token user SID",
            ));
        }
        let sid_text = LocalWideString(sid_text);
        local_wide_to_string(&sid_text)
    }

    struct TokenHandle(HANDLE);

    impl Drop for TokenHandle {
        fn drop(&mut self) {
            // SAFETY: this wrapper uniquely owns the OpenProcessToken handle.
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }

    struct ProcessHandle(HANDLE);

    impl Drop for ProcessHandle {
        fn drop(&mut self) {
            // SAFETY: this wrapper uniquely owns the OpenProcess handle.
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }

    struct LocalWideString(PWSTR);

    impl Drop for LocalWideString {
        fn drop(&mut self) {
            // SAFETY: ConvertSidToStringSidW allocates this string with LocalAlloc.
            unsafe {
                let _ = LocalFree(self.0.cast());
            }
        }
    }

    fn local_wide_to_string(value: &LocalWideString) -> Result<String, String> {
        let mut length = 0;
        // SAFETY: LocalWideString only wraps NUL-terminated allocations returned
        // by Windows security conversion functions.
        unsafe {
            while *value.0.add(length) != 0 {
                length += 1;
            }
            String::from_utf16(std::slice::from_raw_parts(value.0, length))
                .map_err(|error| format!("Windows security text was not valid UTF-16: {error}"))
        }
    }

    fn last_windows_error(context: &str) -> String {
        format!("{context}: {}", std::io::Error::last_os_error())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use std::ffi::OsString;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Restores the shared auth key after each test; leaving it unset made later
    // parallel handler tests fail with 503 in CI.
    struct ScopedEnvVar {
        name: &'static str,
        original: Option<OsString>,
    }

    impl ScopedEnvVar {
        fn set(name: &'static str, value: &str) -> Self {
            let original = std::env::var_os(name);
            std::env::set_var(name, value);
            Self { name, original }
        }
    }

    impl Drop for ScopedEnvVar {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }

    #[tokio::test]
    async fn health_round_trip_uses_versioned_envelope() {
        let _guard = crate::auth::ENV_LOCK.lock().unwrap();
        let api_key = crate::handlers::test_helpers::TEST_API_KEY;
        let _api_keys = ScopedEnvVar::set("SHODH_API_KEYS", api_key);
        let endpoint = test_endpoint("health");
        let listener = LocalIpcServer::bind(endpoint.path()).await.unwrap();
        let shutdown = CancellationToken::new();
        let task_shutdown = shutdown.clone();
        let server = tokio::spawn(listener.serve(
            Router::new().route("/health", get(|| async { axum::Json(json!({"ok": true})) })),
            task_shutdown,
            4,
            Duration::from_secs(2),
        ));

        let client = IpcClient::new(endpoint.path(), api_key.to_string());
        let response: Value = client.get("/health").await.unwrap();
        assert_eq!(response, json!({"ok": true}));

        shutdown.cancel();
        server.await.unwrap().unwrap();
    }

    /// Encode a request envelope frame body (without the trailing newline) for the
    /// process_frame tests.
    fn encode_request(auth: &str, method: &str, path: &str) -> Vec<u8> {
        let id = "test-id";
        let body = raw_null();
        let server_auth = ServerAuth::fixed();
        let proof = request_auth(
            auth,
            &server_auth.nonce,
            PROTOCOL_VERSION,
            id,
            method,
            path,
            body.get().as_bytes(),
        );
        serde_json::to_vec(&RequestEnvelope {
            v: PROTOCOL_VERSION,
            id: id.to_string(),
            auth: proof,
            challenge: String::new(),
            method: method.to_string(),
            path: path.to_string(),
            body,
        })
        .unwrap()
    }

    #[tokio::test]
    async fn streaming_and_public_routes_are_rejected_by_dispatch() {
        // Drives process_frame (with auth satisfied) rather than asserting the const
        // arrays contain their own literals — the previous test would still pass if
        // the policy gates were deleted entirely.
        let _guard = crate::auth::ENV_LOCK.lock().unwrap();
        let api_key = crate::handlers::test_helpers::TEST_API_KEY;
        let _api_keys = ScopedEnvVar::set("SHODH_API_KEYS", api_key);

        for path in STREAMING_PATHS {
            let response =
                process_frame(&encode_request(api_key, "GET", path), Router::new()).await;
            assert_eq!(
                response.status,
                StatusCode::NOT_IMPLEMENTED.as_u16(),
                "streaming path {path} must be refused, not dispatched"
            );
            assert_eq!(response.body["code"], "STREAMING_NOT_SUPPORTED");
        }
        for path in PUBLIC_API_PATHS {
            let response =
                process_frame(&encode_request(api_key, "GET", path), Router::new()).await;
            assert_eq!(
                response.status,
                StatusCode::FORBIDDEN.as_u16(),
                "public path {path} must not be exposed over IPC"
            );
            assert_eq!(response.body["code"], "PATH_NOT_ALLOWED");
        }
    }

    #[tokio::test]
    async fn ipc_enforces_the_api_key_over_ordinary_routes() {
        // Regression: both original auth tests sent auth="" and so only exercised the
        // empty-key short-circuit — stubbing validate_api_key to Ok(()) left the
        // suite green while IPC accepted any key. This drives the real validator.
        let _guard = crate::auth::ENV_LOCK.lock().unwrap();
        let api_key = crate::handlers::test_helpers::TEST_API_KEY;
        let _api_keys = ScopedEnvVar::set("SHODH_API_KEYS", api_key);

        let router = Router::new().route(
            "/api/ping",
            get(|| async { axum::Json(json!({"ok": true})) }),
        );

        let wrong = process_frame(
            &encode_request("wrong-key", "GET", "/api/ping"),
            router.clone(),
        )
        .await;
        assert_eq!(wrong.status, StatusCode::UNAUTHORIZED.as_u16());
        assert_eq!(wrong.body["code"], "INVALID_API_KEY");

        let ok = process_frame(&encode_request(api_key, "GET", "/api/ping"), router).await;
        assert_eq!(ok.status, StatusCode::OK.as_u16());
        assert_eq!(ok.body, json!({"ok": true}));
    }

    #[tokio::test]
    async fn request_proof_hides_the_key_and_binds_the_body() {
        let _guard = crate::auth::ENV_LOCK.lock().unwrap();
        let api_key = crate::handlers::test_helpers::TEST_API_KEY;
        let _api_keys = ScopedEnvVar::set("SHODH_API_KEYS", api_key);

        let frame = encode_request(api_key, "POST", "/api/remember");
        assert!(
            !String::from_utf8_lossy(&frame).contains(api_key),
            "the reusable API key must never appear on the IPC wire"
        );

        let mut request: RequestEnvelope = serde_json::from_slice(&frame).unwrap();
        request.body = RawValue::from_string(r#"{"user_id":"other"}"#.to_string()).unwrap();
        let tampered = serde_json::to_vec(&request).unwrap();
        let response = process_frame(&tampered, Router::new()).await;
        assert_eq!(response.status, StatusCode::UNAUTHORIZED.as_u16());
        assert_eq!(response.body["code"], "INVALID_API_KEY");
    }

    #[test]
    fn pre_auth_limits_are_short_and_separate_from_dispatch_capacity() {
        assert_eq!(PRE_AUTH_READ_TIMEOUT, Duration::from_secs(2));
        assert!(MAX_PRE_AUTH_CONNECTIONS < 200);
    }

    #[tokio::test]
    async fn protocol_errors_carry_their_code_and_echo_a_recoverable_id() {
        // A non-JSON frame cannot yield an id, but its error must still be reported
        // as a protocol error rather than masked as an id mismatch by the client.
        let garbage = process_frame(b"not json", Router::new()).await;
        assert_eq!(garbage.status, StatusCode::BAD_REQUEST.as_u16());
        assert_eq!(garbage.body["code"], "INVALID_REQUEST");
        assert_eq!(garbage.id, "");

        // A malformed envelope that still carries a string id echoes it.
        let with_id = process_frame(br#"{"id":"abc","nope":1}"#, Router::new()).await;
        assert_eq!(with_id.body["code"], "INVALID_REQUEST");
        assert_eq!(with_id.id, "abc");

        // A future protocol version is rejected with UNSUPPORTED_VERSION.
        let future_version = serde_json::to_vec(&json!({
            "v": PROTOCOL_VERSION + 1,
            "id": "v2",
            "auth": "",
            "challenge": "",
            "method": "GET",
            "path": "/api/x",
            "body": null,
        }))
        .unwrap();
        let bad_version = process_frame(&future_version, Router::new()).await;
        assert_eq!(bad_version.body["code"], "UNSUPPORTED_VERSION");
        assert_eq!(bad_version.id, "v2");

        // Authentication runs before protected-route policy, even for a path the
        // transport will later refuse.
        let bad_path_frame = serde_json::to_vec(&RequestEnvelope {
            v: PROTOCOL_VERSION,
            id: "bad-path".to_string(),
            auth: String::new(),
            challenge: String::new(),
            method: "GET".to_string(),
            path: "/secret".to_string(),
            body: raw_null(),
        })
        .unwrap();
        let bad_path = process_frame(&bad_path_frame, Router::new()).await;
        assert_eq!(bad_path.status, StatusCode::UNAUTHORIZED.as_u16());
        assert_eq!(bad_path.body["code"], "MISSING_API_KEY");
    }

    #[tokio::test]
    async fn protected_routes_authenticate_before_policy_rejection() {
        for (method, path) in [("GET", "/api/stream"), ("TRACE", "/api/remember")] {
            let request = RequestEnvelope {
                v: PROTOCOL_VERSION,
                id: "request-one".to_string(),
                auth: String::new(),
                challenge: String::new(),
                method: method.to_string(),
                path: path.to_string(),
                body: raw_null(),
            };
            let frame = serde_json::to_vec(&request).unwrap();
            let response = process_frame(&frame, Router::new()).await;
            assert_eq!(response.status, StatusCode::UNAUTHORIZED.as_u16());
            assert_eq!(response.body["code"], "MISSING_API_KEY");
        }
    }

    #[tokio::test]
    async fn only_exact_get_health_is_an_unauthenticated_probe() {
        for (method, path) in [("POST", "/health"), ("GET", "/health?detail=true")] {
            let request = RequestEnvelope {
                v: PROTOCOL_VERSION,
                id: format!("{method}-{path}"),
                auth: String::new(),
                challenge: String::new(),
                method: method.to_string(),
                path: path.to_string(),
                body: raw_null(),
            };
            let frame = serde_json::to_vec(&request).unwrap();
            let response = process_frame(&frame, Router::new()).await;
            assert_eq!(response.status, StatusCode::UNAUTHORIZED.as_u16());
            assert_eq!(response.body["code"], "MISSING_API_KEY");
        }
    }

    #[tokio::test]
    async fn rejects_more_than_one_frame_per_connection() {
        let (mut writer, mut reader) = tokio::io::duplex(128);
        writer.write_all(b"{}\n{}\n").await.unwrap();
        let error = read_frame(&mut reader, FrameSide::Request)
            .await
            .unwrap_err();
        assert_eq!(error.code, "MULTIPLE_FRAMES");
    }

    #[tokio::test]
    async fn dispatches_at_most_once_when_a_second_frame_arrives_later() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let route_dispatches = Arc::clone(&dispatches);
        let router = Router::new().route(
            "/health",
            get(move || {
                let route_dispatches = Arc::clone(&route_dispatches);
                async move {
                    route_dispatches.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(25)).await;
                    axum::Json(json!({"ok": true}))
                }
            }),
        );
        let request = RequestEnvelope {
            v: PROTOCOL_VERSION,
            id: "delayed-frame".to_string(),
            auth: String::new(),
            challenge: String::new(),
            method: "GET".to_string(),
            path: "/health".to_string(),
            body: raw_null(),
        };
        let mut encoded = serde_json::to_vec(&request).unwrap();
        encoded.push(b'\n');
        let (server_stream, mut client_stream) = tokio::io::duplex(4096);
        let pre_auth = Arc::new(Semaphore::new(1)).acquire_owned().await.unwrap();
        let server = tokio::spawn(handle_connection(
            server_stream,
            router,
            Arc::new(ServerAuth::fixed()),
            Arc::new(Semaphore::new(1)),
            pre_auth,
            Duration::from_secs(1),
        ));

        client_stream.write_all(&encoded).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        client_stream.write_all(&encoded).await.unwrap();
        let mut response = Vec::new();
        client_stream.read_to_end(&mut response).await.unwrap();
        server.await.unwrap();

        assert_eq!(dispatches.load(Ordering::SeqCst), 1);
        assert_eq!(response.iter().filter(|byte| **byte == b'\n').count(), 1);
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let client = IpcClient::new(default_endpoint(), "top-secret".to_string());
        let debug = format!("{client:?}");
        assert!(!debug.contains("top-secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_endpoint_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let endpoint = test_endpoint("perms");
        let listener = LocalIpcServer::bind(endpoint.path()).await.unwrap();

        let socket_mode = std::fs::symlink_metadata(endpoint.path())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(socket_mode, 0o600, "IPC socket must be owner-only (0600)");

        let parent = endpoint.path();
        let parent = parent.parent().unwrap();
        let parent_mode = std::fs::symlink_metadata(parent)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            parent_mode, 0o700,
            "IPC socket parent must be owner-only (0700)"
        );

        drop(listener);
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_pipe_dacl_is_protected_and_user_scoped() {
        let endpoint = test_endpoint("acl");
        let listener = LocalIpcServer::bind(endpoint.path()).await.unwrap();
        let (sddl, user_sid, protected) = platform::inspect_security(&listener.listener).unwrap();
        assert!(protected, "named-pipe DACL must reject inherited ACEs");
        assert!(sddl.contains("SY"), "LocalSystem must retain pipe access");
        assert!(
            sddl.contains(&user_sid),
            "current user SID must have pipe access"
        );
        assert!(
            !sddl.contains(";;;WD)"),
            "Everyone must not have pipe access"
        );
    }

    /// A test endpoint plus, on Unix, the owning temp directory whose Drop cleans up
    /// the socket's parent (the server's EndpointGuard removes only the socket file).
    struct TestEndpoint {
        path: PathBuf,
        #[cfg(unix)]
        _dir: tempfile::TempDir,
    }

    impl TestEndpoint {
        fn path(&self) -> PathBuf {
            self.path.clone()
        }
    }

    fn test_endpoint(label: &str) -> TestEndpoint {
        #[cfg(windows)]
        {
            TestEndpoint {
                path: PathBuf::from(format!(
                    r"\\.\pipe\shodh-memory-test-{label}-{}",
                    Uuid::new_v4()
                )),
            }
        }
        #[cfg(unix)]
        {
            // Keep the socket path short: macOS $TMPDIR is already ~50 bytes and the
            // transport enforces a 103-byte sockaddr_un limit, so a full UUID in the
            // path overflows it (the reason the old fixed test bound nothing on
            // macOS). A TempDir gives a short unique name plus cleanup.
            let dir = tempfile::tempdir().expect("create temp dir for IPC socket");
            let path = dir.path().join(format!("{label}.sock"));
            TestEndpoint { path, _dir: dir }
        }
    }
}
