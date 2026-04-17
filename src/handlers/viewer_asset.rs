//! Serve bundled viewer assets (JS, CSS) under /graph/viewer/{path}.
//!
//! Files are embedded at compile time via rust-embed. The `index.html`
//! template is intentionally excluded — it is served by `graph_view2`
//! with placeholder substitution.

use axum::{
    extract::Path,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "src/handlers/viewer/"]
#[exclude = "index.html"]
#[exclude = "*.rs"]
#[exclude = "tests/*"]
struct ViewerAssets;

fn content_type_for(path: &str) -> &'static str {
    // No .html arm: index.html is excluded (served by graph_view2 with template
    // substitution); any future .html assets would need an explicit arm here.
    if path.ends_with(".js") {
        "application/javascript; charset=utf-8"
    } else if path.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if path.ends_with(".svg") {
        "image/svg+xml"
    } else if path.ends_with(".json") {
        "application/json"
    } else {
        "application/octet-stream"
    }
}

/// GET /graph/viewer/{*rest} — serve embedded viewer bundle assets.
pub async fn viewer_asset(Path(rest): Path<String>) -> Response {
    // Defense-in-depth: reject any `..` segment even though axum's router
    // normalizes paths and rust-embed won't find escape paths anyway.
    if rest.split('/').any(|seg| seg == "..") {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    let Some(file) = ViewerAssets::get(&rest) else {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    };
    (
        [
            (header::CONTENT_TYPE, content_type_for(&rest)),
            (header::CACHE_CONTROL, "public, max-age=31536000, immutable"),
        ],
        file.data,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request, Router};
    use tower::ServiceExt;

    fn router() -> Router {
        Router::new().route(
            "/graph/viewer/{*rest}",
            axum::routing::get(viewer_asset),
        )
    }

    #[tokio::test]
    async fn viewer_asset_serves_app_js() {
        let app = router();
        let req = Request::builder()
            .uri("/graph/viewer/app.js")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/javascript; charset=utf-8"
        );
    }

    #[tokio::test]
    async fn viewer_asset_serves_style_css() {
        let app = router();
        let req = Request::builder()
            .uri("/graph/viewer/style.css")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/css; charset=utf-8"
        );
    }

    #[tokio::test]
    async fn viewer_asset_excludes_index_html() {
        let app = router();
        let req = Request::builder()
            .uri("/graph/viewer/index.html")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn viewer_asset_excludes_test_files() {
        let app = router();
        let req = Request::builder()
            .uri("/graph/viewer/tests/unit/node-style.test.js")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn viewer_asset_rejects_path_traversal() {
        for bad in [
            "../Cargo.toml",
            "js/../../Cargo.toml",
            "js/../../../etc/passwd",
            "%2e%2e/Cargo.toml",
        ] {
            let app = router();
            let req = Request::builder()
                .uri(format!("/graph/viewer/{bad}"))
                .body(Body::empty())
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND, "bad path {bad}");
        }
    }
}
