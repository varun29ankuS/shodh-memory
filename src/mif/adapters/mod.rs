//! MIF Adapter trait and registry for format detection and conversion.
//!
//! Each adapter handles a specific interchange format:
//! - `shodh` — MIF v2 JSON (native, lossless round-trip) + v1 backward compat
//! - `mem0` — mem0 memory format: `[{memory, metadata, ...}]`
//! - `generic` — Generic JSON array: `[{content, timestamp, tags, ...}]`
//! - `markdown` — Markdown with YAML frontmatter (Letta/Obsidian style)

pub mod generic;
pub mod markdown;
pub mod mem0;
pub mod shodh;

use anyhow::Result;

use super::schema::MifDocument;

/// Trait for converting between external formats and MIF v2.
pub trait MifAdapter: Send + Sync {
    /// Human-readable adapter name.
    fn name(&self) -> &str;

    /// MIME type or format identifier for this adapter.
    fn format_id(&self) -> &str;

    /// Sniff the first bytes of data to detect if this adapter can handle it.
    /// Returns true if the data looks like this format.
    fn detect(&self, data: &[u8]) -> bool;

    /// Convert external format bytes into a MifDocument.
    fn to_mif(&self, data: &[u8]) -> Result<MifDocument>;

    /// Convert a MifDocument into the external format bytes.
    #[allow(clippy::wrong_self_convention)]
    fn from_mif(&self, doc: &MifDocument) -> Result<Vec<u8>>;
}

/// Registry of available adapters with auto-detection.
pub struct AdapterRegistry {
    adapters: Vec<Box<dyn MifAdapter>>,
}

impl AdapterRegistry {
    /// Create a registry with all built-in adapters.
    pub fn new() -> Self {
        Self {
            adapters: vec![
                Box::new(shodh::ShodhAdapter),
                Box::new(mem0::Mem0Adapter),
                Box::new(generic::GenericJsonAdapter),
                Box::new(markdown::MarkdownAdapter),
            ],
        }
    }

    /// Auto-detect the format and convert to MifDocument.
    ///
    /// Tries each adapter's `detect()` in order. The shodh adapter is checked
    /// first (most specific), then mem0, generic JSON, and finally markdown.
    pub fn auto_import(&self, data: &[u8]) -> Result<MifDocument> {
        for adapter in &self.adapters {
            if adapter.detect(data) {
                return adapter.to_mif(data);
            }
        }
        anyhow::bail!("No adapter could detect the input format. Supported: shodh (MIF JSON), mem0, generic JSON array, markdown.")
    }

    /// Import using a specific adapter by format ID.
    pub fn import_with(&self, format_id: &str, data: &[u8]) -> Result<MifDocument> {
        for adapter in &self.adapters {
            if adapter.format_id() == format_id {
                return adapter.to_mif(data);
            }
        }
        anyhow::bail!(
            "Unknown format: '{}'. Available: {}",
            format_id,
            self.list_formats().join(", ")
        )
    }

    /// Export using a specific adapter by format ID.
    pub fn export_with(&self, format_id: &str, doc: &MifDocument) -> Result<Vec<u8>> {
        for adapter in &self.adapters {
            if adapter.format_id() == format_id {
                return adapter.from_mif(doc);
            }
        }
        anyhow::bail!(
            "Unknown format: '{}'. Available: {}",
            format_id,
            self.list_formats().join(", ")
        )
    }

    /// List all available format IDs.
    pub fn list_formats(&self) -> Vec<&str> {
        self.adapters.iter().map(|a| a.format_id()).collect()
    }

    /// List all adapters with their names and format IDs.
    pub fn list_adapters(&self) -> Vec<(&str, &str)> {
        self.adapters
            .iter()
            .map(|a| (a.name(), a.format_id()))
            .collect()
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}
