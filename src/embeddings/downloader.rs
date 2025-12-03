//! Model and ONNX Runtime auto-downloader
//!
//! Downloads MiniLM-L6-v2 model and ONNX Runtime on first use.
//! Files are cached in ~/.cache/shodh-memory/
//!
//! Model files (~22MB quantized):
//! - model_quantized.onnx - Quantized MiniLM-L6-v2 model
//! - tokenizer.json - HuggingFace tokenizer
//!
//! ONNX Runtime (~50MB):
//! - onnxruntime.dll (Windows)
//! - libonnxruntime.so (Linux)
//! - libonnxruntime.dylib (macOS)

use anyhow::{Result, Context};
use std::path::PathBuf;
use std::fs;
use std::io::{Read, Write};
use std::sync::Arc;

/// URLs for model files (hosted on HuggingFace)
/// Full model is 90MB, quantized is 23MB - we download quantized for edge devices
const MODEL_ONNX_URL: &str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx";
const MODEL_QUANTIZED_URL: &str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model_quint8_avx2.onnx";
const TOKENIZER_URL: &str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json";

/// ONNX Runtime download URLs by platform (v1.22.0 required by ort 2.0.0-rc.10)
#[cfg(target_os = "windows")]
const ONNX_RUNTIME_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip";
#[cfg(target_os = "linux")]
const ONNX_RUNTIME_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz";
#[cfg(target_os = "macos")]
const ONNX_RUNTIME_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-arm64-1.22.0.tgz";

/// Get the cache directory for shodh-memory
pub fn get_cache_dir() -> PathBuf {
    // Try standard cache locations
    if let Some(cache) = dirs::cache_dir() {
        return cache.join("shodh-memory");
    }

    // Fallback to home directory
    if let Some(home) = dirs::home_dir() {
        return home.join(".cache").join("shodh-memory");
    }

    // Last resort: current directory
    PathBuf::from(".shodh-cache")
}

/// Get the models directory
pub fn get_models_dir() -> PathBuf {
    get_cache_dir().join("models").join("minilm-l6")
}

/// Get the ONNX Runtime directory
pub fn get_onnx_runtime_dir() -> PathBuf {
    get_cache_dir().join("onnxruntime")
}

/// Check if model files are downloaded
pub fn are_models_downloaded() -> bool {
    let models_dir = get_models_dir();
    models_dir.join("model_quantized.onnx").exists() &&
    models_dir.join("tokenizer.json").exists()
}

/// Check if ONNX Runtime is downloaded
pub fn is_onnx_runtime_downloaded() -> bool {
    let onnx_dir = get_onnx_runtime_dir();

    #[cfg(target_os = "windows")]
    let lib_name = "onnxruntime.dll";
    #[cfg(target_os = "linux")]
    let lib_name = "libonnxruntime.so";
    #[cfg(target_os = "macos")]
    let lib_name = "libonnxruntime.dylib";

    onnx_dir.join(lib_name).exists()
}

/// Get the path to the ONNX Runtime library
pub fn get_onnx_runtime_path() -> Option<PathBuf> {
    let onnx_dir = get_onnx_runtime_dir();

    #[cfg(target_os = "windows")]
    let lib_name = "onnxruntime.dll";
    #[cfg(target_os = "linux")]
    let lib_name = "libonnxruntime.so";
    #[cfg(target_os = "macos")]
    let lib_name = "libonnxruntime.dylib";

    let path = onnx_dir.join(lib_name);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Download progress callback type (Arc for clonability)
pub type ProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// Download a file from URL to path with progress
fn download_file(url: &str, path: &PathBuf, progress: Option<&(dyn Fn(u64, u64) + Send + Sync)>) -> Result<()> {
    tracing::info!("Downloading {} to {:?}", url, path);

    // Create parent directories
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .context("Failed to create cache directory")?;
    }

    // Use ureq for simple HTTP downloads (blocking, no async runtime needed)
    let response = ureq::get(url)
        .call()
        .context(format!("Failed to download from {}", url))?;

    let total_size = response.header("content-length")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let mut reader = response.into_reader();
    let mut file = fs::File::create(path)
        .context("Failed to create output file")?;

    let mut downloaded: u64 = 0;
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = reader.read(&mut buffer)
            .context("Failed to read from download stream")?;

        if bytes_read == 0 {
            break;
        }

        file.write_all(&buffer[..bytes_read])
            .context("Failed to write to file")?;

        downloaded += bytes_read as u64;

        if let Some(cb) = progress {
            cb(downloaded, total_size);
        }
    }

    tracing::info!("Downloaded {} bytes to {:?}", downloaded, path);
    Ok(())
}

/// Download MiniLM model files
/// Downloads quantized model (~23MB) by default for edge devices
/// Set use_full_model=true for the larger 90MB model
pub fn download_models(progress: Option<ProgressCallback>) -> Result<PathBuf> {
    download_models_internal(progress, true)
}

/// Download MiniLM model files with option for full or quantized model
pub fn download_models_internal(progress: Option<ProgressCallback>, use_quantized: bool) -> Result<PathBuf> {
    let models_dir = get_models_dir();

    if are_models_downloaded() {
        tracing::info!("Models already downloaded at {:?}", models_dir);
        return Ok(models_dir);
    }

    tracing::info!("Downloading MiniLM-L6-v2 model to {:?}", models_dir);

    // Download model (quantized ~23MB or full ~90MB)
    let (model_url, model_filename) = if use_quantized {
        (MODEL_QUANTIZED_URL, "model_quantized.onnx")
    } else {
        (MODEL_ONNX_URL, "model.onnx")
    };

    let model_path = models_dir.join(model_filename);
    tracing::info!("Downloading model from {} (~{}MB)",
        if use_quantized { "HuggingFace (quantized)" } else { "HuggingFace (full)" },
        if use_quantized { 23 } else { 90 }
    );
    download_file(model_url, &model_path, progress.as_ref().map(|p| p.as_ref()))?;

    // Download tokenizer (~700KB)
    let tokenizer_path = models_dir.join("tokenizer.json");
    tracing::info!("Downloading tokenizer.json");
    download_file(TOKENIZER_URL, &tokenizer_path, progress.as_ref().map(|p| p.as_ref()))?;

    tracing::info!("MiniLM-L6-v2 model downloaded successfully to {:?}", models_dir);
    Ok(models_dir)
}

/// Download ONNX Runtime
pub fn download_onnx_runtime(progress: Option<ProgressCallback>) -> Result<PathBuf> {
    let onnx_dir = get_onnx_runtime_dir();

    if is_onnx_runtime_downloaded() {
        tracing::info!("ONNX Runtime already downloaded at {:?}", onnx_dir);
        return get_onnx_runtime_path()
            .ok_or_else(|| anyhow::anyhow!("ONNX Runtime not found"));
    }

    tracing::info!("Downloading ONNX Runtime to {:?}", onnx_dir);
    fs::create_dir_all(&onnx_dir)?;

    // Download the archive
    let archive_name = if cfg!(target_os = "windows") {
        "onnxruntime.zip"
    } else {
        "onnxruntime.tgz"
    };
    let archive_path = onnx_dir.join(archive_name);

    download_file(ONNX_RUNTIME_URL, &archive_path, progress.as_ref().map(|p| p.as_ref()))?;

    // Extract the archive
    extract_onnx_runtime(&archive_path, &onnx_dir)?;

    // Clean up archive
    let _ = fs::remove_file(&archive_path);

    get_onnx_runtime_path()
        .ok_or_else(|| anyhow::anyhow!("Failed to extract ONNX Runtime"))
}

/// Extract ONNX Runtime from archive
fn extract_onnx_runtime(archive_path: &PathBuf, dest_dir: &PathBuf) -> Result<()> {
    tracing::info!("Extracting ONNX Runtime from {:?}", archive_path);

    #[cfg(target_os = "windows")]
    {
        // Use zip crate for Windows
        let file = fs::File::open(archive_path)?;
        let mut archive = zip::ZipArchive::new(file)?;

        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let name = file.name();

            // Look for the DLL file
            if name.ends_with("onnxruntime.dll") {
                let dest_path = dest_dir.join("onnxruntime.dll");
                let mut outfile = fs::File::create(&dest_path)?;
                std::io::copy(&mut file, &mut outfile)?;
                tracing::info!("Extracted onnxruntime.dll");
                return Ok(());
            }
        }

        anyhow::bail!("onnxruntime.dll not found in archive");
    }

    #[cfg(not(target_os = "windows"))]
    {
        // Use tar + gzip for Unix
        let file = fs::File::open(archive_path)?;
        let gz = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(gz);

        #[cfg(target_os = "linux")]
        let lib_name = "libonnxruntime.so";
        #[cfg(target_os = "macos")]
        let lib_name = "libonnxruntime.dylib";

        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?;
            let name = path.to_string_lossy();

            if name.ends_with(lib_name) || name.contains(lib_name) {
                let dest_path = dest_dir.join(lib_name);
                entry.unpack(&dest_path)?;
                tracing::info!("Extracted {}", lib_name);
                return Ok(());
            }
        }

        anyhow::bail!("{} not found in archive", lib_name);
    }
}

/// Ensure all required files are downloaded
/// Returns paths to model directory and ONNX runtime
pub fn ensure_downloaded(progress: Option<ProgressCallback>) -> Result<(PathBuf, PathBuf)> {
    // Check if ORT_DYLIB_PATH is already set
    if let Ok(existing_path) = std::env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(&existing_path);
        if path.exists() {
            tracing::info!("Using existing ONNX Runtime from ORT_DYLIB_PATH: {:?}", path);
            let models_dir = download_models(progress)?;
            return Ok((models_dir, path));
        }
    }

    // Download models
    let models_dir = download_models(progress.clone())?;

    // Download ONNX Runtime
    let onnx_path = download_onnx_runtime(progress)?;

    // Set ORT_DYLIB_PATH for current process
    std::env::set_var("ORT_DYLIB_PATH", &onnx_path);
    tracing::info!("Set ORT_DYLIB_PATH to {:?}", onnx_path);

    Ok((models_dir, onnx_path))
}

/// Print download status
pub fn print_status() {
    let cache_dir = get_cache_dir();
    let models_downloaded = are_models_downloaded();
    let onnx_downloaded = is_onnx_runtime_downloaded();

    println!("Shodh-Memory Cache Status:");
    println!("  Cache directory: {:?}", cache_dir);
    println!("  Models downloaded: {}", models_downloaded);
    println!("  ONNX Runtime downloaded: {}", onnx_downloaded);

    if models_downloaded {
        let models_dir = get_models_dir();
        println!("  Model path: {:?}", models_dir);
    }

    if onnx_downloaded {
        if let Some(path) = get_onnx_runtime_path() {
            println!("  ONNX Runtime path: {:?}", path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir() {
        let cache_dir = get_cache_dir();
        assert!(cache_dir.to_string_lossy().contains("shodh-memory"));
    }

    #[test]
    fn test_models_dir() {
        let models_dir = get_models_dir();
        assert!(models_dir.to_string_lossy().contains("minilm-l6"));
    }
}
