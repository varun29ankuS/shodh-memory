//! shodh-keyctl — encryption keystore management (encryption-v2 H5).
//!
//! Operates directly on the keystore file (usually `<data-dir>/storage/keystore.json`):
//! rotate the passphrase, rotate the data key, or add a recovery code. Every
//! change is persisted atomically via `Keystore::save_to_path` (temp+fsync+rename,
//! with a `.bak`). The database-side rollback sentinel advances automatically the
//! next time the store opens (it sees the higher keystore generation).
//!
//! Passphrases are taken from flags or env vars to avoid an interactive-prompt
//! dependency; prefer the env form so they don't land in shell history:
//!   SHODH_MASTER_PASSPHRASE (current), SHODH_NEW_MASTER_PASSPHRASE (for rotation).

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use shodh_memory::keystore::Keystore;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "shodh-keyctl", about = "Manage the shodh encryption keystore")]
struct Cli {
    /// Path to keystore.json.
    #[arg(long)]
    keystore: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Re-wrap the master key under a new passphrase (records/index untouched).
    RotatePassphrase {
        #[arg(long, env = "SHODH_MASTER_PASSPHRASE")]
        old: String,
        #[arg(long, env = "SHODH_NEW_MASTER_PASSPHRASE")]
        new: String,
    },
    /// Rotate the active data key to a new epoch.
    ///
    /// WARNING: the storage layer currently serves only the active epoch, so
    /// records written under the previous epoch become UNREADABLE after a
    /// rotation (multi-epoch read is a pending follow-up). Refuses unless
    /// `--force` is given — only safe on an empty or disposable store.
    RotateDek {
        #[arg(long, env = "SHODH_MASTER_PASSPHRASE")]
        passphrase: String,
        /// Rotate anyway, accepting that all existing records become unreadable.
        #[arg(long)]
        force: bool,
    },
    /// Add a one-time recovery code (printed once — store it offline).
    AddRecoveryCode {
        #[arg(long, env = "SHODH_MASTER_PASSPHRASE")]
        passphrase: String,
    },
}

fn load(path: &Path) -> Result<Keystore> {
    let json = std::fs::read_to_string(path)
        .with_context(|| format!("read keystore at {}", path.display()))?;
    Keystore::from_json(&json)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::RotatePassphrase { old, new } => {
            let mut ks = load(&cli.keystore)?;
            let kek = ks.unseal_with_passphrase(&old)?;
            ks.verify_integrity(&kek)?;
            ks.rotate_passphrase(&old, &new)?;
            ks.save_to_path(&cli.keystore)?;
            println!("passphrase rotated (keystore generation {})", ks.generation);
        }
        Cmd::RotateDek { passphrase, force } => {
            if !force {
                anyhow::bail!(
                    "refusing to rotate the data key: storage reads only the active epoch, so \
                     every existing record would become unreadable. Re-run with --force only on \
                     an empty or disposable store (multi-epoch read is a pending follow-up)."
                );
            }
            let mut ks = load(&cli.keystore)?;
            let kek = ks.unseal_with_passphrase(&passphrase)?;
            ks.verify_integrity(&kek)?;
            let epoch = ks.rotate_dek(&kek)?;
            ks.save_to_path(&cli.keystore)?;
            eprintln!(
                "WARNING: rotated to epoch {epoch}; records written under earlier epochs are now \
                 UNREADABLE until multi-epoch read is implemented."
            );
            println!(
                "data key rotated to epoch {epoch} (keystore generation {})",
                ks.generation
            );
        }
        Cmd::AddRecoveryCode { passphrase } => {
            let mut ks = load(&cli.keystore)?;
            let kek = ks.unseal_with_passphrase(&passphrase)?;
            ks.verify_integrity(&kek)?;
            let code = ks.add_recovery_code(&kek)?;
            ks.save_to_path(&cli.keystore)?;
            println!("RECOVERY CODE (store offline, shown once): {code}");
        }
    }
    Ok(())
}
