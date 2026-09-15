//! shodh-keyctl — encryption keystore management.
//!
//! Operates directly on the keystore file (usually `<data-dir>/storage/keystore.json`)
//! while the server is stopped: rotate the passphrase, rotate the data key, add
//! a recovery code, or recover from a lost passphrase. Every change is persisted
//! atomically via `Keystore::save_to_path` (temp + fsync + rename, with a `.bak`)
//! before the command returns, so a rotated data key is on disk before any
//! record can be written under it. The database-side rollback sentinel advances
//! the next time the store opens and sees the higher keystore generation.
//!
//! Secrets are taken from the environment ONLY — never from argv, where they
//! would land in shell history and `ps` output:
//!
//!   SHODH_MASTER_PASSPHRASE      the current passphrase
//!   SHODH_NEW_MASTER_PASSPHRASE  the new passphrase (rotate-passphrase, recover)
//!   SHODH_RECOVERY_CODE          a recovery code (recover)

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use shodh_memory::keystore::Keystore;
use std::path::{Path, PathBuf};
use zeroize::Zeroizing;

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
    /// Re-wrap the master key under a new passphrase (records untouched).
    /// Reads SHODH_MASTER_PASSPHRASE and SHODH_NEW_MASTER_PASSPHRASE.
    RotatePassphrase,
    /// Rotate the active data key to a new epoch. Records written under earlier
    /// epochs stay readable — the keystore retains their DEKs and the store
    /// decrypts each record under its own epoch. Reads SHODH_MASTER_PASSPHRASE.
    RotateDek,
    /// Add a one-time recovery code (printed once — store it offline).
    /// Reads SHODH_MASTER_PASSPHRASE.
    AddRecoveryCode,
    /// Recover from passphrase loss: unseal with a recovery code and install a
    /// new passphrase. The used recovery code is retired and a fresh one printed.
    /// Reads SHODH_RECOVERY_CODE and SHODH_NEW_MASTER_PASSPHRASE.
    Recover,
    /// Print the keystore's non-secret state (versions, epochs, generation).
    Status,
}

/// A secret from the environment, or a clear error naming the variable.
fn secret(var: &str) -> Result<Zeroizing<String>> {
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => Ok(Zeroizing::new(v)),
        _ => Err(anyhow!(
            "{var} is not set; secrets are read from the environment only (not argv)"
        )),
    }
}

fn load(path: &Path) -> Result<Keystore> {
    let json = std::fs::read_to_string(path)
        .with_context(|| format!("read keystore at {}", path.display()))?;
    Keystore::from_json(&json)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::RotatePassphrase => {
            let old = secret("SHODH_MASTER_PASSPHRASE")?;
            let new = secret("SHODH_NEW_MASTER_PASSPHRASE")?;
            let mut ks = load(&cli.keystore)?;
            let kek = ks.unseal_with_passphrase(&old)?;
            ks.verify_integrity(&kek)?;
            ks.rotate_passphrase(&old, &new)?;
            ks.save_to_path(&cli.keystore)?;
            println!("passphrase rotated (keystore generation {})", ks.generation);
        }
        Cmd::RotateDek => {
            let passphrase = secret("SHODH_MASTER_PASSPHRASE")?;
            let mut ks = load(&cli.keystore)?;
            let kek = ks.unseal_with_passphrase(&passphrase)?;
            ks.verify_integrity(&kek)?;
            let epoch = ks.rotate_dek(&kek)?;
            ks.save_to_path(&cli.keystore)?;
            println!(
                "data key rotated to epoch {epoch} (keystore generation {}); \
                 records under earlier epochs remain readable",
                ks.generation
            );
        }
        Cmd::AddRecoveryCode => {
            let passphrase = secret("SHODH_MASTER_PASSPHRASE")?;
            let mut ks = load(&cli.keystore)?;
            let kek = ks.unseal_with_passphrase(&passphrase)?;
            ks.verify_integrity(&kek)?;
            let code = ks.add_recovery_code(&kek)?;
            ks.save_to_path(&cli.keystore)?;
            println!("RECOVERY CODE (store offline, shown once): {code}");
        }
        Cmd::Recover => {
            let recovery_code = secret("SHODH_RECOVERY_CODE")?;
            let new_passphrase = secret("SHODH_NEW_MASTER_PASSPHRASE")?;
            let mut ks = load(&cli.keystore)?;
            let kek = ks.unseal_with_recovery_code(&recovery_code)?;
            ks.verify_integrity(&kek)?;
            ks.set_passphrase(&kek, &new_passphrase)?;
            // Retire the used code: install a fresh recovery wrap.
            let new_code = ks.add_recovery_code(&kek)?;
            ks.save_to_path(&cli.keystore)?;
            println!(
                "passphrase reset via recovery code (keystore generation {})",
                ks.generation
            );
            println!("NEW RECOVERY CODE (store offline, shown once): {new_code}");
        }
        Cmd::Status => {
            let ks = load(&cli.keystore)?;
            println!("crypto_version   {}", ks.crypto_version);
            println!("schema_version   {}", ks.schema_version);
            println!("generation       {}", ks.generation);
            println!("active_epoch     {}", ks.active_epoch);
            println!(
                "epochs           {}",
                ks.deks
                    .iter()
                    .map(|d| format!("{}:{}", d.epoch, d.state))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            println!(
                "unseal providers {}",
                ks.kek_wraps
                    .iter()
                    .map(|w| w.provider.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            println!(
                "kdf              argon2id m={}KiB t={} p={}",
                ks.kdf.m_cost, ks.kdf.t_cost, ks.kdf.p_cost
            );
            println!("sealed           {}", !ks.mac.is_empty());
        }
    }
    Ok(())
}
