#!/usr/bin/env node
/**
 * Postinstall script for @shodh/memory-mcp
 *
 * Downloads the appropriate shodh-memory-server binary for the current platform
 * from GitHub releases with a visual progress bar.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { execFileSync } = require('child_process');

const VERSION = require('../package.json').version;
const REPO = 'varun29ankuS/shodh-memory';
const BIN_DIR = path.join(__dirname, '..', 'bin');

// ─── Visual helpers ──────────────────────────────────────────────────────────

const BAR_WIDTH = 30;
const FILLED = '\u2588'; // █
const EMPTY = '\u2591';  // ░

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(seconds) {
  if (seconds < 1) return '<1s';
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.ceil(seconds % 60);
  return `${m}m ${s}s`;
}

function drawProgress(downloaded, total, startTime) {
  if (total === 0) {
    process.stderr.write(`\r  Downloading... ${formatBytes(downloaded)}`);
    return;
  }

  const pct = Math.min(downloaded / total, 1);
  const filled = Math.round(pct * BAR_WIDTH);
  const bar = FILLED.repeat(filled) + EMPTY.repeat(BAR_WIDTH - filled);
  const pctStr = `${Math.round(pct * 100)}%`.padStart(4);

  const elapsed = (Date.now() - startTime) / 1000;
  let eta = '';
  if (pct > 0.01 && pct < 1) {
    const remaining = (elapsed / pct) * (1 - pct);
    eta = ` | ${formatTime(remaining)} remaining`;
  }

  process.stderr.write(
    `\r  ${bar}  ${pctStr} | ${formatBytes(downloaded)} / ${formatBytes(total)}${eta}   `
  );
}

function printHeader() {
  // Box inner width = 42. Full line before trailing ║ = 2(indent) + 1(║) + 42 = 45.
  // 🐘 surrogate pair is 2 JS chars and 2 visual columns — no special handling needed.
  process.stderr.write('\n');
  process.stderr.write('  \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n');
  process.stderr.write(`  \u2551   \uD83D\uDC18 shodh-memory v${VERSION}`.padEnd(45) + '\u2551\n');
  process.stderr.write('  \u2551   Cognitive Memory for AI Agents         \u2551\n');
  process.stderr.write('  \u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D\n');
  process.stderr.write('\n');
}

function printSuccess(binaryPath) {
  process.stderr.write('\n');
  process.stderr.write(`  \u2713 Binary installed at ${binaryPath}\n`);
  process.stderr.write('\n');
  process.stderr.write('  Next step \u2014 enable automatic memory:\n');
  process.stderr.write('    npx shodh-setup-hooks\n');
  process.stderr.write('\n');
}

function printAlreadyInstalled() {
  process.stderr.write(`  \u2713 Server binary already installed\n`);
  process.stderr.write('\n');
}

// ─── Platform detection ──────────────────────────────────────────────────────

function getPlatformInfo() {
  const platform = process.platform;
  const arch = process.arch;

  if (platform === 'linux' && arch === 'x64') {
    return { name: 'shodh-memory-linux-x64', ext: '.tar.gz', binary: 'shodh-memory-server' };
  } else if (platform === 'linux' && arch === 'arm64') {
    return { name: 'shodh-memory-linux-arm64', ext: '.tar.gz', binary: 'shodh-memory-server' };
  } else if (platform === 'darwin' && arch === 'x64') {
    return { name: 'shodh-memory-macos-x64', ext: '.tar.gz', binary: 'shodh-memory-server' };
  } else if (platform === 'darwin' && arch === 'arm64') {
    return { name: 'shodh-memory-macos-arm64', ext: '.tar.gz', binary: 'shodh-memory-server' };
  } else if (platform === 'win32' && arch === 'x64') {
    return { name: 'shodh-memory-windows-x64', ext: '.zip', binary: 'shodh-memory-server.exe' };
  } else {
    return null;
  }
}

// ─── Download with progress ──────────────────────────────────────────────────

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    const startTime = Date.now();

    const request = (url) => {
      https.get(url, (response) => {
        if (response.statusCode === 302 || response.statusCode === 301) {
          request(response.headers.location);
          return;
        }

        if (response.statusCode !== 200) {
          reject(new Error(`HTTP ${response.statusCode}`));
          return;
        }

        const totalSize = parseInt(response.headers['content-length'] || '0', 10);
        let downloaded = 0;

        response.on('data', (chunk) => {
          downloaded += chunk.length;
          drawProgress(downloaded, totalSize, startTime);
        });

        response.pipe(file);
        file.on('finish', () => {
          file.close();
          // Clear the progress line and show completion
          if (totalSize > 0) {
            drawProgress(totalSize, totalSize, startTime);
          }
          process.stderr.write(' \u2713\n');
          resolve();
        });
      }).on('error', (err) => {
        fs.unlink(dest, () => {});
        reject(err);
      });
    };

    request(url);
  });
}

// ─── Extract archive ─────────────────────────────────────────────────────────

function extract(archive, dest, platformInfo) {
  if (platformInfo.ext === '.tar.gz') {
    execFileSync('tar', ['-xzf', archive, '-C', dest], { stdio: 'pipe' });
  } else if (platformInfo.ext === '.zip') {
    execFileSync('powershell', [
      '-Command',
      `Expand-Archive -Path '${archive}' -DestinationPath '${dest}' -Force`,
    ], { stdio: 'pipe' });
  }
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const platformInfo = getPlatformInfo();

  printHeader();

  if (!platformInfo) {
    process.stderr.write(`  \u2717 Unsupported platform: ${process.platform} ${process.arch}\n`);
    process.stderr.write('  You will need to run the server manually.\n');
    process.stderr.write(`  Download from: https://github.com/${REPO}/releases\n\n`);
    return;
  }

  // Create bin directory
  if (!fs.existsSync(BIN_DIR)) {
    fs.mkdirSync(BIN_DIR, { recursive: true });
  }

  const binaryPath = path.join(BIN_DIR, platformInfo.binary);

  // Check if already installed
  if (fs.existsSync(binaryPath)) {
    printAlreadyInstalled();
    return;
  }

  // Download
  const downloadUrl = `https://github.com/${REPO}/releases/download/v${VERSION}/${platformInfo.name}${platformInfo.ext}`;
  const archivePath = path.join(BIN_DIR, `${platformInfo.name}${platformInfo.ext}`);

  process.stderr.write(`  Downloading server binary for ${process.platform} ${process.arch}...\n`);

  try {
    await download(downloadUrl, archivePath);

    // Extract
    process.stderr.write('  Extracting...');
    extract(archivePath, BIN_DIR, platformInfo);
    process.stderr.write(' \u2713\n');

    // Clean up archive
    fs.unlinkSync(archivePath);

    // Make executable (Unix)
    if (process.platform !== 'win32') {
      fs.chmodSync(binaryPath, 0o755);
    }

    printSuccess(binaryPath);
  } catch (err) {
    process.stderr.write('\n');
    process.stderr.write(`  \u2717 Failed to install binary: ${err.message}\n`);
    process.stderr.write(`  Manual download: https://github.com/${REPO}/releases\n\n`);
  }
}

main();
