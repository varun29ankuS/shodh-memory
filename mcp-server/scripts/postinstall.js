#!/usr/bin/env node
/**
 * Postinstall script for @shodh/memory-mcp
 *
 * Downloads the appropriate shodh-memory-server binary for the current platform
 * from GitHub releases.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { execSync } = require('child_process');

const VERSION = '0.1.6';
const REPO = 'varun29ankuS/shodh-memory';
const BIN_DIR = path.join(__dirname, '..', 'bin');

// Platform detection
function getPlatformInfo() {
  const platform = process.platform;
  const arch = process.arch;

  if (platform === 'linux' && arch === 'x64') {
    return { name: 'shodh-memory-linux-x64', ext: '.tar.gz', binary: 'shodh-memory-server' };
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

// Download file with redirect following
function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);

    const request = (url) => {
      https.get(url, (response) => {
        if (response.statusCode === 302 || response.statusCode === 301) {
          // Follow redirect
          request(response.headers.location);
          return;
        }

        if (response.statusCode !== 200) {
          reject(new Error(`Failed to download: ${response.statusCode}`));
          return;
        }

        response.pipe(file);
        file.on('finish', () => {
          file.close();
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

// Extract archive
function extract(archive, dest, platformInfo) {
  if (platformInfo.ext === '.tar.gz') {
    execSync(`tar -xzf "${archive}" -C "${dest}"`, { stdio: 'inherit' });
  } else if (platformInfo.ext === '.zip') {
    // Use PowerShell on Windows
    execSync(`powershell -Command "Expand-Archive -Path '${archive}' -DestinationPath '${dest}' -Force"`, { stdio: 'inherit' });
  }
}

async function main() {
  const platformInfo = getPlatformInfo();

  if (!platformInfo) {
    console.log('[shodh-memory] Unsupported platform:', process.platform, process.arch);
    console.log('[shodh-memory] You will need to run the server manually.');
    return;
  }

  console.log('[shodh-memory] Installing server binary for', process.platform, process.arch);

  // Create bin directory
  if (!fs.existsSync(BIN_DIR)) {
    fs.mkdirSync(BIN_DIR, { recursive: true });
  }

  const binaryPath = path.join(BIN_DIR, platformInfo.binary);

  // Check if already installed
  if (fs.existsSync(binaryPath)) {
    console.log('[shodh-memory] Binary already installed at', binaryPath);
    return;
  }

  // Download URL
  const downloadUrl = `https://github.com/${REPO}/releases/download/v${VERSION}/${platformInfo.name}${platformInfo.ext}`;
  const archivePath = path.join(BIN_DIR, `${platformInfo.name}${platformInfo.ext}`);

  console.log('[shodh-memory] Downloading from', downloadUrl);

  try {
    await download(downloadUrl, archivePath);
    console.log('[shodh-memory] Downloaded archive');

    // Extract
    extract(archivePath, BIN_DIR, platformInfo);
    console.log('[shodh-memory] Extracted binary');

    // Clean up archive
    fs.unlinkSync(archivePath);

    // Make executable (Unix)
    if (process.platform !== 'win32') {
      fs.chmodSync(binaryPath, 0o755);
    }

    console.log('[shodh-memory] Server binary installed at', binaryPath);
  } catch (err) {
    console.error('[shodh-memory] Failed to install binary:', err.message);
    console.log('[shodh-memory] You can manually download from:', `https://github.com/${REPO}/releases`);
  }
}

main();
