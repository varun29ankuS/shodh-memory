#!/usr/bin/env node
/**
 * Setup Claude Code hooks for shodh-memory.
 *
 * Copies memory-hook.ts to ~/.claude/hooks/shodh-memory/
 * and merges hook configuration into ~/.claude/settings.json.
 *
 * Usage:
 *   npx @shodh/memory-mcp setup-hooks
 *   npx @shodh/memory-mcp setup-hooks --dry-run
 *   npx @shodh/memory-mcp setup-hooks --uninstall
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync } = require('child_process');

// ─── Paths ──────────────────────────────────────────────────────────────────

const HOME = os.homedir();
const CLAUDE_DIR = path.join(HOME, '.claude');
const HOOKS_DEST = path.join(CLAUDE_DIR, 'hooks', 'shodh-memory');
const SETTINGS_PATH = path.join(CLAUDE_DIR, 'settings.json');
const HOOK_SOURCE = path.join(__dirname, '..', '..', 'hooks', 'memory-hook.ts');
// Fallback: when installed via npm, hooks/ is not in the package.
// In that case, we download it from GitHub.
const REPO = 'varun29ankuS/shodh-memory';
const VERSION = require('../package.json').version;
const HOOK_URL = `https://raw.githubusercontent.com/${REPO}/v${VERSION}/hooks/memory-hook.ts`;

// ─── Hook configuration to merge ────────────────────────────────────────────

const HOOK_COMMAND_PREFIX = 'bun run';
const HOOK_SCRIPT = path.join(HOOKS_DEST, 'memory-hook.ts');

function makeCommand(event) {
  // Use forward slashes for cross-platform compat in the command string
  const scriptPath = HOOK_SCRIPT.replace(/\\/g, '/');
  return `${HOOK_COMMAND_PREFIX} ${scriptPath} ${event}`;
}

function buildHookConfig() {
  return {
    SessionStart: [
      {
        matcher: {},
        hooks: [{ type: 'command', command: makeCommand('SessionStart') }],
      },
    ],
    UserPromptSubmit: [
      {
        matcher: {},
        hooks: [{ type: 'command', command: makeCommand('UserPromptSubmit') }],
      },
    ],
    Stop: [
      {
        matcher: {},
        hooks: [{ type: 'command', command: makeCommand('Stop') }],
      },
    ],
    PreToolUse: [
      {
        matcher: { tool_name: ['Edit', 'Write', 'Bash'] },
        hooks: [{ type: 'command', command: makeCommand('PreToolUse') }],
      },
    ],
    PostToolUse: [
      {
        matcher: { tool_name: ['Edit', 'Write', 'Bash', 'TodoWrite', 'Read', 'Task'] },
        hooks: [{ type: 'command', command: makeCommand('PostToolUse') }],
      },
    ],
    SubagentStop: [
      {
        matcher: {},
        hooks: [{ type: 'command', command: makeCommand('SubagentStop') }],
      },
    ],
  };
}

// ─── Visual helpers ─────────────────────────────────────────────────────────

function printHeader() {
  process.stderr.write('\n');
  process.stderr.write('  \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n');
  process.stderr.write('  \u2551   \uD83D\uDC18 shodh-memory hooks setup           \u2551\n');
  process.stderr.write('  \u2551   Automatic memory for Claude Code       \u2551\n');
  process.stderr.write('  \u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D\n');
  process.stderr.write('\n');
}

// ─── Detect bun ─────────────────────────────────────────────────────────────

function isBunInstalled() {
  try {
    execSync('bun --version', { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

// ─── Download hook file ─────────────────────────────────────────────────────

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const https = require('https');
    const file = fs.createWriteStream(dest);

    const request = (url) => {
      https.get(url, (response) => {
        if (response.statusCode === 301 || response.statusCode === 302) {
          request(response.headers.location);
          return;
        }
        if (response.statusCode !== 200) {
          reject(new Error(`HTTP ${response.statusCode} downloading hook file`));
          return;
        }
        response.pipe(file);
        file.on('finish', () => { file.close(); resolve(); });
      }).on('error', (err) => {
        fs.unlink(dest, () => {});
        reject(err);
      });
    };

    request(url);
  });
}

// ─── Merge hooks into settings.json ─────────────────────────────────────────

/**
 * Merges shodh hooks into existing settings.json without clobbering other hooks.
 * For each event type, appends shodh entries to the array (deduplicates by command).
 */
function mergeHooks(existingSettings, newHooks) {
  const settings = JSON.parse(JSON.stringify(existingSettings)); // deep clone
  if (!settings.hooks) {
    settings.hooks = {};
  }

  for (const [event, entries] of Object.entries(newHooks)) {
    if (!settings.hooks[event]) {
      settings.hooks[event] = [];
    }

    for (const entry of entries) {
      const cmd = entry.hooks[0].command;
      // Check if this exact command already exists
      const alreadyExists = settings.hooks[event].some((existing) =>
        existing.hooks && existing.hooks.some((h) => h.command === cmd)
      );

      if (!alreadyExists) {
        settings.hooks[event].push(entry);
      }
    }
  }

  return settings;
}

/**
 * Removes all shodh-memory hooks from settings.json.
 */
function removeHooks(existingSettings) {
  const settings = JSON.parse(JSON.stringify(existingSettings));
  if (!settings.hooks) return settings;

  for (const [event, entries] of Object.entries(settings.hooks)) {
    if (!Array.isArray(entries)) continue;
    settings.hooks[event] = entries.filter((entry) => {
      if (!entry.hooks || !Array.isArray(entry.hooks)) return true;
      return !entry.hooks.some((h) => h.command && h.command.includes('shodh-memory'));
    });
    // Remove empty arrays
    if (settings.hooks[event].length === 0) {
      delete settings.hooks[event];
    }
  }

  // Remove empty hooks object
  if (Object.keys(settings.hooks).length === 0) {
    delete settings.hooks;
  }

  return settings;
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const uninstall = args.includes('--uninstall');

  printHeader();

  // ── Uninstall mode ──────────────────────────────────────────────────────
  if (uninstall) {
    process.stderr.write('  Removing shodh-memory hooks...\n\n');

    // Remove hook files
    if (fs.existsSync(HOOKS_DEST)) {
      if (!dryRun) {
        fs.rmSync(HOOKS_DEST, { recursive: true, force: true });
      }
      process.stderr.write(`  \u2713 Removed ${HOOKS_DEST}\n`);
    } else {
      process.stderr.write('  - Hook directory not found (already removed)\n');
    }

    // Remove from settings.json
    if (fs.existsSync(SETTINGS_PATH)) {
      const settings = JSON.parse(fs.readFileSync(SETTINGS_PATH, 'utf-8'));
      const cleaned = removeHooks(settings);
      if (!dryRun) {
        fs.writeFileSync(SETTINGS_PATH, JSON.stringify(cleaned, null, 2) + '\n');
      }
      process.stderr.write(`  \u2713 Cleaned ${SETTINGS_PATH}\n`);
    }

    process.stderr.write('\n  Hooks removed. Memory capture is now inactive.\n\n');
    return;
  }

  // ── Install mode ────────────────────────────────────────────────────────

  // Step 1: Check for bun
  if (!isBunInstalled()) {
    process.stderr.write('  \u2717 bun is required but not installed.\n');
    process.stderr.write('\n');
    process.stderr.write('  Install bun:\n');
    if (process.platform === 'win32') {
      process.stderr.write('    powershell -c "irm bun.sh/install.ps1 | iex"\n');
    } else {
      process.stderr.write('    curl -fsSL https://bun.sh/install | bash\n');
    }
    process.stderr.write('\n  Then re-run: npx @shodh/memory-mcp setup-hooks\n\n');
    process.exit(1);
  }
  process.stderr.write('  \u2713 bun detected\n');

  // Step 2: Check Claude Code directory
  if (!fs.existsSync(CLAUDE_DIR)) {
    process.stderr.write(`  \u2717 Claude Code config not found at ${CLAUDE_DIR}\n`);
    process.stderr.write('  Make sure Claude Code is installed and has been run at least once.\n\n');
    process.exit(1);
  }
  process.stderr.write('  \u2713 Claude Code detected\n');

  // Step 3: Copy hook file
  fs.mkdirSync(HOOKS_DEST, { recursive: true });

  const destFile = path.join(HOOKS_DEST, 'memory-hook.ts');

  if (fs.existsSync(HOOK_SOURCE)) {
    // Local source available (development or full repo clone)
    if (!dryRun) {
      fs.copyFileSync(HOOK_SOURCE, destFile);
    }
    process.stderr.write(`  \u2713 Hook copied to ${HOOKS_DEST}/\n`);
  } else {
    // Download from GitHub (npm install — hooks/ not in the package)
    process.stderr.write('  Downloading hook script...');
    try {
      if (!dryRun) {
        await downloadFile(HOOK_URL, destFile);
      }
      process.stderr.write(' \u2713\n');
    } catch (err) {
      process.stderr.write(` \u2717\n`);
      process.stderr.write(`  Failed to download hook: ${err.message}\n`);
      process.stderr.write(`  Manual download: https://github.com/${REPO}/blob/main/hooks/memory-hook.ts\n\n`);
      process.exit(1);
    }
  }

  // Step 4: Merge settings.json
  let settings = {};
  if (fs.existsSync(SETTINGS_PATH)) {
    try {
      settings = JSON.parse(fs.readFileSync(SETTINGS_PATH, 'utf-8'));
    } catch {
      process.stderr.write(`  \u26A0 Could not parse ${SETTINGS_PATH} — creating fresh config\n`);
      settings = {};
    }
  }

  // Backup settings.json before modifying
  if (fs.existsSync(SETTINGS_PATH) && !dryRun) {
    const backupPath = SETTINGS_PATH + '.bak';
    fs.copyFileSync(SETTINGS_PATH, backupPath);
  }

  const hookConfig = buildHookConfig();
  const merged = mergeHooks(settings, hookConfig);

  if (!dryRun) {
    fs.writeFileSync(SETTINGS_PATH, JSON.stringify(merged, null, 2) + '\n');
  }
  process.stderr.write(`  \u2713 Settings updated at ${SETTINGS_PATH}\n`);

  // Step 5: Summary
  process.stderr.write('\n');

  if (dryRun) {
    process.stderr.write('  [DRY RUN] No files were modified.\n');
    process.stderr.write('  The following changes would be made:\n');
    process.stderr.write(`    - Copy memory-hook.ts to ${HOOKS_DEST}/\n`);
    process.stderr.write(`    - Merge 6 hook events into ${SETTINGS_PATH}\n`);
    process.stderr.write(`    - Backup existing settings to ${SETTINGS_PATH}.bak\n`);
  } else {
    process.stderr.write('  Hooks will activate on your next Claude Code session.\n');
    process.stderr.write('  Shodh will automatically capture memories from conversations.\n');
    process.stderr.write('\n');
    process.stderr.write('  To remove: npx @shodh/memory-mcp setup-hooks --uninstall\n');
  }

  process.stderr.write('\n');
}

main().catch((err) => {
  process.stderr.write(`\n  \u2717 Setup failed: ${err.message}\n\n`);
  process.exit(1);
});
