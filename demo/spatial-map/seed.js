#!/usr/bin/env node
// seed.js — Warehouse demo: 2 robots, 2 missions, corridor-following paths
// Usage: node seed.js --api-key <key> [--host http://localhost:3030]

const http = require('http');
const https = require('https');

const args = process.argv.slice(2);
let API_KEY = '', HOST = 'http://localhost:3030';
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--api-key' && args[i + 1]) API_KEY = args[++i];
  if (args[i] === '--host' && args[i + 1]) HOST = args[++i];
}
if (!API_KEY) { console.error('Usage: node seed.js --api-key <key> [--host url]'); process.exit(1); }

const BASE = new URL(HOST);
const client = BASE.protocol === 'https:' ? https : http;

function post(path, body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = client.request({
      hostname: BASE.hostname, port: BASE.port, path, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY, 'Content-Length': Buffer.byteLength(data) },
    }, res => {
      let buf = '';
      res.on('data', c => buf += c);
      res.on('end', () => {
        if (res.statusCode >= 300 && res.statusCode !== 202) reject(new Error(`${path}: ${res.statusCode} ${buf}`));
        else { try { resolve(JSON.parse(buf)); } catch { resolve(buf); } }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

const delay = ms => new Promise(r => setTimeout(r, ms));
const RUN_TOKEN = Date.now().toString(36);

const NOW = new Date();
const M1_START = new Date(NOW.getTime() - 14 * 24 * 3600_000); // 14 days ago
const M2_START = new Date(NOW.getTime() - 30 * 60_000);         // 30 min ago

function ts(base, offsetMin) {
  return new Date(base.getTime() + offsetMin * 60_000).toISOString();
}

// Convert warehouse SVG pixel coords to fake geo
function toGeo(x, y) {
  return [37.775 + (y / 500) * 0.004, -122.420 + (x / 800) * 0.006, 0];
}

// ── MISSION 1: Alpha Inspection (14 days ago) ────────────────────────────────
// Alpha systematically scans each aisle via the main corridor
const M1_ID = 'warehouse_alpha_2026_03';
const MISSION1 = [
  { seq: 1, x: 60, y: 250,
    content: 'Entering warehouse through loading dock bay 2. Battery full, all sensors nominal. Beginning systematic aisle-by-aisle inspection via main corridor.',
    action: 'navigate', outcome: 'success', reward: 0.3, terrain: 'indoor',
    sensors: { battery: 100, temperature: 18, humidity: 45 },
    tags: ['start', 'loading-dock'], episode: 'alpha-sweep', importance: 0.3 },

  { seq: 2, x: 170, y: 60,
    content: 'Aisle 1 inspection complete. Traversed full length north to south. Floor condition good, shelving secure, no obstructions detected.',
    action: 'inspect', outcome: 'success', reward: 0.4, terrain: 'indoor',
    sensors: { battery: 95, temperature: 18, humidity: 46 },
    tags: ['aisle-1', 'clear'], episode: 'alpha-sweep', importance: 0.2 },

  { seq: 3, x: 270, y: 440,
    content: 'Aisle 2 inspection complete. Minor scuff marks on floor from forklift traffic. Within acceptable limits. No safety hazards.',
    action: 'inspect', outcome: 'success', reward: 0.4, terrain: 'indoor',
    sensors: { battery: 90, temperature: 18, humidity: 47 },
    tags: ['aisle-2', 'clear'], episode: 'alpha-sweep', importance: 0.2 },

  { seq: 4, x: 370, y: 300,
    content: 'HAZARD: Cracked concrete floor section in Aisle 3, center bay. Crack pattern 1.2m long, 8mm wide at deepest. Probable cause: heavy load impact or foundation settling. Uneven surface creates trip hazard for personnel and navigation risk for autonomous vehicles. Requires structural repair — concrete patch and cure time estimated 5-7 days.',
    action: 'inspect', outcome: 'failure', reward: -0.7, terrain: 'indoor',
    sensors: { battery: 84, temperature: 19, humidity: 48, crack_width_mm: 8.0, crack_length_m: 1.2, surface_deviation_mm: 12.0 },
    tags: ['aisle-3', 'floor-crack', 'structural', 'trip-hazard', 'repair-needed', 'critical'],
    episode: 'alpha-sweep', importance: 0.95, severity: 'critical', is_failure: true },

  { seq: 5, x: 470, y: 60,
    content: 'Aisle 4 inspection complete. Floor and shelving in good condition. Inventory labels 97% readable.',
    action: 'inspect', outcome: 'success', reward: 0.4, terrain: 'indoor',
    sensors: { battery: 78, temperature: 18, humidity: 47 },
    tags: ['aisle-4', 'clear'], episode: 'alpha-sweep', importance: 0.2 },

  { seq: 6, x: 570, y: 300,
    content: 'HAZARD: Damaged racking upright in Aisle 5, section C, level 2. Forklift impact bent the upright 18mm out of plumb. Load capacity reduced — risk of progressive collapse if additional pallets placed on upper levels. Requires structural assessment and upright replacement. Parts on order, ETA 10-14 business days.',
    action: 'inspect', outcome: 'failure', reward: -0.5, terrain: 'indoor',
    sensors: { battery: 72, temperature: 18, humidity: 48, deflection_mm: 18.0, load_capacity_pct: 60.0 },
    tags: ['aisle-5', 'racking-damage', 'forklift-impact', 'structural', 'collapse-risk', 'warning'],
    episode: 'alpha-sweep', importance: 0.85, severity: 'warning', is_failure: true },

  { seq: 7, x: 670, y: 440,
    content: 'Aisle 6 inspection complete. All clear. Shelving secure, floor in good condition.',
    action: 'inspect', outcome: 'success', reward: 0.4, terrain: 'indoor',
    sensors: { battery: 66, temperature: 18, humidity: 47 },
    tags: ['aisle-6', 'clear'], episode: 'alpha-sweep', importance: 0.2 },

  { seq: 8, x: 60, y: 250,
    content: 'Mission complete. 6 aisles inspected via main corridor route. 2 hazards found: cracked floor in Aisle 3 (critical — structural repair needed), damaged racking upright in Aisle 5 (warning — replacement parts on order). Both logged for maintenance tracking.',
    action: 'navigate', outcome: 'success', reward: 0.5, terrain: 'indoor',
    sensors: { battery: 58, temperature: 18, humidity: 46 },
    tags: ['mission-complete', 'summary'], episode: 'alpha-sweep', importance: 0.7 },
];

// ── MISSION 2: Beta Preventive Check (30 min ago, 14 days after Alpha) ──────
// Beta enters same warehouse, gets warned about hazards via memory recall
const M2_ID = 'warehouse_beta_2026_04';
const MISSION2 = [
  { seq: 1, x: 60, y: 250,
    content: 'Entering warehouse for preventive check. Robot Beta assigned — first time in this facility. Prior mission data available from Robot Alpha (14 days ago).',
    action: 'navigate', outcome: 'success', reward: 0.3, terrain: 'indoor',
    sensors: { battery: 100, temperature: 20, humidity: 48 },
    tags: ['start', 'loading-dock', 'preventive'], episode: 'beta-check', importance: 0.4 },

  { seq: 2, x: 170, y: 440,
    content: 'Aisle 1 quick scan. Clear. Proceeding along main corridor toward next aisle.',
    action: 'inspect', outcome: 'success', reward: 0.3, terrain: 'indoor',
    sensors: { battery: 97, temperature: 20, humidity: 49 },
    tags: ['aisle-1', 'clear'], episode: 'beta-check', importance: 0.2 },

  { seq: 3, x: 270, y: 60,
    content: 'Aisle 2 scan complete. No issues. Approaching Aisle 3 zone via main corridor.',
    action: 'inspect', outcome: 'success', reward: 0.3, terrain: 'indoor',
    sensors: { battery: 94, temperature: 20, humidity: 49 },
    tags: ['aisle-2', 'clear'], episode: 'beta-check', importance: 0.2 },

  { seq: 4, x: 340, y: 250,
    content: 'MEMORY RECALL: Approaching Aisle 3 entry. Spatial query surfaced prior finding from Robot Alpha (warehouse_alpha_2026_03, 14 days ago): cracked concrete floor section, 8mm wide, structural repair pending. Hazard may still be present — repair timeline was 5-7 days but no confirmation of completion. Decision: skip Aisle 3, proceed to Aisle 4 via corridor.',
    action: 'navigate', outcome: 'success', reward: 0.4, terrain: 'indoor',
    sensors: { battery: 91, temperature: 20, humidity: 50, prior_findings_nearby: 1 },
    tags: ['decision', 'cross-mission-recall', 'aisle-3-skip', 'floor-crack-avoidance'],
    episode: 'beta-check', importance: 0.9, type: 'Decision' },

  { seq: 5, x: 470, y: 440,
    content: 'Skipped Aisle 3 per recall decision. Aisle 4 scan complete — all clear. Continuing toward Aisle 5.',
    action: 'inspect', outcome: 'success', reward: 0.4, terrain: 'indoor',
    sensors: { battery: 87, temperature: 20, humidity: 49 },
    tags: ['aisle-4', 'clear', 'aisle-3-skipped'], episode: 'beta-check', importance: 0.3 },

  { seq: 6, x: 540, y: 250,
    content: 'MEMORY RECALL: Approaching Aisle 5. Prior finding from Robot Alpha: damaged racking upright, bent 18mm, collapse risk under load. Replacement parts were on order (ETA 10-14 days). May or may not be repaired. Decision: enter Aisle 5 at reduced speed, maintain 2m clearance from damaged section, do not place any loads.',
    action: 'navigate', outcome: 'success', reward: 0.4, terrain: 'indoor',
    sensors: { battery: 84, temperature: 20, humidity: 49, speed_pct: 40.0 },
    tags: ['decision', 'cross-mission-recall', 'aisle-5-caution', 'racking-avoidance'],
    episode: 'beta-check', importance: 0.8, type: 'Decision' },

  { seq: 7, x: 670, y: 60,
    content: 'Aisle 5 traversed at reduced speed. Damaged upright still visible — repair not yet completed. Documented current state. Aisle 6 clear.',
    action: 'inspect', outcome: 'partial', reward: 0.1, terrain: 'indoor',
    sensors: { battery: 78, temperature: 20, humidity: 49, deflection_mm: 17.5 },
    tags: ['aisle-5', 'damage-confirmed', 'aisle-6', 'clear'], episode: 'beta-check', importance: 0.6 },

  { seq: 8, x: 60, y: 250,
    content: 'Mission complete. Both previously flagged hazards confirmed still present. Aisle 3 floor crack: unrepaired. Aisle 5 racking: upright still bent. Escalating maintenance priority. No new hazards found. Zero incidents — prior knowledge enabled safe navigation.',
    action: 'navigate', outcome: 'success', reward: 0.5, terrain: 'indoor',
    sensors: { battery: 72, temperature: 20, humidity: 48 },
    tags: ['mission-complete', 'summary', 'escalation', 'zero-incidents'], episode: 'beta-check', importance: 0.8 },
];

// ── Lineage edges ────────────────────────────────────────────────────────────
const M1_EDGES = [
  [3, 4, 'Caused'],       // A2 clear → found crack in A3
  [4, 5, 'InformedBy'],   // crack → continued to A4
  [5, 6, 'Caused'],       // A4 clear → found racking damage in A5
  [6, 7, 'InformedBy'],   // damage → continued to A6
];
const M2_EDGES = [
  [3, 4, 'TriggeredBy'],  // A2 clear → recall triggered at A3
  [4, 5, 'Caused'],       // recall decision → skipped to A4
  [5, 6, 'TriggeredBy'],  // A4 clear → recall triggered at A5
  [6, 7, 'Caused'],       // recall decision → cautious traverse
];

const CROSS_EDGES = [
  ['M1', 4, 'M2', 4, 'InformedBy'],   // Alpha floor crack → Beta recall at A3
  ['M1', 4, 'M2', 5, 'InformedBy'],   // Alpha floor crack → Beta skip decision
  ['M1', 6, 'M2', 6, 'InformedBy'],   // Alpha racking damage → Beta recall at A5
  ['M2', 4, 'M1', 4, 'TriggeredBy'],  // Beta recall triggered by Alpha finding
];

// ── Main ───────────────────────────────────────────────────────────────────
async function main() {
  console.log('Cleaning old demo data...');
  for (const uid of ['spot-alpha', 'spot-beta']) {
    try {
      const r = await post('/api/forget/age', { user_id: uid, days_old: 0 });
      console.log(`  Cleared ${uid}: ${r.forgotten_count || 0} memories`);
    } catch (e) { console.warn(`  Clear ${uid}: ${e.message}`); }
  }
  await delay(500);

  console.log('\nSeeding warehouse missions...\n');
  const m1Ids = await seedMission(MISSION1, M1_ID, 'alpha-bot', M1_START, 'spot-alpha');
  const m2Ids = await seedMission(MISSION2, M2_ID, 'beta-bot', M2_START, 'spot-beta');
  console.log(`\nMemories: M1=${m1Ids.length}, M2=${m2Ids.length}`);

  let edgeCount = 0;
  edgeCount += await seedEdges(m1Ids, M1_EDGES, 'spot-alpha', 'M1');
  edgeCount += await seedEdges(m2Ids, M2_EDGES, 'spot-beta', 'M2');

  const missionIds = { M1: m1Ids, M2: m2Ids };
  for (const [fromM, fromSeq, toM, toSeq, relation] of CROSS_EDGES) {
    const fromId = missionIds[fromM][fromSeq - 1];
    const toId = missionIds[toM][toSeq - 1];
    if (!fromId || !toId) { console.warn(`  Skip: ${fromM}.WP${fromSeq}->${toM}.WP${toSeq}`); continue; }
    const userId = toM === 'M2' ? 'spot-beta' : 'spot-alpha';
    try {
      await post('/api/lineage/link', { user_id: userId, from_memory_id: fromId, to_memory_id: toId, relation });
      console.log(`  Cross: ${fromM}.WP${fromSeq} --${relation}--> ${toM}.WP${toSeq}`);
      edgeCount++;
    } catch (e) { console.warn(`  Cross-edge failed: ${e.message}`); }
    await delay(50);
  }
  console.log(`\nEdges: ${edgeCount}`);

  console.log('\nReinforcing Alpha floor crack as helpful...');
  try {
    await post('/api/reinforce', { user_id: 'spot-alpha', ids: [m1Ids[3]], outcome: 'helpful' });
    await delay(200);
    await post('/api/reinforce', { user_id: 'spot-alpha', ids: [m1Ids[3]], outcome: 'helpful' });
    console.log('  Reinforced x2');
  } catch (e) { console.warn(`  ${e.message}`); }

  console.log('\nTriggering consolidation...');
  try {
    await post('/api/consolidate', { user_id: 'spot-alpha', min_support: 1, min_age_days: 0 });
    console.log('  Done');
  } catch (e) { console.warn(`  ${e.message}`); }

  console.log('\nSeed complete. Open http://localhost:8080');
}

async function seedMission(waypoints, missionId, robotId, startTime, userId) {
  const ids = [];
  console.log(`  ${missionId} (${waypoints.length} waypoints):`);
  for (let i = 0; i < waypoints.length; i++) {
    const wp = waypoints[i];
    const payload = {
      user_id: userId,
      content: `[${RUN_TOKEN}] ${wp.content}`,
      tags: wp.tags, memory_type: wp.type || 'Observation',
      robot_id: robotId, mission_id: missionId,
      geo_location: toGeo(wp.x, wp.y), local_position: [wp.x, wp.y, 0],
      action_type: wp.action, sensor_data: wp.sensors,
      outcome_type: wp.outcome, reward: wp.reward, terrain_type: wp.terrain,
      episode_id: wp.episode, sequence_number: wp.seq,
      importance: wp.importance, created_at: ts(startTime, i * 3),
    };
    if (wp.severity) payload.severity = wp.severity;
    if (wp.is_failure) payload.is_failure = true;
    try {
      const result = await post('/api/remember', payload);
      ids.push(result.id);
      console.log(`    WP${wp.seq}: ${result.id.slice(0, 8)}${wp.outcome === 'failure' ? ' ⚠' : ''}`);
    } catch (e) { console.error(`    WP${wp.seq} FAILED: ${e.message}`); ids.push(null); }
    await delay(100);
  }
  return ids;
}

async function seedEdges(ids, edges, userId, label) {
  let count = 0;
  for (const [fromIdx, toIdx, relation] of edges) {
    const fromId = ids[fromIdx - 1], toId = ids[toIdx - 1];
    if (!fromId || !toId) continue;
    try {
      await post('/api/lineage/link', { user_id: userId, from_memory_id: fromId, to_memory_id: toId, relation });
      console.log(`  ${label}: WP${fromIdx} --${relation}--> WP${toIdx}`);
      count++;
    } catch (e) { console.warn(`  ${label} edge failed: ${e.message}`); }
    await delay(50);
  }
  return count;
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
