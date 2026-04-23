import { runSync } from "./sync";
import { getTasks, getProjects } from "./todoist";
import { config } from "./config";

async function showOverview(): Promise<void> {
  console.log("\n📋 Todoist Overview");
  console.log("─".repeat(40));

  const projects = await getProjects();
  console.log(`\n📁 Projects (${projects.length}):`);
  for (const p of projects) {
    console.log(`   • ${p.name}`);
  }

  const tasks = await getTasks();
  console.log(`\n✅ Active Tasks (${tasks.length}):`);
  for (const t of tasks.slice(0, 10)) {
    const priority = t.priority === 4 ? "🔴" : t.priority === 3 ? "🟠" : t.priority === 2 ? "🟡" : "⚪";
    const due = t.due ? ` (${t.due.string})` : "";
    console.log(`   ${priority} ${t.content}${due}`);
  }
  if (tasks.length > 10) {
    console.log(`   ... and ${tasks.length - 10} more`);
  }
}

async function startDaemon(): Promise<void> {
  console.log("\n✅ shodh-todoist starting...");
  console.log(`🔑 Todoist: Connected`);
  console.log(`🐘 Memory: ${config.shodh.apiUrl}`);
  console.log(`⏱️  Sync interval: ${config.sync.intervalMs / 1000}s`);

  await showOverview();
  await runSync();

  setInterval(async () => {
    try {
      await runSync();
    } catch (err) {
      console.error("Sync error:", err);
    }
  }, config.sync.intervalMs);

  console.log("\n✅ Daemon running. Press Ctrl+C to stop.");
}

async function main(): Promise<void> {
  if (!config.todoist.apiToken) {
    console.error("❌ TODOIST_API_TOKEN is required");
    process.exit(1);
  }

  const mode = process.argv[2] || "daemon";

  switch (mode) {
    case "sync":
      await runSync();
      break;
    case "overview":
      await showOverview();
      break;
    case "daemon":
    default:
      await startDaemon();
      break;
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
