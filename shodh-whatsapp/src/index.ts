import makeWASocket, {
  DisconnectReason,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  makeCacheableSignalKeyStore,
  type WASocket,
} from "@whiskeysockets/baileys";
import { Boom } from "@hapi/boom";
import pino from "pino";
import qrcode from "qrcode-terminal";
import { handleMessage, getInboxSummary, clearInbox } from "./handler";
import { config } from "./config";

const OWNER_NUMBER = "919810300618@s.whatsapp.net";

async function handleOwnerCommand(sock: WASocket, jid: string, text: string): Promise<boolean> {
  if (jid !== OWNER_NUMBER) return false;

  const cmd = text.toLowerCase().trim();

  if (cmd === "/inbox" || cmd === "/summary" || cmd === "inbox" || cmd === "summary") {
    const summary = getInboxSummary();
    await sock.sendMessage(jid, { text: summary });
    return true;
  }

  if (cmd === "/clear" || cmd === "clear inbox") {
    clearInbox();
    await sock.sendMessage(jid, { text: "✅ Inbox cleared." });
    return true;
  }

  if (cmd === "/help" || cmd === "help") {
    await sock.sendMessage(jid, {
      text: `*Keshav Commands:*\n\n/inbox - Get message summary\n/clear - Clear inbox\n/help - Show this help`,
    });
    return true;
  }

  return false;
}

const logger = pino({ level: config.debug ? "debug" : "silent" });

async function connectToWhatsApp(): Promise<WASocket> {
  const { state, saveCreds } = await useMultiFileAuthState("./auth");
  const { version, isLatest } = await fetchLatestBaileysVersion();

  console.log(`\n🙏 Keshav starting...`);
  console.log(`📱 WhatsApp Web version: ${version.join(".")} (latest: ${isLatest})`);
  console.log(`🐘 Memory: ${config.shodh.apiUrl}`);
  console.log(`🤖 LLM: ${config.llm.provider}`);

  if (config.whatsapp.autoReplyAll) {
    console.log(`✅ Responding to all messages`);
  } else if (config.whatsapp.allowedContacts.length > 0) {
    console.log(`✅ Allowed contacts: ${config.whatsapp.allowedContacts.join(", ")}`);
  }

  if (config.whatsapp.blockedContacts.length > 0) {
    console.log(`🚫 Blocked: ${config.whatsapp.blockedContacts.join(", ")}`);
  }

  const sock = makeWASocket({
    version,
    logger,
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, logger),
    },
    printQRInTerminal: false,
    generateHighQualityLinkPreview: true,
  });

  sock.ev.on("connection.update", async (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      console.log("\n📱 Scan QR code to connect:\n");
      qrcode.generate(qr, { small: true });
    }

    if (connection === "close") {
      const reason = (lastDisconnect?.error as Boom)?.output?.statusCode;

      if (reason === DisconnectReason.loggedOut) {
        console.log("❌ Logged out. Delete ./auth folder and restart.");
        process.exit(1);
      }

      console.log(`⚠️  Connection closed (${reason}), reconnecting...`);
      setTimeout(() => connectToWhatsApp(), 3000);
    }

    if (connection === "open") {
      console.log("\n✅ Keshav is online!");
      console.log("🎧 Listening for messages...\n");
    }
  });

  sock.ev.on("creds.update", saveCreds);

  sock.ev.on("messages.upsert", async ({ messages, type }) => {
    if (type !== "notify") return;

    for (const message of messages) {
      const jid = message.key.remoteJid;
      const text = message.message?.conversation || message.message?.extendedTextMessage?.text;

      // Check for owner commands first
      if (jid && text && !message.key.fromMe) {
        const handled = await handleOwnerCommand(sock, jid, text);
        if (handled) continue;
      }

      await handleMessage(sock, message);
    }
  });

  return sock;
}

async function main() {
  const provider = config.llm.provider;

  if (provider === "anthropic" && !config.llm.anthropic.apiKey) {
    console.error("❌ ANTHROPIC_API_KEY is required for anthropic provider");
    process.exit(1);
  }
  if (provider === "groq" && !config.llm.groq.apiKey) {
    console.error("❌ GROQ_API_KEY is required for groq provider");
    process.exit(1);
  }
  if (provider === "openai" && !config.llm.openai.apiKey) {
    console.error("❌ OPENAI_API_KEY is required for openai provider");
    process.exit(1);
  }

  try {
    await connectToWhatsApp();
  } catch (error) {
    console.error("❌ Fatal error:", error);
    process.exit(1);
  }
}

main();
