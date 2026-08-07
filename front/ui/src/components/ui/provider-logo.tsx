import {
  siAnthropic,
  siCloudflare,
  siDeepseek,
  siGithubcopilot,
  siGooglegemini,
  siHuggingface,
  siKimi,
  siLmstudio,
  siMeta,
  siMinimax,
  siMistralai,
  siMoonshotai,
  siNvidia,
  siOllama,
  siOpencode,
  siOpenrouter,
  siPerplexity,
  siQwen,
  siVercel,
  siVllm,
  siXiaomi,
  siZdotai,
} from "simple-icons";
import { cn } from "@/lib/utils";

/**
 * Provider identification marks.
 *
 * These are IDENTIFICATION, not decoration: they exist so "which model is this
 * running on" is answerable at a glance, from the overlay bar, the picker rows
 * and the Providers page. They render at 14-16px in `currentColor`, so they
 * inherit the surrounding text colour and cannot introduce a second accent —
 * and trademarks are never recoloured into the brand accent.
 *
 * Icons are imported BY NAME from `simple-icons` (MIT for the package, CC0 for
 * the icon data) so the bundler keeps only the paths listed above; the package
 * ships 3,453 icons and importing the namespace would pull all of them.
 *
 * OPENAI IS DELIBERATELY ABSENT. simple-icons does not carry it — the only
 * openai-ish export is `siOpenaigym`, an unrelated project — so OpenAI takes
 * the monogram fallback like any other unmapped provider. A near-miss logo is
 * worse than an honest letter: it would assert a brand identity that is not
 * the one being used.
 *
 * SO ARE, for the same reason and checked against the package rather than
 * assumed: Amazon Bedrock, Azure OpenAI, Groq, Together, Cerebras, Fireworks,
 * Baseten, Radius and Ant Ling. simple-icons carries no mark for any of them.
 * xAI is the sharpest trap here — `siX` exists and is the social network, not
 * the model lab — so xAI takes a monogram too.
 */

/** pi provider id → icon. Every id absent from this map falls back to a
 *  monogram, which is why an unmapped or newly added provider can never render
 *  a broken image or the wrong company's mark.
 *
 *  Ids come from the seat's own listing (`GET /v1/providers`); several
 *  providers appear as a family — a plan tier, a regional endpoint, a gateway —
 *  and each variant is listed explicitly rather than prefix-matched, so a
 *  future id that merely starts with a known string cannot inherit a mark that
 *  was never checked against it. */
const ICONS: Record<string, { path: string; title: string }> = {
  anthropic: { path: siAnthropic.path, title: siAnthropic.title },
  google: { path: siGooglegemini.path, title: siGooglegemini.title },
  "google-vertex": { path: siGooglegemini.path, title: siGooglegemini.title },
  mistral: { path: siMistralai.path, title: siMistralai.title },
  deepseek: { path: siDeepseek.path, title: siDeepseek.title },
  perplexity: { path: siPerplexity.path, title: siPerplexity.title },
  huggingface: { path: siHuggingface.path, title: siHuggingface.title },
  meta: { path: siMeta.path, title: siMeta.title },
  nvidia: { path: siNvidia.path, title: siNvidia.title },
  openrouter: { path: siOpenrouter.path, title: siOpenrouter.title },
  "github-copilot": { path: siGithubcopilot.path, title: siGithubcopilot.title },
  // Gateways carry the mark of whoever operates the gateway, which is what the
  // credential is actually for.
  "vercel-ai-gateway": { path: siVercel.path, title: siVercel.title },
  "cloudflare-ai-gateway": { path: siCloudflare.path, title: siCloudflare.title },
  "cloudflare-workers-ai": { path: siCloudflare.path, title: siCloudflare.title },
  opencode: { path: siOpencode.path, title: siOpencode.title },
  "opencode-go": { path: siOpencode.path, title: siOpencode.title },
  moonshotai: { path: siMoonshotai.path, title: siMoonshotai.title },
  "moonshotai-cn": { path: siMoonshotai.path, title: siMoonshotai.title },
  // Kimi is Moonshot's assistant and has its own mark; the seat lists it as a
  // separate provider ("Kimi For Coding"), so it gets the separate mark.
  "kimi-coding": { path: siKimi.path, title: siKimi.title },
  minimax: { path: siMinimax.path, title: siMinimax.title },
  "minimax-cn": { path: siMinimax.path, title: siMinimax.title },
  "qwen-token-plan": { path: siQwen.path, title: siQwen.title },
  "qwen-token-plan-cn": { path: siQwen.path, title: siQwen.title },
  "qwen-token-plan-individual": { path: siQwen.path, title: siQwen.title },
  xiaomi: { path: siXiaomi.path, title: siXiaomi.title },
  "xiaomi-token-plan-ams": { path: siXiaomi.path, title: siXiaomi.title },
  "xiaomi-token-plan-cn": { path: siXiaomi.path, title: siXiaomi.title },
  "xiaomi-token-plan-sgp": { path: siXiaomi.path, title: siXiaomi.title },
  zai: { path: siZdotai.path, title: siZdotai.title },
  "zai-coding-cn": { path: siZdotai.path, title: siZdotai.title },
  // The three local providers the seat registers (seat/src/models-registry.ts).
  ollama: { path: siOllama.path, title: siOllama.title },
  lmstudio: { path: siLmstudio.path, title: siLmstudio.title },
  vllm: { path: siVllm.path, title: siVllm.title },
};

export function ProviderLogo({
  provider,
  className,
}: {
  provider: string;
  className?: string;
}) {
  const icon = ICONS[provider];
  const size = cn("size-4 shrink-0", className);

  if (icon) {
    return (
      <svg
        viewBox="0 0 24 24"
        role="img"
        aria-label={icon.title}
        // `currentColor`, so a mark never fights the one-accent rule and always
        // reads against whatever surface it lands on.
        fill="currentColor"
        className={size}
      >
        <path d={icon.path} />
      </svg>
    );
  }

  // Monogram fallback: first letter in a rounded square, muted. Never a broken
  // image, never another provider's logo.
  return (
    <span
      aria-label={provider}
      role="img"
      className={cn(
        "bg-muted text-muted-foreground flex items-center justify-center rounded-[3px] text-[9px] font-semibold uppercase",
        size,
      )}
    >
      {provider.slice(0, 1)}
    </span>
  );
}
