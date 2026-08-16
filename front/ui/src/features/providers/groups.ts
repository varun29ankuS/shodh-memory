import type { ProviderInfo } from "@/lib/seat/types";

/**
 * Which pile each provider belongs in, and what a filter does to those piles.
 *
 * Both halves of this were getting a person's question wrong on the real
 * install, and neither failure looks like a bug on screen.
 *
 * READY WAS A CLAIM THE SEAT NEVER MADE. `listProviders`
 * (seat/src/models-registry.ts:251-280) sets `configured` from `checkAuth`,
 * which is a PRESENCE check for a credential — not a network round-trip. The
 * three local providers are keyless, so their check always succeeds and they
 * are permanently `configured: true` whether or not anything is listening on
 * the port. The screen was reading that as "Ready (4)" over Ollama, LM Studio
 * and vLLM with `0 models` each, on a machine where none of them was running.
 * `model_count` is the honest signal — it is the length of the model list the
 * registry actually holds for that provider — so a local endpoint with none is
 * separated out and named for what it is.
 *
 * A FILTER MUST NOT HIDE ITS OWN MATCHES. Unconfigured providers sit behind a
 * collapsed disclosure, which is right for a list of thirty-nine — and was
 * wrong the instant someone typed a name into the filter, because the one row
 * they asked for stayed folded away behind "Available (1)" while the empty
 * Ready group rendered "No provider is configured yet". Filtering therefore
 * overrides the fold; the collapsed state only governs the unfiltered list.
 */

export interface ProviderGroups {
  /** Configured, and the seat holds models for it: callable right now. */
  ready: ProviderInfo[];
  /** A keyless local endpoint with nothing behind it — start it and probe. */
  idle: ProviderInfo[];
  /** Reachable if a key is supplied. */
  available: ProviderInfo[];
  /** A filter is narrowing the list, so every count on screen is a count of
   *  matches and has to say so. */
  filtering: boolean;
  matched: number;
  total: number;
}

/**
 * A local endpoint the seat knows about but has no models from. Local
 * providers need no key, so "not configured" is never their problem and
 * "ready" is never earned by their credential check.
 */
export function isIdleLocal(provider: ProviderInfo): boolean {
  return provider.local && provider.model_count === 0;
}

export function matchesFilter(provider: ProviderInfo, needle: string): boolean {
  if (!needle) return true;
  return (
    provider.name.toLowerCase().includes(needle) || provider.id.toLowerCase().includes(needle)
  );
}

export function groupProviders(providers: ProviderInfo[], filter: string): ProviderGroups {
  const needle = filter.trim().toLowerCase();
  const matches = providers.filter((provider) => matchesFilter(provider, needle));
  const ready: ProviderInfo[] = [];
  const idle: ProviderInfo[] = [];
  const available: ProviderInfo[] = [];
  for (const provider of matches) {
    if (isIdleLocal(provider)) idle.push(provider);
    else if (provider.configured) ready.push(provider);
    else available.push(provider);
  }
  return {
    ready,
    idle,
    available,
    filtering: needle.length > 0,
    matched: matches.length,
    total: providers.length,
  };
}

/**
 * Whether the unconfigured pile is on screen.
 *
 * The fold is a convenience for a thirty-nine row list nobody scrolls; it is
 * not a filter, and it must never outrank one someone typed.
 */
export function showAvailable(groups: ProviderGroups, expanded: boolean): boolean {
  return groups.filtering || expanded;
}
