import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { KeyRound, RefreshCw, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { clearProviderKey, listModels, listProviders, setProviderKey } from "@/lib/seat/client";
import type { ProviderInfo, SeatReachability } from "@/lib/seat/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { Input } from "@/components/ui/input";
import { ProviderLogo } from "@/components/ui/provider-logo";
import { ScrollArea } from "@/components/ui/scroll-area";
import { OAuthFlow } from "./OAuthFlow";

/**
 * Sign in, stated precisely: provider credentials for the model endpoints the
 * seat can call. This is NOT an account system — shodh is local-first and has
 * no accounts. A key submitted here goes over the same-origin proxy to the
 * seat process, lands in its credential file, and never returns to any
 * browser; what this screen reads back is presence and pi's source label
 * ("ANTHROPIC_API_KEY", "stored key", "OAuth"), never material.
 *
 * Kept deliberately distinct from the profile control in the rail: the
 * profile chooses WHOSE MEMORY is on screen; this chooses WHICH MODELS can be
 * reached. Different questions, different lifetimes.
 */

function StatusDot({ configured }: { configured: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "size-1.5 shrink-0 rounded-full",
        configured ? "bg-[var(--live)]" : "bg-muted-foreground/40",
      )}
    />
  );
}

/**
 * What credential is actually working, in the seat's own words.
 *
 * `stored` means "the seat holds a credential for this provider", not "someone
 * pasted a key" — a browser OAuth login stores one too (ModelRegistry
 * .listProviders reads both out of the same credential file). Keying the label
 * off `stored` therefore described a Claude Pro sign-in as a "stored key",
 * which is the one thing about it that was not true. `auth_type` is the field
 * that distinguishes them, so it is the field this reads.
 *
 * Every branch returns something the seat sent — a plan label, pi's source
 * string — rather than a phrase composed here.
 */
function authLabel(provider: ProviderInfo): string {
  if (!provider.configured) return "not configured";
  if (provider.auth_type === "oauth") {
    // A flat-rate plan changes what a token means, so name it where the seat
    // named it; otherwise pi's own source string ("OAuth") is the whole fact.
    return provider.oauth_subscription
      ? (provider.oauth_label ?? provider.source ?? "OAuth")
      : (provider.source ?? "OAuth");
  }
  // An api_key credential in the seat's own file, versus one it found in the
  // environment — `source` names the variable in the second case, which is the
  // only actionable half of "where is this coming from".
  if (provider.stored) return "stored key";
  return provider.source ?? "configured";
}

function ProviderRow({ provider }: { provider: ProviderInfo }) {
  const queryClient = useQueryClient();
  // Three auth shapes, three actions: an API-key form, a browser OAuth
  // flow, or nothing at all for keyless local endpoints. Never one field.
  const [mode, setMode] = useState<"idle" | "key" | "oauth">("idle");
  const editing = mode === "key";
  const setEditing = (open: boolean) => setMode(open ? "key" : "idle");
  const [draft, setDraft] = useState("");

  const patchCache = (updated: ProviderInfo) => {
    queryClient.setQueryData<{ providers: ProviderInfo[] }>(["seat-providers"], (old) =>
      old
        ? { providers: old.providers.map((p) => (p.id === updated.id ? updated : p)) }
        : old,
    );
    // Configured providers gate the model catalog.
    void queryClient.invalidateQueries({ queryKey: ["seat-models"] });
  };

  const save = useMutation({
    mutationFn: (key: string) => setProviderKey(provider.id, key),
    onSuccess: (updated) => {
      patchCache(updated);
      setEditing(false);
      setDraft("");
    },
  });
  const remove = useMutation({
    mutationFn: () => clearProviderKey(provider.id),
    onSuccess: patchCache,
  });

  // The one credential the seat holds for this provider is either an OAuth
  // login or a key, never both, and three controls below phrase themselves
  // differently depending on which. Derived once so they cannot disagree.
  const signedIn = provider.configured && provider.auth_type === "oauth";
  const keyStored = provider.stored && provider.auth_type === "api_key";
  const signOutLabel = signedIn
    ? `Sign out of ${provider.name}`
    : `Remove stored key for ${provider.name}`;

  return (
    <div className="border-border border-b px-4 py-2.5">
      <div className="flex items-center gap-2.5">
        <StatusDot configured={provider.configured} />
        {/* The mark answers "which company is this" before the name is read,
            which is what makes a 43-row list scannable. `currentColor` at
            14px, so it never becomes a second accent. */}
        <ProviderLogo provider={provider.id} className="size-3.5" />
        <span className="min-w-0 flex-1 truncate text-[13px]">{provider.name}</span>
        {provider.local ? <Badge className="mono">local</Badge> : null}
        {/* Fixed columns, so 43 rows read down the page instead of each one
            placing its own metadata wherever its controls happened to end.
            Both hide below `sm`: at 420px the row cannot carry them and the
            provider's NAME, and the name is the row. The status dot still says
            configured or not, and the label is one tap away in `title`. */}
        <span className="text-muted-foreground mono hidden shrink-0 text-right text-[10px] sm:inline sm:w-[74px]">
          {provider.model_count} model{provider.model_count === 1 ? "" : "s"}
        </span>
        <span
          className="text-muted-foreground mono hidden shrink-0 truncate text-right text-[10px] sm:inline sm:w-[172px]"
          title={provider.source ?? undefined}
        >
          {authLabel(provider)}
        </span>
        {/* One gutter for every row's controls, whether it has three or none.
            Without it the metadata columns above cannot line up. */}
        <div className="flex shrink-0 items-center justify-end gap-1 sm:w-[236px]">
        {provider.oauth_available ? (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setMode(mode === "oauth" ? "idle" : "oauth")}
            aria-expanded={mode === "oauth"}
            // Already signed in: the same flow, but offering it as "Sign in"
            // reads as an unfinished step and invites someone to redo work that
            // is done. What it is actually for at that point is a login that
            // expired or a different account.
            aria-label={
              signedIn
                ? `Sign in to ${provider.name} again`
                : `Sign in to ${provider.name}${provider.oauth_label ? ` (${provider.oauth_label})` : ""}`
            }
            title={
              signedIn
                ? "Run the browser sign-in again — for an expired login or a different account"
                : provider.oauth_subscription
                  ? `Flat-rate plan sign-in${provider.oauth_label ? `: ${provider.oauth_label}` : ""}`
                  : undefined
            }
          >
            {signedIn ? "Sign in again" : `Sign in${provider.oauth_subscription ? " · plan" : ""}`}
          </Button>
        ) : null}
        {provider.accepts_api_key ? (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setEditing(!editing)}
            aria-expanded={editing}
            // "Replace key" only when there IS a key. With an OAuth login
            // stored, this control sets a key for the first time — and doing so
            // replaces the login, which is why it must not claim otherwise.
            aria-label={`${keyStored ? "Replace" : "Set"} API key for ${provider.name}`}
            title={keyStored ? "Replace the stored API key" : "Store an API key for this provider"}
          >
            <KeyRound />
            {/* The glyph carries it at 420px, where the words cost the
                provider's name. `aria-label` above is unconditional, so nothing
                is lost to a screen reader. */}
            <span className="hidden sm:inline">{keyStored ? "Replace key" : "Set key"}</span>
          </Button>
        ) : null}
        {provider.stored ? (
          <Button
            size="icon"
            variant="ghost"
            aria-label={signOutLabel}
            title={signOutLabel}
            disabled={remove.isPending}
            onClick={() => remove.mutate()}
          >
            <Trash2 />
          </Button>
        ) : null}
        </div>
      </div>

      {editing ? (
        <form
          className="mt-2 flex items-center gap-2 pl-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (draft.trim()) save.mutate(draft.trim());
          }}
        >
          <Input
            type="password"
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={`${provider.name} API key`}
            aria-label={`${provider.name} API key`}
            autoComplete="off"
            className="max-w-[360px]"
          />
          <Button type="submit" size="sm" disabled={!draft.trim() || save.isPending}>
            Save
          </Button>
          <Button type="button" size="sm" variant="ghost" onClick={() => setEditing(false)}>
            Cancel
          </Button>
        </form>
      ) : null}
      {save.isError ? (
        <p className="text-destructive mt-1 pl-4 text-[11px]">
          {save.error instanceof Error ? save.error.message : "Could not store the key."}
        </p>
      ) : null}

      {mode === "oauth" ? (
        <OAuthFlow
          provider={provider}
          onDone={(updated) => {
            if (updated) patchCache(updated);
            setMode("idle");
          }}
        />
      ) : null}
    </div>
  );
}

export function ProvidersView({ seat }: { seat: SeatReachability }) {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState("");
  const [showUnconfigured, setShowUnconfigured] = useState(false);

  const providersQuery = useQuery({
    queryKey: ["seat-providers"],
    queryFn: ({ signal }) => listProviders(signal),
    enabled: seat.state === "online",
    staleTime: 30_000,
  });

  const refreshLocal = useMutation({
    mutationFn: () => listModels(true),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["seat-providers"] });
      void queryClient.invalidateQueries({ queryKey: ["seat-models"] });
    },
  });

  const providers = providersQuery.data?.providers ?? [];
  const needle = filter.trim().toLowerCase();
  const matches = useMemo(
    () =>
      providers.filter(
        (provider) =>
          !needle ||
          provider.name.toLowerCase().includes(needle) ||
          provider.id.toLowerCase().includes(needle),
      ),
    [providers, needle],
  );
  const configured = matches.filter((provider) => provider.configured);
  const unconfigured = matches.filter((provider) => !provider.configured);
  const localErrors = refreshLocal.data?.local_errors ?? {};

  if (seat.state !== "online") {
    return (
      <EmptyState
        size="page"
        title="Seat not running"
        body="Provider access is managed by the seat harness, a local process that holds every credential server-side. Start it (seat/README.md) and this screen picks up on its own."
      />
    );
  }

  return (
    <div className="flex h-full min-h-0 justify-center overflow-hidden">
      {/* Wider than the 720px this started at. Once the rows gained aligned
          metadata columns and a control gutter, 720 left the provider NAME —
          the thing being scanned for — truncating at "Cloudflare AI Gate…".
          The prose keeps its own narrower measure below. */}
      <div className="flex h-full min-h-0 w-full max-w-[860px] flex-col">
        <header className="shrink-0 px-4 pt-5 pb-3">
          <h2 className="text-[15px] font-medium tracking-tight">Provider access</h2>
          <p className="text-muted-foreground mt-1 max-w-[640px] text-[12px] leading-relaxed">
            {/* "Credentials", not "keys": a browser sign-in stores one here
                too, and calling everything a key is what made a Claude plan
                read as a pasted secret in the rows below. */}
            Which model endpoints the seat can call. Credentials are held by the
            local seat process — set here, stored in its credential file, never
            sent to a browser. Environment variables keep working; anything
            stored here takes precedence for that provider.
          </p>
          <div className="mt-3 flex items-center gap-2">
            <Input
              type="search"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter providers…"
              aria-label="Filter providers"
              className="max-w-[240px]"
            />
            <Button
              size="sm"
              variant="outline"
              disabled={refreshLocal.isPending}
              onClick={() => refreshLocal.mutate()}
              aria-label="Re-probe local endpoints (Ollama, LM Studio)"
            >
              <RefreshCw className={cn(refreshLocal.isPending && "animate-spin")} />
              Probe local endpoints
            </Button>
          </div>
          {Object.entries(localErrors).map(([id, message]) => (
            <p key={id} className="text-warn mt-1.5 text-[11px]" title={message}>
              {id}: not reachable — start it and probe again.
            </p>
          ))}
        </header>

        <ScrollArea className="min-h-0 flex-1">
          {providersQuery.isPending ? (
            <p className="text-muted-foreground px-4 py-6 text-[12px]">Listing providers…</p>
          ) : (
            <>
              <div className="text-muted-foreground/70 border-border border-b px-4 pb-1 text-[10px] tracking-wide uppercase">
                Ready ({configured.length})
              </div>
              {configured.length === 0 ? (
                <p className="text-muted-foreground border-border border-b px-4 py-3 text-[12px] leading-relaxed">
                  No provider is configured yet. Set an API key below, export
                  the provider's environment variable before starting the seat,
                  or start Ollama / LM Studio for keyless local models.
                </p>
              ) : (
                configured.map((provider) => <ProviderRow key={provider.id} provider={provider} />)
              )}

              <button
                type="button"
                onClick={() => setShowUnconfigured((v) => !v)}
                aria-expanded={showUnconfigured}
                className="text-muted-foreground/70 hover:text-foreground focus-visible:ring-ring w-full px-4 pt-4 pb-1 text-left text-[10px] tracking-wide uppercase focus-visible:ring-2 focus-visible:outline-none"
              >
                {showUnconfigured ? "▾" : "▸"} Available ({unconfigured.length})
              </button>
              {showUnconfigured
                ? unconfigured.map((provider) => <ProviderRow key={provider.id} provider={provider} />)
                : null}
            </>
          )}
        </ScrollArea>
      </div>
    </div>
  );
}
