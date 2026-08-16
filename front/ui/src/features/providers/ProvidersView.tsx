import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { KeyRound, RefreshCw, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { clearProviderKey, listModels, listProviders, setProviderKey } from "@/lib/seat/client";
import type { ProviderInfo, SeatReachability } from "@/lib/seat/types";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { InfoHint } from "@/components/ui/info-hint";
import { Input } from "@/components/ui/input";
import { Meta, Stat } from "@/components/ui/meta";
import { ProviderLogo } from "@/components/ui/provider-logo";
import { ScrollArea } from "@/components/ui/scroll-area";
import { McpServers } from "./McpServers";
import { OAuthFlow } from "./OAuthFlow";
import { groupProviders, showAvailable } from "./groups";

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
 * reached.
 *
 * THREE PILES, NOT TWO. The reasoning for splitting a dead local endpoint out
 * of "Ready", and for a filter that outranks the fold, is in groups.ts — both
 * were wrong on the real install in ways that read as ordinary screens.
 *
 * A ROW NEVER LOSES ITS NAME. The action buttons did not shrink and the name
 * did, so at a 700px viewport the flex child holding "Anthropic" measured
 * zero pixels wide and the row overflowed its scroller by 51: a settings row
 * with three buttons and no subject. The row wraps now instead of competing
 * for one line, and the name carries a floor it cannot be squeezed below.
 */

/** Live only for a provider that can be called right now. A dead local
 *  endpoint used to take this green because the seat reports it `configured`;
 *  it is not an alarm either, so it takes the muted dot rather than `--warn` —
 *  nothing is broken, nothing is running. */
function StatusDot({ tone }: { tone: "live" | "muted" }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "size-1.5 shrink-0 rounded-full",
        tone === "live" ? "bg-[var(--live)]" : "bg-muted-foreground/40",
      )}
    />
  );
}

function ProviderRow({
  provider,
  variant,
}: {
  provider: ProviderInfo;
  variant: "ready" | "idle" | "available";
}) {
  const queryClient = useQueryClient();
  // Three auth shapes, three actions: an API-key form, a browser OAuth
  // flow, or nothing at all for keyless local endpoints. Never one field.
  const [mode, setMode] = useState<"idle" | "key" | "oauth">("idle");
  const editing = mode === "key";
  const setEditing = (open: boolean) => setMode(open ? "key" : "idle");
  const [draft, setDraft] = useState("");
  // Removing a stored key is destructive, immediate and unrecoverable — the
  // seat holds the only copy. A single click on an unlabelled icon was the
  // whole gesture; it now costs a second, deliberate one.
  const [confirmRemove, setConfirmRemove] = useState(false);

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
    onSuccess: (updated) => {
      patchCache(updated);
      setConfirmRemove(false);
    },
  });

  return (
    <div className="border-border border-b px-4 py-2">
      {/* `flex-wrap` with the actions in their own `ml-auto` group: at a width
          that cannot hold both, the buttons drop to a second line and the name
          keeps its own. `min-w-[9ch]` is the floor that stops the truncating
          child from being compressed to nothing before wrapping happens. */}
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5">
        <StatusDot tone={variant === "ready" ? "live" : "muted"} />
        {/* Hidden from the a11y tree on purpose. `ProviderLogo` labels itself
            — with the brand title when it has a mark, and with the raw pi id
            ("azure-openai-responses") when it falls back to a monogram — which
            is exactly right where it stands alone in a picker and exactly
            wrong here, an inch from the same provider's readable name. A
            screen reader would announce forty-three providers twice, once in
            machine form. */}
        <span aria-hidden="true" className="flex shrink-0">
          <ProviderLogo provider={provider.id} className="size-3.5" />
        </span>
        <span className="min-w-[9ch] flex-1 truncate text-[13px]">{provider.name}</span>
        <Meta className="shrink-0 flex-nowrap">
          {provider.local ? <span className="mono">local</span> : null}
          <Stat value={provider.model_count} label={provider.model_count === 1 ? "model" : "models"} />
          {/* An idle local's credential source is "local endpoint (keyless)",
              which its group heading, its `local` token and its zero model
              count have each already said. Three phrasings of one fact is how
              a dense row stops being scannable. */}
          {variant === "idle" ? null : (
            <span className="max-w-[180px] truncate" title={provider.source ?? undefined}>
              {provider.configured
                ? (provider.stored ? "stored key" : (provider.source ?? "configured"))
                : "no key"}
            </span>
          )}
        </Meta>
        <div className="ml-auto flex shrink-0 items-center gap-1">
          {provider.oauth_available ? (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setMode(mode === "oauth" ? "idle" : "oauth")}
              aria-expanded={mode === "oauth"}
              aria-label={`Sign in to ${provider.name}${provider.oauth_label ? ` (${provider.oauth_label})` : ""}`}
              title={
                provider.oauth_subscription
                  ? `Flat-rate plan sign-in${provider.oauth_label ? `: ${provider.oauth_label}` : ""}`
                  : undefined
              }
            >
              Sign in{provider.oauth_subscription ? " · plan" : ""}
            </Button>
          ) : null}
          {provider.accepts_api_key ? (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setEditing(!editing)}
              aria-expanded={editing}
              aria-label={`${provider.stored ? "Replace" : "Set"} API key for ${provider.name}`}
            >
              <KeyRound />
              {provider.stored ? "Replace key" : "Set key"}
            </Button>
          ) : null}
          {provider.stored ? (
            confirmRemove ? (
              <>
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-destructive"
                  disabled={remove.isPending}
                  aria-label={`Confirm removing the stored key for ${provider.name}`}
                  onClick={() => remove.mutate()}
                >
                  Remove key
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  aria-label={`Keep the stored key for ${provider.name}`}
                  onClick={() => setConfirmRemove(false)}
                >
                  Keep
                </Button>
              </>
            ) : (
              <Button
                size="icon"
                variant="ghost"
                aria-label={`Remove stored key for ${provider.name}`}
                onClick={() => setConfirmRemove(true)}
              >
                <Trash2 />
              </Button>
            )
          ) : null}
        </div>
      </div>

      {confirmRemove ? (
        <p className="text-muted-foreground mt-1 pl-4 text-[11px] leading-relaxed">
          The seat holds the only copy of this key. Removing it cannot be undone — you would need
          the key itself to set it again.
        </p>
      ) : null}
      {remove.isError ? (
        <p className="text-destructive mt-1 pl-4 text-[11px]">
          {remove.error instanceof Error ? remove.error.message : "Could not remove the key."}
        </p>
      ) : null}

      {editing ? (
        <form
          className="mt-2 flex flex-wrap items-center gap-2 pl-4"
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
            className="max-w-[360px] min-w-0 flex-1"
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

/** A group heading. Static for the two piles that are always meaningful, a
 *  button for the one that folds — never a heading that silently switches
 *  between the two roles. */
function GroupHeading({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-muted-foreground/70 border-border flex items-center gap-1.5 border-b px-4 pt-4 pb-1 text-[10px] tracking-wide uppercase">
      {children}
    </div>
  );
}

export function ProvidersView({ seat }: { seat: SeatReachability }) {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState("");
  const [expanded, setExpanded] = useState(false);

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

  const providers = useMemo(
    () => providersQuery.data?.providers ?? [],
    [providersQuery.data],
  );
  const groups = useMemo(() => groupProviders(providers, filter), [providers, filter]);
  const localErrors = refreshLocal.data?.local_errors ?? {};
  const availableOpen = showAvailable(groups, expanded);

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
      <div className="flex h-full min-h-0 w-full max-w-[720px] flex-col">
        <header className="shrink-0 px-4 pt-5 pb-3">
          <h2 className="text-[15px] font-medium tracking-tight">Provider access</h2>
          <p className="text-muted-foreground mt-1 text-[12px] leading-relaxed">
            Which model endpoints the seat can call. Keys are held by the local
            seat process — set here, stored in its credential file, never sent
            to a browser. Environment variables keep working; a key stored here
            takes precedence for that provider.
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
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
              aria-label="Re-probe local endpoints (Ollama, LM Studio, vLLM)"
            >
              <RefreshCw className={cn(refreshLocal.isPending && "animate-spin")} />
              Probe local endpoints
            </Button>
            {/* A filtered list is a reduced list, and it says so beside the
                control that reduced it. */}
            {groups.filtering && providersQuery.data ? (
              <Meta className="shrink-0">
                <Stat value={groups.matched} label={`of ${groups.total} shown`} />
              </Meta>
            ) : null}
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
          ) : providersQuery.isError ? (
            <p className="text-destructive border-border border-b px-4 py-3 text-[12px] leading-relaxed">
              Could not ask the seat which providers it has
              {providersQuery.error instanceof Error ? `: ${providersQuery.error.message}` : "."}{" "}
              This says nothing about the providers themselves — a key already stored keeps working.
            </p>
          ) : groups.matched === 0 ? (
            // Not two empty group headings and a stale paragraph about
            // configuration: a filter that matched nothing is one fact and one
            // way out of it.
            <div className="px-4 py-6">
              <p className="text-[13px]">Nothing here is called “{filter.trim()}”.</p>
              <p className="text-muted-foreground mt-1.5 text-[12px] leading-relaxed">
                The seat knows {groups.total} providers, matched on name and on the id used in
                configuration.
              </p>
              <Button size="sm" variant="outline" className="mt-2.5" onClick={() => setFilter("")}>
                Clear filter
              </Button>
            </div>
          ) : (
            <>
              {/* An empty pile is worth a heading when it is a statement about
                  the install, and worth nothing when it is an artefact of a
                  filter — under one, the row someone searched for is a few
                  pixels below under its own heading. */}
              {groups.ready.length > 0 || !groups.filtering ? (
                <div className="text-muted-foreground/70 border-border border-b px-4 pb-1 text-[10px] tracking-wide uppercase">
                  Ready ({groups.ready.length})
                </div>
              ) : null}
              {groups.ready.length === 0 ? (
                // Only ever rendered for the WHOLE list. Under a filter this
                // sentence was a false statement about the install — it read
                // "No provider is configured yet" on a machine with a stored
                // Anthropic key, because the filter had excluded it.
                groups.filtering ? null : (
                  <p className="text-muted-foreground border-border border-b px-4 py-3 text-[12px] leading-relaxed">
                    No provider is configured yet. Set an API key below, export the provider's
                    environment variable before starting the seat, or start Ollama / LM Studio for
                    keyless local models.
                  </p>
                )
              ) : (
                groups.ready.map((provider) => (
                  <ProviderRow key={provider.id} provider={provider} variant="ready" />
                ))
              )}

              {groups.idle.length > 0 ? (
                <>
                  <GroupHeading>
                    Local, nothing answering ({groups.idle.length})
                    <InfoHint label="Local, nothing answering">
                      These need no key, so the seat's credential check always passes them — which
                      is why they cannot be told apart from a working endpoint by configuration
                      alone. What separates them is that the seat holds no models for them: the
                      port was not answering when it last looked.
                      <br />
                      <br />
                      Start the program and press “Probe local endpoints”. Nothing polls in the
                      background; a retry loop against a machine with no local runtime would run
                      forever and report nothing.
                    </InfoHint>
                  </GroupHeading>
                  <p className="text-muted-foreground border-border border-b px-4 py-2 text-[12px] leading-relaxed">
                    Keyless, and the seat has no models from them. Start one and probe.
                  </p>
                  {groups.idle.map((provider) => (
                    <ProviderRow key={provider.id} provider={provider} variant="idle" />
                  ))}
                </>
              ) : null}

              {groups.available.length > 0 ? (
                <>
                  <button
                    type="button"
                    onClick={() => setExpanded((v) => !v)}
                    aria-expanded={availableOpen}
                    // Under a filter the pile is forced open and the control
                    // cannot close it — so it stops presenting itself as a
                    // toggle rather than offering one that does nothing.
                    disabled={groups.filtering}
                    className="text-muted-foreground/70 enabled:hover:text-foreground focus-visible:ring-ring w-full px-4 pt-4 pb-1 text-left text-[10px] tracking-wide uppercase focus-visible:ring-2 focus-visible:outline-none disabled:cursor-default"
                  >
                    {groups.filtering ? "" : availableOpen ? "▾ " : "▸ "}
                    Needs a key ({groups.available.length})
                  </button>
                  {availableOpen
                    ? groups.available.map((provider) => (
                        <ProviderRow key={provider.id} provider={provider} variant="available" />
                      ))
                    : null}
                </>
              ) : null}
            </>
          )}
          {/* The other half of "what can my agent reach": model endpoints
              above, the tool servers it can act through below. One question,
              so one screen. */}
          <McpServers />
        </ScrollArea>
      </div>
    </div>
  );
}
