import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { MessageSquarePlus } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { createConversation, listModels } from "@/lib/seat/client";
import type { SeatModelInfo } from "@/lib/seat/types";
import { useSession } from "@/stores/session";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ModelPicker } from "./ModelPicker";

/**
 * The selection step: whose memory, which model — then converse.
 *
 * Profile and model are deliberately separate controls with separate
 * lifetimes. The profile selects which memory corpus the seat recalls from
 * and learns into (it lives in the rail, global); the model is per-
 * conversation and swappable mid-flight. Conflating them is how "sign in"
 * surfaces end up meaning three things at once.
 *
 * A fresh install holds no profiles. Creating one HERE is a deliberate,
 * named act — unlike the Recall search box, where a typo would silently
 * provision an empty store, a conversation is precisely the act of creating
 * memory, so naming a new profile is legitimate. Validation mirrors the
 * seat's own (seat/src/conversation.ts USER_ID_PATTERN, and the 128-char
 * backend cap minus the 13-char ".seat-harness" suffix the seat appends).
 */

const USER_ID_PATTERN = /^[A-Za-z0-9@._-]+$/;
const MAX_PROFILE_LENGTH = 128 - ".seat-harness".length;

function validateProfile(id: string): string | null {
  if (!id) return null;
  if (!USER_ID_PATTERN.test(id) || id.includes("..") || id.startsWith("."))
    return "Letters, digits, @ . _ - only; no leading dot.";
  if (id.length > MAX_PROFILE_LENGTH) return `At most ${MAX_PROFILE_LENGTH} characters.`;
  return null;
}

export function NewConversation({
  profiles,
  onCreated,
}: {
  profiles: string[];
  onCreated: (conversationId: string) => void;
}) {
  const navigate = useNavigate();
  const profile = useSession((s) => s.profile);
  const setProfile = useSession((s) => s.setProfile);
  const [model, setModel] = useState<SeatModelInfo | null>(null);
  const [newProfile, setNewProfile] = useState("");

  // Preselect a model so the page opens ready to converse. Starting at `null`
  // left "Start conversation" disabled with no visible reason — the control
  // reads "Choose model" and the button is simply dead until you find it,
  // which is a dead end rather than a choice.
  //
  // The pick is still the user's: this only fills the first frame, and the
  // ModelPicker below (and the swap control mid-conversation) override it
  // freely. Preference order is the most capable configured Anthropic model,
  // then any configured model at all, so a seat with only a local runtime
  // still lands on something real rather than nothing.
  const { data: catalog } = useQuery({
    queryKey: ["seat-models", "default-pick"],
    queryFn: ({ signal }) => listModels(false, signal),
    staleTime: 60_000,
  });
  useEffect(() => {
    if (model !== null) return;
    const all = catalog?.models ?? [];
    if (all.length === 0) return;
    const anthropic = all.filter((m) => m.provider === "anthropic");
    const preferred =
      anthropic.find((m) => m.id === "claude-opus-4-5") ??
      anthropic.find((m) => m.id.startsWith("claude-opus")) ??
      anthropic[0] ??
      all[0];
    if (preferred) setModel(preferred);
  }, [catalog, model]);

  const fresh = profiles.length === 0;
  const effectiveProfile = fresh ? newProfile.trim() : profile;
  const profileError = fresh ? validateProfile(newProfile.trim()) : null;

  const create = useMutation({
    mutationFn: () =>
      createConversation({
        user_id: effectiveProfile!,
        provider: model!.provider,
        model: model!.id,
      }),
    onSuccess: (created) => {
      if (fresh) setProfile(created.user_id);
      onCreated(created.conversation_id);
    },
  });

  const ready = Boolean(effectiveProfile) && !profileError && model !== null;

  return (
    <div className="grid h-full place-items-center overflow-y-auto px-6">
      <div className="w-full max-w-sm py-8">
        <MessageSquarePlus aria-hidden="true" className="text-muted-foreground size-5" />
        <h2 className="mt-3 text-[15px] font-medium tracking-tight">New conversation</h2>
        <p className="text-muted-foreground mt-1.5 text-[12px] leading-relaxed">
          Every turn shows what the seat recalled, why each memory scored where
          it did, and what it learned from your reaction — inspectable and
          revertible as it happens.
        </p>

        <div className="mt-6 flex flex-col gap-4">
          <div>
            <div className="text-muted-foreground/70 mb-1.5 text-[10px] tracking-wide uppercase">
              Memory profile
            </div>
            {fresh ? (
              <>
                <Input
                  value={newProfile}
                  onChange={(e) => setNewProfile(e.target.value)}
                  placeholder="Name a profile — e.g. varun"
                  aria-label="New profile name"
                />
                <p className="text-muted-foreground mt-1 text-[11px] leading-relaxed">
                  This instance holds no memory yet. The name you pick becomes
                  the store this conversation reads from and writes to.
                </p>
                {profileError ? (
                  <p className="text-destructive mt-1 text-[11px]">{profileError}</p>
                ) : null}
              </>
            ) : (
              <p className="text-[13px]">
                {profile ?? "—"}
                <span className="text-muted-foreground ml-2 text-[11px]">
                  switch profiles from the rail
                </span>
              </p>
            )}
          </div>

          <div>
            <div className="text-muted-foreground/70 mb-1.5 text-[10px] tracking-wide uppercase">
              Model
            </div>
            <ModelPicker
              current={model ? { provider: model.provider, id: model.id, name: model.name } : null}
              onSelect={async (picked) => setModel(picked)}
            />
            <p className="text-muted-foreground mt-1 text-[11px] leading-relaxed">
              Per-conversation, swappable mid-flight — the transcript and every
              piece of recalled evidence survive the switch.{" "}
              <button
                type="button"
                onClick={() => navigate("/providers")}
                className="hover:text-foreground focus-visible:ring-ring rounded underline underline-offset-2 focus-visible:ring-2 focus-visible:outline-none"
              >
                Manage provider access
              </button>
            </p>
          </div>

          <Button
            disabled={!ready || create.isPending}
            onClick={() => create.mutate()}
            className="self-start"
          >
            Start conversation
          </Button>

          {create.isError ? (
            <p className="text-destructive text-[11px] leading-relaxed">
              {create.error instanceof Error ? create.error.message : "Could not create the conversation."}
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}
