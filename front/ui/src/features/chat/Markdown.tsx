import { memo, useMemo } from "react";
import { Marked } from "marked";
import DOMPurify from "dompurify";

import { templateStripper } from "./shared-template";

/**
 * Assistant text: markdown, sanitized, with memory citations as live elements.
 *
 * Two dependencies carry this and both earn their place. `marked` because
 * every model the seat runs emits markdown, and rendering it as plain text
 * fails the bar this UI is held to (code blocks, lists, tables). `dompurify`
 * because the output of an LLM is untrusted input by definition, and this app
 * ships as an embedded page with no CSP — `dangerouslySetInnerHTML` without a
 * sanitizer is not an option. (react-markdown was the alternative: renders to
 * elements without innerHTML, but at roughly four times the bundle weight of
 * these two combined.)
 *
 * Citations: the seat's system prompt asks the model to cite memories inline
 * as `[mem:<8-hex-id>]` (seat/src/conversation.ts BASE_SYSTEM_PROMPT), and the
 * reinforcement loop keys on exactly that pattern (seat/src/feedback.ts). The
 * same pattern is rewritten here into a <button data-mem> BEFORE markdown
 * parsing, so a citation becomes the thing it is: a handle onto the evidence
 * panel. Buttons and data-* attributes survive DOMPurify's defaults; click
 * handling is delegated from the container, so the sanitized HTML carries no
 * handlers at all.
 *
 * The chip shows the MEMORY, not the id. `mem:4a59ea4b` is the protocol
 * between the seat and the model; it is not a thing a reader knows, and a
 * paragraph of them reads as redacted text. The id stays — as `data-mem` for
 * the click target and in the tooltip, because provenance you cannot copy is
 * not provenance — but the visible label is what the memory actually says.
 *
 * Resolution is CONVERSATION-scoped, not turn-scoped, and that took a
 * measurement to learn: a first pass scoped it to the turn on the reasoning
 * that a citation the turn did not surface should honestly show its raw
 * handle. On a live conversation that was 77 citations, 0 resolved. Models
 * cite what they saw three turns ago as readily as what they just recalled,
 * so the fallback was not an edge case, it was the whole feature.
 *
 * The fallback remains for a genuinely unknown id, which is now rare enough to
 * mean something when it appears.
 */

const marked = new Marked({ gfm: true, breaks: false, async: false });

/** Matches seat/src/feedback.ts extractCitations: [mem:<hex-ish id>]. */
const CITATION = /\[mem:([0-9a-fA-F-]{4,36})\]/g;

/** Long enough to identify the memory in a sentence, short enough that three
 *  in a row still read as prose rather than as three quotations. */
const LABEL_CHARS = 46;

/** What a citation resolves to. `type` is shown as a prefix so a Decision and
 *  an Observation with similar wording stay distinguishable. */
export type CitedMemory = { content: string; memory_type?: string };

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  // External links must not inherit this window (embedded, local-first app).
  if (node.tagName === "A" && node.getAttribute("href")) {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noreferrer noopener");
  }
});

/** Escaped at insertion rather than relying on the sanitizer: memory content is
 *  arbitrary text, and a `<` in it would otherwise open a tag mid-attribute
 *  before DOMPurify ever sees well-formed HTML. */
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * One line, collapsed, template stripped, trimmed to fit inline.
 *
 * The type prefix is dropped once the template is gone. It earns its place
 * when memories differ in kind, and on a corpus where every one is a Task it
 * spends six of the forty-six characters saying so on every chip.
 */
function label(memory: CitedMemory, strip: (s: string) => string): string {
  const flat = strip(memory.content.replace(/\s+/g, " ").trim());
  const body = flat.length > LABEL_CHARS ? `${flat.slice(0, LABEL_CHARS - 1).trimEnd()}…` : flat;
  return body;
}

function render(text: string, cited: Map<string, CitedMemory>): string {
  // Computed across every memory the conversation could cite, not just the
  // ones in this message: the template is a property of the corpus, and
  // deriving it per message would make the same memory read differently in
  // two paragraphs.
  const strip = templateStripper([...cited.values()].map((m) => m.content.replace(/\s+/g, " ").trim()));

  const withCitations = text.replace(CITATION, (_all, id: string) => {
    const short = id.slice(0, 8).toLowerCase();
    const memory = cited.get(short);
    const shown = (memory ? label(memory, strip) : `mem:${short}`).replace(/\s+/g, " ");
    // The full id lives in the tooltip so it stays copyable, and the memory's
    // own text lives there too when the label had to be cut. Single line,
    // always. This string goes into an HTML attribute inside
    // MARKDOWN source, and marked ends an inline HTML block at a blank line --
    // a newline here splits the <button> across two paragraphs and the rest of
    // the tag renders as visible prose. Collapse everything.
    const tip = memory
      ? `${memory.content.replace(/\s+/g, " ").trim()} — mem:${short}`
      : `mem:${short}`;
    return `<button type="button" class="mem-cite" data-mem="${escapeHtml(short)}" title="${escapeHtml(tip)}">${escapeHtml(shown)}</button>`;
  });
  const html = marked.parse(withCitations) as string;
  return DOMPurify.sanitize(html);
}

export const Markdown = memo(function Markdown({
  text,
  cited,
  onCitationClick,
}: {
  text: string;
  /** shortId → the memory it names. Absent entries render as the raw id. */
  cited?: Map<string, CitedMemory>;
  /** Receives the short id from the citation (lowercased, as printed). */
  onCitationClick?: (shortId: string) => void;
}) {
  // `cited` is in the deps deliberately: memories stream in during a turn, and
  // memoizing on `text` alone would leave every chip frozen as a raw id for
  // the rest of the conversation.
  const html = useMemo(() => render(text, cited ?? new Map()), [text, cited]);
  return (
    <div
      className="md text-[13px] leading-relaxed"
      // Sanitized above; see file header.
      dangerouslySetInnerHTML={{ __html: html }}
      onClick={(e) => {
        const target = (e.target as HTMLElement).closest?.("[data-mem]");
        if (target instanceof HTMLElement && target.dataset.mem && onCitationClick) {
          e.preventDefault();
          onCitationClick(target.dataset.mem);
        }
      }}
    />
  );
});
