/**
 * Memories from one source usually share a template.
 *
 * An ingested corpus produces content like
 *
 *   [SHOD-15] Todo created in shodh-redb: [shodh-redb audit] M3: StorageError…
 *   [SHOD-22] Todo created in shodh-redb: [shodh-redb audit] H5: Guard clause…
 *
 * where the first ~55 characters are identical and the part a reader needs
 * starts after them. A citation chip has room for roughly 46, so left alone it
 * shows the template on every chip and truncates the content — which is how a
 * page ends up with 134 citations that all look the same.
 *
 * So: find the template a group of citations shares, and spend the label on
 * what differs.
 *
 * Two things this has to get right, and both were learned by being wrong:
 *
 * **A shared template is not a global one.** Requiring every label to share a
 * prefix means two stragglers erase it — measured on a live conversation, 44
 * of 46 memories carried the template and the other two ("Recall found nothing
 * useful for cue…") reduced the common prefix to the empty string. So the
 * template is found per group: any prefix shared by at least MIN_SAMPLES
 * labels counts, and a label that shares none is left alone. A conversation
 * touching two corpora gets two templates, which is the correct answer.
 *
 * **Digit RUNS collapse, not digits.** `[SHOD-15]` and `[SHOD-22]` share no
 * string prefix past `[SHOD-`, but they are the same template with a different
 * number. Collapsing runs makes the normalised string shorter than the
 * original by a different amount per label, so the prefix length is not an
 * index — it is walked back to a per-label cut. A first pass replaced digits
 * one-for-one to keep positions aligned and split identifiers of unequal width
 * down the middle: `[SHOD-1` + `5] Todo created in…`.
 *
 * The identifier is kept rather than stripped with the rest of the template.
 * It is the shortest thing distinguishing two memories, and an operator
 * reading `[SHOD-15]` on a chip can find that row in the evidence panel.
 */

/** Below this, stripping costs more in surprise than it returns in space. */
const MIN_TEMPLATE = 12;

/** A prefix shared by fewer labels than this is a coincidence, not a template. */
const MIN_SAMPLES = 3;

const normalise = (s: string) => s.replace(/\d+/g, "#");

function commonPrefixLength(a: string, b: string): number {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return i;
}

/** Where `cut` normalised characters land in the original string. Walks the two
 *  in step, consuming a whole digit run for each `#`. */
function originalCut(label: string, cut: number): number {
  let i = 0;
  let consumed = 0;
  while (consumed < cut && i < label.length) {
    if (/\d/.test(label[i])) {
      while (i < label.length && /\d/.test(label[i])) i++;
      consumed += 1; // the single "#" this run became
    } else {
      i++;
      consumed += 1;
    }
  }
  return i;
}

/**
 * Returns a function that trims whichever shared template a label belongs to.
 * Identity for labels that belong to none.
 */
export function templateStripper(labels: string[]): (label: string) => string {
  const usable = labels.filter((l) => l.length > 0);
  if (usable.length < MIN_SAMPLES) return (l) => l;

  // Sorting puts labels sharing a prefix next to each other, so the longest
  // prefix shared by MIN_SAMPLES labels is the LCP across some window of that
  // many consecutive entries. Each label takes the longest window it is in.
  const sorted = [...new Set(usable.map(normalise))].sort();
  const cutFor = new Map<string, number>();

  for (let start = 0; start + MIN_SAMPLES <= sorted.length; start++) {
    const end = start + MIN_SAMPLES - 1;
    // Sorted order means the first and last of a window bound the whole window.
    const shared = commonPrefixLength(sorted[start], sorted[end]);
    if (shared < MIN_TEMPLATE) continue;
    for (let i = start; i <= end; i++) {
      cutFor.set(sorted[i], Math.max(cutFor.get(sorted[i]) ?? 0, shared));
    }
  }

  if (cutFor.size === 0) return (l) => l;

  return (label) => {
    const cut = cutFor.get(normalise(label));
    if (!cut) return label;

    let at = originalCut(label, cut);

    // Templates end at a word boundary. Sorted neighbours share a few
    // characters of real content by chance -- three memories beginning
    // "Btree", "Buddy" and "Blob" share a "B" -- and cutting there eats the
    // first letter of the word the reader actually needs, which reads as a
    // rendering fault rather than as a trim.
    while (at > 0 && !/\s/.test(label[at - 1])) at--;
    if (at < MIN_TEMPLATE || label.length <= at) return label;

    // The last digit-bearing token inside the template is the identifier — the
    // one thing in there that differs between memories.
    const head = label.slice(0, at);
    const identifier = [...head.split(/\s+/).filter(Boolean)].reverse().find((t) => /\d/.test(t));

    const tail = label.slice(at).replace(/^[\s:;,\-—]+/, "");
    if (!tail) return label;
    return identifier ? `${identifier} ${tail}` : tail;
  };
}
