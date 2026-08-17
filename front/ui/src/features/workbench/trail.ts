import { DESTINATIONS } from "@/components/layout/destinations";

/**
 * The trail — where you have been, kept in the URL.
 *
 * THE BRIEFING IS NOT A PLACE YOU LEAVE. Selecting a door on the front page
 * used to navigate away from it, and the screen you arrived at had no memory
 * that the briefing existed. The trail replaces that: opening something
 * PROMOTES it to the stage and COMPRESSES what you opened it from into a
 * spine, so the way back is always on screen and always says what it goes
 * back to.
 *
 * ENCODING. The trail is `[briefing, ...via, primary]` where
 *
 *   - the PRIMARY pane is the route's own pathname, and
 *   - `?via=` lists the ancestors between the briefing and it, by id.
 *
 * The primary is the pathname and not a search parameter, and that is a
 * correctness requirement rather than a style choice. Six behaviours outside
 * this module are keyed on `location.pathname` — the conversation dock returns
 * `null` on `/chat` so it does not draw itself on top of itself, the Inspector
 * and the cue field appear on named routes, the header reads its caption from
 * the path, and the rail's active row is a `NavLink` match. Moving the primary
 * into a parameter would silently break every one of them.
 *
 * The briefing is implicit at index 0 rather than written into `via`. It is
 * the base of every trail — a URL that named it would be one more thing that
 * can disagree with itself, and a deep link that omitted it would be a screen
 * with no way home.
 *
 * PROMOTION TRUNCATES. Opening something from pane N discards everything
 * downstream of N. That is the semantic that makes this a trail and not a tab
 * bar: panes that survive being backed out of are tabs, and a tab bar is a set
 * of places you are simultaneously in, which is the opposite of one primary at
 * a time. It also falls out for free from `navigate("/recall")` written
 * anywhere in the product, because a bare path carries no `via` — a promotion
 * from the briefing is the default rather than a thing every caller must
 * remember to encode.
 *
 * OPENING SOMETHING ALREADY IN THE TRAIL GOES TO IT. It does not append a
 * second copy. Two spines with the same title are indistinguishable to a
 * reader, and a trail that can contain `recall → graph → recall` is a
 * navigation history, not a path.
 *
 * AN ABSENT `via` IS NOT AN EMPTY ONE, and this distinction is what lets the
 * whole product participate in the trail without every view knowing it exists:
 *
 *   - `?via=` PRESENT (even empty) means the link states its own ancestry.
 *     `hrefFor` always emits it, so every spine, every back and every rail row
 *     is explicit.
 *   - `via` ABSENT means a bare `navigate("/recall")` written somewhere that
 *     has never heard of a trail — the conversation dock, the evidence panel,
 *     the briefing's doors. The workbench reads that as "open this FROM WHERE
 *     I AM" and promotes it onto the current trail.
 *
 * Without that, every promotion in the product would come from a bare path,
 * every trail would be two panes deep, and the stack would never be a stack.
 *
 * A RAIL CLICK RESETS THE TRAIL to `[briefing, destination]`, and this is not
 * in the spec, so it is recorded here: the rail is a jump to a place, not an
 * opening-from-here. That is why its rows link to `railHref` — an explicit
 * empty ancestry — rather than to a bare path.
 */

export interface Pane {
  /** URL token. Stable, and short enough to read in a shared link. */
  readonly id: string;
  /** The route this pane renders. Also the primary's pathname. */
  readonly path: string;
  /** What the spine says, at full size. */
  readonly title: string;
  /** What the pane is, for the spine's accessible name. */
  readonly caption: string;
}

/** The search parameter carrying the ancestors. */
export const VIA_PARAM = "via";

const PANES: readonly Pane[] = DESTINATIONS.map((d) => ({
  id: d.id,
  path: d.path,
  title: d.label,
  caption: d.caption,
}));

const BY_ID = new Map(PANES.map((p) => [p.id, p]));
const BY_PATH = new Map(PANES.map((p) => [p.path, p]));

const root = BY_PATH.get("/");
if (!root) {
  // A programming error, not a runtime condition: every trail is rooted at the
  // briefing, so a destination table with no `/` entry would leave every
  // screen in the product with no way back. Failing at module load is the only
  // place this can be caught before it reaches a person.
  throw new Error("destinations.ts must contain an entry whose path is '/'");
}

/** The pane every trail starts at. */
export const ROOT: Pane = root;

export function paneById(id: string): Pane | null {
  return BY_ID.get(id) ?? null;
}

export function paneByPath(path: string): Pane | null {
  return BY_PATH.get(path) ?? null;
}

/**
 * The trail a location describes.
 *
 * Unknown ids in `via` are dropped rather than rendered as a pane with no
 * name — a stale or hand-edited link degrades to a shorter trail, which is
 * still a working screen. Duplicates and a `via` entry naming the primary or
 * the briefing are dropped for the same reason: they would produce two spines
 * a reader cannot tell apart.
 *
 * An unknown pathname yields the briefing alone. The router sends unknown
 * hashes home anyway; this makes the trail agree with it instead of rendering
 * a spine for a pane that is not on screen.
 */
export function parseTrail(pathname: string, search: string): Pane[] {
  const primary = BY_PATH.get(pathname);
  if (!primary || primary.id === ROOT.id) return [ROOT];

  const raw = new URLSearchParams(search).get(VIA_PARAM);
  if (!raw) return [ROOT, primary];

  const via: Pane[] = [];
  const seen = new Set<string>([ROOT.id, primary.id]);
  for (const token of raw.split(",")) {
    const pane = BY_ID.get(token.trim());
    if (!pane || seen.has(pane.id)) continue;
    seen.add(pane.id);
    via.push(pane);
  }
  return [ROOT, ...via, primary];
}

/**
 * The link that makes pane `index` the primary, keeping everything before it.
 *
 * Everything AFTER it is dropped, which is what a spine click means: you have
 * gone back to that pane, and what you opened from it is no longer where you
 * are. Re-opening is one click, and it is the click you are already looking
 * at.
 */
export function hrefFor(trail: readonly Pane[], index: number): string {
  const pane = trail[index];
  if (!pane || index <= 0) return ROOT.path;
  // Ids are lowercase slugs from our own table, so the comma is the only
  // character with meaning here and encodeURIComponent leaves the rest alone.
  // Built by hand rather than with URLSearchParams, which would percent-encode
  // the separator and make a shared link unreadable.
  const list = trail
    .slice(1, index)
    .map((p) => encodeURIComponent(p.id))
    .join(",");
  // The parameter is emitted even when the list is empty. An absent `via` is a
  // different fact — see the note at the top — and a link that dropped it when
  // there happened to be no ancestors would be read as "open this from
  // wherever I am" the moment it was followed from somewhere else.
  return `${pane.path}?${VIA_PARAM}=${list}`;
}

/** Where a rail row points: this destination, with an ancestry stated to be
 *  empty. The rail is a jump, not an opening-from-here. */
export function railHref(path: string): string {
  return path === ROOT.path ? ROOT.path : `${path}?${VIA_PARAM}=`;
}

/**
 * The trail that results from opening `targetId` from pane `fromIndex`.
 *
 * Everything downstream of `fromIndex` is discarded. A target already in the
 * trail goes to its existing position rather than being appended, so the trail
 * never holds the same pane twice.
 */
export function promoteTrail(
  trail: readonly Pane[],
  fromIndex: number,
  targetId: string,
): Pane[] {
  const from = Math.max(0, Math.min(fromIndex, trail.length - 1));
  const target = BY_ID.get(targetId);
  // An unknown target leaves you exactly where you are. Nothing in the product
  // can produce one — every caller names a pane from the same table — and
  // moving somewhere arbitrary would be a worse answer than not moving.
  if (!target) return trail.slice(0, from + 1);
  // The briefing is the base of every trail, so opening it is a reset rather
  // than an append: there is no arrangement in which it is not index 0.
  if (target.id === ROOT.id) return [ROOT];

  const existing = trail.findIndex((p) => p.id === target.id);
  if (existing >= 0) return trail.slice(0, existing + 1);

  return [...trail.slice(0, from + 1), target];
}

/** The link for that promotion. */
export function promoteHref(
  trail: readonly Pane[],
  fromIndex: number,
  targetId: string,
): string {
  const next = promoteTrail(trail, fromIndex, targetId);
  return hrefFor(next, next.length - 1);
}

/* ================================================================== *
 * WHAT A COMPRESSED PANE SAYS
 *
 * Both of these are rules rather than markup, and both were got wrong in the
 * shipped spine in ways nothing in this project could have caught: there is no
 * DOM harness here, so a rule that lives in JSX is a rule with no way to pin
 * it. They live beside `Pane` because they are functions of a pane's own
 * fields and of its position in the trail.
 * ================================================================== */

/**
 * The two strings a spine carries, derived from ONE source.
 *
 * THE INVARIANT IS THAT THE ACCESSIBLE NAME OPENS WITH THE VISIBLE TEXT, and
 * that is the whole reason this returns a pair rather than leaving two
 * independent template literals in the component. The shipped spine DREW the
 * destination alone — `Briefing` — while its `aria-label` said `Back to
 * Briefing — What is in here, and what changed`. A screen reader was told what
 * the control does; a sighted reader was given a bare noun beside a chevron
 * and left to infer it, and two people including the product's owner did not.
 * Deriving both from one string makes that divergence impossible to
 * reintroduce quietly.
 *
 * The caption goes on the accessible name only. It says what is AT the
 * destination, which is worth hearing when a control's whole purpose has to
 * arrive through speech, and is more than a 40px column can hold without
 * becoming a paragraph turned on its side.
 */
export function spineText(
  title: string,
  caption: string,
): { visible: string; accessible: string } {
  const visible = `Back to ${title}`;
  return { visible, accessible: `${visible} — ${caption}` };
}

/**
 * The number to draw on the spine at `index`, or null to draw none.
 *
 * A POSITION IS ONLY A FACT WHEN THERE IS SOMETHING TO BE POSITIONED AMONG.
 * The trail is `[briefing, …ancestors, primary]` and every pane but the last is
 * a spine, so a two-pane trail — which is every screen in this product reached
 * from the rail — has exactly one. A `1` above it is a digit with no series,
 * sitting immediately above an unrelated word, and it was the least
 * decipherable mark in the column.
 *
 * From two spines up it does real work: it says which of the stacked panes is
 * nearer the base, in a column where they are otherwise told apart only by a
 * word read vertically.
 *
 * `length` is the whole trail and ordinals are 1-based over it, so a spine's
 * number matches its position in `Briefing │ Recall │ Graph` as read.
 */
export function spineOrdinal(length: number, index: number): number | null {
  if (length - 1 < 2) return null;
  return index + 1;
}

/**
 * One level back, or `null` at the briefing.
 *
 * `null` is the answer, not `"/"`: Escape at the base of the trail must do
 * nothing at all rather than re-navigate to the screen already on display,
 * which would push a history entry and reset the briefing's scroll.
 */
export function backHref(trail: readonly Pane[]): string | null {
  if (trail.length < 2) return null;
  return hrefFor(trail, trail.length - 2);
}
