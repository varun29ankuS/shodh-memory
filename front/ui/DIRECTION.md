# Visual direction

**Decided 2026-08-07: adopt the Gridline Dashboard structure. This supersedes
the Linear direction.**

Both were real directions and they are not compatible. Linear is sans-serif,
labelled text navigation, calm indigo, list-first. Gridline is monospace body
copy, an icon rail, teal accent, and a full-bleed spatial canvas. Taking half of
each gets neither. The Linear token layer in `src/index.css` stays as the
mechanism — semantic shadcn variables — but its *values* are now up for change.

## Why Gridline

Judged from the live preview, not from a description:

- **The map is the interaction shodh's graph needs.** Region pins that lazy-load
  on click is the same motion as "click a cluster to drill in", and it is what
  the geotemporal work wants.
- **`Published / Validated / In Review` plus `v2.4.1 → v2.4.2`** in the header is
  a review-state and version-diff affordance. That maps almost directly onto
  belief drift and provenance, which is the differentiator.
- **Progressive disclosure is already its idiom** — every pin says "Click to
  load" rather than showing everything at once.

## Changes to make, not copy verbatim

0. **Orange accent, not Gridline's teal/cyan.** This is the one place the brand
   already has an answer: the shodh mark is a low-poly elephant in reds and
   oranges (`src/assets/shodh-mark.png`), and the previous UI ran a warm accent
   for the same reason. Teal would leave the mark as the only warm thing on
   screen, fighting everything around it.

   Take the accent from the mark rather than picking an orange: its dominant
   tones are roughly `#e8342a` through `#f4622e` with a lighter `#f6893f`. The
   accent should be the brighter end — around **`#f4622e`** — so it stays legible
   as small text and thin strokes on `#08090a`, with a dimmer variant for
   resting borders.

   Two constraints this must not break:
   - **Still one accent.** It marks focus, the primary action, and active nav.
     Warm accents are louder than indigo, so it needs *less* usage, not more.
   - **Orange must not also mean "anomaly".** The graph already uses warm hues
     for active/anomalous nodes, and the anomalies view is built on deviation.
     If the chrome and the alarm are the same colour, the alarm stops reading as
     one. Move node/anomaly warmth to red (`--destructive`) and amber, and
     reserve the accent orange for chrome. Check this before shipping — it is
     the failure mode that actually bites.

1. **Smaller icons** in the rail. Gridline's are oversized for the density this
   product needs.
2. ~~**Hover expands the rail** into a labelled column.~~ **SUPERSEDED
   2026-08-15 — the rail is permanently labelled and does not expand.** The
   diagnosis below still holds and is why the rail carries labels at all; the
   *mechanism* was wrong. In use the column animated on every accidental
   pointer pass, so it moved when nobody asked it to, and no decision could be
   made until the labels had finished arriving — it broke "nothing ever jumps"
   and "hover reveals, never reflows" simultaneously, which is worse than the
   memory game it was fixing. The rail is now 244px, permanently labelled, at
   Linear's shipped density (28×220px rows, inset 12px, 8px radius, 13px text
   at weight 450, no rules between rows). ~190px of width buys an instant,
   motionless decision. The overlay/close-delay/focus-mirrors-hover
   requirements below existed only to serve the expansion and are retired with
   it; `aria-label` on every control and honouring `prefers-reduced-motion`
   survive, because neither was about expanding. See
   `src/components/layout/Sidebar.tsx`. Original text, kept for the reasoning:

   Icon-only navigation fails
   the unaided-decipherability test — seven unlabelled glyphs is a memory game.
   Requirements:
   - Expand must **overlay**, not push content. Reflowing a map or a graph on
     mouse-over is disorienting and re-lays-out the thing being pointed at.
   - Keyboard must reach it: focus expands the same as hover, and labels are in
     the DOM at all times so screen readers are never given bare icons. Use
     `aria-label` on every control regardless.
   - Honour `prefers-reduced-motion` — the expand is a width transition and must
     collapse to an instant state change.
   - A short close delay, so crossing the rail diagonally toward content does not
     flicker it shut.

## Source

Captured from `ui.watermelon.sh/dashboard/gridline-dashboard` (Source Code tab)
into `front/ui/.reference/gridline-src.json` — 13 files, ~72 kB, gitignored. The
registry endpoint `registry.watermelon.sh/r/gridline-dashboard.json` returns the
SPA 404 page as HTML, so `shadcn add` cannot fetch it; the site advertises a
command that does not work. The files are lifted from that JSON instead.

## Verified 2026-08-07 — the premise holds

All three open questions are now settled by reading the captured source, not by
inference.

**The token layer carries.** Across all 13 files: **0** raw hex literals, **148**
semantic-token utilities, **22** hardcoded palette classes. Every one of those 22
is in `modelling-dashboard.tsx` or `flexibility-dashboard.tsx` — the two *content*
views, replaced wholesale — where they serve as chart-series and status palettes.
The chrome we actually port (`app-sidebar`, `top-navbar`, `dashboard-shell`,
`sidebar-navigation-item`) is 100% token-driven: `bg-sidebar`,
`border-sidebar-border`, `text-primary`, `bg-primary/10`. Setting `--primary` to
orange propagated through it with no component edits. Confirmed in the browser.

**Dependencies: `lucide-react`, not `@tabler/icons-react`** (an earlier note here
was wrong). Plus Radix via `@/components/ui/{tooltip,switch,sheet,dropdown-menu,
button,card,input,drawer}`, and a Watermelon-specific `icon-lg` button size that
stock shadcn does not have.

**No map library.** The region-pin map is hand-built SVG inside
`flexibility-dashboard.tsx`. There is nothing to adopt and nothing to install.

**Net new dependencies for the port: zero.** Gridline hangs its rail labels off
Radix Tooltip; expanding the rail instead makes tooltips redundant, so that
dependency — and Sheet, which only existed for the mobile drawer the expanding
rail also covers — is not needed.

> Since written: the Geo destination added `topojson-client` (~10 kB) to decode
> the vendored basemap. That is not part of the Gridline port — the finding
> above still holds for the chrome — but "zero new dependencies" is no longer
> true of the app as a whole. There is still no *map library* and no tile
> client: `d3-geo` came in transitively with `d3`, and the basemap is a
> committed file, because the single-file no-network constraint forbids
> fetching one.

## Also dropped from Gridline, deliberately

- **Its theme switch**, and the `MutationObserver` + `localStorage` plumbing
  behind it. This product is dark-only; the control would toggle nothing, which
  is exactly the inert chrome we removed everywhere else.
- **Its version strip** (`v2.4.1 → v2.4.2`). The affordance is a good match for
  belief drift, but there is no version feed behind it yet, and chrome that
  displays invented data is worse than chrome that is absent. It comes back when
  something real feeds it.

## Layout decision — where the Inspector lives

Gridline's shell is `absolute rail + absolute header + main`. It has no third
column, and `WORKFLOWS.md` makes the Inspector the spine. Resolved: the
Inspector is **absolutely positioned**, out of the flex flow, so the d3 canvas
can take the stage's full width with the panel floated over its right edge.

The canvas has since landed, and the decision held: `main` reserves the
Inspector's width and the graph and geo canvases fill the stage beneath it,
with no structural change from when the stage held only the explainer diagram.
The Inspector now accompanies `/geo` as well as `/recall` — both render the
same recall result set and both select into it.

It also renders **only when the server is reachable**. Offline there is nothing
to select, so it could only repeat what the stage already says; the offline
screen previously had four panels apologising separately for one problem.
