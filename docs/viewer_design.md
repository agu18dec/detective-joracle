# Lens-readout viewer: the design contract

The one look every readout viewer in this family should have (Agam, 2026-09-22). Reference
implementation of the design system: the wsbench viewer
(https://need-c10-a-camila-agam--wsbench-viewer-web.modal.run/). Reference for the interaction
model — clickable read tokens, a sticky per-layer grid at the selected position, arrow keys — the
lie-detection sweep viewer. `scripts/weirdchat/build_site.py` implements this contract for
WeirdChat.

## Principles

- **Light, dense, app-shell, keyboard-driven.** The page is a tool, not a document: `body` is a
  flex column with `overflow:hidden`, panes scroll independently, every action has a key.
- **The text IS the token strip.** Never render prose and a tokenized copy of it side by side; the
  clickable tokens are the transcript.
- **The text pane is big and resizable.** It lives in a LEFT column with a draggable splitter
  (width persisted), not in a strip across the top.
- **Metadata first.** Before any grid: what the record is, the primary hypothesis (top-ranked
  mechanism or prediction, with confidence), the agent's summary, the counts that matter (reads,
  mechanisms, how many cited cells verify verbatim), and any ground truth or judge label.
- **Cells are whole and verbatim**, with the exact conversation a read came from one click away.
- **Unverified means a banner**, on every view, not a footnote.

## Design tokens

```
:root{
  --ground:#f3f5f7; --surface:#fff; --surface-2:#e9edf0; --line:#cfd8de; --line-soft:#e1e7eb;
  --text:#152230; --text-dim:#5a6b78; --text-faint:#8695a2;
  --accent:#1d6f8b; --accent-soft:#dbeaf0; --accent-ink:#0d4c62;      /* the token in view */
  --hit:#20714c; --hit-soft:#d9eee2;                                   /* a target / quoted phrase found */
  --miss:#a8431c; --miss-soft:#f6e0d5;                                 /* a miss / the behavior side */
  --hold:#8a6410; --hold-soft:#f5e9cf;                                 /* on hold / secondary */
  --sample:#8a6d00; --sample-soft:#fff3a8; --sample-line:#b89a1e;      /* the position of record */
  --find:#5b3fa8; --find-soft:#e6ddff; --find-ink:#3d2a80;             /* search hit */
  --sans:"IBM Plex Sans",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,monospace;
  --serif:"IBM Plex Serif",Georgia,serif;
}
:root[data-theme="dark"]{ --ground:#0e161d; --surface:#152029; --surface-2:#1b2833; --line:#2a3d4a;
  --line-soft:#223341; --text:#e6edf2; --text-dim:#9fb2bf; --text-faint:#6d8393;
  --accent:#57b6d0; --accent-soft:#16333f; --accent-ink:#9ad8e8; --hit:#5cc294; --hit-soft:#12332a;
  --miss:#e08b62; --miss-soft:#3a2118; --hold:#d9b45e; --hold-soft:#33290f;
  --sample:#f0d24a; --sample-soft:#4a3d05; --sample-line:#e2c53a; --find:#b9a3ff; --find-soft:#2c1f5c; --find-ink:#d6c8ff }
```

Body 13px / 1.45 in the sans; tokens, cells, badges and inputs in the mono; the serif only for a
title or a long-form note. The Google Fonts stylesheet for IBM Plex is the one external asset.

## Layout

```
#bar      top bar: h1 (14px) · pickers (<select>, mono 12px) · ‹ › · [/ find in readouts] [; search records]
          · column toggles · c context · w compact rows · ◐ theme · m manual · ? keys
#ctx      metadata header: a grid of panes (max-height ~150px, scroll), toggled with c
#main     two panes with a draggable splitter
  #text   LEFT  the transcript as clickable tokens, grouped by role (system clipped + "show all", user, assistant)
  #grid   RIGHT sticky-header table at the token in view: rows = layers, columns = readers / compared reads
#hits     under the bar when a search is live: the matching tokens as chips
#note     footer: provenance, what was thinned, keys hint
<dialog>  manual (what the colours mean) and keyboard
```

## Tokens (left pane)

`.tok` — mono 12px, `padding:2px 4px`, 3px radius, transparent 1px border, `white-space:pre`.
States: `.cur` accent inverted (the token in view) · `.read` sample-yellow (the position of record)
· `.hit` green top bar (a target or quoted phrase is in this token's cells) · `.found` violet
underline (search) · `.mark` dotted underline (a computed marker: fork, R1 / "about to speak") ·
`.nodata` 40% opacity (a position that was thinned away). Newlines render as `⏎`.

## Grid (right pane)

Sticky `thead` (uppercase 11px, letter-spaced, text-dim) and a sticky mono layer column. `.cell`
mono 11.5px, pre-wrap, whole text, `max-height:var(--rowmax)` under the compact-rows toggle.
Drag a header's right edge to widen a column, a layer label's bottom edge for row height;
double-click resets. The focused row (↑/↓) is tinted `--accent-soft`. `mark.find` for search
hits; `.cell.hit` green inset bar when a target phrase is found. When compared columns share the
same token at this position, say so once above the table ("identical prefix — same activation,
different lens samples").

## Keys

`← →` token · `shift ← →` ×10 · `home end` · `p` position of record · `↑ ↓` layer · `j k`
next / previous record · `[ ]` family · `1…9` toggle a column · `c` context · `w` compact rows ·
`/` find in this record · `;` search across records · `t` theme · `m` manual · `?` keys.

## Delivery

Split build — a small `index.html` plus `data/<key>.json` fetched when a record opens — hosted on
Modal with a static FastAPI app (`scripts/weirdchat/serve_site_modal.py` is the pattern);
`single=true` inlines everything into one file for hand-offs. Print sizes at build time and warn
when a file passes 4 MB.
