# WeirdChat viewer: what each part shows

Viewer: https://need-c10-a-camila-agam-weirdchat--weirdchat-site-web.modal.run
One pattern per page. A pattern = one WeirdChat prompt + the behavior it triggers in Qwen3.6-27B.

## Top bar
- **Pattern dropdown**: behavior → pattern, with WeirdChat's flagged rate after the dot
  (0.45 = 45% of the study's ~64 replies were flagged). Arrow keys step patterns/behaviors.
  Some entries read "Similar prompt produces observed behavior": that is Transluce's own name
  for that pattern group, not ours.
- **Header**: the behavior, the exact prompt, the flagged rate, and one sentence:
  *detective-joracle's hypothesis (unverified)*. That is the OLens investigator's top mechanism,
  rewritten in plain English. Click it for the investigator's full text and cited cells. If the
  pattern was intervened on, a line says so and points to the interventions tab.

## Left pane: the reply
- The reply as **clickable tokens** (drag the divider to resize; `x` switches to plain text).
- **Reply selector** (above the grid, key `5` cycles): *flagged reply A* (default), *clean reply
  A*, and B replies when a lens read them. Flagged = WeirdChat's judge said the reply shows the
  behavior; clean = same prompt, judge said it does not. **These are different rollouts.** They
  share the user turn and the chat header word for word; the reply text differs from the first
  reply token. The page shows one reply at a time and never lines two rollouts up side by side.
- **Evidence replies**: flagged A/B and clean A/B as plain text, so you can read what the judge
  flagged and what it cleared.
- **Find** (`/`): highlight a word in the tokens and in every lens cell; `;` cycles hits.

## Right pane: the grid
- **Rows = layers 20 to 60 (step 4). Columns = positions of the reply shown.** Each cell is what
  a lens says the model is "thinking" at that token and layer.
- **Lens toggles** (keys `1` to `4`):
  1. *OLens · investigator's read*: the OLens verbalizer sample the agent itself pulled during
     its run. Default column.
  2. *J-lens*: a bag of ~10 tokens per cell (the Jacobian lens; words, not sentences).
  3. *NLA L42*: one scene label per position from the NLA verbalizer, layer 42 only.
  4. *OLens · study read*: our own diagnostic OLens pass over flagged A and clean A for every
     pattern. Same lens, second sample. Gemini's green quotes were verified against this read.
  Where the investigator never read the reply shown, the study read stands in and says so.
  Where J-lens or NLA never read it, a dashed **"not read for this reply"** column appears;
  click its header to collapse all placeholders to a thin dot column.
- **An OLens cell** is one verbalizer sample: a short bulleted list written into a fixed token
  budget, so long bullets give 2 lines, short ones 4, most cells 3. Chinese lines carry an
  **EN** translation underneath (Haiku, literal).
- **Green cells**: OLens readouts that Gemini 3.8 Flash found support the hypothesis, verified
  verbatim. Default shows *contrastive* cells only: the content is in this reply's cell and not
  in the other reply's cell at the same position and layer. Toggle *shared* to also see cells
  where both replies carry it (these are the propensity cells). ⚑ flags (an earlier
  "interesting cell" pass) are off by default.
- **Header position**: the token before the model writes ("about to speak"). Identical text
  for every reply of the prompt, so what it reads out is the prompt's effect, not the outcome.

## Clicking a token
- Opens the **matrix for that position**: one column per lens toggled on, one row per layer,
  for the reply shown. Switch the reply selector to compare the other rollout's cells at the
  same position index.
- A green cell opens its **annotation card**: the verified quote, which mechanism it supports,
  the other reply's cell at this position and layer as plain text, and Gemini's one-line
  contrast. The card states that the other reply is a different rollout.
- A cell can also show a **quote marker**: a phrase the investigator cited in a mechanism is
  verbatim in that cell.

## Tabs and menus
- **interventions** (5 patterns): the arms written from the investigator's proposed tests, 64
  replies each, judged by Gemma 4 31B with the pattern's rubric. Columns: arm, flagged rate,
  Δ vs the unchanged prompt, Fisher p, truncation. Each investigator arm's prediction is
  marked right / wrong / not predicted.
- **more ▸ themes**: the 12 themes an Opus clustering pass filed the 110 OLens mechanisms
  under, with member patterns and cited cells.
- **more ▸ arms compared**: the blind reader's comparison of this pattern's OLens mechanisms
  with the black-box arm (and J-lens, NLA where run): counterparts, lens-only, black-box-only.
- **more ▸ manual / keys**: the same explanations in-page, and the keyboard map.

## Reading rules
- Nothing on the page verifies a hypothesis except the interventions tab.
- Before the first reply token, clean and flagged cells read the same (the text is the same);
  a difference there is verbalizer sampling noise. From the first reply token the cells follow
  the reply that was written.
- A verbalizer bullet is a sample, not a measurement: read several layers before trusting one.
