# AgentSentinel-Lite research poster

## Authorized deliverable

Create the poster now; no approval gate. The supplied reference requires portrait orientation,
two vertical columns, left-to-right/top-to-bottom reading, IMRD organization, concise bullets,
graphs, consistent colors/shapes, and a framed conclusion.

Working format: A1 portrait, 594 × 841 mm. Produce an English presentation version and a Chinese
version, each as print PDF, 300 dpi PNG, and editable SVG, plus small preview PNGs.
Use Matplotlib's existing PDF/SVG backends; its PDF output can embed/subset fonts.
No need to install ReportLab or a browser. Do not invent authors, institutions, affiliations,
logos, or online success claims. Do not commit/push this work unless requested.

## Visual direction

Warm ivory `#F6F4EF`, dark ink/navy `#12313B`, muted teal `#26867C`, amber `#D89845`,
restrained cool gray. Teal = success / useful early warning; amber = failure / success loss.
Use generous spacing and consistent thin borders; avoid dense report pages or many tiny charts.
Target body text around 25–28 pt at A1, labels at least 18–20 pt, strong section hierarchy.

Decorative conceptual illustration is already generated and reviewed:
`visual_v1/seedream_visual.png` (2512 × 1680). It shows a monitoring gate scanning a sequence
and branching into a teal continuation path and amber stop. Use near the upper-right of the
header. It is not a data figure. Generation metadata is in `visual_v1/metadata.json`.
Returned model: `doubao-seedream-5-5-261028`; one successful request, 28.316 seconds.
No further paid generation needed unless visual review exposes a specific defect.

Seedream endpoint authorized by user: `ep-m-20261106162157-7d2jm`.
API client: `scripts/generate_poster_visual.py`; it loads `.env` without exposing the key.
Never print or copy the API key into documents, prompts, source, or metadata.

## Suggested A1 layout in millimeters

- Full-width header, y≈15–105: small research category; `AgentSentinel-Lite`; title
  “Early failure warning from tool-call traces”; decorative illustration at right.
- Takeaway banner y≈112–154: “Risk signals identify failure, but early stopping still costs successes.”
- Body columns: left x=20, width=270; right x=304, width=270. Footer y≈810.
- Left: 01 Introduction y≈174–251; 02 Methods y≈267–541;
  03 Results / Training EDA y≈557–797.
- Right: 03 Results / Main test y≈174–594; 04 Discussion y≈609–691;
  framed Conclusion y≈704–795.
- Use approximate anchors only; adjust after inspecting full previews and crops for overflow.

## Content and verified results

Source: `docs/RESEARCH_REPORT_6000_FINAL.md`. Keep the poster focused on the frozen MAIN
22-feature LR experiment; do not mix it with the later exploratory 48-feature model.

Introduction:
- Tool errors can recover; a normal tool return can still violate the user's task.
- Question: can observable tool events warn early enough to save resources while preserving success?

Methods:
- Real Kimi model (`kimi-k3-1-pro-261024`) in a local simulated service desk.
- Ticket and incident workflows; 20 templates in 10 parent families; five conditions:
  normal, timeout, rate limit, schema change, partial response.
- 6,000 runs; 106 excluded infrastructure/protocol failures; 5,894 valid tasks;
  47,184 eligible prefixes; maximum 12 tool calls per run.
- Family-separated train/validation/test: 3,542 / 1,176 / 1,176 tasks; 6 / 2 / 2 families.
- Deterministic success: required writes, final state, no unexpected/duplicate writes.
- Runtime inputs: errors/retries, tool context, tokens/time/cost, writes; never future or verifier data.
- Pipeline diagram: Tool events → 22 runtime features → LR failure score → Continue / Stop.
- Compare LR, KNN, fixed and step rules; select threshold on validation with a 5% false-stop budget.
- All resource and success outcomes are offline replay estimates, not online gains.

Training EDA (only n=3,542):
- Success 2,570; failure 972.
- Of successes: 1,316 experienced errors (51.21%); 77 first met strict completion at step 12.
- 80 finally failed runs had once met strict completion before further writes.
- Draw a real boxplot of total tokens using `artifacts/expanded_eda_v2/diagnostic_sessions.csv`:
  incident success n=1,003 median 14,407; failure n=766 median 24,945.5;
  ticket success n=1,567 median 12,410; failure n=206 median 24,061.5.
- Keep fliers, 1.5 IQR whiskers. Use thousands of tokens on y-axis.
- Early contrast from `fixed_step_summary.csv`: step-3 mean tokens incident success/failure
  4,277/4,226, ticket 4,246/4,289. Complete-run separation does not establish early separability.

Main test:
- n=1,176 tasks: 855 success, 321 failure; 9,288 eligible prefixes.
- LR full-prefix ROC AUC .8413 / AP .8030; KNN .7749 / AP .7295.
- Step-3 AUC LR .7422 vs KNN .6099 (n=1,176 tasks). Suggested compact comparison chart.
- Main LR detects 267/321 failures = 83.18%; false-stops 17/855 successes = 1.99%.
- Saves 9.74% cost: baseline ¥206.1985612, saving ¥20.087795, remaining ¥186.1107662.
- Strict replay success: 855 −17 lost +3 rescued =841 (net −14).
- Timing of 267 alerted failures: 32 before first violation, 55 same step, 154 after,
  26 without a recorded violation. Suggested stacked bar with counts and denominator.
- True pre-violation coverage is 32/268 violation-bearing tasks =11.94%, a DIFFERENT denominator.
- Main threshold .7380596938; tool-return-then-score design cannot undo the just-completed write.

Discussion / conclusion:
- Signals capture failure and task stage, but recovery remains hard to protect.
- One LLM, two workflows, only two test families limit generalization.
- No online intervention experiment in this batch.
- “Useful for risk monitoring. Not yet a loss-free stop policy.”
- Report success loss alongside cost savings.

Footer:
Source/reproducibility URL: `github.com/Flysheep1223/agentsentinel-lite`.
Source report may be hyperlinked in PDF. No fabricated author block.
Optional small references: ReliabilityBench, arXiv:2601.06112; AgentRx, arXiv:2602.02475.

## Build and verification

Use project `.venv` with existing PIL/Matplotlib/Pandas; no source/data/model changes.
Create `scripts/make_research_poster.py`, optionally an editable content JSON.
Fonts available: Arial regular/bold; DejaVu Sans; Arial Unicode MS at
`/Library/Fonts/Arial Unicode.ttf` for Chinese.
English SVG should retain text; PDF should embed fonts. Save source hashes / dimensions / key
metrics in poster-specific provenance rather than editing prior sealed manifests.
Render preview first, inspect full page and dense crops, adjust layout, then export final
A1 PDF/PNG/SVG in both languages. Check text fits, figure values, no clipping, correct dimensions.
