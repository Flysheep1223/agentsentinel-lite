# Dense, complete-project poster — current authorized revision

## User correction (highest priority)

“还是不对，可以写得更多，整体项目全都要写到，字再密集一些，着重写下eda，并可以放eda的那几张图”

Previous two posters were too sparse and overcorrected by reducing the model/project to two lines.
This revision must be a substantially denser COMPLETE research poster with EDA the largest section.
Execute without approval; no git commit/push. Preserve prior versions and all original study files.

User already told:
“这版会保留‘项目背景 → 数据采集与验收 → EDA → 22 项特征主实验 → 48 项特征探索 → 结论与局限’的完整研究链条。
EDA 选用报告里的 6 张原图，并补充相关性、数据质量和资源统计的文字解读；缩小页首装饰、减少留白，
提供密排 A1 版和更便于阅读的 A0 打印版。”

## Deliverables and visual direction

- New directory: `docs/poster/agentsentinel_complete_eda/`.
- New script: `scripts/make_complete_poster.py`.
- Chinese main version and corresponding English body version; original report figure lettering is preserved.
- A1 594×841 mm dense composition; A0 841×1189 mm print edition (~sqrt(2) enlargement).
- Both languages: A1/A0 PDF, A1 300 dpi PNG, A0 editable SVG, lightweight preview.
- White background so embedded original report figures integrate cleanly.
- Two columns; margin16 mm, column275 mm, gap12 mm; right x303; center divider297.
- Compact header ends y95, body y102–812, footer y823–837. Previous headers ended y175 — reclaim that space.
- Body ~16.5–18 pt at A1 (23–25.5 pt at A0), dense but still readable. Title~56 pt, sections22,
  subsection19.5. More connected explanation and actual findings, not just big metrics.
- Target roughly 800–1,000 English words / substantial Chinese text, plus six full original data figures
  and compact partition/model/extension tables. Do not turn this back into a sparse summary.
- Reuse Seedream art only as SMALL header decoration. No further paid call needed.

## Existing implementation and export mechanics

`scripts/make_research_poster.py` provides Poster, fonts, t/rect/line/ax/check_layout, sha256,
verify_inputs, check_exports. Its global W/H=594/841, COL270/RIGHT304 and OUT target older poster;
do not accidentally overwrite old artifacts using its export/header/section helpers.
`scripts/make_eda_poster.py` has exact EDA assertions and evidence calculations; reuse read-only.

Use Matplotlib/PIL in project `.venv`. Arial regular/bold and `/Library/Fonts/Arial Unicode.ttf`.
PDF uses pdf.fonttype42; SVG svg.fonttype=none; save PDFs/SVG dpi300 so art is not downgraded to72dpi.
For A0 render with same design coordinates but physical figure 841×1189; multiply font sizes by
sqrt(2) and scale text-box measurement back to design units. Or equivalent robust vector scaling.
Figure positions normalized to594×841 remain valid. Verify both exact PDF MediaBox sizes.
PNG for A1 is7015×9933 @300dpi. A0 high-res PNG unnecessary (avoid excess memory).
`sips -s format png <PDF> --out <print_check.png>` successfully renders PDFs locally; use for actual-PDF inspection.
No browser required.

## Six original EDA images to embed (do not redraw)

Under `docs/figures/research_report_6000/`:

1. `01_failure_composition.png` 1540×700, English, outcome composition by all12 training templates.
2. `05_condition_failure.png` 2376×1039, Chinese, incident/ticket paired panels and5 assigned conditions.
3. `06_token_boxplot.png` 2378×1096, Chinese, total-token distributions by workflow and final outcome.
4. `04_success_completion_step.png` 1540×560, English, first strict completion step among successes.
5. `03_step3_error_repeat.png` 991×543, English, step3 error/repetition joint failure-rate heatmap.
6. `02_error_trends_survivors.png` 1540×560, English, cross-step errors with surviving cohort counts.

All original figures have white backgrounds. Use original aspect ratio, full image, preserve legends/labels.
Other EDA topics must be covered IN TEXT: time/tool-step distributions, Spearman correlations, concurrency
and batch confounding, data-quality/outlier checks. This covers full EDA without shrinking9 images into illegibility.
Original `09_step3_spearman.png` is2149×4191 and would be unreadably tiny at full-page-fit;
do not add as a thumbnail. Its key findings appear in text.

## Suggested A1 design coordinates (adjust after fit inspection)

Header y8–95: project title, descriptive project subtitle, small art atx477 y8 w117 h58,
four-stat strip aroundy67–95 (6000 attempts,5894 valid,47184 prefixes,3542 training).

LEFT x16 width275:
- 01 Introduction y102, text117 (~4 lines).
- 02 Methods / system & data y148.
  Small4-step architecture y165–182. Environment paragraph188–213.
  Partition table219–251. Label/logging/no-leakage text258–280.
- 03 Results / EDA y293; train-only/quality note306–319.
- A Failure evidence & heterogeneity title326, ~3 prose lines340–359.
  Figure01 y367 w260 h118 (center within column).
- B Conditions × workflow title495, ~3 prose lines509–527.
  Figure05 y535 w260 h114.
- C Resources and early comparison title657, ~4 prose lines671–696.
  Figure06 y702 w240 h111, ends813.

RIGHT x303 width275:
- 03 Results / EDA continued y102.
- D Recovery and completion title119, ~4 prose lines133–156.
  Figure04 y162 w236 h86, ends248.
- E Early thresholds and correlation title255; ~4 lines269–292.
  Figure03 y298 w228 h125, ends423.
- F Survivor/batch effects title430; ~3 lines444–462.
  Figure02 y468 w226 h82, ends550.
- Results / prediction & intervention section y561.
  Model specification paragraph575–603.
  LR/KNN metric table610–633.
  Small3-metric strip638–655:83.18% recall /9.74% saving /−14 net successes.
  Strict replay/timing/baselines/ablations prose663–696.
- 48-feature exploration title702, detailed prose/table715–758.
- 04 Discussion / framed conclusion y770–813; 3–4 dense lines under label.
Footer y823 onward: source/report link, small references, note illustration provenance.

## Complete-project content to include

Source of truth: `docs/RESEARCH_REPORT_6000_FINAL.md` (709 lines).
Python/data paths in old poster briefs. Do not modify source reports or frozen src/models/artifacts/data.

Background/objective:
Tool errors can recover; normal tool returns can still violate the user's task.
Use observable tool events to predict final failure and evaluate alert timing, savings and success protection together.
Prediction, preventing first violation, and stopping without success loss are different objectives.

System/data:
Real kimi-k3-1-pro-261024 in LOCAL SIMULATED service desk, ticket+incident read/write workflows.
20 templates×20 parameter instances×5 conditions×3 repeats=6000 runs,10 parent families.
Conditions normal/timeout/rate-limit/schema-change/partial-response. Max12 tool calls,24 LLM calls.
106 infrastructure/protocol exclusions;5894 valid;47184 eligible prefixes.
Family-separated train3542 /validation1176 /test1176; families6/2/2; successes2570/997/855;
failures972/179/321; prefixes27367/10529/9288.
Strict deterministic label: required writes+final state+no unexpected/duplicate historical writes.
Log session.json, raw events.jsonl (SSE), trajectory.json,result.json; audit/immutable hashes.
Labels/verifier outputs/future events/fault metadata never model features.

EDA TRAIN ONLY:
- Valid resource/error fields and22 runtime features complete, no negative/non-finite entries;
  unique session IDs.62 time and31 token outliers retained, raw records audited.
- Failures mutually exclusive:411 violation+missing;253 violation-only;277 missing-only;31 state-only.
  757/972 budget failures(77.88%);664 with violations (overlap).
- Template failure rate0.67%–90.41%;2 incident parent families contribute658/972=67.70%.
- Normal condition170 failures,159 involve violations. Partial response incident45.9% vs ticket8.7%.
  Conditions are assigned, not guaranteed triggered; descriptive comparisons, not causal rankings.
- Successes2570:1316 with errors(51.21%),548 with≥3 errors(21.32%).
  Read-error event recovery4031/5926=68.02%, median1 step; incomplete-policy recovery590/804=73.38%, median2 steps.
  Recovery means later same-tool/same-arguments read returns normally (not automatically task success).
  77 successes first strictly complete at12;80 failures had once strictly completed then deteriorated.
- Full-run token medians incident S/F14407/24945.5; ticket12410/24061.5.
  Time medians incident S/F67.37/92.61 seconds; ticket56.90/86.52.
  Tool-step medians incident7/12;ticket6/12. Preserve all fliers.
  STEP3 token means incident S/F4277/4226;ticket4246/4289 (~4.2–4.3k), so full-run gaps don't prove early separability.
- Step3 error/repeat lacks stable monotonic threshold: incident no-error/no-repeat42.8% failure vs no-error/repeat15.8%.
  Ticket high-risk cell50% has n2; small groups cannot define rules; no causal "repetition helps" claim.
- Spearman step3: retry/repeat.981;total tokens/cost.884;tokens/reasoning.894.
  Within workflow output/reasoning.999/.998; constant columns undefined, notzero.
  Correlated coefficients not independent or causal effects; motivates joint features and ablation.
- Incident success eligible cohort1003 early→84 atstep11;failures705 atstep11.
  Report surviving n and common cohorts; changing population affects trends/later AUC.
- Concurrency50 vs10 invalid8.00% vs1.03%; date/provider/restart confound, not causal.
  Unknown costs notzero; simulated tool timeout not real network wait.

Model/main:
Frozen MAIN22 observable features: errors/retries/tool context/progress+writes/tokens+cost/time.
LR C1, no class weighting; KNN k9 distance weighted; fixedOR/step/never-stop baselines.
Training-only preprocessing; grouped parent-family CV; validation false-stop budget5%;
chosen LR failure probability threshold.7380596938 (not5% risk).
Tool returns THEN scoring, cannot undo just-completed write.
Test1176 tasks855S/321F and9288 prefixes.
Full-prefix ROC AUC LR.8413/KNN.7749; AP.8030/.7295; step3 AUC.7422/.6099.
LR267/321 failure recall83.18%;17/855 false stops1.99%;9.74% estimated cost saving:
¥206.20→¥186.11; strict replay855−17+3=841, net−14.
Timing among267 alerts32before/55same/154after/26no recorded violation.
True pre-violation coverage32/268=11.94% (different denominator).
Under5% budget KNN+fixed/step rules select no-stop; no implied comparison of impossible AUC for rules.
Ablation: stage-only saves6.96%; removing write count saves5.37% with net+2success (0false stops).
All17 main false stops and nearly all savings concentrate in incident family; two test families limit inference.

48-feature extension (MUST now include; user asks whole project):
Add same-argument history, call sequences, recovery, public response completeness. No hidden/future features.
Exploratory22-feature comparator C10 vs48-feature C.1, task-weighted fitting; not the frozen main C1 model.
Shared auxiliary LR current-completion and twoRidge remaining cost/token heads (not main inputs).
90%failure +2consecutive steps +remaining cost≥0: false stops4→3;cost saving5.39%→5.83%;
failure recall237/321 unchanged; failures detected bystep6:14→60;strict replay851→852 (<855).
Validation mixed:step6 AUC.80856→.86201 butstep3 Brier.12525→.15199.
Task-weighted full-prefix test AUC.83404→.82522; not universally improved.
Case:zero explicit errors, incomplete policy;stoppedstep8, original succeedsstep12.
Reuses already seen test set =>exploratory reevaluation, NOT new independent blind test.

Discussion/conclusion:
Signals help risk monitoring and save some later waste, no evidence of loss-free stopping or online success improvement.
OneLLM,2simulatedworkflows,6train/2testfamilies; verifier omits reading adequacy/final-reply quality.
All stop outcomes offline, no online intervention; time/concurrency/provider confounding; totalcost savings aren't deployment gains.
Next:new independent task families, preregister protection thresholds, richer acceptance checks,
compare warnings/deferred decisions/completion detection. Keep difficult late recoveries.
Small refs ReliabilityBench arXiv2601.06112,AgentRx2602.02475 and report/codeURL.
