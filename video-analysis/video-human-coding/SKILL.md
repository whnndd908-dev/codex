---
name: video-human-coding
description: Use when building a dataset by coding attributes of people or objects from long video (clothing, posture, demographics, behaviour) with a VLM, where a human must first establish WHO is WHO before any model runs — covers the human-verified registry pattern, stable identifiers that survive re-detection, evidence-sheet design, composing the human's corrections, keeping model judgements independent of ground truth, staleness detection, mutation-testing work whose errors look like valid output, people present for the whole recording rather than one segment, per-cell reassignment across person-classes, coding a second class of person without breaking the first, stamping the artifact the human returns so a stale export is diagnosable, keeping a human's negative mark visible and revocable and round-tripping it back into the page, choosing thresholds and quality floors from measured distributions rather than convenient flags, sharing a four-cell evidence sheet across several labelled views instead of letting the first fill it, confirming a human-certified exception without relaxing the validation rule, checking a panel's missing member before reporting it absent, the re-run loop (re-sampling only what is needed, incremental detection, frame-rate and extraction cost, and undoing a dense pass without destroying the human's labels), and how to split the work between machine, server, and human.
---

# Video → Human Verification → VLM Coding

Turning long video into a per-person coded dataset. The machine finds and tracks bodies; a **human decides identity**; only then does a VLM describe attributes.

**Core principle:** the model can describe *what it sees* far better than it can decide *whose body it is looking at*. Every failure in this class of project traces back to blurring those two questions. Keep them in separate stages with a human-verified artifact between them.

## When to Use

- Coding attributes (clothing, colour, posture, expression) of specific, identified individuals from video.
- The identity assignment is a research variable — "presenter 2 of company X wore navy" — not just "someone wore navy".
- The video is long enough that a human cannot watch all of it, but identity cannot be automated reliably.

Not for: object counting, action recognition where identity does not matter, or anything with reliable face/ID metadata.

## The Pipeline Shape

```
① sample frames        machine, IO-bound
② parse the schedule   machine (who should appear when, from an external table)
③ detect + track       machine, GPU-bound  ← the expensive stage
④ geometric ordering   machine, PRODUCES A BASELINE GUESS ONLY
─────────────────────  human verifies identity here, once
⑤ VLM attribute coding machine, costs money
⑥ review pages         machine
```

Stage ④ never decides anything final. It produces a *baseline* the human corrects. Treating its output as truth is the single most expensive mistake available here: every downstream number inherits a silent identity error that no test can catch.

**Number the stages so the numbers are the order, with the human step on a boundary.** Two rules, both learned the hard way:

*The paid stage must sort after the human one.* When numbering grew organically, the model-coding stage ended up numbered before the page that the human labels on. Nobody ran them in numeric order, so a separate "run these, skip that one" command appeared purely to work around the numbering, and every explanation of the pipeline began by apologising for it.

*No stage may straddle the human boundary.* One stage built both the labelling page (before the human) and the results gallery (after). The gallery needs model output; the page must not. So a fresh dataset always hit a missing-prerequisite error, and a special case grew — "if there is no model output yet, build only the page and print that there is no gallery". Splitting the stage deleted the special case outright. **A branch shaped like "if the later artifact is missing, do half of this stage" is a sign that the stage spans a boundary it should not.**

Also give every step a number, including the ones that feel like utilities. The build-the-registry and package-the-bundle steps sat outside the numbering as side commands — and were duly forgotten, producing exactly the "the human checked an image the model never saw" failure this whole design exists to prevent.

Renumbering is cheap early and expensive later: old logs, docs and habits all encode the numbers. Ship a mapping table when you do it.

References:

| File | Covers |
|---|---|
| `registry.md` | the artifact carrying the human's decision; schema versioning; the two freshness checks |
| `identifiers.md` | labels that survive re-detection; keying side-channels; group-level people and realigning position-derived ids |
| `corrections.md` | how the human's several kinds of correction compose; key collisions; wiring |
| `evidence-sheets.md` | what may enter the model's image, sharing a 4-slot budget, truthful prompt text, model independence, ordering, neighbouring segments |
| `iterating.md` | **the re-run loop**: what re-sampling breaks, incremental detection, sampling grids, named moments, undo |
| `verification.md` | test-first, mutation testing, the fixture traps that let real bugs through |
| `division-of-labour.md` | what to put on a server, and what that does and does not buy |

## A worked example

One 5.5-hour recording, 17 companies, 42 people. Useful as a shape to expect, and because every number below was measured rather than estimated.

**The venue defeated the automatic stage completely.** The camera faced the judges and the projection screen; the entrepreneurs' table sat at the frame edge, so during the presentation only their hands and forearms were in shot. The geometric baseline still produced 28 of 42 "seats" — including two people who were **photographs inside a PowerPoint slide** and a dozen boxes containing nothing but hands. Final result after human verification: **42/42 seats, of which 0 came from the seated stage** — every single person was identified from the seconds they walked past the camera.

Read that as a warning about baselines. It was confident, complete, plausible, and ~11% of it was slide content. Nothing in the output said so; a human looking at the pictures did.

**A giveaway worth checking automatically:** a person standing in the room cannot have their feet above the middle of the frame. Boxes whose *bottom edge* sits in the upper half are screen content, posters, or reflections. On this recording that rule flagged exactly the three bad seats and touched none of the 107 verified seats in three other recordings.

**Cost profile of the loop** (see `iterating.md`): first pass 7,600 frames; after the human asked for denser transitions and three targeted re-extractions, 23,800. Detection went from 1.5 s/frame (silently on CPU) to 0.15 s (GPU + fp16), and a full pass from 9.5 hours to 55 minutes — then to **5 minutes** once only changed groups were re-detected. The human's labelling, not the machine, was the schedule from that point on, which is the state you are aiming for.

## Non-Negotiables

**Never let the model see an image the human did not verify.** Build the evidence image with *one* function, called from both the review page and the model call. Two code paths that "should" produce the same crop will diverge, and the divergence shows up as a human who signed off on a picture the model never saw. No test catches this; only a shared function does.

**Every gap must be visible.** A seat with no usable frame must appear as a gap in the UI, not as a missing dict key, an empty list, or a silently skipped row. "Determined" and "nobody looked at it" must be different states with different colours.

**Human input beats machine input; specific beats general; newer beats older.** "This picture is person 3" outranks "the automatic label here is wrong". Two human statements about the same person **merge** — combine their images. A human statement never merges with an unverified machine one. Write the full interaction table before the code (`corrections.md`); every pair you leave undefined becomes a silent wrong answer.

**Never guess a second thing from one statement.** If the human says "this image is actually seat 2", do not also infer that seat 2's previous occupant is seat 1. Move what they named; drop the displaced one back to undetermined and let them say. Adjacent-swap is a plausible guess and plausible guesses are how wrong data enters.

**Describe the evidence truthfully to the model.** If the sheet has 3 images, say 3. If some are standing shots, do not tell the model that legs are never visible. Prompt text that describes a composition the image does not have is a silent, systematic bias — and it is the failure most likely to survive to publication.

**Never put the ground truth in the prompt.** Human-coded gender, role or seat number must not reach the model. The moment they do, its output stops being independent of what you are validating against, and every agreement statistic built on it is void. The box on the image already says whom to describe.

**Prove your tests can fail.** Green means the code passes the tests; it does not mean the tests would notice a wrong answer. In a project where wrong answers are indistinguishable from right ones, mutation testing is the actual verification — see `verification.md`.

## Failure Modes That Do Not Raise

Ranked by how long they survive undetected:

| Failure | Looks like | Guard |
|---|---|---|
| Human verified image A, model saw image B | nothing | one shared build function; a preview page rendered *from the delivered artifact* |
| Labels re-aligned to a new detection run onto the wrong person | nothing | content-derived stable IDs + fingerprint check; refuse to load on mismatch |
| Registry rebuilt from stale labels | nothing | store a digest of the labels file inside the registry; raise, do not print |
| Exclusion/selection keyed by tracker ID | works until re-detection, then silently voids | key side-channels by (segment, seat) |
| Prompt claims N images, sheet has N-1 | model hallucinates the missing one | recompute the composition text after every filter |
| Multiple human labels for one person, only the first used | quietly fewer images | accumulate, never `setdefault` |
| Two artifacts at one position share a `(segment, slot)` key | the human's per-cell edit lands on an image they never saw — wrong cells dropped, no error | only the path that renders the control may read it; the correction follows the image, not the slot |
| A correction channel is listed as reserved but never re-attached on load | file has it, page shows it, pipeline ignores it | test that each channel survives **every** load branch, re-align included |
| A new correction channel not added to the reserved list at all | treated as a segment id, matches nothing, counted as failed matches and dropped — a day of work gone, hidden inside a plausible "50 of 57 matched" line | adding a channel = reserved list + re-attach on every branch + a test across a fingerprint mismatch |
| The match-rate gate voids a file that has no candidate labels yet | channels keyed by seat or position — stable across re-detection — are discarded with it | "nothing to align" is not staleness; gate only when the rate exists |
| Position-derived ids for always-present people shift on any re-sample | exact-string lookup reports the person was never labelled | realign by nearest position; exact matches first; refuse ties; print the match rate |
| Page and pipeline each compute "is this record complete" | the two disagree; the human trusts the screen and stops early | one counting function per side, each commented with the other; mutate one, expect the other's test to go red |
| Growing a candidate pool renumbers its first entries | every existing per-cell label silently re-points | the first N must be a byte-identical prefix of the old selection |
| Spare candidate images drawn by score from the same stretch | the replacement is the same wrong person | take the best frame from each unused segment before a second from a used one |
| A spare sheet painted with cell numbers 1–4 while its controls say 5–8 | the human reads one number and records another | the numbers on the image are the numbers in the controls |
| A person-class harvested by region, one member sitting outside it | that slot is empty and the "not seated" list is empty too — nothing to point at | provide a cross-class reassignment path; never widen the region filter |
| A conditional control rendered by two code paths, one updated | works in the shape you tested, absent in the shape the human meets | grep every place the control's label appears; test the shape the user will actually hit |
| One widget meaning "seat 1–3" for one role and "judge 1–5" for another | the human reads a panel position as a table position | rename the label per role, in place |
| Both the whole-card role and its per-cell marks collected | the same frame arrives twice; the evidence sheet shows a duplicate | cells win when cells exist; the card role is the fallback for one-person boxes |
| A second person-class stored as a flag on the first class's row type | one consumer forgets the branch and counts a judge as a company's presenter | give the group-level class its own artifact, and add it to the deliverable list |
| A second prompt copied rather than derived | an anchor is tuned on one copy; two rulers, both plausible | slice the shared scales out of the original; hash the original in a test |
| The model-call cache keyed on slot or seat number | renumbering re-bills valid results; swapped cells reuse the wrong ones | key on a digest of the frames that went into the sheet |
| A default resolved deep inside the call chain | no caller or test can assert which prompt/version actually went out | resolve defaults at the entry point, pass the real value down |
| A new parameter added, call sites not updated | same as above, and the signature looks correct | grep every call site; the default silently wins |
| A deliverable regenerated by calling the renderer instead of the official entry point | page opens and looks right, missing the post-processing (save-as, pre-filled labels) | one blessed entry point per deliverable, documented with the consequence of bypassing it |
| Correctness resting on dict insertion order | passes today; reorder a merge and it inverts, with no test red | make one guard load-bearing and assert the invariant |
| A display-only cap (top N shown) also shrinks the pool used to match labels | a person labelled in the collapsed remainder matches nothing; the label vanishes | keep selection, display and matching pools separate |
| Cross-segment flow windowed in the plan but not in the candidate pool | the plan places someone the pool cannot resolve; the code skips it silently | apply the same window in both, and test with the person owned by the neighbour |
| Ground-truth attributes leak into the prompt | model output correlates with truth for the wrong reason | keep the prompt free of anything you intend to validate against |
| Model call failed, agreement flag defaults to "agree" | "never compared" is recorded as "compared and matched" | failure must record disagreement |
| Automatic stage accepts people *inside* the projected slides | a confident, complete registry, ~11% of it PowerPoint | a physical-plausibility rule (feet cannot float); show the human the picture |
| The human's deletion is undone by the file on next load | they clear a label, reopen, and it is back | a deletion is a statement — store it explicitly, don't just remove the key |
| Occupancy computed from the pre-correction baseline | the human's replacement label is skipped as "seat taken" | every "is this slot free" test must read the corrected state |
| Displayed images silently degraded to fit a size budget | human signs off on a blurrier image than the model receives | budget for the real page; if you must compress, say so on the page |
| The command prints one plan and executes another | "0 frames added" after a run that reported the right window | the printed plan and the executed plan must be the same object |
| Sampling finer than the source's *distinct*-frame rate | half the frames are duplicates; detection pays for all of them | probe content fps by run length, not the container's declared rate |
| A faster extraction path that returns a different frame for the same timestamp | filenames still line up, images still open, only the contents moved | decide explicitly what error the filename→content contract tolerates; verify on the real source, never on synthetic media |
| The list of roles that *show* an input and the list that *export* it kept separately | the human marks a number on every one of them, the export writes `null`, the consumer collects nothing — marking is a no-op and no stage errors | bind the two in a test that walks every role: shows a control ⇒ exports its value (this omission recurred across two different roles) |
| The file the human returns carries no mark of which page version produced it | you fix the export, ask them to re-export, and get the same wrong file — "the page is right" and "the file is wrong" are both true and nothing distinguishes a stale tab from a second bug | stamp a fingerprint of the *page's own code* into the export and warn on mismatch; the detection fingerprint cannot do this job — it is unchanged precisely when the page code changes |
| The pipeline finds the returned labels by filename alone | a file named after a different video, holding the right content, is not found: the run completes on the automatic baseline and reports nothing missing | identify the file by the fingerprint inside it, not its name; print which file was used and what it matched |
| A thumbnail on the human's page sourced from an artifact a *later* stage writes | it is empty exactly when the human needs it and appears only after they have finished — and it is not the image the model will see | the human's view may only be built from what exists at that point in the pipeline; if the real evidence is already on the card, delete the second image rather than explaining it |
| A domain-knowledge priority ("prefer this camera angle") applied unconditionally | a candidate the system itself rejected as unusable wins over a clean one, because the priority never asks whether its own premise holds | gate the priority on the premise: skip candidates too small or too dark to judge, and fall through to the plain rule when none qualify |
| The obvious flag reused as a quality proxy | measured against real data it condemns the very candidate the human wants — the existing filter answered a different question | measure the proxy against the case that prompted the change *before* implementing it; a floor built from the wrong flag degrades to the old behaviour and looks fixed |
| A slot's evidence topped up from every candidate the human assigned to it | two tracks blend into one sheet and the model averages two people's clothing, with no error anywhere | top-up is right, but record per-frame provenance and show the human which frames came from which candidate |
| A control whose effect deletes the element carrying that control | marking someone "not one of these people" removes their card — and with it the only checkbox that could undo it; the mark lives on in storage with nothing on screen to point at | a negation must leave the item visible somewhere, with its own control still on it and still showing the marked state |
| The undo exists, but only in a collected section far from the click | the human looks where the thing used to be, sees a vacancy notice, and reports "I can't find how to get it back" | say it where it happened — the vacated position must name the human's own action and link to the undo — and add a route from the table of contents |
| A vacancy that the human caused, worded as a detection failure | "no candidate — they may be absent, or we may have missed them" sends them to investigate the pipeline for something they did themselves | distinguish "emptied by a human mark" from "nothing found" in the text, not only in the data |
| A correction channel that exports and is honoured downstream but is never sent back into the page | the page reads only browser storage: on another machine, another browser, or after cleared site data, the human's negation evaporates and the person is back in their slot with no trace it ever happened | close the loop for every channel — page → file → pipeline **and** file → page — and test the round trip, not just the export |
| File-to-page backfill decided by whether the value is truthy | the human's explicit undo (an empty string) loses to the file's older mark: they clear it, reload, and it is back | decide on whether the **key is present**, and let the page's state win over the file |
| Swapping which table supplies the schedule | groups whose two sources disagree get renumbered, and every existing label points at the wrong segment | before switching, diff the whole corpus: row counts, row order, and times; name the groups that would move, and confirm the already-delivered ones are byte-identical |
| The first evidence source keeps all its cells; other labelled views only fill gaps | the sheet shows one pose four times (median 0.5 s apart) while the other views the human labelled never appear — measured on 30 of 99 multi-source seats | every source takes turns, the first included; a source's second cell is the frame farthest in time from its first |
| An ordering rule kept "because controls refer to positions" without checking where positions are read | the rule protects nothing and blocks the fix the human is asking for | find every place a position is interpreted before relying on — or relaxing — the rule |
| Composition inferred from how much the sheet grew, or by slicing after the first source | once sources share the budget, 2 + 2 has the same length as 4 + 0: the model is told "no standing shots, legs never visible" about a sheet that has two | count kinds from the final cells; after any selection-rule change, grep for every length-difference or positional slice |
| De-duplication compares records (path, segment, window) rather than pictures | overlapping windows save one instant twice; the human labels both; the same image fills two cells | picture identity is video time + box — not path, not time alone (two people share a frame) |
| A status line from an early stage reports failure that a later stage repairs | "not placed — seat taken" printed for flows whose images were merged into that seat downstream; relayed to the human as outstanding problems for several rounds | check the claim against the delivered artifact before relaying it; word the line for the final outcome, and pin that outcome with a test before rewording |
| A human-certified out-of-range value handled by relaxing the validator | the check stops protecting every other group from the typo it exists for | keep the rule; add a confirmation list keyed by (group, segment, side, **value**), print when it is used — an edited cell loses its exemption |
| "Panel member missing" reported straight from the candidate count | a merged pair, a member just past the expected position, or one cut by the frame edge is reported as absent — or an absent one gets a neighbour's frames | extrapolate the row spacing, test for two boxes coexisting in one candidate, read edge candidates by height and coverage; then leave the slot visibly empty and let the human number it |

## Working Rhythm

Human identity work is the bottleneck and it is not parallelisable *within* a video — but it is across videos. Package each video as a self-contained folder (review page, labels, registry, preview, evidence images) so several people can label different videos at once and return one JSON each.

Re-running detection invalidates frame-level records but *not* seat-level ones. Design side-channels so a human's work survives a re-detect; otherwise every density change costs a full re-label and people stop asking for re-detects they need. Scope every re-run to the smallest unit that answers the question — `iterating.md` has the measured cost of not doing so.

**With several videos in flight, establish which one the report is about before you rebuild anything.** The packages are interchangeable by design — same page, same controls, same wording — so a bug report reads identically whatever group it came from. Twice in one session a fix was applied and verified against the group named in the *previous* message while the person was labelling a different one; the code was right, their page was untouched, and the next message was "I refreshed and it's still not there". The report names a symptom, not a file. Ask, or infer it from which package they last exported (the label file's timestamp), and **do the verification on the artifact they actually have open**.

Expect the human to report **one symptom** that has several causes behind it. "These two cells became one" turned out to be a key collision, a stale rejection blocking a new correction, a missing merge rule, and a parameter that was never passed — four defects, one observation. Fix one, re-run against their real data, and look again; stopping at the first plausible cause leaves the rest in place and they look like new bugs later.

**When you change a selection rule, scan every delivered artifact for its invariants — before and after.** Switching the evidence sheet to take turns across sources surfaced one image appearing twice. The same scan on the untouched registries found the duplicate already present in two other groups: the old rule had hidden it, not prevented it. Without the "before" scan, a pre-existing defect gets blamed on the new rule (or the new rule gets rolled back for a bug it only exposed).

**Verify a log line against the artifact before passing it on.** A warning printed by an early stage describes what that stage knew. Several rounds of "N flows not placed" were reported to the human as open problems; checked against the registry, every one of those images had been merged into its seat later on. Relaying a stage's opinion as the outcome is the same mistake as quoting counts from a stale registry.

**After each re-run, report the diff, not the totals.** The human changed two labels; "36/36 seats, 5/5 judges" tells them nothing about whether those two took effect. Keep the previous registry, compare seat by seat, and say which seats moved and how ("3 → 4 cells, now from the exit candidate you added"). It is also the fastest way to notice that a re-run changed something nobody asked it to.

Quote numbers only from a freshly rebuilt artifact. Reporting counts from a registry built on older labels is easy to do, sounds authoritative, and sends the human off to investigate a change that never happened.

**A threshold is a measurement, not a choice — and neither is the proxy you measure.** Asked to stop a rule from preferring unusable images, the obvious move was to reuse the existing "this candidate was filtered out" flag. Measured against the case that prompted the request, that flag would have condemned the candidate the human wanted (it was filtered for a *different* reason) and left the bad one winning — a fix that looks applied and changes nothing. Measure the proxy on the real case first. Then measure the threshold: the venue was uniformly dim, so the "too dark" cut-off that seemed obvious would have condemned half of all candidates; the defensible value came from the actual distribution (1st percentile, 10th percentile, median) and the cut landed near the bottom decile. Finally, replay the original failure end to end and show the before/after — and check the new rule against every group you have already delivered, so you can say whether anything shipped changes.
