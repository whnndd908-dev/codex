# The Evidence Sheet Is a Scarce Resource

The model sees one composite image per person — typically 2×2, so **at most four frames**. Every design question follows from that cap: what is allowed in, how the slots are shared, and what the prompt is told about the result.

## What may enter

Only frames the human has tied to *this* person. That sounds obvious until the pipeline is short of images and a plausible filler is at hand.

This project banned transition (entry/exit) frames outright at one point: the human reviewed the deliverable and found the same venue staff member — dark coat, walking back and forth, back to camera — appearing in the transition cells of five different companies' sheets. Speed, path length, track completeness: none of them separated staff from participants cleanly. Rather than tune a threshold that would be wrong in an unknown fraction of cases, the frames were excluded.

They were re-enabled later on **exactly one** justification: *the human pointed at that candidate and named it for that seat.* That is the whole warrant. An automatic rule that merely looks reliable is not a substitute — the failure mode is another person's clothing recorded under this person's name, in a dataset nobody will re-check.

## Sharing a four-slot budget

When the human has labelled the same person in several places, the cap forces a choice. **Every source takes turns — including the first one.**

An earlier version of this rule round-robined only among the *additional* sources and kept the first source's frames whole, "because per-cell controls refer to positions". Measured on seven recordings, that left **30 of 99** multi-source seats with every cell from a single source: the first candidate filled all four, and the other views the human had labelled — often the other end of the segment — never appeared. Meanwhile a single source's four frames spanned a median of **0.5 seconds**: the same pose, four times.

What works:

- **One source**: leave it exactly as it was. There is nothing to swap in, and four similar frames beat empty cells.
- **Several sources**: take one cell from each in turn, first source first — two sources give 2 + 2, three give 2 + 1 + 1, four or more give one each.
- **A source's second cell**: take the frame *farthest in time* from what that source already contributed, not its second-best by quality score (typically 0.1 s away).
- **After choosing, group cells by source.** One row per source reads better than an alternating pattern, and "the first source comes first" stays true for every one of its cells.

Before relaxing an ordering rule that exists "because of positions", **find every place a position is actually interpreted.** Here the per-cell exclusions were applied in exactly two places, both on the seated sheet *before* any merging, and the preview page had no per-cell controls — the constraint protected nothing. It is still the right instinct; check it rather than obey it.

**De-duplicate on the picture, not on the record.** Identity is *video time + box*. Comparing detection records (which carry file path, segment and window kind) fails when two sampling windows overlap: the same instant is saved under both windows' directories, the human labels the person in both, and one image fills two cells. Comparing path + time fails the other way — two people in one frame collapse into one. The old rule had hidden this because the first source filled the sheet before the twin could arrive; a scan of already-delivered registries found it in two of them.

## Tell the model what is actually there

The prompt text describing the composition must be recomputed **after** every filter, merge and top-up — and it must be computed before any caching decision that uses it.

Concretely wrong, and each shipped at some point:

- Saying "four images" after one was excluded — the model looks for, and hallucinates, the missing one.
- Reusing the seated-only wording ("lower body and posture are never visible") on a sheet that now contains standing full-body shots. The most useful evidence in the image is explicitly denied to the model.
- Describing frames as "cells 2 and 3 of one grid" after merging images from two different parent grids, where cell numbers mean different things.

Cell numbers are **per source sheet**. The moment a sheet merges frames from two sources, drop the "cell N of the grid" phrasing and describe counts and kinds instead.

**Count the composition from the final cells' own kinds — never infer it from lengths or positions.** Code that asked "did the sheet grow beyond the seated frames?" to decide whether standing shots were merged in was correct only while merged frames could never *replace* seated ones. Once sources share the budget, four seated + one labelled exit becomes 2 + 2 at the same length: the prompt said "four seated images, no transitions, lower body never visible" while two cells were standing shots — and slicing "everything after the seated frames" reported 3 + 1 for the same 2 + 2 sheet. Any rule change that alters *which* frames fill the slots must be followed by a search for every place that inferred composition from how many there were or where they sat.

## The composition text is also the cache key

If the pipeline re-uses a previous model result to avoid paying twice, the composition text is the right signature to compare: change the cells, add a top-up frame, exclude one — the text changes, and the stale result is correctly rejected. Compare the underlying track identity too, so that a different person who happens to produce identical wording cannot inherit the other's answer.

## The human must see the image the model gets — at the same quality

The review page usually inlines every evidence image, and inlining has a size budget. When the group grew, a budget-fitting routine quietly downscaled and re-encoded **162 of 183** images to fit — and its return value was discarded, so neither the page nor the console said a word.

That is not a cosmetic issue. The model receives the full-resolution sheet; the human signs off on a blurrier one. It is the same failure as building the image twice, arrived at by a different route.

Budget for the real page, and if compression must happen anyway, **say so on the page**, with the numbers and a pointer to the original frames. A degradation nobody is told about is indistinguishable from no degradation.

A related trap: an image that looks hazy is not necessarily processed. Check by comparing against the *uncompressed* copy before hunting for a filter. On this project the haze was motion blur in the source — people walking fast past a close camera in a dim room — and the frame selector was already choosing the sharpest frames available. Sometimes the honest answer is "the video does not contain a better image of this person", and the useful response is more sampling at the right moment, not a sharpening pass.

## A second class of person needs a second prompt — derived, not copied

When you extend coding to another class (judges, staff, the audience), their prompt differs in only two places: who the picture is of, and which "known facts" are true for them. A judge belongs to no company and has no headcount.

**Derive the second prompt by slicing the first.** The scale anchors, the lighting/white-balance instructions and the JSON schema must be *literally the same string*. Copy them and the two drift the first time someone tunes an anchor: nothing errors, and numbers measured with two different rulers look equally reasonable side by side. One function that cuts the original at two markers costs less than the divergence.

**Hash the original prompt in a test.** It is the version that already coded everyone you have coded so far; a change to it silently makes the existing rows incomparable with the new ones. A stored sha256 turns that into a red test instead of a footnote nobody writes.

**Resolve the default at the boundary.** `ask(sheet, ctx, prompt=None)` that falls back to `prompt or DEFAULT` deep inside means no caller and no test can assert *which* version was used — and the one thing you need to be sure of is that the judge sheet went out with the judge prompt. Resolve it in the entry point and pass the real value down.

## Never feed the model the answer

Human-coded attributes (gender, role, seat number) must not appear in the prompt. Once they do, the model's output is no longer independent of the ground truth, and every validation built on agreement between them is void — a batch of validation work was discarded for exactly this. A box drawn on the image already says *whom to describe*; the model does not need, and must not be told, who that is.

The same logic argues for a second, non-model measurement (e.g. pixel statistics) recorded alongside, with an explicit agreement flag. When the model call **fails**, that flag must record disagreement, not agreement — defaulting to "agree" turns "never compared" into "compared and matched".

## Ordering: selection and display are different questions

Pick the best frames by whatever criterion serves quality; **present them in time order**. A human scanning a page assumes chronology and reports a bug when it is absent.

Beware caps that are meant for display leaking into logic. A "show the top N candidates" limit must never shrink the pool used to match the human's labels — a person labelled in the collapsed remainder then matches nothing, and their label silently disappears. Anything the human can label must be labelable everywhere it is shown, including inside a collapsed section.

## Neighbouring segments

A person may enter before their own segment begins or leave after it ends, so the human marks them on a *neighbouring* segment's candidate. Whatever window you use to find candidates must be applied in **both** places: the plan that decides which seat gets filled, and the candidate pool that resolves an identifier to an image. Getting one right and the other wrong makes the flow silently vanish — the plan places a person the pool cannot find, and the code skips it.

For the same reason, "segment N+1 does not exist" warnings must distinguish *absent* from *outside the window I was handed*, or every windowed call emits false alarms and people stop reading them.
