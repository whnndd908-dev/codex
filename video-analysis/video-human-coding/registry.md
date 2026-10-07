# The Human-Verified Registry

The artifact that carries the human's identity decision from the review stage to the model stage. Everything downstream reads it; nothing downstream re-derives it.

## What it stores

Per (segment, subject): who this is, **how it was decided**, which exact frames and boxes are the evidence, whether a human confirmed it, and any provenance that a later reader would otherwise have to guess.

```
{"_meta": {"schema", "source_video_or_sheet", "detections_fingerprint",
           "built_at", "labels_file", "labels_digest", "labels_mtime",
           "n_expected", "n_resolved", "counts_by_source"},
 "segments": {"<order>": {
     "presenter_num", "rejected_seats", "unresolved_seats",
     "unresolved_reason": {"<seat>": "no_candidate" | "not_reviewed"},
     "seats": {"<seat>": {
        "seat", "source", "confirmed", "confirmed_via",
        "track_id", "kind", "stable_id", "cell_indices", "note",
        "frames": [{"frame_path", "t", "box", "conf", "kind", "track_id"}]}}}}}
```

`source` is not decoration. "Decided by geometry", "picked by a human from a candidate", "split out of a multi-person box by a human" have different trust levels, and the next stage must be able to tell them apart — e.g. to skip re-review of confirmed rows, or to report how much of the dataset is machine-ordered.

`unresolved_reason` separates "a human looked and there is genuinely no usable frame" from "nobody has looked yet". They demand opposite actions (re-sample vs. assign a labeller) and collapsing them wastes the most expensive resource in the project.

## Schema version

Give it an explicit version string and **void the whole file on mismatch**. Field shapes change (a scalar becomes a list); old files then read as "this field is absent", which is not an error, not a crash, and not correct. *Being able to parse it is not the same as reading it right.* That is the only reason the version constant exists.

## Two independent freshness checks

**Detection fingerprint** — a hash of the detection output. The registry stores frame paths and pixel boxes; after re-detection those boxes describe different bodies. On mismatch: void the file. There is no "re-align" path here, unlike labels — a box is a box.

**Labels digest** — a hash of the human's label file. On mismatch: **raise, do not print.** The distinction matters. A changed detection is an external fact and the caller may reasonably proceed without the registry. A changed label file means a human step was skipped, and if you merely print, they will keep reviewing yesterday's pictures. Provide an `allow_stale` escape hatch for diagnostics only, and never use it on a production path — one "just to get it running" and the check is gone.

Also record the labels file's mtime as a *human-readable hint* and print times in local time. People compare against their file manager. A UTC timestamp turns a ten-minute discrepancy into something nobody can see.

## Breaking the assignment circularity

A trap worth naming because the obvious design deadlocks: the model's judgement is wanted as an input to assigning identities (e.g. matching judged gender against a known roster), the prompt should not carry the seat number, and the seat number is what you are trying to produce. Written naively this needs two full passes — run everything, re-sort, run everything again.

It resolves in three steps inside one pass:

1. Measure and query the model **per candidate track**, with no identity in the prompt. The box says whom to describe; nothing else is needed.
2. Assign final identities by matching the model's judgements against the roster — the information arrives in this same call, no re-run required.
3. Emit one record per final identity, taking the roster fields from the assignment.

Only do step 2 when the counts match; comparing multisets of different sizes is meaningless. When they differ, keep the baseline order and let the missing slots surface as gaps.

Everything downstream — evidence, prompts, top-ups, human corrections — must read the *single* place where the final identity is decided. Applying a correction at any later point leaves the earlier consumers on the old number, and the symptom is a review page and a result file that disagree about who somebody is.

## A group-level class gets its own output file

The row type you built first is shaped by the first class: segment, company, seat index, headcount, who-spoke-when. A class that exists at group level — a panel that sits through the whole recording — has none of those. Two ways to store them, and only one survives:

- *A flag on the existing row* (`seg_order = 0`). Every consumer now needs a branch, and the one place that forgets it counts a judge as one company's presenter. Nothing errors.
- *A separate artifact* keyed by (group, slot). Consumers that care load it; the rest are unaffected.

Take the second. The same reasoning kills the tempting shortcut of copying the group-level rows into every segment "so the join works": five judges across seventeen segments becomes eighty-five rows of identical clothing, and any later aggregation counts that outfit seventeen times.

Add the new file to the deliverable-name list in the same commit. A file that is not on that list gets left behind by the packaging step and the migration step, and the symptom is that it silently stops being delivered.

## Preview page

Render a page from the registry showing, per subject, the exact image the model will receive — built by the same function the model path uses, from the same stored frames. Print on it: which label file it came from (digest prefix + time), when it was built, and per card whether that image is genuinely reconstructible. If a frame file is gone, that card must say so rather than showing a blank and implying agreement.

Add a left-hand table of contents for segments with a per-segment "resolved/expected" count and an explicit missing marker. Never encode status in colour alone — add a glyph and words. These pages get screenshotted, printed, and read by people with colour vision deficiency.
