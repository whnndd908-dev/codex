# Re-running Without Destroying the Human's Work

The first pass through a video is the cheap part. What actually consumes the project is the loop: the human labels, finds a gap, asks for more frames somewhere, and everything downstream is recomputed. Each turn of that loop can silently invalidate work that took hours.

Design the loop before you need it. The three questions are: **what does re-sampling break, how little can you re-run, and how do you undo it.**

## Re-sampling breaks the identifiers in the segments you touch

Content-derived identifiers (`kind@t_mid#cx,y2` — see `identifiers.md`) survive re-detection because they describe the video, not the tracker. But they are computed from *the frames that were sampled*. Add frames inside a window and the merged track's median time and position shift; past the realignment tolerance, the human's label for that candidate matches nothing and is dropped.

Measured on one 17-segment recording, same labels each time:

| What was re-sampled | Labels surviving |
|---|---|
| every segment's windows widened | 36 / 50 |
| exact previous frame set restored | 48 / 50 |
| three segments densified | 44 / 50 (all six losses inside those three) |
| one segment, one window | 51 / 51 |

The pattern is the whole lesson: **breakage is confined to the segments you re-sample, and is roughly proportional to how much you change inside them.** So:

- Re-sample **the smallest unit that answers the question** — one segment, one window, ten seconds around a named moment. Never "the whole group, while I'm here".
- Say plainly, before running, which segments' labels are at risk. The human can re-do three segments; they cannot re-do seventeen, and they will not notice which ones changed unless you say.
- Report the match rate every time, with the count from the file and the count that loaded. Two numbers, one line. This is the only visible sign that work was lost.

## Only re-detect what changed — it is provably equivalent, not an approximation

Detection dominates wall-clock. But if the tracker is reset between `(segment, window)` groups **and the reset also zeroes the global track-ID counter**, then each group's output depends only on that group's frames. Re-running one group and keeping the rest verbatim gives byte-identical results to a full pass.

Verify that premise in your tracker before relying on it, then:

- Store a digest of each group's frame set (sorted, order-independent) **beside** the detections, not inside them. Inside changes the detections' own fingerprint, which is what tells a label file whether it belongs to this detection run — breaking it to speed something up invalidates every existing label.
- Re-run the groups whose digest changed; **copy the others' records verbatim**. Recomputing "equivalently" turns a proof into a hope.
- No digest (older artifact) → full run. Do not guess which groups are still valid.
- Print how many groups were reused and which were redone. Reuse that nobody can see is indistinguishable from reuse that was wrong.
- The equivalence test is the one that matters: run incremental and full on the same input and assert the outputs are **identical, record for record**.

Measured effect: 22,000 frames, one changed window → 55 minutes becomes 5. And because only that group's identifiers move, the label breakage above drops to zero elsewhere.

## Key the paid stage's cache on the evidence, not on the identity

Re-running the model is the one stage that costs money, so it needs a "has this changed?" key. The obvious key — the person's slot or seat number — is the wrong one in both directions:

- The human renumbers constantly (that is what the review page is *for*). Renumbering without changing any image invalidates a result that is still perfectly valid, and you pay again.
- Conversely a slot can keep its number while its images change completely — after a re-sample, or after the human swaps cells between people. Reusing the old result then attributes one person's clothing to another, and nothing errors.

Key it on a digest of **the frames that went into the sheet** — the same thing the model actually saw. That is stable under renumbering and changes exactly when the picture does.

## Keep every sampling grid an integer subdivision of the first pass

If the first pass samples every 0.2 s, every later pass must use 0.2/n. The natural formula — *target frame count ÷ window length* — violates this the moment a window is clipped by a neighbour: 57.2 s ÷ 600 gives 0.0953 s, and the whole grid shifts a few milliseconds. Nothing on disk lines up, the same seconds are sampled a second time, and you get pairs of frames 5 ms apart: wasted disk, wasted inference, and near-duplicate candidates for the human to wade through. Snapping to the ladder cut one re-extraction from 8,167 new frames to 5,890 and removed the duplicates.

Corollary for a *named moment*: sample it at the source frame rate — the human pointed there, so take everything — but never finer. Probe the actual rate (it is a rational like `30000/1001`; rounding it to 30 samples finer than the source) and pick the finest ladder value that is still ≥ one frame period.

## Probe the *content* frame rate, not the container's

`r_frame_rate` is what the container claims. What matters is how many **distinct** images per second the file actually carries. One recording declared 50 fps and carried 25 fps of content — every frame duplicated. Sampled at five points across the whole file: 50–51% distinct, longest run of identical frames 2–3.

The cost of not checking: a dense re-extraction at one frame period (0.02 s) produced 16,000 frames of which ~8,000 were duplicates, and **detection ran on all 16,000** — half of a 41-minute GPU pass spent on images already seen. The correct floor for that source was 0.04 s.

Detect it by **run length of consecutive identical frames**, not by the ratio of distinct frames. Frame doubling gives runs of exactly 2; a static scene gives one long run, and those must not be treated alike or a quiet stretch will coarsen the whole file's grid. Sample several points, take the **smallest** median run (most conservative — one static stretch must not speak for the file), and clamp the reduction (÷4 is enough) so an entirely still clip cannot drive the interval to nonsense.

## The filename is a contract: it names the frame inside

Frames named by timestamp are the join key for everything downstream — window-boundary validation, candidate identity, grid alignment on the next dense pass. Any change to how frames are produced must preserve *that* correspondence, and "the bytes differ from the old method" is the wrong test to apply to it.

Extraction speed, measured on one 50 fps source, per 20 frames:

| Method | Time | Notes |
|---|---|---|
| one process per frame (`-ss t -frames:v 1`) | 3.55 s | seeks to a keyframe and decodes forward *for every frame* |
| same, 8 concurrent | 0.9 s | identical bytes by construction — same command, just parallel |
| one process per window (`-ss` + `-t` + `fps=`) | 0.36 s | decodes the span once |
| `-hwaccel` | slower | failed to initialise and silently fell back to software |

GPU acceleration is a red herring for extraction; concurrency and batching are where the time is. But batching changes *which* frame lands under a given name: a per-frame seek rounds each timestamp up independently, a batch counts forward from the seek point, so they disagree by 1–3 source frames (and the offset is not constant within a window, because the source is variable-rate). Decide explicitly whether your pipeline can absorb an error of one sampling interval in the filename→content mapping. If it can, batch and say so; if it cannot, run the per-frame command concurrently and keep the bytes.

Batching pays only where the span is dense. A long, sparsely-sampled window (1000 s at one frame per 10 s) must decode the whole span to emit 107 frames — measured 30.2 s against 4.2 s for concurrent seeks. Gate batching on the window being dense, and let the sparse ones seek.

## Do not thin the transition windows to save time

Tempting, and wrong. Measured, same windows, every 5th frame instead of every frame:

| Segment | candidates at 0.1 s | at 0.5 s |
|---|---|---|
| A | 9 | 3 |
| B | 8 | 3 |
| C | 9 | 2 |

And the losses are not random: the best shots vanish. The near-full-frame images — a person passing close to the camera, 0.74 of frame height — exist for **2 to 9 frames**. Sparse sampling misses that window outright; and even when it catches one frame, the tracker cannot associate a fast-moving person across 0.5 s, so it never becomes a candidate at all (raw track count fell from ~35 to ~15). Thinning removes exactly the frames that identify people.

If detection is slow, fix detection (see `division-of-labour.md`), not the sampling.

## Windows for a named moment must be scoped by the named segment

Two different jobs, two different rules:

- **"Here is a timestamp"** → you must infer which segment it belongs to, so require it to fall inside one, and refuse rather than guess. Guessing puts another company's footage in this segment's directory.
- **"Here is a timestamp *for segment N seat M*"** → the segment is given. The timestamp is then allowed to fall in the **gap between segments** — and it usually will, because entries and exits happen in those gaps. Clip the window to `[previous segment end, next segment start]`; only a timestamp outside *that* range is an error.

Getting this wrong rejects the feature's most common use. One real case: a human marked 18512 s for a segment that ended at 18506.8 — the five seconds during which the person stood up and walked out.

## Let the human hand you the moment

The cheapest input in the project. Two forms, both worth building:

- **Per-segment anchors** (when the group entered / left). Sampling windows are otherwise derived from the schedule by a guessed offset, and that guess can miss entirely: measured entries preceded the scheduled start by 26 to 130 seconds, and for one segment in seventeen the guessed 40-second window and the real entry did not overlap at all.
- **Per-seat moments** attached to "I could not find this person" — a checkbox alone tells you only that a gap exists; a timestamp tells you where to look. Accept **several** per seat (people appear more than once; one real entry was `1:01:12,1:24:21` — the entry and the exit). Accept clock and seconds forms. Parse in **one** place; never mirror the parser into the UI for a live preview, or the seconds shown and the seconds sampled will drift apart. Report unreadable pieces verbatim and use the rest; never guess what "about an hour" means.

Give the named moment **its own mode that samples only that window.** The re-sampling command's default, handed a seat marked "no usable image" with a moment attached, also doubled the density of that segment's whole seated window — re-detection there can reshuffle seats the human has already settled, and none of it was asked for. A `--spots-only` switch cut the job to one 20-second window; all 65 of the human's labels realigned.

And **anchor the window on the segment the label names, not on the time.** The moment fell in the gap between two segments — exactly where people walk in and out — and the time-based lookup refused it as "belongs to no segment". The label already says which segment the seat is in; use that.

## Wider is not better

Widening the anchored windows (40 s → 60 s, symmetric) looked like a free improvement and was reverted the same day. Two costs, neither obvious in advance:

1. The entry window reaches back into the *previous* group's departure, so two sets of people crowd the same candidate list and the human's job gets harder, not easier.
2. Every segment's identifiers move, so the widening alone cost 14 of 50 labels.

If a window genuinely misses the action, the fix is a better *anchor* (above), not a wider net.

## Every dense pass needs an undo

Record what a dense pass added — the exact file list — and provide a command that removes exactly those files and their index entries. Constraints, both learned:

- **Delete only files that pass recorded.** Everything else on disk carries the human's labels and other passes' results.
- **Keep files the current windows still want.** Decide by the current windows' *timestamp set*, not by interval range: after a parameter change the grid itself may differ, and a range test leaves orphan frames that no later pass will overwrite or clean.
- Refuse loudly when the manifest is missing. Without it you cannot tell added frames from original ones, and deleting by window alone destroys the first pass.

Changing the sampling parameters back is **not enough on its own**: the extra frames stay in the index and keep being detected, so the candidate list still reflects the old settings and the parameter change looks like it did nothing.

## Watch for stale inputs

The schedule and annotation files are inputs, and they change under you. A human edited one field in the annotation table; the derived per-segment file was never regenerated, so the review page kept displaying the old value for days. Nothing warned.

Store a digest of every input inside the derived artifact and check it before the stages that consume it. The check costs nothing and this class of staleness is invisible by construction — the artifact is well-formed, just answering an older question.
