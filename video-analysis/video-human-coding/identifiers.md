# Identifiers That Survive Re-Detection

Re-detection is not an edge case. Sampling more densely to find a missing person is a routine, necessary operation — and it renumbers every tracker ID in the video. If human labels are keyed by tracker ID, each re-detect silently reassigns them to different people.

Silently is the operative word: some old IDs still exist, so nothing is missing and nothing raises.

## Derive identifiers from content

Build a stable ID from what is visible, not from what the tracker assigned:

```
<kind>@<midpoint-time-seconds>#<median-cx>,<median-y2>
```

Kind (entry/exit/seated), when it happened, and where the body was. All three survive re-detection because they describe the video, not the run.

On load, re-align each label to the current candidate set with a tolerance (e.g. 3 s and 80 px). Report three counts: matched, re-aligned, unmatched. **Unmatched labels are dropped, never guessed.** Print them individually — a human can often re-label ten items in a minute, but only if told which ten.

## Key side-channels by (segment, subject), not by ID

Anything the human asserts *about a position* rather than about a specific detection:

- this seat has no usable frame anywhere
- exclude cell 2 of this evidence sheet
- this seat's automatic assignment is wrong
- this image actually belongs to seat N

All of these must key on `(segment_order, seat_number)`. Seat numbers are human-assigned and stable across re-detection; tracker IDs are not. Get this wrong and the human's work evaporates on the next re-sample, invisibly.

**But a position can host more than one artifact** — the automatic sheet at seat 2 and a human-named candidate also at seat 2. Two code paths reading that one key for two different images is its own silent failure; see `corrections.md` for the rule that keeps them apart.

## Reserved top-level keys

Side-channels live at the top level of the label file, beside segment numbers. Maintain an explicit list of reserved keys and skip them on load. Miss one and it gets treated as a segment number, expanding into entries that match no candidate — silently.

Keep deprecated keys in the reserved list forever. Users have old exported files, and an unlisted key is worse than a removed feature.

**Never attach an empty side-channel object.** Callers test `if labels:` to decide whether human input exists. An empty `{"seat_overrides": {}}` makes a file with zero labels truthy, and the pipeline announces that human review is active while doing nothing.

Being on the reserved list is only half the wiring: the channel must also be **re-attached** to the loaded object, on **every** load branch. See `corrections.md` § *Wiring is where these die*.

## Cell numbering inside an evidence sheet

When one detection box contains several people, the human splits it by grid position. Fix the mapping *by rendering the sheet and reading the numbers off it*, not by trusting a comment about row-major order. Use one function for both the split and the downstream fetch. Two implementations of "which image is cell 3" will diverge, and the symptom is one person's clothing recorded under another's name, with no error anywhere.

When one person occupies several cells, keep all of them. When merging images from *different* parent sheets, cell numbers stop being globally meaningful — carry each frame's number alongside it, and change the description text to stop claiming a single grid.

## Group-level people need a different identifier

Not everyone in the frame belongs to a segment. A panel of judges, a moderator, a camera operator sit through the whole recording; their clothing is one fact about the whole session, not one per company. Give them:

- **Group-level records**, not one copy per segment. Five judges across seventeen segments is five rows, not eighty-five — and the eighty-five would all be the same clothing, inviting a later join to double-count.
- **An identifier with no time in it.** The pattern used for people who come and go (`kind@t_mid#cx,y2`) is wrong here: a person present for the entire recording has no characteristic moment, and whichever moment the sampler happened to pick becomes a fake discriminator. Position alone is the identity: `role#cx,y2`.
- **Pooling across all segments**, because the tracker renumbers at every window. Merge fragments by position tolerance, not by tracker id, or one seated person becomes N candidates and the human identifies the same face N times.

**A fixed number of slots, with visible gaps.** If the panel is five, always render five. Detecting four and reporting "this group has four" destroys the distinction between "only four attended" and "the fifth was missed" — exactly the judgement you want a human to make.

**Ordering by geometry puts one person out of place and shifts everyone after them.** Sorting a row of seated people by depth is right until someone at the frame edge is picked up: on one recording a 266-frame detection at the far left edge took slot 3, pushing the real occupant (625 frames, in the middle of the row) out of the panel entirely and shifting two others by one. Print each slot's coordinates in the navigation so the outlier is visible — five positions listed together make a broken monotone sequence obvious at a glance, which no per-card view does.

### When the panel comes up one short, check before you report it

Four of five found is a finding about the *detector* until proven otherwise. Three checks, cheapest first, each answered by the detections already on disk:

1. **Extrapolate where the missing member would sit.** A seated row is close to evenly spaced in the image (one recording: 43 → 225 → 386 → 554, steps of 182 / 161 / 168). Look for tracks at the next step on both ends. Boxes whose bottom edge lies in the upper half of the frame are screen content, not a person.
2. **Test for two people merged into one candidate.** Within one segment, count frames where *two* boxes coexist in that candidate's area. Merged neighbours show up as sustained co-presence; one person shifting in their chair shows none (measured 0–3 stray frames per segment against ~46). Position spread alone cannot tell these apart — the same person drifted by 50 px between segments.
3. **Read the edge candidate both ways.** A box hugging the frame edge with far fewer frames than its neighbours was an intruder on one recording (it displaced the real occupant). On two others it was the member nearest the camera, partly out of frame — tallest box in the row, present in almost every segment. Height and segment coverage decide which; edge proximity alone does not.

If all three come back empty, say so and leave the slot visibly empty for the human. **The numbering is theirs too:** on one recording the human renumbered the four found members 2–5 and left slot 1 empty — the absent member was at the near end, off-frame, not at the far end where the geometric baseline had put the gap. A baseline that assumes the gap is always last would have been wrong for every person.

### Position-derived ids move when you re-sample

A position median is not as stable as it looks. Re-sampling **any** segment changes what feeds the median, and an identifier rounded to whole pixels flips on a sub-pixel shift. Measured: densifying 5 of 17 segments (28% of that person-class's detections) moved one person's median x from 245.48 to 244.45 — `#245,801` became `#244,801`, and the exact-string lookup then reported that nobody had ever labelled them.

So these ids need the same realignment layer the segment-level ones have:

- Match **exact ids first** and remove those candidates from the pool, before any nearest-match pass, or a shifted label can steal the candidate belonging to an unshifted one.
- Then nearest-match within a tolerance smaller than the gap between neighbours (25 px against a 56 px minimum spacing, on one recording).
- **Refuse when two candidates are within tolerance.** Picking one is a coin flip that attributes a person's clothing to someone else.
- **Print the match rate every time.** `5/5 matched` is the only visible evidence that a day of work survived.
