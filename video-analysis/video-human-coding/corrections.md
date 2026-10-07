# Composing the Human's Corrections

Once a human can correct the machine, they will accumulate several *kinds* of correction, and those kinds interact. Every interaction you leave undefined becomes a silent wrong answer.

A realistic set, all pointing at the same seat:

| Channel | What the human means |
|---|---|
| reject | "the automatic result here is wrong" |
| exclude cell N | "cell N of *this* evidence sheet is a different person" |
| reassign to seat M | "*this* image is actually seat M" |
| name a candidate | "looking at this picture, that is seat M" |
| no usable frame | "I looked; nothing in this segment works" |

## Write the interaction table before writing the code

For every ordered pair, answer: **which wins, or do they merge?** Then encode it in one place and test each cell. The table is short; discovering it one bug report at a time is not.

Rules that held up in practice:

- **A newer, more specific statement supersedes an older, vaguer one.** "This image is seat 2" outranks a stale "seat 2's automatic result is wrong" — reassigning *is* how they are now saying it is wrong. Leaving the old rejection in force silently discards the new correction.
- **A rejection attached to a slot refers to the artifact that was in that slot.** After the artifact moves, the rejection does not follow it, and it does not apply to whatever arrives.
- **Two human statements about the same person merge; a human statement never merges with an unverified machine one.** If they name a candidate as seat 2 *and* reassign a seated image to seat 2, both are that person — combine the images. If the seated image merely landed at seat 2 automatically, it must not be merged in; that is exactly the unverified guess the human is there to override.
- **Never infer a second correction from one statement.** "This is seat 2" does not license swapping seat 2's previous occupant into seat 1. Move what they named; return the displaced one to undetermined and let them speak. Adjacent-swap is a plausible guess, and plausible guesses are how wrong data enters a dataset that nobody re-checks.
- **Contradictions that cannot be resolved should raise, not pick.** Two seats reassigned onto the same target: refuse the segment and say so. Picking one makes the result depend on dict ordering, which changes with edit order and reports nothing.

## The key-collision trap

Side-channels are naturally keyed by *position* — `(segment, seat)` — because positions survive re-detection while tracker IDs do not. But a position can host **more than one artifact**: the automatic seated sheet at seat 2, and a human-named candidate also at seat 2.

If two code paths read the same `(segment, seat)` key for two different images, the human's "exclude cell 2" is applied to an image they never looked at. The cells removed are simply the wrong ones — no error, no missing key, no size mismatch.

Two rules keep this straight:

1. **Only the path that renders the control may consume it.** If the checkbox exists solely on the seated sheet, only the seated path reads it. Other paths must not "also" apply it for symmetry — that is precisely the bug.
2. **A correction follows the image, not the slot.** When an image is reassigned from seat 1 to seat 2, look its exclusions up under seat **1** — the number that was on the card when the human ticked it.

If you ever need per-cell corrections on a second artifact at the same position, the key must name the artifact, not just the seat. Do not overload the existing key.

## Every "is this slot free?" test must read the corrected state

A correction that vacates a slot is only half-applied if the *occupancy map* still shows it taken. This one bites in several places at once, because occupancy is consulted by more code than you think: the fill planner, the "which seats can this candidate be pointed at" dropdown, the progress counter, the gap report.

The symptom is a pair of contradictory statements in one run — "seat 1 already has someone, not overwriting" alongside "seat 1: nobody has looked at it". Whenever you see two outputs disagree about the same slot, suspect two different occupancy computations, one of them reading the pre-correction baseline.

Two rules:

- Derive occupancy **once**, from the baseline *after* applying vacate/move corrections, and pass that everywhere. Grep for every place that asks "does this slot have someone".
- When a move frees a slot, express it in that map as **removing the source entry**, not as re-pointing it at the destination. Re-pointing makes the moved artifact "occupy" the destination, which then blocks the more specific correction — a candidate the human named for exactly that slot. Who ends up at the destination is decided by the priority table above, not by the occupancy map.

Wording follows the same rule. A control that says "adding to the existing person here, not replacing them" must not appear on a slot whose occupant was moved away or rejected — there is nobody there to add to, and the label tells the human their action will not do what it will.

## Merge, don't discard, when the target is under-filled

"Do not overwrite an occupied slot" is right about *replacing* and wrong about *adding*. If the human names a second image for a person who already has one, and the evidence sheet is not full, the second image belongs in the empty cells. Anything else throws away work they did on purpose.

This holds across segment boundaries too. A label written on segment N that points at segment N+1 must be able to take cells in N+1's sheet — not only to fill N+1's empty slots, and (once sources share the budget, see `evidence-sheets.md`) even when N+1's sheet is already full. The cross-segment path is usually written later than the same-segment one and inherits only the "fill" half; the symptom is a log line saying "not placed, that seat already has someone" while the target's sheet visibly has empty cells.

The reverse symptom appears after you fix it: the planning stage still prints "not placed — seat taken" (it only decides who *occupies* a seat), while the images were merged into that seat when the sheet was composed. Word that line for what actually happens ("same person, images merged into this seat's sheet"), keep genuinely unplaceable flows — no such segment, seat number beyond the headcount — under the warning, and pin the "merged" claim with a test before rewording it.

Record the provenance per record when a sheet ends up mixed: a flag meaning "this whole record came from a neighbour" cannot describe two frames from here and two from next door, and someone will later check the timestamps against this segment's range and find they do not fit.

The same top-up has a second, sharper edge: if the human marks *two different candidates* as the same person, the sheet is filled from both, and the model then averages two bodies' clothing. That is the human's claim, not a bug — but they made it one image at a time and never saw the consequence. Keep the per-frame provenance (which candidate, which track, which cell) in the record, and surface it on the preview page, so "these four cells are two different people" is visible to them rather than inferable only by tracing boxes back to detections. Note that the *plan* stage may be strict here — refusing to mix cells from two candidates — while the *registry* top-up is permissive; when both exist, the strict test passes and the mixing still happens downstream.

## Confirm the exception, not the rule

A validation that rejects implausible input is also rejecting, occasionally, input that is merely unusual. When the human looks at the video and certifies such a value ("the timestamps are right, process it"), do not relax the check and do not add a global "warn only" mode — the rule is protecting every other group from the typo it exists to catch (a mistyped exit time silently samples another company's people).

Record the human's confirmation as **data**, keyed so it cannot outlive what was confirmed: `(group, segment, side, value)`. Include the *value*. If the source cell is later edited, the key no longer matches and the check is back in force — an old confirmation must never wave through a new typo. Print a line each time a confirmation is used, so the exemption is visible in every run log rather than only in the commit that added it.

When you raised the doubt in the first place, state the evidence once, with the exact values and why they look wrong; if the human reaffirms, stop arguing and make their answer durable. Re-asking on every re-run, or on another machine, is the failure — not the original question.

## A deletion is a statement, not an absence

If the human can clear a label, that clearing must survive a reload. The natural implementation — delete the key from local state — is undone the next time the page merges in the stored file, which still contains the label. From the human's side: they clear it, it disappears, they reopen the page, it is back, and nothing explains why.

Store the clearing **explicitly** (a tombstone, or an explicit `false`), have the merge skip keys the human has cleared, and drop the tombstone as soon as they label that item again. The same reasoning applies to un-ticking any checkbox whose ticked state is persisted: an absent key means "never said", not "said no".

Two further halves of the same rule, both learned by shipping only one of them:

**A negation must not hide its own undo.** Marking a person "not one of the people we are coding" removed their card from the page — and that card carried the only checkbox that could take the mark back. The mark persisted in storage while nothing on screen referred to it. Keep the negated item rendered somewhere of its own, with its control still on it and still showing the marked state, and say plainly what the empty position now means: the vacancy left behind must name the human's own action ("marked as not a judge"), because the generic "nothing found here — absent, or missed by detection" reads as a pipeline failure and sends them to debug the wrong thing. Put a route to it in the table of contents too; a section at the bottom of a long page is not where someone looks after a mis-click.

**The round trip has two directions and the second one gets forgotten.** The negation was written on export and honoured by the pipeline, so every test passed and the registry was correct — but the page itself only ever read browser storage. Open the package on another machine, in another browser, or after cleared site data, and the human's decision is simply gone: the person sits back in their slot with no trace. For each correction channel, test page → file → pipeline **and** file → page. When you add that backfill, decide precedence on whether the **key is present**, not whether the value is truthy, or the human's explicit undo (an empty string) loses to the stale mark in the file — the same "clear it, reload, it's back" failure as above, arriving by a different route.

## Wiring is where these die

Three ways a correction channel silently stops working. All three have shipped:

- **Not listed as a reserved key at all.** Worse than the next item and easy to miss when adding a channel. Top-level keys that are not on the reserved list get treated as segment identifiers: the loader tries to match the channel's contents against that segment's candidates, matches nothing, counts every entry as a **failed match**, and drops the lot. Observed: a whole day of per-cell verification vanished, and the only trace was the re-alignment line reading `57 labels, 50 matched, 7 dropped` — five of those seven were the new channel, invisible in the number. Adding a channel means: reserved list, re-attach on **every** load branch, and a test that the channel survives a fingerprint mismatch.
- **The match-rate gate voids the stable channels too.** A file with zero candidate-level labels scored 0% and was discarded whole — taking with it channels keyed by seat number and by position, which are stable across re-detection and had nothing to do with the rate. "No labels to align" is not evidence of staleness. Gate on the rate only when there is something to rate.
- **Listed as a reserved key, never re-attached** to the loaded object. The file has it; nothing reads it.
- **A parameter added, no call site updated.** The function accepts `reassign=`; all three call sites still pass four arguments. Grep for *every* call site, not just the one you were looking at.
- **Only one load path updated.** Loading typically has several branches — fingerprint matches, no fingerprint supplied, fingerprint differs so re-align. A correction must survive **all** of them. The re-align branch is the one that gets forgotten, and it is the one that runs after re-sampling — so the symptom is "everything I fixed disappeared after the re-detect", reported long after the change.

Test each load path separately. A single test on the happy path passes while two of three branches drop the data.

## Regenerating deliverables

Give each deliverable exactly **one** blessed entry point, and use it. Pages are often assembled as *render + post-process injection* (save-to-path handling, pre-filling existing labels). Calling the renderer directly produces a page that opens, looks right, and is quietly missing the injected half — including, in one case, the pre-fill, so the human's existing work appeared to have vanished.

Document the entry point next to the command, with the consequence of bypassing it spelled out. "Use the official path" is not memorable; "bypassing it makes the human think their labels are gone" is.

## One rule, one implementation — including across languages

Most of the failures above are the same shape: a rule that exists in two places, where one copy was updated. When the review UI is a browser page and the pipeline is another language, the temptation to mirror a rule is strongest and the divergence is hardest to see, because each side is self-consistent.

Two that shipped:

- A quantity control ("how many people are in this box") governed whether per-cell assignments were *offered* in the UI but not whether they *counted* in the pipeline. Setting it back to 1 hid the rows on screen while the pipeline still split the person into three. Whatever a control means, both sides must mean it.
- Timestamp parsing. Resist adding a live preview that re-implements the parser in the UI; keep one parser and have the consuming stage print what it understood. A preview that disagrees with the sampler is worse than no preview.

When you must have a value on both sides, generate one from the other or test them against the same table. And prefer *hiding* to *deleting*: a control that means "ignore these" should leave the underlying data intact, so toggling back restores the human's work rather than silently destroying it.

## "Who is it" beats "not this person"

The cheapest per-cell control to build is a reject: *this image is not the person it is filed under*. It is also the one that throws data away. When a tracker merges a neighbour into someone's track, the rejected frames are not corrupt — they are **filed under the wrong person**, and the human looking at them usually knows whose they are.

Offer the identity instead of the veto: a per-cell choice of *this person / actually person N / nobody*. Rejected cells then move to the record they belong to rather than being deleted. Measured on one group: five people had 7 of 20 cells rejected, leaving two of them with a single usable image each; re-expressed as reassignments, the same 7 cells refilled the records they belonged to and every person reached a full sheet.

Consequences to design for:

- **A reassigned cell must outrank an unverified one.** The receiving record's own cells are a machine baseline nobody has looked at; the incoming cell carries an explicit human statement. If the receiver's slots are already full of baseline, the verified cell silently loses. Decide the precedence and write it down; whichever way you choose, the loser must be *reported*, not dropped.
- **A donation the target cannot fit must be visible.** The human pointed a cell at someone; if it does not appear, they will assume they mis-clicked. List it as "could not place".
- **Reassignment only lands when the target has made room.** Twice the human reassigned cells and nothing happened, because the receiving record's own four slots were still occupied. The page must show the receiver's state, or this reads as the feature being broken.

## Growing the candidate pool without moving the labels

Per-cell labels are keyed by cell number. The moment you offer *more* candidate images — a spare pool for the human to pick from when rejects leave a record short — the numbering is at risk: any change to the first N entries silently re-points every label the human already made.

Make the pool a **prefix-stable extension**: entries 1..N must be byte-identical to what the old selection produced, with spares appended after. That is automatic if both come from the same ranking and you only widen the slice.

Two details that matter more than they look:

- **Draw the spares from different segments.** The frames that get rejected are usually a neighbour merged in during one stretch of the recording; the next-best frames by score come from that same stretch and are the same wrong person. Take the best frame from each *unused* segment first, and only then fall back to more of an already-used one.
- **Number the spares by pool position on the image itself.** If the second sheet is drawn with cells labelled 1–4 while the human's control says 5–8, they are reading one number and recording another. Whatever numbers the controls use must be the numbers painted on the image.

## The class filter will exclude a member of the class

Any rule that says "these people are the ones on the left / in this region / of this size" is a heuristic about *where a class usually sits*, and some member of the class will sit elsewhere. One panel of five judges had its fifth member seated on the **opposite side of the room**, next to the projector. The detector that harvests judges filtered to the left region, so that person produced **zero** judge candidates: the panel showed four of five, the "not seated yet" list was empty, and there was nothing anywhere for the human to point at.

Do not widen the filter — widening it pulls in everyone on the right and the list stops being usable. The person is not lost; they are in the *other* class's candidate list, where the region filter put them. So the fix is a **cross-class reassignment**: let the human mark a cell of an entrepreneur candidate as "actually judge N", reusing the same donation path that moves cells between judges. One extra option in a dropdown recovered a person the detector could not reach.

The general shape: for every class you harvest by position, provide a path from the *other* classes into it, and make the receiving record able to exist with no candidate of its own.

**A slot filled only by donations has no identifier.** It has no detection of its own, so leave its stable id null and keep it out of the "N of M resolved" count. Minting one would fabricate a detected person, and nothing downstream could then tell a harvested judge from a hand-assembled one.

## One control, two renderers

A control that appears conditionally will have more than one code path, and you will fix one of them. The seat-number selector here is rendered twice: once per cell when a box holds several people, and once at card level when it holds one. Adding "judges have numbers 1..5" to the per-cell path looked complete and tested green. The user's very next message was that selecting *judge* on a one-person candidate produced no number field at all — the card-level renderer still had `show only when role == entrepreneur` from before judges existed.

Two habits that would have caught it: grep for every place the control's label appears before declaring the change done, and write the test against the *specific shape* the user will meet (one person in the box), not the shape you happened to be editing.

**When a role changes what a number means, change its label too.** The same widget showing "seat 1–3" for an entrepreneur and "seat 1–5" for a judge invites the human to read a panel position as a table position. Rename it in place — "which judge" — so the two readings cannot be confused.

**Per-cell marks outrank the whole-card role.** Support both (a one-person card has no cells to mark), but when cells exist, use only those. Collecting both means the same frame arrives twice and the evidence sheet shows a duplicate.
