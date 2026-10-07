# Verifying Work Nobody Will Re-Check

The defining hazard of this project class: **the wrong answer looks exactly like the right one.** A mis-assigned identity produces a complete, plausible, correctly-typed row. No exception, no missing key, no failing assertion. If your verification only proves the code ran, it proves nothing.

## Test-first is not ceremony here

Write the failing test, watch it go **red for the reason you expect**, then implement. A test written after the fix is a test that passes against the fix — it has never demonstrated it can detect the bug.

Read the red output, don't just note the exit code. Twice in one session a test failed for a reason different from the one intended: the fixture's dict happened to iterate in the lucky order, and a rendered-HTML assertion matched the JavaScript *source* rather than any rendered element. Both would have passed as "red then green" while testing nothing.

## Mutation testing is the real check

Green tests prove the code passes the tests. Mutation testing asks the only question that matters: **would these tests notice if the code were wrong?**

Keep an explicit list — each entry a plausible wrong version of a specific line, plus the test file that should catch it:

```
for name, path, anchor, replacement, testfile in MUTATIONS:
    assert src.count(anchor) == 1     # zero = never applied; two = changed more than intended
    write(mutated); clear_pyc(); run(testfile); write(original); clear_pyc()
    assert md5(path) == before        # a script that corrupts the source is worse than a survivor
    survived = (returncode == 0)
```

Non-negotiable mechanics:

- **The anchor must match exactly once.** Zero matches means the mutation never applied and "survived" is a lie. Two means you changed something else too.
- **Verify the restore by hash.** A mutation script that leaves the file damaged and doesn't say so is worse than any bug it was hunting.
- **Clear `__pycache__` and set `PYTHONDONTWRITEBYTECODE=1`.** CPython caches on `(mtime_seconds, size)`. A script that rewrites a file twice within the same second gets the *previous* bytecode back — every mutation "survives", and the list looks perfect.

## Reading survivors honestly

A survivor is one of three things, and they demand different responses:

**Weak fixture.** The most common. Observed causes, all real:
- uniform-colour frames, so colour-measurement mutations changed nothing
- identical boxes across a track's frames, so "which frame" mutations were invisible
- `track_id` equal to the seat number, so confusing the two was undetectable
- an assertion matching a string that also appeared in a comment in the code under test
- the fixture's target slot was empty, so a "yield the previous occupant" rule was never exercised
- only one direction (exit, never entry) tested, so a direction-swap survived
- **the dict happened to iterate in the order that hides the bug**

Fix the fixture, don't relax the mutation.

**Dead code.** The condition can never be true. Delete it. One guard read `if card_seat in moved_away and card_seat == seat` — reassignment guarantees old ≠ new, so it was unreachable from birth.

**Genuinely equivalent.** Two guards enforcing the same thing, so mutating either changes nothing. Do not leave both: that is a *fake second line of defence*, and whichever one someone later breaks will fail silently. Pick the one that should be load-bearing, make it so, and assert the invariant it maintains so removing it fails loudly.

Never rely on dict insertion order for correctness. It passes today, inverts when someone rewrites a merge as `{**a, **b}`, and no test goes red.

## Synthetic media is the fixture trap that bites hardest

A generated test clip is *too well-behaved to reproduce the bug you are guarding against*. Every frame a keyframe, a constant frame rate, no duplicates, no variable timing. Measured, same assertion, one session:

| Fixture | Frames matching the reference extraction |
|---|---|
| generated test pattern | 30 / 30 — shipped as proof of equivalence |
| the real recording | 16 / 30, and 2 / 30 when the window start was not frame-aligned |

The synthetic green was not a weak assertion; it was the right assertion against media that lacks the property under test. **Any claim of the form "the new extraction path is equivalent to the old one" must be verified on the real source.** Guard the real-media test with a skip so it degrades on machines without the file, rather than dropping it.

The mirror image also bites: to reproduce a *defect* of the real source in a fixture, you may have to disable the lossy step that hides it. Building a clip whose frames come in identical pairs needed lossless encoding — with normal compression the encoder wrote the two frames slightly differently, the decoded bytes differed, and the fixture contained no duplicates at all. The test then verified nothing and passed.

## Vacuous fixtures: the assertion that cannot fail

Distinct from a weak fixture. Here the *scenario* makes the assertion true regardless of the code. Four from one session, all initially green, all caught only by mutation:

- **The set you assert is empty is empty by construction.** "When every seat is filled, the leftover-candidates row disappears" — tested with three candidates for three seats, so there were never any leftovers. Squeeze the capacity (three candidates, two seats) so the thing you claim is hidden actually exists.
- **Every sample is identical, so competing implementations agree.** Choosing the *minimum* run length versus the *maximum* is only observable on a mixed input; on a uniform clip both mutations pass. Build the mixed case (part static, part moving).
- **You truncated a suffix where only a middle gap discriminates.** Dropping the tail of a sequence cannot distinguish "stop at the first gap" from "skip the gap and continue" — only a hole in the middle can. Test the mapping rule as a pure function and hand it the hole.
- **The string you slice on also occurs inside the element.** Splitting rendered HTML on a person's display name to isolate their card cut it short, because that same text appears inside the card's own dropdown options. Slice on structural boundaries (the element's class or id), never on display text. The same trap catches *labels*: searching for "seat number" found the phrase in the card's help text ("clears the seat number, count and note") rather than the control.
- **The key you write the fixture state under is not the key the code reads.** A UI test seeded browser storage under `order:id` while the page composes `order::id`. The label was simply absent, the page rendered its default, and the assertion about the default passed. When a test drives a UI through its storage, assert first that the state *arrived* — check the control reflects it — before asserting anything about behaviour.

Mutation testing is what exposes all four: each produced a survivor whose only explanation was "the scenario cannot discriminate".

## A mutation that did not land looks exactly like a test that works

Mutation by text substitution silently targets the *first* match. Two different functions in one file each had a local named `dropped`; the substitution hit the one in the section renderer while the assertion under test was about the table of contents. The suite stayed green, and the honest-looking conclusion — "this test does not hold the behaviour down, delete or rewrite it" — was wrong.

A mutation that changes nothing relevant is indistinguishable from a test that cannot fail. So before believing a survivor: check the edit landed where you meant. Substitute on a unique anchor, or count occurrences first and address them by position; then confirm the mutated source really differs in the spot you care about. The same applies to mutating a value into something the renderer swallows — replacing a count with `true` made a template print the word "true" rather than always rendering the row, which tested nothing.

## A test that never runs, and a helper that never matches

Two ways a green suite hides dead assertions. Both were found only by changing unrelated code and noticing the numbers move.

**Duplicate test names.** Python binds the later `def` over the earlier one, and the collector only ever sees the survivor. Three pairs had accumulated in one file; in each pair the *dead* one was the richer test (it checked the rendered page, not just the data structure) and had rotted years-of-commits deep — one asserted on base64 image data from before images became files, and would have raised had it ever executed. The live one was a strict subset, so coverage looked fine. Scan for it directly: `grep -o "^def test_[A-Za-z0-9_]*" file | sort | uniq -d`, in CI. When you find a pair, read both before deleting either — the dead one usually tests more.

**A selector that matches nothing.** A helper extracting "the *still missing* line" from rendered HTML looked for a CSS class the page had stopped emitting. It returned `None` for every input, which made one assertion ("no such line when everything is filled") vacuously true and another ("the line names seats 2 and 3") impossible — and the impossible one happened to live inside a shadowed test, so nobody saw it fail. Any helper that *finds* something in output needs its own test that it finds it, or it will quietly become a constant.

## Two surfaces computing the same number will drift

Whenever a browser page and the pipeline each compute "how complete is this record", they will disagree, and the human trusts whichever one they are looking at. Observed: the review page reported a person's evidence sheet complete while the registry recorded it two cells short — the page counted only that person's own cells, the registry also counted cells the human had reassigned *in from* other candidates.

Neither number was wrong on its own terms; there were two definitions. When you cannot share code across the language boundary, **write the rule once in a comment on both sides naming the other**, and put the counting in one function per side rather than inline at each use. Then mutate one side and confirm a test of the other side's behaviour goes red.


## Assert behaviour, not source text

A whole family of features was "tested" by searching the generated page for the source of the function that implements them. Every one of those tests passed while the feature was broken end to end, because the source really was present — it just never ran, or ran on the wrong pool.

Symptoms of this anti-pattern: assertions containing a function name, a regex over the emitted script, `assert "someFlag" in html`. Replace them with the observable outcome — run the emitted code, or inspect the emitted data structure, and assert what the human would see. Two concrete cases:

- A command printed the plan it had computed and then let the worker recompute its own. The test asserted the printed interval appeared in stdout; it did. The run added **zero** frames. The honest assertion is "more files exist on disk at the finer setting than at the coarser one".
- A correction channel had four tests, all matching strings in the emitted script. The channel had never affected a single downstream number.

Rule of thumb: if the test would still pass with the feature's call site deleted, it is testing text.

## Beware assertions that match your own comments

`assert "不解析" not in html` broke the moment a comment in the code under test used the same phrase. Choose a string that only the user-facing artifact can contain — a full sentence from the rendered text, not a word that also reads naturally in prose about the code.

## Fixtures must resemble the real thing

Repeatedly, a fixture passed while the real data failed, because the fixture omitted a precondition that is always present in practice:

- Candidates cannot fill a slot the automatic stage already occupies. A fixture without the human's "vacate this slot" produced a record of the wrong kind entirely, and the test silently verified something else.
- Merging two sources with entirely disjoint frames never exercises de-duplication.
- Distinguishing "which source did this image come from" requires the sources to use **different files**; a fixture where every frame shares one image can only count, never identify.
- The destination slot was *already empty*, so "put it in the pool of selectable slots" could not be observed — the mutation removing that line changed nothing. Build the fixture in the state where the line matters (here: every slot occupied).
- Only one of a pair of symmetric fields was set, so "read the wrong one of the pair" fell back to the right value by accident. Real records carry both; give the fixture both.

When a test's premise depends on the pipeline reaching a particular branch, assert the premise (`assert rec["source"] == "cell_split", "fixture precondition"`). A premise assertion that fires is a five-second fix; a premise that quietly stops holding wastes an afternoon.

## Data-validation tests fail differently

Some tests check the *code*; others check the *incoming data* against a regularity someone observed — "position 1 always does X, it held for all 16 rows". Both go red the same way, and it is easy to treat the second kind as a broken test.

It usually isn't. When a data-validation test fires on a new batch, the finding is **about the data**, and it belongs to whoever owns the research question. Report the counter-example — which record, which field, which value — and stop. Do not rewrite or weaken the assertion to get the suite green: that silently discards the one signal that the new batch differs from the assumption the analysis rests on. Whether the regularity was a real invariant or a coincidence of the first batch is not a question the test author can settle alone, and it may change how the data is analysed.

Worth knowing which kind a red test is before touching it: a fixture built in the test file is checking code; one that loads whatever dataset the configuration currently points at is checking data.

## Tests against live human-labelled data

Useful — they catch what synthetic fixtures cannot — but they break every time the human relabels, and a test that fails for that reason trains people to ignore failures.

Derive expectations **from the label file at runtime** and assert invariants ("every seat the human named appears in the registry"), not literals ("segment 3 has 4 images"). When the data cannot support the check, skip with a reason that says plainly that **nothing was verified** — a silent skip reads as a pass.

Watch the skip count. It going from 4 to 9 after a refactor was the only visible sign that path resolution had broken.

## Editing source with scripts

Anchored replacement over line numbers, always, and `assert count == 1` before writing.

Never slice with two searched indices without checking their order. `s[s.index(A):s.index(B)]` where B precedes A yields the empty string, and `str.replace("", new)` inserts the replacement **between every character in the file**. That destroyed a 2000-line module in one command; only version control saved it.

After changing anything shared, run the **whole** test file, not the `-k` subset you were iterating on. Filtering to the tests you were thinking about hid 21 failures caused by reading a variable outside its guard.
