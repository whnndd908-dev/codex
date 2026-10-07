# Splitting the Work: Machine, Server, Human

## What is actually the bottleneck

Wall-clock cost and human cost are different resources, and only one of them is scarce.

| Stage | Cost type | Parallelisable | Automatable |
|---|---|---|---|
| Frame sampling | IO/CPU minutes | across videos | fully |
| Detection + tracking | GPU minutes — usually the longest | across videos | fully |
| Geometric ordering | seconds | — | fully, but **produces a guess** |
| Identity verification | **human hours** | **across videos only** | not at all |
| VLM coding | money + minutes | across subjects | fully, *after* verification |
| Page rendering | minutes | across videos | fully |

Moving stages to a server removes **waiting**, not **work**. The human hours are unchanged. Say this out loud before anyone plans a schedule around a GPU purchase.

## What genuinely accelerates the project

**Parallelise across videos, not within one.** Make each video a self-contained folder: review page, label file, registry, preview page, evidence images, and a README naming exactly which file to send back. Three people label three videos simultaneously and each returns one JSON. This is usually the largest available speedup and it needs no new infrastructure.

**Over-sample up front instead of iterating.** The naive loop is: label → discover a subject has no usable frame → re-sample densely → re-detect → re-label. Each turn is serialised on a human. If server time is cheap, sample densely everywhere *before* the human starts, so they label once. Disk and GPU are the cheap resources here.

**Buy the anchor timestamps from the human before spending GPU on a guessed window.** Sampling windows are usually derived from a schedule — "the entry happens shortly before the talk starts". That offset is a guess, and the cost of it being wrong is total: if the window misses the event, no amount of extra density inside the window produces a single usable frame, and the gap is invisible — the run completes, the page renders, the subject simply has no standing shot. Marking the actual moment takes a human seconds per segment (they scrub, they read off a timestamp) and it is the cheapest input in the whole project. Measured on one 5.5-hour recording: the entry preceded the scheduled start by anywhere from 26 to 130 seconds, and for one segment in seventeen the guessed 40-second window and the real entry did not overlap at all.

Three things follow. Treat the mark as optional per side and per segment — real annotation is ragged, and one missing mark must not void the other or the segment. Validate it hard against its neighbours: a mistyped magnitude anchors the window inside a *different* subject's footage, which is the same silent cross-contamination that frame-to-schedule mismatch causes. And keep the window derivation in one function so the dense pass inherits the corrected anchor automatically — a second copy of the interval arithmetic is how the dense pass ends up sampling the old, wrong place.

Better anchors, **not wider windows**: widening them symmetrically was tried and reverted the same day, because a wider entry window reaches into the previous subject's departure and mixes two sets of people in one candidate list — and because changing the windows at all costs labels (`iterating.md`).

**Check what device the inference framework actually chose.** It reports nothing and it is not obliged to pick the fast one. On one machine the framework's auto-selection returned CPU while a GPU sat idle: 0.955 s/frame instead of 0.192 s with the GPU and half precision — five times slower, for a stage that processes tens of thousands of frames, with no log line to suggest anything was wrong. Print the chosen device and precision on every run.

Half precision is worth measuring rather than assuming: on the same 150 frames it produced the *same number of boxes*, coordinates within ~0.1 px, confidences within 0.05, and the only structural difference was the track numbering shifting by one — which is already treated as unstable across runs. That is a fine trade; a silent 5× penalty is not.

Two accelerations that did **not** work here, recorded so nobody re-derives them: multiple worker processes over a single GPU gave identical total throughput at 1, 2, 3 and 4 workers (and worse at 4) — the device was already saturated; and handing the framework a list of paths for "streaming" made it batch the whole list and exhaust GPU memory. The real win was elsewhere — re-running only the groups whose input changed (`iterating.md`).

**Detection is sequential by construction and that caps what tuning can buy.** A tracker carries state from frame to frame, so batched inference is not available: the frames must go through one at a time. Two further measurements on the same stage, each ruling out an obvious idea:

- *Decoding is not the cost.* Reading and decoding a 1440×1080 JPEG took 3 ms against 103 ms of inference. Pre-decoding in worker threads and handing the model arrays instead of paths changed the total by under 1%.
- *Shrinking the inference size is not free, and the damage is in exactly the wrong place.* Measured end-to-end on one segment, not by box counts:

  | inference size | ms/frame | candidates produced | tallest boxes (fraction of frame height) |
  |---|---|---|---|
  | 1280 | 120 | 23 | 0.773, 0.743, 0.740, 0.735, 0.733 |
  | 960 | 89 | 23 | 0.95, 0.946, 0.942, 0.941, 0.94 |
  | 640 | 43 | **18** | **1.0, 1.0**, 0.629, 0.437, 0.381 |

  At 640 the pipeline lost a fifth of its candidates and the two "best" boxes became full-frame-height — junk. The near-camera, full-body frames are the ones that let a human identify anyone, and they are the first thing to go. Judge this metric by *candidates and their geometry*, never by raw box counts, which went **up** as accuracy went down.

So the honest accounting for a long run of recordings: extraction can be cut several-fold with concurrency and batching (`iterating.md`), but detection cannot be tuned down without paying in identification accuracy. What remains is hardware and parallelism across videos — which is where to spend, because videos are independent and the code already takes a device override.

**Run parameter variants in parallel, pick before handing over.** Different detector weights or thresholds can run concurrently on a server. Choose the best output, then give the human one page — not three rounds.

**Never run the paid model before verification.** Coding an unverified registry buys wrong data at full price, and the errors are invisible in the output. Order is not negotiable.

## What must stay local or gated

- **Credentials.** API keys and relay endpoints belong in server-side environment injection, never in an image, a log line, a report, or a commit.
- **Source media**, if the recordings are sensitive or licence-restricted. Check before uploading anything.
- **The verification UI itself** is a static page; it needs no server, and keeping it static means a labeller needs nothing but a browser.

## Getting the labels back out of the browser

A static page cannot write to a chosen path. Plan for this or the human's work lands somewhere neither of you expects:

- Offer a real save-to-path where the browser supports one, remembering the chosen location across reloads (a file handle survives in IndexedDB; a string in `localStorage` does not).
- Degrade in steps — remembered handle → re-request permission → ask again → plain download — and wrap **every** step, because private-browsing modes throw on the storage API alone.
- Keep a plain download as the final fallback; some browsers have nothing else.
- Have the pipeline also collect the file from the default download folder, and print where it looked.
- Write the expected path **on the page, next to the button**. Saving to the wrong folder produces "no labels at all": the run completes, everything falls back to the automatic baseline, and nothing reports a problem.

Wrapping every level matters more than it looks: a single stray character inside that script once disabled export entirely while the page still rendered and the button still clicked. If the export path is assembled by string surgery, syntax-check the result in CI.

## Handing a video to someone else

The bundle must be verifiable without you. It should contain the exact images the model will receive, so the reviewer's sign-off means something. It should carry a stamp naming which label file it came from and when. And the README should state the single file to return.

When they send it back, re-run the registry build and check the freshness digest. If it does not change, either they changed nothing or they sent the wrong file — both are worth catching before you spend money on model calls.

Do not trust the *name* of the file they return, and do not let the pipeline depend on it. One came back named after a different video while containing the right content — the person had re-used an earlier download's name. A lookup by filename finds nothing, the run falls back to the automatic baseline, and the output looks complete. Match on the fingerprint inside the file instead, say which file was used, and if you must rename it, copy rather than move and leave their original untouched.
