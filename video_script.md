# Video Talking Points — Assignment 3 (target 7 min, mic + cam on)

Tone: talk through the story, don't read the report aloud. The one thread that ties it
all together: **a model is only worth what it scores on patients it has never seen.**

Times are cumulative. Each block lists what to *say* (cues, not a script to read word for
word) and what to *show* on screen.

---

## 0:00 — 0:40 · Intro

**Show:** title slide / notebook top.

- "This assignment classifies single colon cells from histopathology images."
- Two tasks: **Task A**, is the cell cancerous (binary); **Task B**, which of four cell
  types (epithelial, inflammatory, fibroblast, other).
- "The whole project is really about one idea — does the model generalise to *new
  patients* — so keep that in mind, it's where the interesting result is."

## 0:40 — 1:40 · Data and EDA

**Show:** `eda_class_distribution.png`, then `eda_pca_kmeans.png`.

- 20,280 cells, tiny 27×27 RGB patches, from 98 patients.
- Both tasks are **imbalanced** — point at the bars; Task B's "other" class is small.
  "That imbalance isn't noise, it's the difficulty of Task B."
- PCA + k-means: first two components explain very little variance, and clusters match
  the real labels at an adjusted Rand index of only ~0.11 — basically chance.
- Key line: "So the classes are **not linearly separable** in pixel space. That's a
  prediction — simple linear models should fail, and a model that learns its own local
  features should win. Hold onto that."

## 1:40 — 2:40 · Pre-processing — the patient split

**Show:** the patient-split code cell (Section C).

- "The single most important decision in this whole assignment is here."
- Cells from one patient share staining, scanner, tissue — visual quirks unrelated to the
  label. "If the same patient is in train and test, the model can cheat by recognising the
  patient, not the biology."
- So I split **by patient**: 60/20/20, no patient in more than one set, verified disjoint.
- "This gives lower but *honest* scores. It's exactly what makes the Task B finding later
  trustworthy."

## 2:40 — 3:40 · Models (Section D)

**Show:** `model_comparison.png` chart, then `loss_A_cnn.png`.

- Built models of rising capacity: decision tree → logistic regression → MLP → CNN.
- Point at the chart: "Performance climbs exactly as the EDA predicted — linear models
  weak, the CNN best, because convolution looks at local pixel neighbourhoods, which is how
  images actually work." (One sentence on **inductive bias**.)
- Tuned the CNN's learning rate; 0.001 was best.
- Switch to loss curve: "But the CNN **overfit** — training loss keeps dropping while
  validation loss turns back up around epoch five. That's high variance, memorising instead
  of learning. That's the problem Section E fixes."

## 3:40 — 5:00 · Advanced techniques (Section E)

**Show:** `loss_A_cnn_aug.png` for augmentation; then mention ensemble numbers.

**Augmentation (the win):**
- "First technique: data augmentation — random flips, rotations, colour jitter on every
  training image."
- Valid here because "a cell has no fixed orientation — a rotated nucleus is still the same
  cell type."
- Point at the fixed loss curve: "Validation loss now tracks training loss — overfitting
  gone. It gave the best model on **both** tasks: 0.878 on A, 0.646 on B."
- "Note the gain is small — augmentation doesn't add information, it reduces variance."

**Ensemble (the honest failure):**
- "Second technique: a soft-voting ensemble of the MLP, the CNN and the augmented CNN."
- "It didn't help — it actually *lost* to its best member, dropping to 0.604 on Task B."
- Why: "Averaging only works when the models are diverse **and** comparably good. The MLP
  is much weaker, so its equal vote dragged the average down. The bias it adds outweighs
  the variance averaging removes."
- "I'm reporting that honestly — a technique has preconditions, and this one wasn't met."

## 5:00 — 6:30 · Final results (Section G) — the heart of it

**Show:** `cm_A_final_test.png`, then `cm_B_final_test.png`.

- "Final model is the augmented CNN. I evaluated it **once** on the held-out test patients."
- **Task A:** "Validation 0.878, test 0.876 — a gap of 0.002. It generalises almost
  perfectly, catches 89% of cancerous cells. Just under the 0.90 target, which is a real
  resolution ceiling, not a tuning failure."
- **Task B:** "This is the important one. Validation looked fine at 0.646, but on new
  patients it **collapses to 0.475** — well under target."
- Why (keep it tight — name the forces): "Cell typing depends on subtle features that vary
  per patient, the rare classes live in only a few patients, and macro-F1 punishes their
  collapse. On test the model just floods the easy inflammatory class."
- Punchline: "And this gap is the *point*. A careless image-level split would have hidden it
  and reported a fake 0.65. The patient split is what exposed the truth."

## 6:30 — 7:00 · Ethics and conclusion

**Show:** back to camera / closing slide.

- Ethics: "Neither model is a medical device — Task A still misses 1 cancer in 9, Task B
  doesn't transfer. At most it assists a pathologist, never replaces one. And 98 patients
  from one source is a narrow, possibly biased sample."
- Conclusion in one breath: "The binary cancer task is nearly solved at this resolution;
  reliable cell typing needs class-balanced training and, above all, more patients. The
  biggest lesson: validation can lie, and honest patient-level testing is what tells you
  the truth."
- "Thanks for watching."

---

## Delivery tips
- ~7 minutes leaves buffer under the 8-minute cap; if running long, trim the model-ladder
  detail (3:40) first.
- Have the notebook open and scroll to each figure as you mention it — examiners want to
  see the actual plots, not just slides.
- Say the numbers out loud (0.876, 0.475) — the val→test gap is the thing they're listening
  for.
- Face the camera for the intro, the Task B result, and the conclusion; screen-share the
  rest.
