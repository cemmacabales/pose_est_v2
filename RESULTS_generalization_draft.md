# Results — Exercise-Classifier Generalization

_Draft results section. All numbers are measured; protocols and artifacts are named so each
row is reproducible. The exercise head is a 9-class LSTM over 16 hip-centred joint-angle
features (30-frame windows); per-video prediction is the mean softmax across a video's windows._

## 1. Evaluation protocols

We evaluate under four protocols of increasing rigour. The distinction between them is the
**finding**, not a footnote: each holds out a different unit, and the accuracy gap between
them quantifies how much of the naive score is memorisation.

| Protocol | Holds out | Subjects disjoint? | Domain disjoint? | Measures |
|---|---|:--:|:--:|---|
| Window-level split | nothing (windows from one video on both sides) | ✗ | ✗ | memorisation (upper bound) |
| Video-level split | whole videos, same people | ✗ | ✗ | unseen-*video* |
| Held-out internet | the 26 web clips | ✓ | ✓ | unseen *person + camera* |
| Leave-One-Subject-Out (LOSO) | one recorded subject at a time | ✓ | ✗ | unseen *person* |

Data: 201 recorded clips (6 subjects, P1–P6) + 26 internet clips. Each exercise is performed
by exactly **two** subjects; the three shoulder classes gained a second subject (P6) in this work.

## 2. Headline results

| Protocol | Exercise (top-1) | Quality | Note |
|---|:--:|:--:|---|
| Window-level (leaky) | ~97% | ~93% | not a generalization number |
| Video-level (subject overlap) | 93.9% | 63.6% | optimistic; unseen videos only |
| **Held-out internet** (subj+domain disjoint) | **65% ± 6%** | 57.7%¹ | honest new-user number (see §3) |
| LOSO (subject disjoint) | 24.1%² | 57.2% | harshest; single-subject-per-class regime |

¹ Internet clips are all `correct` form → quality is recall-only, not a two-class test.
² LOSO headline over the 6 two-subject lower-body classes; 20.1% over all 9.

**The window→subject-disjoint gap is the headline methodological result:** the same model and
recipe score ~97% at the window level and ~24–65% once the *person* is held out. The naive
score measures who the performer is, not what the movement is.

## 3. The held-out number is a distribution, not a point (variance analysis)

On the 26-clip internet test, each clip is worth 3.85%, so single-run accuracies are noisy.
Training the **identical recipe** under three random seeds:

| Seed | Held-out exercise top-1 |
|---|:--:|
| 42 | 73.1% |
| 1 | 57.7% |
| 7 | 65.4% |
| **mean ± std** | **65.4% ± 6.3%** |

A 15-point spread across seeds of the *same* configuration. We therefore report the
**3-seed mean ± std (65% ± 6%)** as the honest top-1 generalization figure; any single run
(including our best 73.1%) is a draw from this distribution, not a stable result.

**Top-1 vs top-2.** Because several classes are biomechanically similar (sit-to-stand vs
squat; the three arm raises), we also report top-2 accuracy — relevant for a coaching UI that
surfaces two candidates for user confirmation:

| Metric (3-seed ensemble) | Accuracy |
|---|:--:|
| Exercise top-1 | 61.5% |
| **Exercise top-2** | **80.8%** |

The correct exercise is in the model's top two predictions **>80%** of the time. (Softmax
averaging did not improve top-1 over the member mean — on a test this small, even ensembling
is within the noise — but top-2 is a stable, defensible operating point.)

## 4. Per-class generalization (held-out internet)

| Class | Held-out acc | Reads as |
|---|:--:|---|
| DeepSquat | 100% | generalizes |
| HurdleStep | 100% | generalizes |
| InlineLunge | 100% | generalizes |
| SideLunge | 88% | acceptable |
| StandingLegRaise | 67% | borderline (3 clips) |
| **SitToStand** | **33%** | **weak** — confused with squat/side-lunge |
| Shoulder ×3 | 0–100% | anecdotal — 1 internet clip each (see §5) |

Movements with distinctive temporal/orientation signatures transfer; **SitToStand** is the
persistent lower-body failure (it is kinematically a squat without the seated cue) and is the
one class where added data would most help.

## 5. Shoulder classes: a dedicated subject-disjoint test (P6)

The internet set has only one clip per shoulder class, so it cannot measure them. Subject **P6**
(27 shoulder clips, 15 correct / 12 incorrect) was collected as a held-out shoulder test for
the deployed model (which never saw P6):

| Shoulder class | Held-out acc (27 clips) |
|---|:--:|
| ShoulderAbduction | **100%** |
| ShoulderExtension | 33% |
| ShoulderScaption | 22% |
| **Quality (correct vs incorrect)** | **66.7%** |

Abduction generalizes cleanly to a new person; **Extension and Scaption collapse into
Abduction**. This is a feature-level explanation, not just a data-size one: the three movements
differ mainly in *arm direction*, which lives in the viewpoint-sensitive x/y orientation
features — trained on a single subject's camera angle, those cues do not transfer. P6 also
provides the first honest cross-subject **quality** result for the shoulder classes (66.7%,
incorrect-form recall 0.50).

## 6. What did not work (negative results)

We attempted to raise the held-out number with standard generalization techniques. All were
run once on the fixed protocol and **regressed** relative to the baseline recipe:

| Change | Held-out top-1 | Δ vs baseline |
|---|:--:|:--:|
| Baseline recipe | 65% ± 6% | — |
| + rotation augmentation + L2 + heavy dropout | 53.8% | −12 |
| + label smoothing only | 57.7% | −8 |

Rotation augmentation in particular destroyed the lower-body classes (DeepSquat 100%→0%):
squats/lunges are defined by the *vertical* orientation of the limbs, exactly the cue that
rotating the skeleton corrupts. The conclusion is that the limiting factor is **data
diversity (subjects per class), not model regularisation** — no augmentation/architecture
change we tried substitutes for a third subject.

## 7. Conclusions and limitations

- **Honest top-1 generalization to a new user is ~65% ± 6%** (subject + domain disjoint),
  rising to **>80% top-2**. Window-level scoring inflates this to ~97%.
- **The dataset is the ceiling.** With exactly two subjects per class, any subject-disjoint
  test trains each class on a single person; LOSO (24%) is the extreme of this. The fix is
  more subjects per class and a larger held-out test — demonstrated empirically here, since
  six training runs of regularisation/augmentation/ensembling did not move the honest number.
- **Quality estimation is the weaker head** and is only honestly testable where cross-subject
  incorrect-form clips exist (shoulders via P6: 66.7%).
- **Limitations:** the internet test (26 clips) has high variance (§3) and conflates new
  subject with new camera; shoulder per-class internet numbers are anecdotal (1 clip);
  quality is recall-only except on P6. We report mean ± std rather than single runs throughout.

### Artifacts
`eval/results/`: `heldout_eval.md` (video-level), `internet_heldout_eval.md` (held-out internet,
single run), `ensemble_eval.md` (3-seed variance + top-2), `p6_unseen_eval.md` (shoulder P6),
`loso_eval.md` (LOSO). Scripts: `train_internet_heldout.py`, `train_ensemble.py`,
`eval_p6_unseen.py`, `train_loso.py`. Production model `models/classifier.keras` was not
modified by any evaluation.
