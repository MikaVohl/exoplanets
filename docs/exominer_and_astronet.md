# ExoMiner and AstroNet, in plain English

Both are published systems that do what our model does: look at a repeating dip in a star's
brightness and score how likely it is to be a real planet. Neither one searches for dips itself;
each is handed a list of dips found by someone else's search pipeline.

## The shared idea

1. **Fold:** cut the brightness data at the planet's period and stack every dip on top of each other,
   so random noise averages out and the true dip shape shows.
2. **Make snapshots:** average the stack into a fixed number of points (a "view"): one of the whole
   orbit, one zoomed on the dip. Rescale so normal brightness = 0 and the dip bottom = -1.
3. **Neural network:** small pattern detectors slide along each snapshot (U-shaped dip? bump halfway?),
   their findings are combined, and the network outputs a score from 0 to 1.
4. **Learn from experts:** train on dips that humans already classified as planet or fake.
5. **Average several copies:** train ~10 networks from different random starts and average their scores.

Our model follows the same recipe (section "How we compare" below).

---

## AstroNet (Google + Harvard, 2018)

**Goal:** prove a neural network can vet Kepler planet candidates as well as human experts, and use
it to find planets humans missed.

**Track record:**
- Kepler data. Ranked a real planet above a fake **98.8% of the time** (AUC 0.988).
- Found **Kepler-90i** (making Kepler-90 the first known 8-planet system besides ours) and
  **Kepler-80g**, by scoring weak dips in known multi-planet systems.
- Paper: [Shallue & Vanderburg 2018](https://arxiv.org/abs/1712.05044).
  Code: [google-research/exoplanet-ml](https://github.com/google-research/exoplanet-ml) (frozen 2018-19
  research code, old TensorFlow 1).

**What it looks at:** only **two snapshots**, no numbers:
- whole orbit (2,001 points),
- close-up of the dip (201 points).

**The network:** two separate stacks of pattern detectors, one per snapshot (14 detector layers in
total), joined and passed through 4 layers that weigh everything into one score. 10 copies averaged.

**Labels:** Kepler's "Autovetter" catalog: planet candidate vs false positive vs "not a transit".

**On TESS:** Google never ported it. MIT did ([Yu et al. 2019](https://arxiv.org/abs/1904.02726)):
- **Astronet-Triage:** a first-pass filter for MIT's TESS pipeline (QLP). Keeps anything planet-like,
  throws out ~93% of junk before humans look. **Used in production at MIT.** A v2
  ([Tey et al. 2023](https://arxiv.org/abs/2301.01371)) is still in use.
- **Astronet-Vetting:** planet vs everything else. Adds a halfway-dip snapshot and one number (does
  the dip depth change with the camera aperture, i.e. is it a neighbour star?). The authors said it
  was **not ready for production**: it confused star pairs with planets because it had no odd/even view.

**Main weaknesses:** sees only the dip shape (no star details, no position shift, no odd/even in the
original); relies on the upstream search getting the period right.

---

## ExoMiner (NASA Ames, 2022 to now)

**Goal:** vet NASA's own Kepler and TESS candidate lists the way a NASA expert reads a "Data Validation
report" (the standard diagnostic sheet for each candidate), and confirm planets statistically without
waiting for other telescopes.

**Track record:**
- **Kepler:** validated **301 new planets** ([2022 paper](https://arxiv.org/abs/2111.10009)), plus **69**
  more in a follow-up that uses "planets come in families" as extra evidence.
- **TESS (ExoMiner++, [2025](https://arxiv.org/abs/2502.09790)):** scored **147,568** unlabeled TESS
  signals and kept **7,330** as planet candidates, including **50 brand-new** ones.
- Code: [nasa/ExoMiner](https://github.com/nasa/ExoMiner), with a ready-to-run container: give it TESS
  star IDs, get back scores and PDF reports.

**What it looks at:** each test from the NASA diagnostic sheet becomes one input "branch":

| Branch | What it shows | Catches |
|---|---|---|
| Whole orbit, close-up | Dip shape | U vs V shape |
| Odd vs even | Alternating dips | Star pairs |
| Halfway point | Faint second dip | Star pairs |
| Star position | Light centre moving during dips | Neighbour star |
| **Pixel images** | Camera pixels during vs outside the dip | Exactly which star dimmed |
| **Each dip separately** | 20 dips, not stacked | One-off glitches |
| Detrending curve | What the smoothing removed | Star wobbles mistaken for dips |
| Thruster firings | When the spacecraft adjusted itself | Fake dips from the spacecraft |
| Periodogram | Which repeat times stand out | Stars that pulse |
| ~35 numbers | Star size, temperature, density, fit quality, dip count... | Physically impossible "planets" |

**The network:** one small stack of pattern detectors per branch (about 10), each joined with its own
related numbers; everything is combined and weighed by a 4-layer head into one score.
About **1.4 million** adjustable settings. 10 copies per fold averaged; the spread between copies is
reported as "how unsure the model is".

**Labels and training:**
- TESS answers come from NASA/ExoFOP (confirmed planet vs false positive), a catalog of known star
  pairs, and an automatic "not a transit" list.
- TESS labels are noisy, so it also trains on about **30,000 Kepler examples** (cleaner labels).
- Tested with **10-fold, star-grouped** splits (a star is never in both training and testing).

**Results on TESS:** precision 0.93, recall 0.95, ranking quality (ROC AUC) 0.998. These are on NASA's
full candidate list, which includes many easy "obviously not a transit" cases, so they aren't directly
comparable to our "planets vs NASA-flagged fakes" test.

**Weaknesses (authors' own list):**
- TESS is harder than Kepler (shorter observations, noisier, bigger pixels).
- Fakes caused by a faint star behind the target are caught only ~half the time.
- Labels contain mistakes, which both mislead training and blur the scores.
- Scores are for ranking, not true probabilities.

---

## How we compare

| | AstroNet | ExoMiner++ | Ours |
|---|---|---|---|
| Finds its own dips | No | No (uses NASA's search) | **Yes** (our TLS search) |
| Snapshots | 2 | ~10 kinds + pixel images | 6 |
| Numbers alongside | None | ~35 | 12 |
| Star details | No | Yes | Not yet (being tested) |
| Pixel images | No | Yes | No |
| Training data | Kepler (~15k signals) | TESS + ~30k Kepler | TESS only, ~3,200 signals |
| Averaged copies | 10 | 10 per fold | 5 |
| Test split | Random | 10-fold by star | 5-fold by star |
| Ranking quality (AUC) | 0.988 (Kepler) | 0.998 (TESS, incl. easy cases) | 0.87 (TESS, hard cases only) |

**Takeaways for us:**
- We're on the standard approach; our model is closest to MIT's Astronet-Vetting, plus odd/even,
  position-shift and 12 measurements.
- Biggest missing pieces, in order of likely payoff: **star details** (cheap, in progress),
  **pixel images** (the best way to catch neighbour-star fakes), **more labeled data** (ExoMiner leans
  on Kepler for this; we've chosen TESS-only).
- Ideas worth copying cheaply: report the **spread between averaged copies** as an uncertainty flag,
  and a **"each dip separately"** view to catch one-off glitches.
- ExoMiner++ publishes scores for TESS candidates, so it's a free benchmark for our ranking.
