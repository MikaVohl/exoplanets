# How the pipeline works (plain-English overview)

**Goal:** look at how a star's brightness changes over time, find small dips that repeat on a regular
schedule (a planet passing in front of the star), and judge whether each dip is a real planet or a fake.

```mermaid
flowchart LR
  A[TESS brightness data] --> B[Clean it]
  B --> C[Search for repeating dips]
  C --> D[Measure each dip]
  D --> E[Model: planet or fake?]
```

## 1. The data

- **TESS** is a NASA satellite that measures the brightness of stars every 2 minutes, for about a
  month at a time per patch of sky ("sector"). Some stars get observed for many months over the years.
- We download this data with **lightkurve**, a free Python library.

## 2. Cleaning

- Stars slowly brighten and dim on their own (spots, rotation), and the camera drifts.
- We **flatten** the data: estimate that slow wobble with a smooth curve (33-hour window) and divide it
  out, so only short dips remain.
- We remove upward spikes (glitches), and average the data into 10-minute points to speed things up.

## 3. Searching for dips

- **TLS (Transit Least Squares)** tries thousands of possible repeat times ("periods") and asks, for
  each: if a planet repeated this often, would the dips line up? It returns the best period, when the
  dips happen, how long and how deep they are, and a score for how strongly it stands out (**SDE**).
- **Two passes (100-star benchmark only):** the smoothing curve tends to dip along with a real dip and
  eat part of it. So after the first search, we redo the smoothing while **ignoring the dips we found**,
  then search again near that period. This gives a truer depth. The training dataset skips this step;
  its depths come from the dip measurements in section 5.
- **Second search:** we hide the first signal's dips and search again. This finds a weaker second
  planet that the bigger one was drowning out.

## 4. Stars with many months of data

- Searching years of patchy data in one go gave wrong dip lengths and broke every later measurement.
- So we search in **blocks of up to 4 back-to-back months** (at most 2 blocks per star), then
  **combine**: the same signal found in several blocks is merged, and we use all the months together to
  pin down its exact period.

## 5. Measuring each dip (the clues)

Each signal gets a set of measurements. These are the clues the model learns from.

| Clue | What it checks | Planet | Fake |
|---|---|---|---|
| **Signal strength** | How much the dip stands out from the noise, per dip | strong | often weaker |
| **Odd/even depths** | Do every-other dips have different depths? Two stars of different brightness give deep, shallow, deep, shallow when we catch half their orbit | match (~1.0) | differ more (~1.3) |
| **Faint second dip** | A small dip exactly halfway between the main ones: the brighter star of a pair passing in front of the dimmer one. Planets give off almost no light, so they don't cause it | none (~0.6) | often present (~1.2) |
| **Position shift** | Where the light lands on the camera during dips vs between dips. A big shift means the dip is really on a neighbouring star blended into our pixels | small (~2.8) | larger (~5.3) |
| **Depth, length, period** | How deep and long the dip is and how often it repeats | | very deep or very long dips are suspicious |
| **Dip counts** | How many individual dips we actually saw and how many were clear on their own | | |

(Typical values are medians over NASA-confirmed planets vs NASA-known fakes in our dataset. Signal
strength, depth and length matter most to the model; the other clues help less but catch specific fakes.)

## 6. Labels: where the right answers come from

- NASA's TESS team flags promising dips ("TOIs"), and astronomers follow them up with other telescopes.
  Each gets an expert verdict: **confirmed planet** or **fake** (usually two stars eclipsing each other).
- We **don't label anything ourselves**. If a dip we found repeats at the same rate as a NASA candidate on
  that star (within 1%), it gets NASA's verdict.
- Dips that match no NASA candidate are left out, on every star: they could be real planets NASA
  hasn't listed, or plain noise.
- Our dataset: 2,454 stars sampled, 2,204 with 2-minute data; 1,725 labeled dips (1,111 planets,
  614 fakes).

## 7. Keeping the comparison fair

- **Planet stars get observed more** (NASA keeps watching confirmed planets), so "more data" alone hints
  "planet". To stop the model cheating this way, strength measures are **per dip**, not totals.
- We then check the model separately on stars with the same amount of data (1, 2, 3-4, 5-8 months).

## 8. The model

- **Boosted decision trees** (scikit-learn): hundreds of small yes/no flowcharts built one after
  another, each fixing the previous ones' mistakes. Their votes add up to a score from 0 to 1:
  how likely the dip is a real planet.
- **Testing:** the stars are split into 5 groups; the model learns from 4 and is tested on the 5th,
  rotating so every star is tested once by a model that never saw it.
- **Shape model** (PyTorch, on the GPU): instead of only a summary, it also looks at "snapshots" of
  each dip: the whole orbit, a zoom on the dip, odd vs even dips, the halfway point, and the star's
  position. A small neural network slides pattern detectors along these curves (e.g. U-shaped planet
  dip vs V-shaped star-pair dip) and combines what it sees with the measurements.

## 9. Results so far

On NASA-confirmed planets vs NASA-known fakes:

| | Planets called planet | Fakes called planet | Overall correct |
|---|---|---|---|
| Shape model | 90% | 29% | 83% |
| Decision-tree model | 88% | 37% | 79% |
| Old fixed rules | 60% | 30% | 64% |

- At the same false-alarm level as the rules, the shape model catches about **90%** of planets, the
  tree model 83%, the rules 60%.
- Both win in every same-amount-of-data group, so they learn from the dips, not the data amount.
- They're **ranking tools**, not final judges: about 1 in 3 fakes still gets through. Best use: put the
  most promising candidates at the top for a person to check.

## 10. Known limits

- Some fakes look exactly like planets in brightness data alone; NASA often needed other telescopes.
- Very faint planets, or ones with only one dip in the data, can't be found.
- The snapshots alone (without the measurements) do no better than the tree model: the gain comes
  from combining shape with the measurements.
