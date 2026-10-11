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
- **Two passes:** the smoothing curve tends to dip along with a real dip and eat part of it. So after
  the first search, we redo the smoothing while **ignoring the dips we found**, then search again
  near that period. This gives a truer depth.
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
| **Faint second dip** | A small dip exactly halfway between the main ones: the brighter star of a pair passing in front of the dimmer one. Planets give off almost no light, so they don't cause it | none (~0.4) | often present (~1.2) |
| **Position shift** | Where the light lands on the camera during dips vs between dips. A big shift means the dip is really on a neighbouring star blended into our pixels | small (~2.7) | larger (~5.4) |
| **Depth, length, period** | How deep and long the dip is and how often it repeats | | very deep or very long dips are suspicious |
| **Dip counts** | How many individual dips we actually saw and how many were clear on their own | | |

(Typical values are medians from our dataset.)

## 6. Labels: where the right answers come from

- NASA's TESS team flags promising dips ("TOIs"), and astronomers follow them up with other telescopes.
  Each gets an expert verdict: **confirmed planet** or **fake** (usually two stars eclipsing each other).
- We **don't label anything ourselves**. If a dip we found repeats at the same rate as a NASA candidate on
  that star (within 1%), it gets NASA's verdict.
- Extra fakes: other dips we find on stars NASA marked as fake. Unmatched dips on planet stars are left
  out, since they could be real planets NASA hasn't listed.
- Our dataset: about 2,450 stars, 3,198 labeled dips (1,112 planets, 2,086 fakes).

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

## 9. Results so far

On NASA-confirmed planets vs NASA-known fakes:

| | Planets called planet | Fakes called planet | Overall correct |
|---|---|---|---|
| Our model | 79% | 28% | 77% |
| Old fixed rules | 60% | 30% | 63% |

- At the same false-alarm level as the rules, the model catches about **4 in 5** planets vs 3 in 5.
- It wins in every same-amount-of-data group, so it's learning from the dips, not the data amount.
- It's a **ranking tool**, not a final judge: about 1 in 4 fakes still gets through. Best use: put the
  most promising candidates at the top for a person to check.

## 10. Known limits

- Some fakes look exactly like planets in brightness data alone; NASA often needed other telescopes.
- Very faint planets, or ones with only one dip in the data, can't be found.
- The model only sees a summary of each dip, not its shape. A model that looks at the dip shape
  directly (using the GPU) is a possible next step.
