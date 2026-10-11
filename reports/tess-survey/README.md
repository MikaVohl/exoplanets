# TESS — The moving sky

Open `index.html` directly in a browser. It is a single, self-contained file:
JavaScript, catalog stars, land boundaries, observation times, pointing frames,
and ephemerides are embedded. There are no runtime downloads or API requests.
A browser with WebGL is required.

Drag to rotate, Shift/right-drag to pan, and scroll/pinch to zoom. The canvas
also accepts arrow keys and +/−. Elsewhere on the page, Space plays/pauses,
`[` / `]` step to the previous/next sector, and `1`–`3` switch views. Use
**Earth**, **TESS**, or **Sun** to inspect a body, **Face sector** to center the
selected field, and **Reset** to restore the view; camera moves are short eased
flights that any pointer, wheel or key input cancels (instant with reduced
motion). Close-up views follow the selected body during playback. All time
inputs and displayed observation times are UTC, regardless of the browser's
timezone.

The three views show an observer-centered celestial globe, the Earth-centered
TESS trajectory, and the Sun–Earth geometry. Choose a survey year or sector,
scrub time, or play the mission. The timeline draws each sector to its
duration, with published orbit windows inside it set against downlink gaps.
Planned sectors (after the October 10, 2026 snapshot) are drawn at reduced
opacity, and the comet window is marked over Sector 99. Colored fields are the
four camera envelopes (dashed) and their calibrated CCD footprints. During a gap
the retained reference field is dimmed. In the orbit and Sun–Earth views the
fields are faint and the camera sight lines carry the pointing. Other sectors
and accumulated coverage can be toggled. The sky globe draws the ecliptic, a 30°
ecliptic graticule and the celestial equator, which is computed from the
obliquity. Scale notes, accuracy caveats, sources and keyboard shortcuts are in
the **About this data** sheet. The star picker and clicking a star report
nominal camera/CCD membership. Star point size follows HYG V magnitude and color
follows B−V through a temperature/blackbody approximation. HYG visual magnitudes
are labeled V, not TESS magnitudes.

## Mission Control

Mission Control shows the TLS masked-search benchmark
(`tls_masked_search/04_benchmark_100_stars.py`) working star by star. Click
**Run pipeline** (top of the page) to start a run, or open the panel with the
pulse button in the toolbar or `M`. Every target is a ring on the sky
globe, colored by outcome once known: green recovered, cyan harmonic, red
missed, amber false alarm, grey quiet, purple error, dark grey pending; colours
fade in as verdicts arrive. Sector numbers are drawn on the globe: the active
sector large, every run target's sector dimmer.

Each star is told as a short story. The camera flies to the star on the globe
(the timeline jumps to its sector), then into TESS's own view along the
boresight of the camera that holds it, while that CCD breathes and a soft scan
band sweeps across it. The camera keeps diving toward the star and a
**star-system stage** fades in over the 3D view: the star as a limb-darkened
disk coloured by its temperature (`teff`; sun-like if unknown), and a
brightness trace scrolling underneath. During the wide search, faint ghost
orbits at trial periods spin and then converge on the best period, or fade if
no SDE reaches the threshold. A locked-in planet candidate then orbits nearly
edge-on and the trace dips exactly while it crosses the disk. The dip's shape is
the TLS folded model once the fold is known, and a box before that. Refinement
eases the orbit and size to the refined values (before → after numbers). A
second signal above the threshold gets its own orbit. Until the verdict, every
signal is drawn as a candidate (dashed outline). At the verdict, only signals
that match a catalog planet turn solid. A detected signal that matches no known
planet, and any signal on a comparison star, stays hollow with a dashed orbit.
On a known host the card adds a line saying it matches no known planet and is
likely leftover noise or systematics, or an unknown planet. The decision card is
worded honestly: the pipeline finds planet candidates, and the benchmark checks
them against the catalog. A missed or harmonic catalog planet is drawn as a
dashed ghost orbit. Everything on the stage is schematic: orbit radius
follows a ∝ P^(2/3), the planet radius is √depth of the star's (small planets
are enlarged and labelled), and every orbit lasts a few seconds on screen.

Under the stage, a filmstrip collects each step's graph in order: cleaned light
curve, period search, masked light curve, refined period, folded transit, then
the same for signal 2. The latest graph is drawn large and animated; click a
thumbnail to enlarge it. While the stage is up, the panel keeps only the
controls, the step rail with its plain-English caption, the scoreboard and the
ledger. Click a finished star in the ledger to review its stage and graphs, and
close the stage with ×. Reduced-motion settings skip the animations.

Events are shown from a queue at a readable pace, about 60–70 s per star at 1×
(signal-2 steps are shorter). Use the speed menu (0.5–4×), pause, step and the
position slider. Seeking jumps straight to the right state without replaying
animations.

- **Live:** start `cd tls_masked_search && python live.py`; it serves this page
  at http://127.0.0.1:8765/ and opens it. Click **Run pipeline** and choose the
  stars: the 7-star tour (the sample's stars), the first N stars, or TIC IDs
  (suggestions list each target's TIC and host name). **Start** runs the real
  benchmark; a new run replaces the one in progress. Over HTTP the page follows
  `/events` (Server-Sent Events) and shows whether the pipeline is running,
  finished, failed (with its exit code) or stopped (**Stop**). The pipeline is
  usually ahead of the presentation; "N events queued" then appears, with
  **Skip to latest** to catch up.
- **Replay:** open `index.html` directly and click **Run pipeline** (or
  **Watch sample run**) to play the embedded sample run; a real run needs
  `live.py`. You can also open/drop a recorded JSONL event log (for example
  `tls_masked_search/output/live/events.jsonl`). The recorded compute times are
  not replayed.

The **Milky Way background** (toggle under Display) is the NASA SVS Deep Star
Maps 2020 Milky Way layer, drawn at infinity in J2000 RA/Dec behind the globe.
The build decodes the 4k EXR with zlib and numpy, checks its orientation (the
Large Magellanic Cloud and Galactic Centre bright, the galactic poles dark), and
embeds a 4096 × 2048 JPEG cached as `assets/milkyway.jpg`.

The sample is `runs/sample.jsonl`, embedded at build time. Each line is one
JSON event: `run_start` (config, targets), `star_start` (position, sector and
stellar `teff`, `radius`, `mass`; also on each target), `loaded`, `cleaned`
(wide or masked pass), `search` (wide or fine pass, SDE spectrum),
`signal_done` (folded data), `hide`, `verdict` (outcome) and `run_end`;
signal-scoped events carry `signal` 1 or 2.

## Scientific scope

The file includes sectors 1–121 and special sector 1751, covering July 2018 to
September 2027. Future sectors are planned pointings. Sector boundaries and
science windows come from [MIT TESS](https://tess.mit.edu/observations/), sampled
on October 10, 2026. The date-level January 15–22, 2026 comet override comes from
[MIT's special pointing page](https://tess.mit.edu/observations/3i_atlas/).

[NASA/JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) provides geometric
Earth-centered ICRF position/velocity vectors for TESS (−95) every three hours
and the Sun (10) daily. Cubic Hermite interpolation uses the velocities.
Horizons' TESS trajectory was revised September 30, 2026, with last tracking
data September 27, 2026; trajectories beyond that date are predictions. The
timeline uses UTC while source epochs use TDB. Their conversion uses the first
sample's offset, ignoring subsequent millisecond periodic variations.

The published spacecraft RA/Dec/roll is fixed within each nominal sector. The
[TESS-Point](https://github.com/tessgi/tess-point) focal-plane model generates
four camera frames and 16 CCD boundaries, including CCD gaps and distortion.
The source tables round spacecraft angles to 0.01°. The calibrated camera
centers differ from the published rounded camera center table by at most
0.075°. The visualization omits aberration and pointing jitter. It retains the
previous field as a reference during downlink or inter-sector gaps; it does not
reconstruct actual downlink attitudes, slews, momentum dumps or safe modes.
Published orbit windows are not complete telemetry of every science exposure.

Earth uses ERFA/SOFA IAU 2006 precession/nutation and sidereal rotation, with
UT1 approximated by UTC and polar motion omitted. Its painted prime meridian
and land rotate in the correct direction. The Sun uses the IAU reference pole
RA 286.13°, Dec 63.87°, and W = 84.176° + 14.1844° × days since J2000. This is a
rigid reference rotation, not measured differential rotation or sunspot data.
Stars have fixed catalog J2000 directions, without proper motion or parallax;
unresolved stellar spin is not shown.

Object sizes are deliberately enlarged. Sky and orbit views compress the Sun's
distance while preserving its observer-relative direction. The Earth orbit
view uses Earth-radius units for spacecraft distances. The Sun–Earth view uses
true heliocentric Earth positions, with the spacecraft displacement enlarged
160×. The sky-globe radius represents directions, not physical star distances.
The spacecraft body is schematic. The camera axes are calibrated; the solar
panels illustrate ideal single-axis Sun tracking, not panel-angle telemetry.

Accumulated coverage uses nominal 24° camera envelopes on coarse 5° cells and
published science-window durations where available. It is not a detector-level
exposure map, ignores small intra-window interruptions, and updates in daily
buckets. The coverage color scale uses absolute days, not the maximum visible
in the selected year.

## Validation

Chrome/Playwright checks pass for all three views, offline operation, sector
selection, the comet override, downlink gaps, years, playback, camera toggles,
coverage, mouse pan/zoom, keyboard rotation, and a 390 px mobile layout. No
JavaScript or WebGL shader errors were reported, and no HTTP requests occurred.

Independent checks:

- The spacecraft +Z axis matches the selected published boresight.
- Attitude stays fixed during a six-hour interval within a sector.
- Earth rotates 90.2464° in six hours, consistent with sidereal rotation.
- Pi Mensae is on Camera 4, CCD 2 in Sector 1, matching the separate TESS-Point
  forward pixel prediction.
- Interpolated TESS positions were compared with 1,107 additional Horizons
  samples at 90-minute cadence in July–August 2018, August 2023, and October
  2026. Maximum position differences were 6.617, 1.855, and 0.682 km respectively.
  This is a check across those windows, not a global accuracy guarantee.

## Editing and rebuilding

Edit `template.html`; `index.html` is generated. Rebuild using the project's
existing Python environment:

```sh
.venv/bin/python scripts/build_tess_visualization.py
```

Raw source downloads are cached in the Git-ignored `assets/` directory. A fresh
build needs network access; its generated HTML does not. The current sources
are upstream snapshots, so rebuilding with an empty cache may use updated data.
For HTML-only edits, `--render-only` rebuilds offline from the cache. On a
fresh clone (empty cache), first recover the cache from the committed
`index.html`; rendering then reproduces it byte for byte:

```sh
.venv/bin/python scripts/build_tess_visualization.py --extract-cache
.venv/bin/python scripts/build_tess_visualization.py --render-only
```

`verify.cjs` runs the browser checks with Playwright in `NODE_PATH` and installed
Chrome. Set `TESS_CHROME` to override the Chrome executable. Diagnostic JSON and
screenshots are saved in the ignored `assets/` directory.

## Credits

HYG v4.1: David Nash / Astronexus, derived subset of 5,070 stars with V ≤ 6,
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); that embedded
derived catalog remains under this license. Natural Earth land boundaries are
public domain. TESS-Point by Burke, Levine and collaborators, and Three.js r128
with OrbitControls, are MIT licensed. Their license notices are embedded in the
standalone file. The Milky Way background is from
[Deep Star Maps 2020](https://svs.gsfc.nasa.gov/4851/): NASA/Goddard Space
Flight Center Scientific Visualization Studio; Gaia DR2: ESA/Gaia/DPAC. NASA SVS
material is public domain unless marked otherwise, with this credit requested.
Scientific source links are also included in the page.
