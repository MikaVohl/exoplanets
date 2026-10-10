# TESS — The moving sky

Open `index.html` directly in a browser. It is a single, self-contained file:
JavaScript, catalog stars, land boundaries, observation times, pointing frames,
and ephemerides are embedded. There are no runtime downloads or API requests.
A browser with WebGL is required.

Drag to rotate, Shift/right-drag to pan, and scroll/pinch to zoom. The canvas
also accepts arrow keys and +/−. Use **Earth**, **TESS**, or **Sun** to inspect a
body, **Face sector** to center the selected field, and **Reset** to restore the
view. Close-up views follow the selected body during playback. All time inputs
and displayed observation times are UTC, regardless of the browser's timezone.

The three views show an observer-centered celestial globe, the Earth-centered
TESS trajectory, and the Sun–Earth geometry. Choose a survey year or sector,
scrub time, or play the mission. Colored fields are the four camera envelopes
and their calibrated CCD footprints. Other sectors and accumulated coverage
can be toggled. The star picker and clicking a star report nominal camera/CCD
membership. HYG visual magnitudes are labeled V, not TESS magnitudes.

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
For HTML-only edits with the existing cache:

```sh
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
standalone file. Scientific source links are also included in the page.
