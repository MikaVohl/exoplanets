"""Build the offline TESS explorer from embedded, attributed source snapshots.

Run with .venv/bin/python scripts/build_tess_visualization.py. Raw source files
are cached under reports/tess-survey/assets; index.html is the only runtime file.
--render-only rebuilds index.html offline from that cache; --extract-cache
recreates the cache from the committed index.html (for a fresh clone).
"""
from pathlib import Path
import csv
import importlib.util
import json
import math
import struct
import re
import subprocess
import zlib
from base64 import b64decode, b64encode
import sys
from datetime import datetime, timezone

import numpy as np
from bs4 import BeautifulSoup
from astropy.time import Time
import erfa

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/tess-survey'
CACHE = OUT / 'assets'
CACHE.mkdir(parents=True, exist_ok=True)
# Each notice's first line, used to split the concatenated notices back apart.
LICENSES = [('tess-point-LICENSE', 'MIT License\n'), ('three-LICENSE', '\nThe MIT License\n'), ('HYG-LICENSE', '\nShield: ')]
NOTICES = '<!-- Third-party notices\n'


def download(url, name):
    path = CACHE / name
    if not path.exists():
        subprocess.run(['curl', '-fLsS', '--retry', '2', url, '-o', str(path)], check=True)
    return path


def stamp(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()


def horizons(target, name, step):
    path = CACHE / name
    if not path.exists():
        params = dict(format='json', COMMAND=f"'{target}'", MAKE_EPHEM="'YES'",
                      EPHEM_TYPE="'VECTORS'", CENTER="'500@399'",
                      START_TIME="'2018-07-25'", STOP_TIME="'2027-09-20'",
                      STEP_SIZE=f"'{step}'", REF_PLANE="'FRAME'", REF_SYSTEM="'ICRF'",
                      OUT_UNITS="'KM-S'", VEC_TABLE="'2'", CSV_FORMAT="'YES'", VEC_CORR="'NONE'")
        cmd = ['curl', '-fLsS', '--retry', '2', '--get', 'https://ssd.jpl.nasa.gov/api/horizons.api']
        for key, val in params.items():
            cmd += ['--data-urlencode', f'{key}={val}']
        result = subprocess.check_output(cmd, timeout=240)
        parsed = json.loads(result)
        if '$$SOE' not in parsed.get('result', ''):
            raise RuntimeError(str(parsed)[:1500])
        path.write_bytes(result)
    raw = json.loads(path.read_text())['result']
    rows = []
    for line in raw.split('$$SOE')[1].split('$$EOE')[0].strip().splitlines():
        row = next(csv.reader([line]))
        rows.append([float(row[0]), *[float(x) for x in row[2:8]]])
    array = np.array(rows)
    print(name, array.shape, flush=True)
    return {'t0': round(float(Time(array[0, 0], format='jd', scale='tdb').utc.unix), 3),
            'step': round((array[1, 0] - array[0, 0])*86400),
            'states': [[*[round(v, 3) for v in row[1:4]], *[round(v, 7) for v in row[4:7]]] for row in array]}, array


def unit(ra, dec):
    a, d = np.deg2rad([ra, dec])
    return np.array([math.cos(d)*math.cos(a), math.cos(d)*math.sin(a), math.sin(d)])


def milky_way():
    """Return the cached Milky Way background JPEG, converting it on first use.

    Source: NASA SVS Deep Star Maps 2020, Milky Way layer (bright Hipparcos/Tycho
    stars omitted, so the page's own HYG stars sit on top), plate carrée in ICRF
    RA/Dec: centered on RA 0h, RA increasing to the left, north up. The EXR is
    linear half-float RGB with ZIP compression, decoded here with zlib + numpy.
    """
    jpg = CACHE / 'milkyway.jpg'
    if jpg.exists():
        return jpg
    from PIL import Image
    data = download('https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/milkyway_2020_4k.exr', 'milkyway_2020_4k.exr').read_bytes()
    header, p = {}, 8
    while data[p]:
        name, kind, rest = data[p:].split(b'\0', 2)
        p += len(name) + len(kind) + 2
        size = struct.unpack_from('<i', data, p)[0]
        header[name.decode()] = data[p+4:p+4+size]
        p += 4 + size
    assert header['compression'] == b'\x03', 'expected ZIP-compressed EXR'
    assert header['channels'].count(b'\x01\x00\x00\x00\x00\x00\x00\x00\x01') == 3, 'expected three half-float channels (B, G, R)'
    x0, y0, x1, y1 = struct.unpack('<4i', header['dataWindow'])
    w, h = x1-x0+1, y1-y0+1
    image = np.empty((h, 3, w), '<f2')
    for offset in np.frombuffer(data, '<u8', (h+15)//16, p+1):
        y, size = struct.unpack_from('<ii', data, int(offset))
        delta = np.frombuffer(zlib.decompress(data[int(offset)+8:int(offset)+8+size]), np.uint8)
        raw = (np.cumsum(delta.astype(np.int64)-128)+128).astype(np.uint8)  # undo the byte predictor
        half = (len(raw)+1)//2
        bytes_ = np.empty_like(raw)
        bytes_[0::2], bytes_[1::2] = raw[:half], raw[half:]  # re-interleave the split byte halves
        rows = min(16, h-y+y0)
        image[y-y0:y-y0+rows] = bytes_.view('<f2').reshape(rows, 3, w)
    rgb = image[:, ::-1].transpose(0, 2, 1).astype(np.float32)  # B,G,R planes -> RGB pixels
    lum = rgb.mean(2)

    def sky(ra, dec):
        x, y = int((180-ra) % 360/360*w), int((90-dec)/180*h)
        return lum[y-20:y+20, x-20:x+20].mean()
    # Orientation check: Large Magellanic Cloud and Galactic Centre bright, galactic poles dark.
    assert min(sky(80.9, -69.8), sky(266.4, -28.9)) > 10*max(sky(192.9, 27.1), sky(12.9, -27.1)), 'Milky Way image orientation'
    # Black point removes most of the unresolved-star haze; gamma lifts the faint Milky Way.
    out = np.clip((rgb-.006)/.6, 0, 1)**(1/1.8)
    Image.fromarray((out*255+.5).astype(np.uint8)).save(jpg, 'JPEG', quality=75, optimize=True, progressive=True)
    return jpg


def main():
    if '--extract-cache' in sys.argv:
        extract_cache()
        return
    if '--render-only' in sys.argv:
        render()
        return
    for url, name in [
        ('https://tess.mit.edu/public/files/TESS_orbit_times.csv', 'orbit-times.csv'),
        ('https://raw.githubusercontent.com/tessgi/tess-point/master/tess_stars2px.py', 'tess_stars2px.py'),
        ('https://raw.githubusercontent.com/tessgi/tess-point/master/LICENSE', 'tess-point-LICENSE'),
        ('https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv', 'stars.csv'),
        ('https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/LICENSE', 'HYG-LICENSE'),
        ('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson', 'land.geojson'),
        ('https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js', 'three.min.js'),
        ('https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js', 'OrbitControls.js'),
        ('https://cdn.jsdelivr.net/npm/three@0.128.0/LICENSE', 'three-LICENSE'),
    ]:
        download(url, name)
    milky_way()
    sectors = []
    for year in range(1, 10):
        page = download(f'https://tess.mit.edu/tess-year-{year}-observations/', f'year-{year}.html')
        soup = BeautifulSoup(page.read_text(), 'html.parser')
        for tr in soup.select('table tr'):
            cells = [x.get_text(' ', strip=True).replace('−', '-') for x in tr.find_all('td')]
            if len(cells) < 7 or not cells[0].isdigit():
                continue
            start, end = cells[1].replace('–', '-').split('-')
            start = datetime.strptime(start.strip(), '%m/%d/%y').replace(tzinfo=timezone.utc).timestamp()
            end = datetime.strptime(end.strip(), '%m/%d/%y').replace(tzinfo=timezone.utc).timestamp()
            points = [[float(v.strip()) for v in cell.split(',')] for cell in cells[2:7]]
            sectors.append(dict(id=int(cells[0]), year=year, start=start, end=end, pointing=points[0], cameras=points[1:]))
    sectors.sort(key=lambda x: x['id'])
    assert [s['id'] for s in sectors] == list(range(1, 122)), 'Missing sector table rows'
    orbits = []
    for row in csv.DictReader((CACHE / 'orbit-times.csv').read_text().splitlines()):
        if not row['Sector'].isdigit() or not row.get('End of Orbit'):
            continue
        orbits.append(dict(sector=int(row['Sector']), orbit=int(row['Orbit']),
                           start=stamp(row['Start of Orbit']), end=stamp(row['End of Orbit'])))
    for sector in sectors:
        intervals = [o for o in orbits if o['sector'] == sector['id']]
        sector['intervals'] = intervals
        if intervals:
            sector['start'] = min(x['start'] for x in intervals)
            # An incomplete future sector must keep its published full end date.
            if sector['id'] <= 110:
                sector['end'] = max(x['end'] for x in intervals)
    special = dict(id=1751, year=8, start=stamp('2026-01-15T00:00:00'), end=stamp('2026-01-22T00:00:00'),
                   pointing=[109.492,21.842,84.336], cameras=[], intervals=[], special=True)
    sectors.append(special)
    spec = importlib.util.spec_from_file_location('tesspoint', CACHE / 'tess_stars2px.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []
    for sector in sectors:
        fpg = module.Levine_FPG(sc_ra_dec_roll=np.array(sector['pointing']))
        sector['basis'] = fpg.rmat1.T.round(9).tolist()
        sector['cameraBasis'] = [m.T.round(9).tolist() for m in fpg.rmat4]
        sector['ccd'] = []
        for cam in range(4):
            cam_ccds = []
            if sector['cameras']:
                center = fpg.rmat4[cam].T[:,2]
                checks.append(math.degrees(math.acos(np.clip(np.dot(center, unit(*sector['cameras'][cam][:2])), -1, 1))))
            for ccd in range(4):
                corners = [(44.5, .5), (2092.5, .5), (2092.5, 2048.5), (44.5, 2048.5)]
                perimeter = []
                for p, q in zip(corners, corners[1:]+corners[:1]):
                    for f in np.linspace(0, 1, 6, endpoint=False):
                        xy = np.array(p)*(1-f)+np.array(q)*f
                        ra, dec = fpg.pix2radec_nocheck_single(cam, ccd, xy-np.array([45.,1.]))
                        perimeter.append(unit(ra, dec).round(8).tolist())
                cam_ccds.append(perimeter)
            sector['ccd'].append(cam_ccds)
    print('Max camera-center difference from MIT table (deg):', max(checks), flush=True)
    assert max(checks) < .1
    tess, tess_raw = horizons('-95', 'horizons-tess.json', '3 h')
    sun, sun_raw = horizons('10', 'horizons-sun.json', '1 d')
    # Daily inertial Greenwich direction and pole; a spin interpolation preserves
    # sub-day rotation. IAU 2006 precession/nutation, UT1 ~= UTC, no polar motion.
    earth_axes = []
    for row in sun_raw:
        t = Time(row[0], format='jd', scale='tdb')
        ut = t.utc
        rot = erfa.c2t06a(t.tt.jd1,t.tt.jd2,ut.jd1,ut.jd2,0.,0.).T
        earth_axes.append(rot.round(10).tolist())
    stars = []
    for row in csv.DictReader((CACHE / 'stars.csv').open()):
        if not row['mag'] or row['id'] == '0': continue
        if float(row['mag']) > 6.0: continue
        stars.append([round(float(row['ra'])*15,6),round(float(row['dec']),6),float(row['mag']),row['proper'] or row['bf'].strip() or f"HIP {row['hip']}",float(row['ci']) if row['ci'] else .6])
    # The project's example planet host is easy to find, even without a common name.
    for star in stars:
        if abs(star[0]-84.29)<.1 and abs(star[1]+80.47)<.1: star[3]='Pi Mensae'
    land = json.loads((CACHE / 'land.geojson').read_text())
    data = dict(sectors=sectors, tess=tess, sun=sun, earthAxes=earth_axes,
                stars=stars, land=land, asof='2026-10-10', trackingEnd='2026-09-27',
                validation=dict(maxCameraErrorDeg=round(max(checks),6)))
    (CACHE / 'scene-data.json').write_text(json.dumps(data, separators=(',', ':')))
    render()
    print(len(stars), 'stars', flush=True)


def render():
    missing = [f for f in ['three.min.js', 'OrbitControls.js', 'scene-data.json', *(n for n, _ in LICENSES)] if not (CACHE / f).exists()]
    if missing:
        sys.exit(f'Missing cached {", ".join(missing)} in {CACHE}.\nRun --extract-cache to recover them '
                 'from the committed index.html, or run without flags for a full (network) build.')
    template = (OUT / 'template.html').read_text()
    for token, file in [('__THREE__','three.min.js'),('__CONTROLS__','OrbitControls.js'),('__DATA__','scene-data.json')]:
        template = template.replace(token, (CACHE / file).read_text().replace('</script', '<\\/script'))
    sample = OUT / 'runs/sample.jsonl'
    template = template.replace('__SAMPLE_RUN__', sample.read_text().replace('</script', '<\\/script') if sample.exists() else '')
    jpg = CACHE / 'milkyway.jpg'
    template = template.replace('__MILKYWAY__', b64encode(jpg.read_bytes()).decode() if jpg.exists() else '')
    licenses = '\n\n'.join((CACHE / name).read_text() for name, _ in LICENSES)
    template = template.replace('</body>', NOTICES+licenses.replace('--','—')+'\n-->\n</body>')
    (OUT / 'index.html').write_text(template)
    print('Built', OUT / 'index.html', (OUT / 'index.html').stat().st_size, 'bytes', flush=True)


def extract_cache():
    """Recreate render()'s cached inputs from the committed index.html (reverse of render)."""
    html = (OUT / 'index.html').read_text()
    template = (OUT / 'template.html').read_text()
    # Match the template's embedded-script block, tokens -> captures, so tags/order come from the template.
    start = template.rindex('<script', 0, template.index('__THREE__'))
    end = template.index('</script>', template.index('__MILKYWAY__')) + len('</script>')
    pattern = re.escape(template[start:end])
    for token in ['__THREE__', '__CONTROLS__', '__DATA__', '__SAMPLE_RUN__', '__MILKYWAY__']:
        pattern = pattern.replace(re.escape(token), f'(?P<{token.strip("_")}>.*?)')
    m = re.search(pattern, html, re.S)
    if not m:
        sys.exit('index.html does not contain the template script block; cannot extract the cache.')
    for group, file in [('THREE', 'three.min.js'), ('CONTROLS', 'OrbitControls.js'), ('DATA', 'scene-data.json')]:
        (CACHE / file).write_text(m[group].replace('<\\/script', '</script'))
    if m['MILKYWAY']:
        (CACHE / 'milkyway.jpg').write_bytes(b64decode(m['MILKYWAY']))
    # The '--' -> '—' substitution is not reversed: the upstream notices contain no '--'.
    notices = html[html.rindex(NOTICES) + len(NOTICES):html.rindex('\n-->\n</body>')]
    cuts = [0, *(notices.index('\n\n' + head) for _, head in LICENSES[1:]), len(notices)]
    for (name, _), a, b in zip(LICENSES, cuts, cuts[1:]):
        (CACHE / name).write_text(notices[a + (2 if a else 0):b])
    print('Extracted cache into', CACHE, flush=True)


if __name__ == '__main__':
    main()
