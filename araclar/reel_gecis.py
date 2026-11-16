"""Reels motoru v2 — CapCut tarzı geçişler (whip pan, zoom, spin, glitch, flaş), vuruşa oturan kesmeler.

Kalp atışı gibi her vuruşta zıplama YOK; hareket geçişlerden ve sahne içi yavaş yakınlaşmadan gelir.
Kullanım: python3 reel_gecis.py spec.json

spec:
  out, logo, fonts (@fontsource klasörü)
  stems: {"dir","base","drums":1,"bass":1,"other":0.55,"vocals":0}  (ya da "music": mp3)
  start_at: ops. müzik başlangıcı (sn); yoksa enerjili bir bölüm başı seçilir
  shots: [{"src","ss","beats":2,"speed":1.0,"tr":"whip_l|whip_r|whip_up|zoom_in|spin|glitch|flash|cut","drift":[1.0,1.06]}]
         ("tr" = bir sonraki sahneye geçiş)
  cards: [{"from":i,"to":j,"html":"..."}]   (sahne indeksleri; html .wrap içine girer)
  end: {"slogan":"...", "beats":6}
"""
import json, os, sys, math, shutil, subprocess, tempfile
import numpy as np, cv2, librosa

spec = json.load(open(sys.argv[1]))
W, H, FPS = 1080, 1920, 30
HALF = spec.get('half', 0.10)  # geçiş yarı süresi (sn) — kısa ve vuruşa oturan
work = tempfile.mkdtemp(prefix='gecis_')
NM = spec['fonts']; logo = spec['logo']
rng = np.random.default_rng(7)


def run(a):
    subprocess.run(a, check=True)


# ---------- müzik ----------
if spec.get('stems'):
    st = spec['stems']; ins = []; ws = []
    for k in ('drums', 'bass', 'other', 'vocals'):
        if st.get(k, 0) > 0:
            ins += ['-i', os.path.join(st['dir'], f"{st['base']}__{k}.mp3")]; ws.append(str(st[k]))
    music = os.path.join(work, 'mix.wav')
    run(['ffmpeg', '-v', 'error', '-y', *ins, '-filter_complex',
         ''.join(f'[{i}:a]' for i in range(len(ws))) + f"amix=inputs={len(ws)}:weights={' '.join(ws)}:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11[a]",
         '-map', '[a]', '-ar', '44100', music])
else:
    music = spec['music']
y, sr = librosa.load(music, sr=22050)
hop = 512
_, bf = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, tightness=200)
bt = librosa.frames_to_time(bf, sr=sr, hop_length=hop)
shots = spec['shots']
cum = [0]
for s in shots:
    cum.append(cum[-1] + s.get('beats', 2))
end_beats = spec.get('end', {}).get('beats', 6)
need = cum[-1] + end_beats
if 'start_at' in spec:
    i0 = int(np.argmin(np.abs(bt - spec['start_at'])))
else:
    C = librosa.util.sync(np.vstack([librosa.util.normalize(librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)),
                                     librosa.util.normalize(librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13))]), bf)
    bounds = sorted(set(librosa.segment.agglomerative(C, max(4, len(bt) // 32))))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rb = np.array([rms[f:f + 8].mean() for f in bf])
    cands = []
    for b in bounds:
        if b + need >= len(bt):
            continue
        inside = [x for x in bounds if b < x < b + need - 4]
        cands.append((len(inside), -rb[b:b + need].mean(), b))
    i0 = sorted(cands)[0][2] if cands else 0
# davul katmanından kick (bas davul) zamanları: kesimleri gerçek vuruşa oturt, ölçü başını bul
kick_env = None
if spec.get('stems'):
    yd, _ = librosa.load(os.path.join(spec['stems']['dir'], f"{spec['stems']['base']}__drums.mp3"), sr=22050)
    kick_env = librosa.onset.onset_strength(y=yd, sr=sr, hop_length=hop, fmax=150, n_mels=64)
    kicks = librosa.onset.onset_detect(onset_envelope=kick_env, sr=sr, hop_length=hop, units='time')
    snapped = bt.copy()
    for k, t in enumerate(bt):
        j = np.argmin(np.abs(kicks - t))
        if abs(kicks[j] - t) < 0.06:
            snapped[k] = kicks[j]
    bt = snapped
    # ölçü başı: 4 fazdan kick'i en güçlü olanı
    def kstr(i):
        f = int(bt[i] * sr / hop); return kick_env[max(0, f - 2):f + 3].max() if f < len(kick_env) else 0
    ph = max(range(4), key=lambda p: np.mean([kstr(i0 + p + 4 * q) for q in range(8) if i0 + p + 4 * q < len(bt)]))
    if spec.get('downbeat_detect'):  # bölüm başı zaten ölçü başıdır; dört dörtlükte kick her vuruşta olduğundan varsayılan kapalı
        i0 += ph
assert i0 + need < len(bt), 'müzik bu süre için kısa'
start = float(bt[i0])
beats = bt[i0:i0 + need + 1] - start
cut_t = [float(beats[k]) for k in cum]
end_t = float(beats[need])
P = float(np.mean(np.diff(beats)))
print(json.dumps({'start': round(start, 2), 'bpm': round(60 / P, 1), 'dur': round(end_t, 2)}))

# ---------- kaynak videodaki kendi kesmeleri: sahne bunların üstüne denk gelmesin ----------
_cuts = {}
def src_cuts(path):
    if path not in _cuts:
        if path.lower().endswith(('.jpg', '.jpeg', '.png')):
            _cuts[path] = []
        else:
            o = subprocess.run(['ffmpeg', '-hide_banner', '-i', path, '-vf', 'scale=270:-2,scdet=threshold=12', '-an', '-f', 'null', '-'],
                               capture_output=True, text=True).stderr
            _cuts[path] = [float(x.split()[-1]) for x in o.split('\n') if 'lavfi.scd.time' in x for x in [x[x.index('lavfi.scd.time'):]]]
    return _cuts[path]
for i, s_ in enumerate(shots):
    sp = s_.get('speed', 1.0); span = (cut_t[i + 1] - cut_t[i] + 2 * HALF) * sp
    a0 = s_['ss'] - HALF * sp
    for c in src_cuts(s_['src']):
        if 0.03 < c > a0 and c < a0 + span:
            new = c + 0.05 + HALF * sp if (c - a0) < span / 2 else c - 0.05 - span + HALF * sp
            print(f"sahne {i}: kaynak kesmesi {c:.2f}s, ss {s_['ss']} -> {new:.2f}"); s_['ss'] = round(new, 2)
            a0 = s_['ss'] - HALF * sp

# ---------- sahne kaynakları (önceden ölçekle) ----------
srcs = []
for i, s in enumerate(shots):
    t0 = cut_t[i] - HALF; t1 = cut_t[i + 1] + HALF
    sp = s.get('speed', 1.0)
    ss = max(0.0, s['ss'] - HALF * sp)
    n = int(math.ceil((t1 - t0) * FPS)) + 2
    p = os.path.join(work, f's{i:02d}.mp4')
    vf = f"setpts=PTS/{sp},fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"
    run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{ss:.3f}', '-i', s['src'], '-vf', vf, '-frames:v', str(n), '-an',
         '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '12', p])
    srcs.append((p, t0))


class Reader:
    def __init__(self, path, t0):
        self.cap = cv2.VideoCapture(path); self.t0 = t0; self.k = -1; self.last = None

    def at(self, t):
        want = max(0, int(round((t - self.t0) * FPS)))
        while self.k < want:
            ok, fr = self.cap.read()
            if not ok:
                break
            self.k += 1; self.last = fr
        return self.last


readers = {}


def shot_frame(i, t):
    if i not in readers:
        readers[i] = Reader(*srcs[i])
    fr = readers[i].at(t).copy()
    d0, d1 = shots[i].get('drift', [1.0, 1.05])
    u = np.clip((t - cut_t[i]) / max(1e-3, cut_t[i + 1] - cut_t[i]), 0, 1)
    return zoom(fr, d0 + (d1 - d0) * u)


def zoom(fr, s, cx=0.5, cy=0.5):
    if abs(s - 1) < 1e-3:
        return fr
    M = np.float32([[s, 0, (1 - s) * W * cx], [0, s, (1 - s) * H * cy]])
    return cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def ease(p):
    return p * p * (3 - 2 * p)


def tr_frame(kind, A, B, p):
    """p: 0..1 geçiş ilerlemesi; A çıkan, B giren sahne."""
    if kind == 'whip_l' or kind == 'whip_r' or kind == 'whip_up':
        e = ease(p); dirx = -1 if kind == 'whip_l' else 1
        if kind == 'whip_up':
            off = int(e * H); canvas = np.vstack([A, B])[off:off + H]
            k = 1 + int(160 * math.sin(math.pi * p)); return cv2.blur(canvas, (1, k)) if k > 1 else canvas
        off = int(e * W)
        canvas = np.hstack([A, B])[:, off:off + W] if dirx < 0 else np.hstack([B, A])[:, W - off:2 * W - off]
        k = 1 + int(180 * math.sin(math.pi * p))
        return cv2.blur(canvas, (k, 1)) if k > 1 else canvas
    if kind == 'zoom_in':
        if p < 0.5:
            q = ease(p / 0.5); s = 1 + 2.2 * q; src = A
        else:
            q = ease((1 - p) / 0.5); s = 1 + 2.2 * q; src = B
        layers = [zoom(src, s * f).astype(np.float32) for f in (1.0, 0.94, 0.88, 0.82)]
        out = sum(layers) / len(layers)
        out += 60 * math.sin(math.pi * p)
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == 'spin':
        if p < 0.5:
            q = ease(p / 0.5); ang = 70 * q; src = A
        else:
            q = ease((1 - p) / 0.5); ang = -70 * q; src = B
        s = 1 + 1.3 * q
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, s)
        r = cv2.warpAffine(src, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        k = 1 + int(40 * q)
        return cv2.GaussianBlur(r, (0, 0), k / 6) if k > 1 else r
    if kind == 'glitch':
        src = A if p < 0.5 else B
        inten = math.sin(math.pi * p)
        out = src.copy()
        dx = int(40 * inten)
        out[..., 2] = np.roll(src[..., 2], dx, axis=1); out[..., 0] = np.roll(src[..., 0], -dx, axis=1)
        for _ in range(int(7 * inten) + 1):
            y0 = int(rng.integers(0, H - 80)); hgt = int(rng.integers(20, 160))
            out[y0:y0 + hgt] = np.roll(out[y0:y0 + hgt], int(rng.integers(-120, 120) * inten), axis=1)
        return out
    if kind == 'flash':
        e = ease(p)
        mix = (A.astype(np.float32) * (1 - e) + B.astype(np.float32) * e)
        mix += 255 * 0.85 * math.sin(math.pi * p)
        return np.clip(mix, 0, 255).astype(np.uint8)
    return A if p < 0.5 else B


# ---------- yazı kartları ve kapanış ----------
from playwright.sync_api import sync_playwright
css = f"""@import url("file://{NM}/montserrat/800.css");@import url("file://{NM}/montserrat/900.css");
@import url("file://{NM}/pt-serif/700.css");@import url("file://{NM}/montserrat/500.css");@import url("file://{NM}/montserrat/700.css");
*{{margin:0;padding:0;box-sizing:border-box}}html,body{{width:{W}px;height:{H}px;background:transparent;overflow:hidden}}
.wrap{{position:absolute;left:50px;right:50px;top:300px;display:flex;flex-direction:column;align-items:center;gap:16px}}
.big{{font-family:Montserrat;font-weight:900;font-size:120px;line-height:1;color:#fff;letter-spacing:-2px;text-shadow:0 6px 0 rgba(0,0,0,.25),0 10px 40px rgba(0,0,0,.55)}}
.huge{{font-family:Montserrat;font-weight:900;font-size:300px;line-height:.9;color:#fff;text-shadow:0 8px 0 rgba(0,0,0,.25),0 14px 50px rgba(0,0,0,.6)}}
.box{{font-family:Montserrat;font-weight:900;font-size:96px;line-height:1;color:#fff;background:#111;padding:18px 34px 22px;border-radius:14px}}
.gold{{font-family:Montserrat;font-weight:800;font-size:76px;line-height:1;color:#1E1E1E;background:#F6F1EA;padding:16px 30px 20px;border-radius:12px}}
.num{{width:260px;height:260px;border-radius:50%;background:#B89B6E;color:#fff;font-family:Montserrat;font-weight:900;font-size:190px;line-height:260px;text-align:center;box-shadow:0 12px 40px rgba(0,0,0,.35)}}
"""
end = spec.get('end', {})
end_html = f"""<html lang="tr"><head><meta charset="utf-8"><style>{css}
html,body{{background:#F6F1EA}}.e{{position:absolute;top:400px;left:70px;right:70px;text-align:center;color:#1E1E1E}}.e img{{height:520px}}
.e h2{{font-family:'PT Serif';font-weight:700;font-size:80px;margin-top:64px;line-height:1.12}}
.l{{width:100px;height:4px;background:#B89B6E;margin:44px auto}}
.a{{font-family:Montserrat;font-weight:500;font-size:42px;white-space:nowrap;margin-bottom:14px;color:#2a2a2a}}
.c{{font-family:Montserrat;font-weight:700;font-size:50px;line-height:1.4;color:#2a2a2a}}
</style></head><body><div class="e"><img src="file://{logo}"><h2>{end.get('slogan', 'En Rahat Gelin<br>Ayakkabısı Markası')}</h2>
<div class="l"></div><div class="a">Altıparmak Cad. No:9, Osmangazi/BURSA</div>
<div class="c">Sipariş İçin<br>karincaayakkabi.com.tr'ye<br>Tıkla veya DM'den Yaz</div></div></body></html>"""
cards = []
with sync_playwright() as pw:
    br = pw.chromium.launch(executable_path=shutil.which('google-chrome'), args=['--no-sandbox', '--allow-file-access-from-files'])
    pg = br.new_page(viewport={'width': W, 'height': H})
    for ci, c in enumerate(spec.get('cards', [])):
        hp = os.path.join(work, f'k{ci}.html')
        open(hp, 'w').write(f'<html lang="tr"><head><meta charset="utf-8"><style>{css}</style></head><body><div class="wrap">{c["html"]}</div></body></html>')
        pg.goto('file://' + hp); pg.wait_for_timeout(500)
        pp = os.path.join(work, f'k{ci}.png'); pg.screenshot(path=pp, omit_background=True)
        im = cv2.imread(pp, cv2.IMREAD_UNCHANGED)
        cards.append((cut_t[c['from']], cut_t[c['to']], im))
    hp = os.path.join(work, 'end.html'); open(hp, 'w').write(end_html)
    pg.goto('file://' + hp); pg.wait_for_timeout(600)
    pp = os.path.join(work, 'end.png'); pg.screenshot(path=pp)
    endimg = cv2.imread(pp)
    br.close()

# üst gölge (beyaz yazı her zeminde okunsun)
grad = np.zeros((H, 1, 1), np.float32)
for yy in range(1150):
    grad[yy] = 0.55 * (1 - yy / 1150) ** 1.5


def put_card(fr, im, t, t0, t1):
    a = t - t0
    if a < 0 or t > t1:
        return fr
    if a < 0.28:  # pop: küçükten büyüğe, hafif taşma
        q = a / 0.28; s = 0.6 + 0.48 * math.sin(q * math.pi * 0.62) / math.sin(math.pi * 0.62) - 0.08 * q
        alpha_mul = min(1, q * 2.5)
    else:
        s = 1.0; alpha_mul = 1.0
    if t1 - t < 0.12:
        alpha_mul *= max(0, (t1 - t) / 0.12)
    ys, xs = np.where(im[..., 3] > 0)
    cy = (ys.min() + ys.max()) / 2 if len(ys) else 500
    M = np.float32([[s, 0, (1 - s) * W / 2], [0, s, (1 - s) * cy]])
    imw = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    al = imw[..., 3:4].astype(np.float32) / 255 * alpha_mul
    return (fr.astype(np.float32) * (1 - al) + imw[..., :3].astype(np.float32) * al).astype(np.uint8)


# ---------- kare kare oluştur ----------
vid = os.path.join(work, 'v.mp4')
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS),
                        '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', vid], stdin=subprocess.PIPE)
nfr = int(round(end_t * FPS)); last_cut = cut_t[-1]
for f in range(nfr):
    t = f / FPS
    if t >= last_cut + HALF:
        fr = endimg
    else:
        i = max(0, min(len(shots) - 1, int(np.searchsorted(cut_t, t, side='right') - 1)))
        # hangi kesime yakınız?
        fr = None
        for j in (i, i + 1):
            if 1 <= j <= len(shots) and abs(t - cut_t[j]) < HALF:
                p = (t - (cut_t[j] - HALF)) / (2 * HALF)
                A = shot_frame(j - 1, t)
                B = endimg if j == len(shots) else shot_frame(j, t)
                fr = tr_frame(shots[j - 1].get('tr', 'cut'), A, B, p)
        if fr is None:
            fr = shot_frame(i, t)
        if t < last_cut:
            fr = (fr.astype(np.float32) * (1 - grad)).astype(np.uint8)
            for t0, t1, im in cards:
                fr = put_card(fr, im, t, t0, t1)
    enc.stdin.write(np.ascontiguousarray(fr).tobytes())
enc.stdin.close(); enc.wait()
run(['ffmpeg', '-v', 'error', '-y', '-i', vid, '-ss', f'{start:.3f}', '-t', f'{end_t:.3f}', '-i', music, '-map', '0:v', '-map', '1:a',
     '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-af', f'afade=t=in:d=0.05,afade=t=out:st={end_t - 1.5:.2f}:d=1.5', '-shortest', spec['out']])
print('ok', round(end_t, 2), spec['out'])
