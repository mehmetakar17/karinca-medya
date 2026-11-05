"""Müşteri yorumu reels'i: her sahnede yorum fotoğrafı (Ken Burns) + yıldızlı yorum kartı, müziğin vuruşlarına kesilir.

Kullanım: python3 yorum_reels.py SPEC.json
SPEC: music, out, fonts, logo, search_from,
  scenes: [{"kind":"intro"|"review"|"stat", "img":yol, "focus":[fx,fy], "beats":n,
            "lines":[...] (intro/stat başlık satırları), "quote":..., "name":...}]
  end: {"slogan":..., "beats":6}
"""
import sys, os, json, subprocess, shutil, tempfile, html
import numpy as np, librosa
from PIL import Image, ImageOps

spec = json.load(open(sys.argv[1])); W, H, FPS = 1080, 1920, 30
work = tempfile.mkdtemp(prefix='yorum_'); NM = spec['fonts']


def run(a):
    subprocess.run(a, check=True, stderr=subprocess.DEVNULL if a[0].endswith('chrome') else None)


scenes = spec['scenes']; end_beats = spec['end'].get('beats', 6)
need = sum(s['beats'] for s in scenes) + end_beats
y, sr = librosa.load(spec['music'], sr=22050, duration=150); hop = 256
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
_, fr = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, tightness=200)
bt = librosa.frames_to_time(fr, sr=sr, hop_length=hop)
cands = [i for i in range(len(bt) - need - 1) if bt[i] >= spec.get('search_from', 0)]
best = max(cands, key=lambda i: oenv[fr[i:i + need]].mean())
phase = max(range(4), key=lambda p: oenv[fr[best + p:best + need:4]].mean())
i0 = best + phase; start = float(bt[i0]); beats = bt[i0:i0 + need + 1] - start
P = float(np.mean(np.diff(beats)))
bidx = [0]
for s in scenes: bidx.append(bidx[-1] + s['beats'])
bidx.append(bidx[-1] + end_beats)
cut_t = [float(beats[b]) for b in bidx]; frames = [round(t * FPS) for t in cut_t]
print(json.dumps({'start': round(start, 2), 'bpm': round(60 / P, 1), 'dur': round(cut_t[-1], 2)}))

STAR = '<svg viewBox="0 0 24 24" width="58" height="58"><path fill="#E8B04B" d="M12 2l2.9 6.6 7.1.7-5.4 4.8 1.6 7-6.2-3.7-6.2 3.7 1.6-7L2 9.3l7.1-.7z"/></svg>'
css = f"""
@import url("file://{NM}/montserrat/600.css");@import url("file://{NM}/montserrat/700.css");@import url("file://{NM}/montserrat/900.css");
@import url("file://{NM}/pt-serif/700.css");@import url("file://{NM}/pt-serif/400-italic.css");
*{{margin:0;padding:0;box-sizing:border-box}} html,body{{width:{W}px;height:{H}px;background:transparent;overflow:hidden}}
.badge{{position:absolute;left:60px;right:60px;top:250px;display:flex;align-items:center;gap:16px;background:#111;color:#fff;
  border-radius:22px;padding:18px 26px;font-family:Montserrat;opacity:0;animation:fade .35s forwards;box-shadow:0 12px 40px rgba(0,0,0,.35)}}
.badge.inwrap{{position:static;margin-top:6px}}
.badge .bs{{display:flex;gap:2px}} .badge b{{font-weight:900;font-size:46px;color:#E8B04B}} .badge .bt{{font-weight:700;font-size:36px;line-height:1.15;white-space:nowrap}}
.card{{position:absolute;left:60px;right:60px;top:400px;background:rgba(246,241,234,.96);border-radius:34px;padding:44px 50px 46px;
  box-shadow:0 20px 60px rgba(0,0,0,.35);opacity:0;animation:up .45s cubic-bezier(.2,.9,.3,1.15) .05s forwards}}
.stars{{display:flex;gap:6px;margin-bottom:22px}} .stars span{{opacity:0;display:inline-block;animation:pop .3s cubic-bezier(.2,.9,.3,1.4) forwards}}
.q{{font-family:'PT Serif','Noto Color Emoji';font-style:italic;font-size:56px;line-height:1.24;color:#1E1E1E}}
.q span{{opacity:0;animation:fade .25s ease-out forwards}}
.who{{margin-top:28px;font-family:Montserrat;font-weight:700;font-size:40px;color:#8C6D3F;opacity:0;animation:fade .3s forwards}}
.who small{{font-weight:600;color:#6b6b6b;font-size:36px}}
.wrap{{position:absolute;left:60px;right:60px;top:330px;display:flex;flex-direction:column;align-items:center;gap:18px}}
.w{{font-family:Montserrat;font-weight:900;color:#fff;line-height:1;opacity:0;display:inline-block;
  text-shadow:0 6px 0 rgba(0,0,0,.25),0 10px 40px rgba(0,0,0,.6);animation:pop .32s cubic-bezier(.2,.9,.3,1.2) forwards}}
.big{{font-size:124px;letter-spacing:-2px}} .box{{font-size:104px;background:#111;padding:18px 34px 22px;border-radius:14px;text-shadow:none}}
.gold{{font-size:76px;background:#F6F1EA;color:#1E1E1E;padding:16px 30px 20px;border-radius:12px;text-shadow:none;font-weight:900}}
.title{{font-size:112px;letter-spacing:-2px}}
.kicker{{font-size:62px;background:#F6F1EA;color:#1E1E1E;padding:14px 28px 18px;border-radius:12px;text-shadow:none;font-weight:900}}
.sub{{font-family:'PT Serif';font-weight:700;font-size:64px;background:#111;color:#fff;padding:16px 34px 22px;border-radius:14px;text-shadow:none;margin-top:14px}}
.line{{display:flex;gap:22px;justify-content:center;flex-wrap:wrap;align-items:center}}
@keyframes up{{0%{{opacity:0;transform:translateY(80px) scale(.92)}}100%{{opacity:1;transform:none}}}}
@keyframes pop{{0%{{opacity:0;transform:scale(.3)}}60%{{opacity:1;transform:scale(1.18)}}100%{{opacity:1;transform:scale(1)}}}}
@keyframes fade{{0%{{opacity:0;transform:translateY(10px)}}100%{{opacity:1;transform:none}}}}
"""


RATING = spec.get('rating')  # örn. {"score":"5,0","count":40,"note":"Hepsi 5 Yıldız"}


def rating_badge(delay=0.05):
    if not RATING:
        return ''
    st = ''.join(STAR.replace('width="58" height="58"', 'width="44" height="44"') for _ in range(5))
    return (f'<div class="badge" style="animation-delay:{delay:.2f}s"><span class="bs">{st}</span>'
            f'<b>{RATING["score"]}</b><span class="bt">{RATING["count"]} Değerlendirme<br>{RATING["note"]}</span></div>')


def scene_html(s):
    if s['kind'] == 'review':
        words = s['quote'].split(); t0 = 0.45
        n = len(s['quote'])
        fs = 56 if n < 140 else 50 if n < 220 else 45 if n < 300 else 41
        # kelimeler sahnenin ~%35'inde yazılır, kalan süre okumak için ekranda kalır
        step = min(0.16, max(0.05, (s['beats'] * P * 0.35) / max(1, len(words))))
        words = ['“' + words[0]] + words[1:]; words[-1] += '”'
        q = ' '.join(f'<span style="animation-delay:{t0 + i * step:.2f}s">{html.escape(w)}</span>' for i, w in enumerate(words))
        tw = t0 + len(words) * step + 0.05
        body = (f'{rating_badge()}<div class="card"><div class="q" style="font-size:{fs}px">{q}</div>'
                f'<div class="who" style="animation-delay:{tw:.2f}s">{html.escape(s["name"])} <small>· Trendyol yorumu</small></div></div>')
    else:
        k = 0; lines = []
        for line in s['lines']:
            ws = []
            for text, st in line:
                ws.append(f'<span class="w {st}" style="animation-delay:{0.05 + k * P * 0.5:.2f}s">{text}</span>'); k += 1
            lines.append(f'<div class="line">{"".join(ws)}</div>')
        badge = rating_badge(0.05 + k * P * 0.5 + 0.1) if s.get('show_rating') else ''
        body = f'<div class="wrap">{"".join(lines)}<div style="height:10px"></div>{badge}</div>'
        if badge:
            body = body.replace('class="badge"', 'class="badge inwrap"')
    return f'<html lang="tr"><head><meta charset="utf-8"><style>{css}</style></head><body>{body}</body></html>'


# ---- görüntü sahneleri (Ken Burns) ----
parts = []
for k, s in enumerate(scenes):
    n = frames[k + 1] - frames[k]
    im = ImageOps.exif_transpose(Image.open(s['img'])).convert('RGB'); iw, ih = im.size
    fx, fy = s.get('focus', [0.5, 0.5])
    if iw / ih > W / H:
        cw = int(ih * W / H); x0 = int((iw - cw) * fx); im = im.crop((x0, 0, x0 + cw, ih))
    else:
        ch = int(iw * H / W); y0 = int((ih - ch) * fy); im = im.crop((0, y0, iw, y0 + ch))
    src = os.path.join(work, f'i{k}.jpg'); im.resize((W * 2, H * 2), Image.LANCZOS).save(src, quality=95)
    z0, z1 = s.get('zoom', [1.0, 1.08]) if k % 2 == 0 else s.get('zoom', [1.08, 1.0])
    vf = (f"zoompan=z='{z0}+({z1}-{z0})*on/{max(1, n - 1)}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':d={n}:s={W}x{H}:fps=30,"
          f"eq=saturation=1.05:contrast=1.03,setsar=1,format=yuv420p")
    p = os.path.join(work, f'c{k:02d}.mp4'); parts.append(p)
    run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-i', src, '-vf', vf, '-frames:v', str(n), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', p])

# ---- kapanış ----
e = spec['end']; logo = spec['logo']
eh = os.path.join(work, 'end.html')
open(eh, 'w').write(f"""<html lang="tr"><head><meta charset="utf-8"><style>
@import url("file://{NM}/pt-serif/700.css");@import url("file://{NM}/montserrat/500.css");@import url("file://{NM}/montserrat/700.css");
*{{margin:0;padding:0}}html,body{{width:{W}px;height:{H}px;background:#F6F1EA;overflow:hidden}}
.e{{position:absolute;top:400px;left:70px;right:70px;text-align:center;color:#1E1E1E}}.e img{{height:520px}}
.e h2{{font-family:'PT Serif';font-weight:700;font-size:80px;margin-top:64px;line-height:1.12}}.l{{width:100px;height:4px;background:#B89B6E;margin:44px auto}}
.a{{font-family:Montserrat;font-weight:500;font-size:42px;white-space:nowrap;margin-bottom:14px;color:#2a2a2a}}
.c{{font-family:Montserrat;font-weight:700;font-size:50px;line-height:1.4;color:#2a2a2a}}</style></head>
<body><div class="e"><img src="file://{logo}"><h2>{e['slogan']}</h2><div class="l"></div>
<div class="a">Altıparmak Cad. No:9, Osmangazi/BURSA</div><div class="c">Sipariş İçin<br>karincaayakkabi.com.tr'ye<br>Tıkla veya DM'den Yaz</div></div></body></html>""")
ep = os.path.join(work, 'end.png')
run([shutil.which('google-chrome'), '--headless=new', '--no-sandbox', '--disable-gpu', '--allow-file-access-from-files', '--hide-scrollbars',
     f'--window-size={W},{H}', '--virtual-time-budget=3000', f'--screenshot={ep}', 'file://' + eh])
pe = os.path.join(work, 'cend.mp4'); parts.append(pe)
run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-t', f'{(frames[-1] - frames[-2]) / FPS:.4f}', '-i', ep, '-vf', 'fps=30,format=yuv420p',
     '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', pe])
lst = os.path.join(work, 'l.txt'); open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
body = os.path.join(work, 'body.mp4'); run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', body])

# ---- katmanlar ----
from playwright.sync_api import sync_playwright
movs = []
with sync_playwright() as pw:
    br = pw.chromium.launch(executable_path=shutil.which('google-chrome'), args=['--no-sandbox', '--allow-file-access-from-files'])
    pg = br.new_page(viewport={'width': W, 'height': H})
    for k, s in enumerate(scenes):
        hp = os.path.join(work, f'o{k}.html'); open(hp, 'w').write(scene_html(s))
        pg.goto('file://' + hp); pg.wait_for_timeout(400); pg.evaluate("document.getAnimations().forEach(a=>a.pause())")
        fd = os.path.join(work, f'o{k}'); os.makedirs(fd)
        for f in range(frames[k + 1] - frames[k]):
            pg.evaluate(f"document.getAnimations().forEach(a=>{{a.currentTime={f / FPS * 1000}}})")
            pg.screenshot(path=os.path.join(fd, f'{f:04d}.png'), omit_background=True)
        mv = os.path.join(work, f'o{k}.mov')
        run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(FPS), '-i', os.path.join(fd, '%04d.png'), '-c:v', 'qtrle', '-pix_fmt', 'argb', mv])
        movs.append((mv, cut_t[k], s['kind']))
    br.close()

g = Image.new('RGBA', (W, H), (0, 0, 0, 0)); a = Image.linear_gradient('L').resize((W, 1100)).transpose(Image.FLIP_TOP_BOTTOM)
g.paste((0, 0, 0, 255), (0, 0, W, 1100), a.point(lambda v: int(120 * (v / 255) ** 1.4))); gp = os.path.join(work, 'grad.png'); g.save(gp)
vid_end = cut_t[-2]
punch = "+".join(f"0.035*exp(-max(t-{b:.3f},0)*14)*gte(t,{b:.3f})*lt(t,{b + 0.4:.3f})" for b in cut_t[:-1])
fc = (f"[0:v]scale=w='trunc({W}*(1+{punch})/2)*2':h='trunc({H}*(1+{punch})/2)*2':eval=frame,crop={W}:{H},setsar=1[z];"
      f"[z][1:v]overlay=enable='lt(t,{vid_end:.3f})'[g0];")
last = 'g0'; inputs = ['-i', body, '-loop', '1', '-i', gp]
for j, (mv, t0, kind) in enumerate(movs):
    inputs += ['-i', mv]; fc += f"[{j + 2}:v]setpts=PTS+{t0:.3f}/TB[t{j}];[{last}][t{j}]overlay=eof_action=pass[o{j}];"; last = f'o{j}'
fc += f"[{last}]format=yuv420p"
vid = os.path.join(work, 'vid.mp4'); dur = cut_t[-1]
run(['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex', fc, '-t', f'{dur:.3f}', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17', vid])
run(['ffmpeg', '-v', 'error', '-y', '-i', vid, '-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', spec['music'], '-map', '0:v', '-map', '1:a',
     '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-af', f'afade=t=in:d=0.3,afade=t=out:st={dur - 1.5:.2f}:d=1.5', '-shortest', spec['out']])
print('ok', spec['out'])
