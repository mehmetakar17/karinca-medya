"""Karınca reels motoru: müziğin gerçek vuruşlarına kesilen, kinetik yazılı reels.

Kullanım: python3 reel_motoru.py SPEC.json
SPEC alanları:
  music: mp3 yolu
  out: çıktı mp4
  beats_per_cut: her sahne kaç vuruş (varsayılan 2)
  search_from: müzikte en güçlü bölümü aramaya başlanacak saniye (varsayılan 0)
  clips: [{"type":"video","src":..., "ss":sn, "crop":[w,h,x,y] (ops.), "fill":true/false},
          {"type":"image","src":..., "focus":[fx,fy] (0-1, zoom merkezi), "zoom":[z0,z1]}]
  cards: [{"from":sahne_no, "to":sahne_no(hariç), "lines":[[["METİN","stil"],...], ...]}]
         stiller: big, huge, box, gold
  end: {"slogan":"En Rahat Gelin<br>Ayakkabısı Markası", "beats":6}
  logo: dikey siyah logo png yolu
  fonts: @fontsource node_modules yolu
"""
import sys, os, json, subprocess, shutil, tempfile
import numpy as np, librosa
from PIL import Image

spec = json.load(open(sys.argv[1]))
W, H, FPS = 1080, 1920, 30
work = tempfile.mkdtemp(prefix='reel_')
NM = spec['fonts']


def run(a):
    subprocess.run(a, check=True)


# ---------- müzik vuruşları ----------
music = spec['music']; bpc = spec.get('beats_per_cut', 2); clips = spec['clips']
end_beats = spec.get('end', {}).get('beats', 6)
y, sr = librosa.load(music, sr=22050, duration=120)
hop = 256
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
_, fr = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, tightness=200)
bt = librosa.frames_to_time(fr, sr=sr, hop_length=hop)
need = bpc * len(clips) + end_beats
cands = [i for i in range(len(bt) - need) if bt[i] >= spec.get('search_from', 0)]
best = max(cands, key=lambda i: oenv[fr[i:i + need]].mean())
phase = max(range(4), key=lambda p: oenv[fr[best + p:best + need:4]].mean())
i0 = best + phase; start = float(bt[i0])
beats = bt[i0:i0 + need + 1] - start
cut_t = [float(beats[k * bpc]) for k in range(len(clips) + 1)]
end_t = float(beats[min(len(beats) - 1, len(clips) * bpc + end_beats)])
frames = [round(t * FPS) for t in cut_t] + [round(end_t * FPS)]
P = float(np.mean(np.diff(beats)))
print(json.dumps({"start": round(start, 3), "bpm": round(60 / P, 1)}))

# ---------- sahneler ----------
parts = []
for k, c in enumerate(clips):
    d = (frames[k + 1] - frames[k]) / FPS; n = frames[k + 1] - frames[k]
    p = os.path.join(work, f'c{k:02d}.mp4'); parts.append(p)
    if c['type'] == 'video':
        crop = c.get('crop')
        pre = f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}," if crop else ''
        if c.get('fill'):
            vf = (f"[0:v]{pre}fps=30,split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                  f"boxblur=40:2,eq=brightness=-0.06:saturation=0.8[bg];[b]scale={W}:-2[fg];[bg][fg]overlay=0:(H-h)/2,setsar=1,format=yuv420p")
            run(['ffmpeg', '-v', 'error', '-y', '-ss', str(c['ss']), '-t', f'{d:.4f}', '-i', c['src'], '-filter_complex', vf,
                 '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', p])
        else:
            vf = f"{pre}fps=30,scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,format=yuv420p"
            run(['ffmpeg', '-v', 'error', '-y', '-ss', str(c['ss']), '-t', f'{d:.4f}', '-i', c['src'], '-vf', vf,
                 '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', p])
    else:  # image: Ken Burns
        im = Image.open(c['src']).convert('RGB'); iw, ih = im.size
        # 9:16 kadraj
        if iw / ih > W / H:
            cw = int(ih * W / H); im = im.crop(((iw - cw) // 2, 0, (iw - cw) // 2 + cw, ih))
        else:
            ch = int(iw * H / W); im = im.crop((0, (ih - ch) // 2, iw, (ih - ch) // 2 + ch))
        big = im.resize((W * 2, H * 2), Image.LANCZOS); src = os.path.join(work, f'i{k:02d}.png'); big.save(src)
        z0, z1 = c.get('zoom', [1.0, 1.12]); fx, fy = c.get('focus', [0.5, 0.5])
        z = f"{z0}+({z1}-{z0})*on/{max(1, n - 1)}"
        vf = (f"zoompan=z='{z}':x='(iw-iw/zoom)*{fx}':y='(ih-ih/zoom)*{fy}':d={n}:s={W}x{H}:fps=30,setsar=1,format=yuv420p")
        run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-i', src, '-vf', vf, '-frames:v', str(n),
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', p])

# ---------- kapanış kartı ----------
end = spec.get('end', {}); logo = spec['logo']
end_html = f"""<html lang="tr"><head><meta charset="utf-8"><style>
@import url("file://{NM}/pt-serif/700.css");@import url("file://{NM}/montserrat/500.css");@import url("file://{NM}/montserrat/700.css");
*{{margin:0;padding:0}}html,body{{width:{W}px;height:{H}px;background:#F6F1EA;overflow:hidden}}
.e{{position:absolute;top:400px;left:70px;right:70px;text-align:center;color:#1E1E1E}}.e img{{height:520px}}
.e h2{{font-family:'PT Serif';font-weight:700;font-size:80px;margin-top:64px;line-height:1.12}}
.l{{width:100px;height:4px;background:#B89B6E;margin:44px auto}}
.a{{font-family:Montserrat;font-weight:500;font-size:42px;white-space:nowrap;margin-bottom:14px;color:#2a2a2a}}
.c{{font-family:Montserrat;font-weight:700;font-size:50px;line-height:1.4;color:#2a2a2a}}
</style></head><body><div class="e"><img src="file://{logo}"><h2>{end.get('slogan', 'En Rahat Gelin<br>Ayakkabısı Markası')}</h2>
<div class="l"></div><div class="a">Altıparmak Cad. No:9, Osmangazi/BURSA</div>
<div class="c">Sipariş İçin<br>karincaayakkabi.com.tr'ye<br>Tıkla veya DM'den Yaz</div></div></body></html>"""
eh = os.path.join(work, 'end.html'); open(eh, 'w').write(end_html)
ep = os.path.join(work, 'end.png')
run([shutil.which('google-chrome'), '--headless=new', '--no-sandbox', '--disable-gpu', '--allow-file-access-from-files',
     '--hide-scrollbars', f'--window-size={W},{H}', '--virtual-time-budget=3000', f'--screenshot={ep}', 'file://' + eh])
d_end = (frames[-1] - frames[-2]) / FPS
pe = os.path.join(work, 'cend.mp4'); parts.append(pe)
run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-t', f'{d_end:.4f}', '-i', ep, '-vf', 'fps=30,format=yuv420p',
     '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', pe])
lst = os.path.join(work, 'l.txt'); open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
body = os.path.join(work, 'body.mp4')
run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', body])

# ---------- kinetik yazı kartları ----------
from playwright.sync_api import sync_playwright
css = f"""
@import url("file://{NM}/montserrat/800.css"); @import url("file://{NM}/montserrat/900.css");
*{{margin:0;padding:0;box-sizing:border-box}} html,body{{width:{W}px;height:{H}px;background:transparent;overflow:hidden}}
.wrap{{position:absolute;left:60px;right:60px;top:300px;display:flex;flex-direction:column;align-items:center;gap:18px}}
.line{{display:flex;gap:24px;justify-content:center;flex-wrap:wrap}}
.w{{font-family:Montserrat;font-weight:900;color:#fff;line-height:1;display:inline-block;opacity:0;transform-origin:50% 60%;
   text-shadow:0 6px 0 rgba(0,0,0,.25),0 10px 40px rgba(0,0,0,.6)}}
.big{{font-size:124px;letter-spacing:-2px}} .huge{{font-size:138px;letter-spacing:-3px}}
.box{{font-size:108px;background:#111;padding:18px 34px 22px;border-radius:14px;text-shadow:none}}
.gold{{font-size:80px;background:#F6F1EA;color:#1E1E1E;padding:16px 30px 20px;border-radius:12px;text-shadow:none;font-weight:800}}
@keyframes pop{{0%{{opacity:0;transform:scale(.4) translateY(40px) rotate(-4deg)}} 55%{{opacity:1;transform:scale(1.14) translateY(-6px) rotate(1deg)}}
 75%{{transform:scale(.96)}} 100%{{opacity:1;transform:scale(1) translateY(0) rotate(0)}}}}
"""
cards = spec.get('cards', []); movs = []
with sync_playwright() as pw:
    br = pw.chromium.launch(executable_path=shutil.which('google-chrome'), args=['--no-sandbox', '--allow-file-access-from-files'])
    pg = br.new_page(viewport={'width': W, 'height': H})
    for ci, card in enumerate(cards):
        t0, t1 = cut_t[card['from']], cut_t[card['to']]
        k = 0; lines = []
        for line in card['lines']:
            ws = []
            for text, st in line:
                ws.append(f'<span class="w {st}" style="animation:pop .32s cubic-bezier(.2,.9,.3,1.2) {k * P * 0.5:.3f}s forwards">{text}</span>'); k += 1
            lines.append(f'<div class="line">{"".join(ws)}</div>')
        hp = os.path.join(work, f'k{ci}.html')
        open(hp, 'w').write(f'<html lang="tr"><head><meta charset="utf-8"><style>{css}</style></head><body><div class="wrap">{"".join(lines)}</div></body></html>')
        pg.goto('file://' + hp); pg.wait_for_timeout(400); pg.evaluate("document.getAnimations().forEach(a=>a.pause())")
        fdir = os.path.join(work, f'k{ci}'); os.makedirs(fdir)
        for f in range(round((t1 - t0) * FPS)):
            pg.evaluate(f"document.getAnimations().forEach(a=>{{a.currentTime={f / FPS * 1000}}})")
            pg.screenshot(path=os.path.join(fdir, f'{f:04d}.png'), omit_background=True)
        mv = os.path.join(work, f'k{ci}.mov')
        run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(FPS), '-i', os.path.join(fdir, '%04d.png'), '-c:v', 'qtrle', '-pix_fmt', 'argb', mv])
        movs.append((mv, t0))
    br.close()

# üst gölge
g = Image.new('RGBA', (W, H), (0, 0, 0, 0)); a = Image.linear_gradient('L').resize((W, 1100)).transpose(Image.FLIP_TOP_BOTTOM)
a = a.point(lambda v: int(150 * (v / 255) ** 1.4)); g.paste((0, 0, 0, 255), (0, 0, W, 1100), a)
gp = os.path.join(work, 'grad.png'); g.save(gp)

vid_end = frames[-2] / FPS
punch = "+".join(f"0.05*exp(-max(t-{b:.3f},0)*14)*gte(t,{b:.3f})*lt(t,{b + 0.4:.3f})" for b in beats if b < vid_end)
fc = (f"[0:v]scale=w='trunc({W}*(1+{punch})/2)*2':h='trunc({H}*(1+{punch})/2)*2':eval=frame,crop={W}:{H},setsar=1[z];"
      f"[z][1:v]overlay=enable='lt(t,{vid_end:.3f})'[g0];")
last = 'g0'; inputs = ['-i', body, '-loop', '1', '-i', gp]
for j, (mv, t0) in enumerate(movs):
    inputs += ['-i', mv]
    fc += f"[{j + 2}:v]setpts=PTS+{t0:.3f}/TB[t{j}];[{last}][t{j}]overlay=eof_action=pass[o{j}];"; last = f'o{j}'
fc += f"[{last}]format=yuv420p"
vid = os.path.join(work, 'vid.mp4'); dur = frames[-1] / FPS
run(['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex', fc, '-t', f'{dur:.3f}', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17', vid])
run(['ffmpeg', '-v', 'error', '-y', '-i', vid, '-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', music, '-map', '0:v', '-map', '1:a',
     '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-af', f'afade=t=out:st={dur - 1.5:.2f}:d=1.5', '-shortest', spec['out']])
print('ok', round(dur, 2), spec['out'])
