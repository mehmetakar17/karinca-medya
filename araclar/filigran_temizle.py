import cv2, numpy as np, sys, json
from PIL import Image, ImageOps
sys.path.insert(0,__import__('os').path.dirname(__file__))
from filigran_bul import A, find, hpf
def place(shape,t,x,y):
    M=np.zeros(shape,np.float32); h,w=t.shape
    x0,y0=max(0,x),max(0,y); x1,y1=min(shape[1],x+w),min(shape[0],y+h)
    if x1<=x0 or y1<=y0: return None
    M[y0:y1,x0:x1]=t[y0-y:y1-y,x0-x:x1-x]; return M
def corr(g,M):
    sel=cv2.dilate((M>0.02).astype(np.uint8),np.ones((9,9),np.uint8))>0
    a=hpf(g,3)[sel]; b=hpf(M*255,3)[sel]
    if a.std()<1e-3 or b.std()<1e-3: return 0
    return float(np.corrcoef(a,b)[0,1])
def remove(img,M):
    I=img.astype(np.float32); g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    mk=cv2.dilate((M>0.05).astype(np.uint8),np.ones((7,7),np.uint8))*255
    bgi=cv2.inpaint(g,mk,9,cv2.INPAINT_TELEA).astype(np.float32)
    sel=(M>0.6)&(np.abs(cv2.Laplacian(bgi,cv2.CV_32F,ksize=5))<20)
    c=176.0
    if sel.sum()>500:
        Bs=bgi[sel]; Is=g.astype(np.float32)[sel]; d=Bs-c
        a=float(np.clip(np.sum((Bs-Is)*d)/max(np.sum(d*d),1),0.2,0.6))
    else: a=0.4
    B=np.clip((I-a*M[...,None]*c)/(1-a*M[...,None]),0,255).astype(np.uint8)
    edge=cv2.morphologyEx((M>0.3).astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))*255
    B=cv2.inpaint(B,edge,3,cv2.INPAINT_TELEA)
    # rebuild plain bright background inside the logo footprint
    area=cv2.dilate((M>0.02).astype(np.uint8),np.ones((21,21),np.uint8))>0
    gB=cv2.cvtColor(B,cv2.COLOR_RGB2GRAY).astype(np.float32)
    # object = strong texture or darker than local bg
    big=cv2.GaussianBlur(gB,(0,0),40)
    tex=cv2.GaussianBlur(np.abs(gB-cv2.GaussianBlur(gB,(0,0),4)),(0,0),6)
    obj=(gB<big-25)|(tex>6)|(gB<170)
    obj=cv2.dilate(obj.astype(np.uint8),np.ones((25,25),np.uint8))>0
    plain=area&~obj
    if plain.sum()>100:
        sc=0.25; small=cv2.resize(B,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
        pm=cv2.resize(plain.astype(np.uint8)*255,(small.shape[1],small.shape[0]),interpolation=cv2.INTER_NEAREST)
        sm=cv2.inpaint(small,cv2.dilate(pm,np.ones((3,3),np.uint8)),20,cv2.INPAINT_TELEA)
        sm=cv2.GaussianBlur(cv2.resize(sm,(B.shape[1],B.shape[0]),interpolation=cv2.INTER_CUBIC),(0,0),6).astype(np.float32)
        # keep original fine grain
        grain=B.astype(np.float32)-cv2.GaussianBlur(B,(0,0),2).astype(np.float32)
        wt=cv2.GaussianBlur(plain.astype(np.float32),(0,0),8)[...,None]
        B=np.clip(B*(1-wt)+(sm+grain*0.6)*wt,0,255).astype(np.uint8)
    return B, a
def clean(path,out):
    img=np.array(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
    mv,w,(x,y),t=find(img)
    log=[[round(mv,2),w,x,y]]
    if mv<0.45: Image.fromarray(img).save(out,quality=97); return log
    step=round(w*583/608)
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.float32)
    for k in range(-3,4):
        for dy in (0,):
            M=place(g.shape,t,x+k*step,y+dy)
            if M is None: continue
            # small local refine for k!=0
            best=(corr(g,M),0,M)
            if k!=0:
                for ddx in range(-8,9,2):
                    M2=place(g.shape,t,x+k*step+ddx,y)
                    if M2 is not None:
                        cc=corr(g,M2)
                        if cc>best[0]: best=(cc,ddx,M2)
            cc,ddx,M=best
            if cc>0.35:
                img,a=remove(img,M); g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.float32)
                log.append([k,ddx,round(cc,2),round(a,2)])
    Image.fromarray(img).save(out,quality=97); return log
for p in sys.argv[2:]:
    print(p, clean(p, sys.argv[1]+'/'+p))
