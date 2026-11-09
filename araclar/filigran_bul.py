import cv2, numpy as np, sys, json
from PIL import Image, ImageOps
S='/tmp/claude-0/-home-claude/23a739f0-ee9d-5809-9295-a5cb4a62f5e7/scratchpad'
logo=np.array(Image.open(S+'/brand/dikey_siyah.png'))
A=logo[...,3].astype(np.float32)/255  # logo mask
ys,xs=np.where(A>0.05); A=A[ys.min():ys.max()+1, xs.min():xs.max()+1]
def hpf(x,sg=6):
    return np.abs(cv2.GaussianBlur(x,(0,0),sg)-x)
def find(img):
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.float32)
    small=0.5; gs=cv2.resize(g,None,fx=small,fy=small,interpolation=cv2.INTER_AREA)
    hp=hpf(gs,3)
    best=None
    for w in range(360,1180,20):
        ws=int(w*small)
        t=cv2.resize(A,(ws,int(A.shape[0]*ws/A.shape[1])),interpolation=cv2.INTER_AREA)*255
        if t.shape[0]>=gs.shape[0] or t.shape[1]>=gs.shape[1]: break
        th=hpf(t,3)
        r=cv2.matchTemplate(hp,th,cv2.TM_CCOEFF_NORMED); r[~np.isfinite(r)]=0; r[r>1.001]=0
        _,mv,_,ml=cv2.minMaxLoc(r)
        if best is None or mv>best[0]: best=(mv,w,(int(ml[0]/small),int(ml[1]/small)))
    mv,w,(x,y)=best
    t=cv2.resize(A,(w,int(A.shape[0]*w/A.shape[1])),interpolation=cv2.INTER_AREA)
    # fine search full res
    hpF=hpf(g,3); bb=None
    for dw in range(-12,13,4):
        tt=cv2.resize(A,(w+dw,int(A.shape[0]*(w+dw)/A.shape[1])),interpolation=cv2.INTER_AREA)
        x0,y0=max(0,x-12),max(0,y-12); x1,y1=min(g.shape[1],x+w+dw+12),min(g.shape[0],y+tt.shape[0]+12)
        roi=hpF[y0:y1,x0:x1]
        if roi.shape[0]<tt.shape[0] or roi.shape[1]<tt.shape[1]: continue
        r=cv2.matchTemplate(roi,hpf(tt*255,3),cv2.TM_CCOEFF_NORMED); r[~np.isfinite(r)]=0; r[r>1.001]=0; _,m2,_,l2=cv2.minMaxLoc(r)
        if bb is None or m2>bb[0]: bb=(m2,w+dw,(x0+l2[0],y0+l2[1]),tt)
    return bb
def clean(path,out):
    img=np.array(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
    Image.fromarray(img).save(out,quality=97)
    res=[]
    for it in range(3):
        img=np.array(Image.open(out).convert('RGB'))
        r=clean1(img,out)
        if r is None: break
        res.append(r)
    return dict(f=path,passes=res)
PAD=500
def clean1(img0,out):
    img=cv2.copyMakeBorder(img0,PAD,PAD,PAD,PAD,cv2.BORDER_REPLICATE)
    mv,w,(x,y),t=find(img)
    if mv<0.45: return None
    I=img.astype(np.float32)
    h_,w_=t.shape
    M=np.zeros(I.shape[:2],np.float32); M[y:y+h_,x:x+w_]=t
    # background estimate: inpaint masked area on gray
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    mk=(cv2.dilate((M>0.05).astype(np.uint8),np.ones((7,7),np.uint8))*255)
    bgi=cv2.inpaint(g,mk,9,cv2.INPAINT_TELEA).astype(np.float32)
    sel=(M>0.6)&(cv2.Laplacian(bgi,cv2.CV_32F,ksize=5).__abs__()<20)
    Bs=bgi[sel]; Is=g.astype(np.float32)[sel]
    slope,icpt=np.polyfit(Bs,Is,1)
    c=176.0; d=Bs-c; a=float(np.clip(np.sum((Bs-Is)*d)/max(np.sum(d*d),1),0.05,0.7))
    B=np.clip((I-a*M[...,None]*c)/(1-a*M[...,None]),0,255).astype(np.uint8)
    # kill residual outline: inpaint thin band around mask edges
    edge=cv2.morphologyEx((M>0.3).astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))*255
    B=cv2.inpaint(B,edge,3,cv2.INPAINT_TELEA)
    # plain bright background inside logo area -> rebuild from surroundings (removes faint ghost)
    gB=cv2.cvtColor(B,cv2.COLOR_RGB2GRAY)
    area=cv2.dilate((M>0.02).astype(np.uint8),np.ones((15,15),np.uint8))
    tex=cv2.GaussianBlur(np.abs(cv2.Laplacian(bgi,cv2.CV_32F,ksize=3)),(0,0),4)
    plain=(area>0)&(gB>175)&(tex<3)
    plain=cv2.morphologyEx(plain.astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    if plain.sum():
        small=cv2.resize(B,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)
        pm=cv2.resize(plain*255,(small.shape[1],small.shape[0]),interpolation=cv2.INTER_NEAREST)
        sm=cv2.inpaint(small,cv2.dilate(pm,np.ones((3,3),np.uint8)),15,cv2.INPAINT_TELEA)
        sm=cv2.GaussianBlur(cv2.resize(sm,(B.shape[1],B.shape[0]),interpolation=cv2.INTER_CUBIC),(0,0),3)
        wt=cv2.GaussianBlur(plain.astype(np.float32),(0,0),3)[...,None]
        noise=(B.astype(np.float32)-cv2.GaussianBlur(B,(0,0),2).astype(np.float32))*0  # keep simple
        B=np.clip(B*(1-wt)+sm*wt,0,255).astype(np.uint8)
    B=B[PAD:-PAD,PAD:-PAD]
    Image.fromarray(B).save(out,quality=97)
    return [round(float(mv),2),w,x,y,round(a,2)]
if __name__=='__main__':
    for p in sys.argv[2:]:
        print(json.dumps(clean(p, sys.argv[1]+"/"+p.rsplit("/",1)[-1]),default=lambda o: o.tolist() if hasattr(o,"tolist") else str(o)))
