/* Local MediaPipe try-on; shared by online and local pages. */
const cv=$('#cv'),cx=cv.getContext('2d');
let ML=null,MLp=null,base=null,user=null,tid=null,sel=null,photoVersion=0;const LAY={},jobs={};
const SLOT={shirt:'top',tee:'top',polo:'top',knit:'top',jeans:'bot',pants:'bot',puffer:'out',leather:'out',blazer:'out',coat:'out',jeanjacket:'out',shoe:'shoe',sneak:'shoe',loafer:'shoe',belt:null,hat:null};
const ORDER=['bot','top','out','shoe'];
function busy(m){const b=$('#busy');b.hidden=!m;b.textContent=m||''}
function stat(m){$('#tstat').textContent=m||''}
function errText(e){return t(e.message==='cors'?'e_cors':e.message==='img'?'e_img':e.message==='empty'?'e_empty':e.message==='file'?'e_file':'e_ml')}
function loadML(){
 if(location.protocol==='file:')return Promise.reject(new Error('file'));
 if(ML)return Promise.resolve(ML);if(MLp)return MLp;
 MLp=(async()=>{
  const mod=await import('./mp/vision_bundle.mjs');
  const fs=await mod.FilesetResolver.forVisionTasks(new URL('mp/wasm',location.href).href);
  let pose=null,seg=null;
  try{
   pose=await mod.PoseLandmarker.createFromOptions(fs,{baseOptions:{modelAssetPath:new URL('models/pose_landmarker_lite.task',location.href).href,delegate:'CPU'},runningMode:'IMAGE',numPoses:1});
   seg=await mod.ImageSegmenter.createFromOptions(fs,{baseOptions:{modelAssetPath:new URL('models/selfie_multiclass_256x256.tflite',location.href).href,delegate:'CPU'},runningMode:'IMAGE',outputCategoryMask:true,outputConfidenceMasks:false});
   return ML={pose,seg};
  }catch(e){pose?.close();seg?.close();throw e}
 })().catch(e=>{MLp=null;throw e});return MLp;
}
function prep(img,max){const w=img.naturalWidth||img.width,h=img.naturalHeight||img.height,k=Math.min(1,max/Math.max(w,h));const c=document.createElement('canvas');c.width=Math.round(w*k);c.height=Math.round(h*k);c.getContext('2d').drawImage(img,0,0,c.width,c.height);return c}
function body(res,W,H){const l=res&&res.landmarks&&res.landmarks[0];if(!l)return null;const P=i=>({x:l[i].x*W,y:l[i].y*H,v:l[i].visibility??1});
 const o={Ls:P(11),Rs:P(12),Lh:P(23),Rh:P(24),La:P(27),Ra:P(28),Lk:P(25),Rk:P(26),n:P(0)};if(o.Ls.v<.5||o.Rs.v<.5)return null;
 o.Sc={x:(o.Ls.x+o.Rs.x)/2,y:(o.Ls.y+o.Rs.y)/2};o.sw=Math.hypot(o.Ls.x-o.Rs.x,o.Ls.y-o.Rs.y);
 if(!Number.isFinite(o.sw)||o.sw<2)return null;
 o.hipsOk=o.Lh.v>.5&&o.Rh.v>.5&&Math.abs(o.Lh.y-o.Sc.y)>o.sw*.6;
 if(!o.hipsOk){const d=o.sw*1.55;o.Lh={x:o.Ls.x,y:o.Ls.y+d};o.Rh={x:o.Rs.x,y:o.Rs.y+d}}
 o.Hc={x:(o.Lh.x+o.Rh.x)/2,y:(o.Lh.y+o.Rh.y)/2};o.torso=Math.hypot(o.Hc.x-o.Sc.x,o.Hc.y-o.Sc.y);
 o.legsOk=o.La.v>.4&&o.Ra.v>.4&&o.hipsOk;o.Ac={x:(o.La.x+o.Ra.x)/2,y:(o.La.y+o.Ra.y)/2};return o}
function solve(S,D){/* affine from 3 point pairs -> [a,b,c,d,e,f] for ctx.setTransform */
 const [p,q,r]=S,[P,Q,R]=D;const det=p.x*(q.y-r.y)+q.x*(r.y-p.y)+r.x*(p.y-q.y);if(Math.abs(det)<1e-6)return null;
 const sol=(u,v,w)=>{const A=(u*(q.y-r.y)+v*(r.y-p.y)+w*(p.y-q.y))/det,B=(u*(r.x-q.x)+v*(p.x-r.x)+w*(q.x-p.x))/det,C=(u*(q.x*r.y-r.x*q.y)+v*(r.x*p.y-p.x*r.y)+w*(p.x*q.y-q.x*p.y))/det;return[A,B,C]};
 const X=sol(P.x,Q.x,R.x),Y=sol(P.y,Q.y,R.y);return[X[0],Y[0],X[1],Y[1],X[2],Y[2]]}
function whiteCut(c){/* flat product shot on light background -> alpha by distance from background */
 const x=c.getContext('2d'),d=x.getImageData(0,0,c.width,c.height),a=d.data;const bg=[0,1,2].map(k=>(a[k]+a[(c.width-1)*4+k]+a[(c.height-1)*c.width*4+k])/3);
 for(let i=0;i<a.length;i+=4){const dist=Math.hypot(a[i]-bg[0],a[i+1]-bg[1],a[i+2]-bg[2]);a[i+3]=Math.max(0,Math.min(255,(dist-22)*9))}x.putImageData(d,0,0);return c}
function bbox(c){const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let x0=1e9,y0=1e9,x1=-1,y1=-1;for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++)if(d[(y*c.width+x)*4+3]>40){if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y}return x1<0?null:{x0,y0,x1,y1}}
function feather(c){const o=document.createElement('canvas');o.width=c.width;o.height=c.height;const x=o.getContext('2d');x.filter='blur(1.1px)';x.drawImage(c,0,0);return o}
async function makeLayer(item,slot){
 const ml=await loadML();
 const im=await new Promise((ok,no)=>{const i=new Image();if(new URL(item.im[0],location.href).origin!==location.origin)i.crossOrigin='anonymous';i.referrerPolicy='no-referrer';i.onload=()=>ok(i);i.onerror=()=>no(new Error('img'));i.src=item.im[0]});
 const c=prep(im,720);try{c.getContext('2d').getImageData(0,0,1,1)}catch(e){throw new Error('cors')}
 let mb=null,cut=document.createElement('canvas');cut.width=c.width;cut.height=c.height;const cc=cut.getContext('2d');
 if(slot!='shoe'){try{mb=body(ml.pose.detect(c),c.width,c.height)}catch(e){}}
 if(mb){/* garment on a model: keep only "clothes" pixels in the right body band */
  const r=ml.seg.segment(c);const mask=r.categoryMask;if(!mask){r.close();throw new Error('empty')}const m=mask.getAsUint8Array(),mw=mask.width,mh=mask.height;cc.drawImage(c,0,0);const d=cc.getImageData(0,0,c.width,c.height),a=d.data;
  const hy=mb.Hc.y,T=mb.torso;const y0=slot=='bot'?hy-.14*T:-1e9,y1=slot=='top'?hy+.5*T:slot=='out'?hy+1.05*T:1e9;
  for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++){const i=y*c.width+x;a[i*4+3]=(m[Math.min(mh-1,Math.floor(y*mh/c.height))*mw+Math.min(mw-1,Math.floor(x*mw/c.width))]==4&&y>=y0&&y<=y1)?255:0}
  cc.putImageData(d,0,0);r.close()}
 else{cc.drawImage(c,0,0);whiteCut(cut)}
 const f=feather(cut);const fc=f.getContext('2d');fc.globalCompositeOperation='source-in';fc.drawImage(c,0,0);/* colour from original, soft alpha */
 /* soft alpha from blurred mask over original colours */
 const out=document.createElement('canvas');out.width=c.width;out.height=c.height;const ox=out.getContext('2d');ox.drawImage(c,0,0);ox.globalCompositeOperation='destination-in';ox.drawImage(f,0,0);
 const box=bbox(out);if(!box||box.x1<=box.x0||box.y1<=box.y0)throw new Error('empty');return{slot,item,cv:out,mb,box,dx:0,dy:0,k:1}}
function placeT(L){/* returns affine for layer onto user photo */
 if(!user||!L.box||L.box.x1<=L.box.x0||L.box.y1<=L.box.y0)return null;
 const u=user,m=L.mb,e=1.1;const S=m?[m.Ls,m.Rs,m.Hc]:null;
 const ex=(a,b,k)=>({x:b.x+(a.x-b.x)*k,y:b.y+(a.y-b.y)*k});
 let tf=null,cen;
 if(L.slot=='top'||L.slot=='out'){
  if(S){const sc={x:u.Sc.x,y:u.Sc.y};const Ls=ex(u.Ls,sc,e),Rs=ex(u.Rs,sc,e);const Hc={x:u.Hc.x,y:u.Hc.y+(L.slot=='out'?.04:.0)*u.torso};tf=solve(S,[Ls,Rs,Hc]);cen=sc}
  else if(L.box){const w=u.sw*(L.slot=='out'?1.75:1.6),k=w/(L.box.x1-L.box.x0);tf=[k,0,0,k,u.Sc.x-k*(L.box.x0+L.box.x1)/2,u.Sc.y-u.torso*.12-k*L.box.y0];cen=u.Sc}}
 else if(L.slot=='bot'){
  const ank=u.legsOk?u.Ac:{x:u.Hc.x,y:u.Hc.y+u.torso*1.75};
  if(S&&m.legsOk){const T=[m.Lh,m.Rh,m.Ac];tf=solve(T,[ex(u.Lh,u.Hc,1.08),ex(u.Rh,u.Hc,1.08),ank]);cen=u.Hc}
  else if(L.box){const w=u.sw*1.25,k=w/(L.box.x1-L.box.x0);tf=[k,0,0,k,u.Hc.x-k*(L.box.x0+L.box.x1)/2,u.Hc.y-u.torso*.1-k*L.box.y0];cen=u.Hc}}
 else if(L.slot=='shoe'&&L.box){const h=u.legsOk?Math.abs(u.Ac.y-u.n.y)*1.1:u.torso*3.2;const w=h*.17,k=w/(L.box.x1-L.box.x0);
  tf=[k,0,0,k,0,0];L.pair=true;cen=u.legsOk?u.Ac:{x:u.Hc.x,y:u.Hc.y+u.torso*1.9}}
 L.tf=tf;L.cen=cen;return tf}
function draw(){if(!base)return;cv.width=base.width;cv.height=base.height;cx.drawImage(base,0,0);
 ORDER.forEach(s=>{const L=LAY[s];if(!L||!L.tf)return;const[a,b,c,d,e,f]=L.tf,k=L.k,cn=L.cen;
  const draw1=(mx,my)=>{cx.save();cx.translate(cn.x+L.dx,cn.y+L.dy);cx.scale(k,k);cx.translate(-cn.x,-cn.y);cx.transform(a,b,c,d,mx!=null?mx:e,my!=null?my:f);cx.drawImage(L.cv,0,0);cx.restore()};
  if(L.pair&&L.box){const bw=(L.box.x1-L.box.x0),bh=(L.box.y1-L.box.y0);const kk=a;
   [[user.La,false],[user.Ra,true]].forEach(([p,fl])=>{const pp=user.legsOk?p:{x:user.Hc.x+(fl?1:-1)*user.sw*.3,y:user.Hc.y+user.torso*1.9};cx.save();cx.translate(pp.x+L.dx,pp.y+L.dy);cx.scale(k*(fl?-1:1),k);cx.translate(-kk*(L.box.x0+L.box.x1)/2,-kk*L.box.y1*.82);cx.scale(kk,kk);cx.drawImage(L.cv,0,0);cx.restore()})}
  else draw1()})}
function layersUI(){$('#layers').innerHTML=ORDER.filter(s=>LAY[s]).map(s=>`<span class="opt lay ${sel==s?'on':''}" onclick="selL('${s}')">${t('sl_'+s)}<b onclick="event.stopPropagation();rmL('${s}')" style="margin-inline-start:4px">×</b></span>`).join('')||`<span class="note">—</span>`}
function selL(s){sel=s;$('#tsc').value=Math.round(LAY[s].k*100);layersUI()}
function rmL(s){delete jobs[s];delete LAY[s];if(sel==s)sel=ORDER.find(x=>LAY[x])||null;layersUI();draw()}
function pickUI(){$('#pick').innerHTML=D.filter(x=>SLOT[x.s]&&!x.imageUnavailable).map(x=>`<img src="${x.im[0]}" referrerpolicy="no-referrer" class="${x.id==tid?'on':''}" onclick="tryOn(${x.id},true)" onerror="this.remove()" alt="">`).join('')}
async function tryOn(id,stay){
 const x=D.find(y=>y.id==id);if(!x)return;tid=id;pickUI();
 if(!stay)$('#tryon').scrollIntoView({behavior:'smooth'});
 const slot=SLOT[x.s];if(!slot){stat(t('noslot'));return}
 if(x.imageUnavailable){stat(t('e_img'));return}
 if(!base){stat(t('needph'));return}
 if(!user){stat(t('nopose'));return}
 const version=photoVersion,job=Symbol();jobs[slot]=job;
 try{
  busy(t('fit'));stat();const layer=await makeLayer(x,slot);
  if(version!==photoVersion||jobs[slot]!==job)return;
  if(!placeT(layer))throw new Error('empty');
  LAY[slot]=layer;sel=slot;$('#tsc').value=100;layersUI();draw();
  stat(layer.mb||slot==='shoe'?t('fit_ok'):t('fit_flat'));
 }catch(e){if(version===photoVersion&&jobs[slot]===job)stat(errText(e))}
 finally{if(jobs[slot]===job){delete jobs[slot];if(!Object.keys(jobs).length)busy()}}
}
async function wearSet(){if(!base){stat(t('needph'));return}if(!user){stat(t('nopose'));return}for(const x of cur_out()){if(SLOT[x.s]&&!x.imageUnavailable)await tryOn(x.id,true)}}
$('#file').onchange=async e=>{
 const f=e.target.files[0];if(!f)return;
 const version=++photoVersion;drag=null;user=null;base=null;cv.hidden=true;cx.clearRect(0,0,cv.width,cv.height);
 Object.keys(LAY).forEach(k=>delete LAY[k]);Object.keys(jobs).forEach(k=>delete jobs[k]);sel=null;layersUI();busy();
 if(!f.type.startsWith('image/')||f.size>25*1024*1024){stat(t('e_upload'));return}
 const url=URL.createObjectURL(f),im=new Image();
 try{
  await new Promise((ok,no)=>{im.onload=ok;im.onerror=()=>no(new Error('upload'));im.src=url});
  if(version!==photoVersion)return;
  base=prep(im,900);cv.hidden=false;$('#thint').style.display='none';draw();busy(t('ld'));stat();
  const ml=await loadML();if(version!==photoVersion)return;
  user=body(ml.pose.detect(base),base.width,base.height);
  if(!user)stat(t('nopose'));else{stat(t('pose_ok'));if(tid)await tryOn(tid,true)}
 }catch(err){if(version===photoVersion)stat(err.message==='upload'?t('e_upload'):errText(err))}
 finally{URL.revokeObjectURL(url);if(version===photoVersion&&!Object.keys(jobs).length)busy()}
};
let drag=null;
cv.addEventListener('pointerdown',e=>{if(!sel||!LAY[sel])return;cv.setPointerCapture(e.pointerId);const r=cv.getBoundingClientRect(),k=cv.width/r.width;const L=LAY[sel];drag={x:(e.clientX-r.left)*k-L.dx,y:(e.clientY-r.top)*k-L.dy}});
cv.addEventListener('pointermove',e=>{if(!drag||!sel||!LAY[sel]){drag=null;return}const r=cv.getBoundingClientRect(),k=cv.width/r.width,L=LAY[sel];L.dx=(e.clientX-r.left)*k-drag.x;L.dy=(e.clientY-r.top)*k-drag.y;draw()});
['pointerup','pointercancel'].forEach(n=>cv.addEventListener(n,()=>drag=null));
$('#tsc').oninput=()=>{if(sel&&LAY[sel]){LAY[sel].k=$('#tsc').value/100;draw()}};
$('#twear').onclick=wearSet;$('#tclr').onclick=()=>{Object.keys(jobs).forEach(k=>delete jobs[k]);busy();drag=null;Object.keys(LAY).forEach(k=>delete LAY[k]);sel=null;layersUI();draw()};
$('#tdl').onclick=()=>{if(!base){stat(t('needph'));return}try{const a=document.createElement('a');a.download='reda-tryon.png';a.href=cv.toDataURL('image/png');a.click()}catch(e){toast(t('e_save'))}};
$('#tadd').onclick=()=>{const ids=ORDER.filter(s=>LAY[s]).map(s=>LAY[s].item.id);(ids.length?ids:[tid]).filter(Boolean).forEach(i=>quick(i))};

