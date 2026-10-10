#!/usr/bin/env python3
"""Render the source-verified, editable scientific architecture figure (stdlib only)."""
import html
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / 'assets/model-layout.json').read_text())
a = spec['architecture']
assert a == dict(layers=28, hidden_size=2048, intermediate_size=6144, query_heads=16,
                 kv_heads=8, head_dim=128, vocab_size=151936, rope_theta=1000000,
                 tie_word_embeddings=True), 'Reconcile diagram annotations with changed model configuration.'
P=[]
INK='#243344'; GRAY='#68727e'; LINE='#526274'; BLUE='#dceaf5'; B='#497695'; TEAL='#dceee9'; T='#457e71'; PURPLE='#ece5f3'; V='#806496'; ORANGE='#f6e7d5'; O='#af8757'
def add(s): P.append(s)
def text(x,y,s,size=25,anchor='middle',color=INK,bold=False,italic=False):
    content = html.escape(s)
    for sub in ('gate', 'down', 'up', 'Q', 'K', 'V', 'O'):
        content = content.replace('W'+sub, f'W<tspan baseline-shift="sub" font-size="{size*.72}">{sub}</tspan>')
    add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}"'+(' font-weight="600"' if bold else '')+(' font-style="italic"' if italic else '')+'>'+content+'</text>')
def rect(x,y,w,h,fill='white',stroke='#c4cbd1',r=5,dash=False):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"'+(' stroke-dasharray="7 6"' if dash else '')+'/>')
def path(d,color=LINE,width=2,arrow=False,dash=False):
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"'+(' stroke-dasharray="7 6"' if dash else '')+'/>')
    if arrow:
        points = re.findall(r'[ML]([0-9.]+),([0-9.]+)', d)
        (x1,y1),(x2,y2) = [(float(x),float(y)) for x,y in points[-2:]]
        theta = math.atan2(y2-y1,x2-x1)
        bx,by = x2-9*math.cos(theta), y2-9*math.sin(theta)
        dx,dy = 4*math.sin(theta), -4*math.cos(theta)
        add(f'<polygon points="{x2},{y2} {bx+dx},{by+dy} {bx-dx},{by-dy}" fill="{color}"/>')
def line(x1,y1,x2,y2,**k): path(f'M{x1},{y1} L{x2},{y2}',**k)
def dot(x,y,mark='+',fill='white',r=19):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{LINE}" stroke-width="1.8"/>'); text(x,y+8,mark,28)
def box(x,y,w,h,label,sub=None,fill=BLUE,stroke=B):
    rect(x,y,w,h,fill,stroke)
    if sub: text(x+w/2,y+h/2-3,label,27); text(x+w/2,y+h/2+24,sub,21,color=GRAY)
    else: text(x+w/2,y+h/2+9,label,27)
def panel(x,y,w,h,label):
    rect(x,y,w,h,'white','#c9cdd2',8); text(x+24,y+39,label,29,'start',bold=True)
def tensor(x,y,w,h,rows=5,cols=8,fill=BLUE,stroke=B):
    # Back planes and gridded front communicate a tensor rather than a process block.
    rect(x+12,y-12,w,h,fill,stroke,0); rect(x+6,y-6,w,h,fill,stroke,0); rect(x,y,w,h,fill,stroke,0)
    for i in range(1,cols): line(x+w*i/cols,y,x+w*i/cols,y+h,color=stroke,width=.7)
    for j in range(1,rows): line(x,y+h*j/rows,x+w,y+h*j/rows,color=stroke,width=.7)
def arrow(x1,y1,x2,y2): line(x1,y1,x2,y2,arrow=True)
def head(x,y,w,h,name,fill,stroke): rect(x,y,w,h,fill,stroke,3); text(x+w/2,y+h/2+8,name,23)

add('<svg xmlns="http://www.w3.org/2000/svg" width="2200" height="1470" viewBox="0 0 2200 1470" role="img" aria-labelledby="title desc">')
add('<title id="title">Meridian Text 1.7B — decoder architecture, attention, feed-forward and training objective</title>')
add('<desc id="desc">Four-panel scientific diagram of the unchanged Qwen3 architecture: 28 pre-norm decoder layers, grouped-query attention with 16 query and eight key-value heads, QK normalization and rotary positions, a 6144-channel SwiGLU, and the local next-token adaptation recipe. Attention cells are schematic.</desc>')
add('<defs><marker id="arr" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4.5 L0,9 Z" fill="#526274"/></marker></defs>')
add('<style>text{font-family:"Times New Roman",Times,"Liberation Serif",serif}path,rect,circle{vector-effect:non-scaling-stroke}</style>')
add('<rect width="2200" height="1470" fill="white"/>')
panel(40,35,840,1000,'(a)  Pre-norm causal decoder')
panel(920,35,1240,575,'(b)  Grouped-query attention')
panel(920,640,1240,395,'(c)  SwiGLU feed-forward network')
panel(40,1070,2120,355,'(d)  Local domain adaptation')

# (a) Information flows upward; residuals bypass the two pre-norm sublayers.
text(460,1010,'Meridian Text 1.7B  ·  Qwen3 backbone',26,bold=True)
for i,s in enumerate(['The','force','is','…']):
    rect(295+i*80,928,68,42,'#f4f5f7','#b8c0c8',3); text(329+i*80,957,s,23)
text(230,956,'x₁:T',26,italic=True)
arrow(455,925,455,898)
box(295,842,320,56,'Token embedding E','',BLUE,B)
text(640,876,'T × 2048',24,'start',italic=True)
arrow(455,842,455,803)
# repeated layer backing
rect(260,323,420,477,'#fafbfc','#cad0d7',10)
rect(250,313,420,477,'#fafbfc','#cad0d7',10)
rect(240,303,420,477,'#f8fafc','#9eafbe',10)
text(264,337,'Decoder layer ℓ',24,'start',bold=True)
text(687,345,'× 28',30,bold=True)
text(210,810,'H⁽ℓ⁾',27,italic=True)
arrow(455,803,455,752)
box(315,700,280,52,'RMSNorm',fill='#eef1f4',stroke='#9eacb9')
arrow(455,700,455,677)
box(315,614,280,63,'GQA attention',fill=BLUE,stroke=B)
text(282,649,'(b)',23,'end',color=B)
arrow(455,614,455,588)
dot(455,568)
path('M455,786 L700,786 L700,568 L474,568',arrow=True)
text(718,675,'identity',22,'start',color=GRAY,italic=True)
arrow(455,549,455,528)
box(315,476,280,52,'RMSNorm',fill='#eef1f4',stroke='#9eacb9')
arrow(455,476,455,454)
box(315,391,280,63,'SwiGLU',fill=TEAL,stroke=T)
text(282,429,'(c)',23,'end',color=T)
arrow(455,391,455,381)
dot(455,362)
path('M455,542 L735,542 L735,362 L474,362',arrow=True)
text(750,458,'identity',22,'start',color=GRAY,italic=True)
arrow(455,343,455,281)
box(315,231,280,50,'RMSNorm',fill='#eef1f4',stroke='#9eacb9')
arrow(455,231,455,210)
box(315,153,280,57,'LM head Eᵀ',fill=PURPLE,stroke=V)
arrow(455,153,455,130)
tensor(338,100,230,25,1,12,PURPLE,V)
text(620,120,'Z ∈ ℝᵀˣ¹⁵¹⁹³⁶',25,'start',italic=True)
text(622,150,'next-token logits',22,'start',color=GRAY)
path('M295,870 L140,870 L140,182 L315,182',color=V,arrow=True,dash=True)
text(120,515,'tied E',23,italic=True)
# Fine tensor labels reinforce dimensions rather than a generic chart.
text(695,951,'T ≤ 4096',24,italic=True)
text(695,985,'training length',21,color=GRAY)

# (b) Query/key normalizations precede RoPE, V is unrotated.
text(951,114,'U = RMSNorm(H⁽ℓ⁾) ∈ ℝᵀˣ²⁰⁴⁸',26,'start',italic=True)
text(2131,114,'dₕ = 128',25,'end',italic=True)
for y,label,shape in [(159,'WQ','2048 → 2048'),(291,'WK','2048 → 1024'),(423,'WV','2048 → 1024')]:
    box(952,y,157,66,label,shape,BLUE,B)
box(1135,159,134,66,'Q norm','per head','#eef1f4','#9eacb9')
box(1135,291,134,66,'K norm','per head','#eef1f4','#9eacb9')
box(1294,159,115,66,'RoPE',fill=PURPLE,stroke=V)
box(1294,291,115,66,'RoPE',fill=PURPLE,stroke=V)
arrow(1109,192,1135,192); arrow(1269,192,1294,192)
arrow(1109,324,1135,324); arrow(1269,324,1294,324)
arrow(1409,192,1440,192); arrow(1409,324,1491,324)
arrow(1109,456,1491,456)
text(1350,252,'θ = 10⁶',22,color=V,italic=True)
# Representative KV group, repeated 8 times.
rect(1445,146,230,353,'#fcfcfd','#abb8c5',7,True)
text(1560,530,'8 KV groups',24,bold=True)
head(1460,166,93,53,'q₂g',BLUE,B); head(1565,166,93,53,'q₂g+1',BLUE,B)
head(1499,297,121,53,'k g',TEAL,T)
head(1499,429,121,53,'v g',ORANGE,O)
text(1560,264,'2 queries / KV',22,color=GRAY)
# one group's queries and key meet at the causal score map; V routed below.
path('M1675,192 L1720,192 L1720,244 L1745,244',arrow=True)
path('M1620,324 L1710,324 L1710,274 L1745,274',arrow=True)
text(1840,160,'QKᵀ / √128 + M',27,italic=True)
text(1840,193,'softmax over keys',23,color=GRAY)
x0,y0,cell=1768,220,24
colors=['#bdd5e6','#87adc8','#5f8da9','#a6c5da','#d8e6f0','#719bb7']
for i in range(6):
    for j in range(6):
        rect(x0+j*cell,y0+i*cell,cell,cell,colors[(i*3+j)%len(colors)] if j<=i else '#f1f2f4','white',0)
        if j>i: text(x0+j*cell+12,y0+i*cell+16,'×',14,color='#adb4bc')
text(1840,394,'causal A ∈ ℝᵀˣᵀ',23,italic=True)
text(1840,421,'Mᵢⱼ = −∞  if j > i',21,color=GRAY,italic=True)
arrow(1915,292,1946,292)
dot(1968,292,'×',fill=ORANGE,r=22)
path('M1620,456 L1968,456 L1968,315',arrow=True)
arrow(1990,292,2015,292)
tensor(2026,249,90,90,6,5,BLUE,B)
text(2075,376,'Concat',23)
text(2075,404,'16 heads',22,color=GRAY)
path('M2075,417 L2075,461',arrow=True)
box(2005,464,142,58,'WO',fill=BLUE,stroke=B)
text(2075,563,'T × 2048',24,italic=True)
text(975,576,'Each KV head serves two query heads; shown for one representative group.',22,'start',color=GRAY)

# (c) Explicit SwiGLU branching, expansion, gate and contraction.
text(950,722,'U′ = RMSNorm(Ĥ⁽ℓ⁾)',26,'start',italic=True)
tensor(960,787,105,100,5,6,'#eef1f4','#8195a7')
text(1012,927,'T × 2048',24,italic=True)
path('M1076,837 L1100,837 L1100,771 L1128,771',arrow=True)
path('M1100,837 L1100,906 L1128,906',arrow=True)
box(1133,742,184,60,'Wgate',fill=TEAL,stroke=T)
box(1133,876,184,60,'Wup',fill=TEAL,stroke=T)
arrow(1317,771,1345,771)
box(1350,742,140,60,'SiLU',fill=PURPLE,stroke=V)
arrow(1490,771,1531,771)
tensor(1540,742,180,60,3,10,TEAL,T)
arrow(1317,906,1531,906)
tensor(1540,876,180,60,3,10,ORANGE,O)
text(1625,841,'T × 6144',24,italic=True)
path('M1732,771 L1768,771 L1768,819',arrow=True)
path('M1732,906 L1768,906 L1768,863',arrow=True)
dot(1768,841,'⊙',fill='#fbf6ed',r=22)
arrow(1790,841,1830,841)
box(1835,810,154,63,'Wdown',fill=TEAL,stroke=T)
arrow(1989,841,2020,841)
tensor(2030,797,95,89,5,5,TEAL,T)
text(2077,927,'T × 2048',24,italic=True)
text(1197,984,'FFN(U′) = [SiLU(U′Wgate) ⊙ U′Wup] Wdown',27,'start',italic=True)

# (d) Effective shifted targets show what the causal objective supervises.
text(68,1151,'Document-level split → tokenize + EOS → right-pad batch',26,'start')
text(68,1199,'Input',24,'start',bold=True)
text(68,1264,'Target',24,'start',bold=True)
tokens=['The','force','is','F','EOS','PAD']; targets=['force','is','F','EOS','−100','−100']
for i,(tok,targ) in enumerate(zip(tokens,targets)):
    x=165+i*94
    rect(x,1168,84,43,BLUE if tok!='PAD' else '#f0f1f3',B if tok!='PAD' else '#b5bdc5',3)
    text(x+42,1196,tok,23)
    rect(x,1233,84,43,TEAL if targ!='−100' else '#f0f1f3',T if targ!='−100' else '#b5bdc5',3)
    text(x+42,1261,targ,23)
    if i<4: path(f'M{x+42},1214 L{x+42},1229',arrow=True,color='#95a6b2',width=1.5)
text(165,1312,'Effective next-token targets; −100 excludes padding.',22,'start',color=GRAY)
text(165,1350,'EOS remains supervised even when EOS = PAD.',22,'start',color=GRAY)
# loss mathematics with an actual summation and mask, no invented benchmark chart.
line(785,1140,785,1388,color='#d2d8de',width=1)
text(835,1160,'Causal objective',26,'start',bold=True)
text(1190,1220,'ℒ = − 1/N  ∑ᵦ,ₜ mᵦ,ₜ log pθ(xᵦ,ₜ₊₁ | xᵦ,≤ₜ)',29,italic=True)
text(835,1274,'mᵦ,ₜ = 1 for valid next-token labels; N = ∑ mᵦ,ₜ',24,'start',color=GRAY,italic=True)
text(835,1327,'Full-parameter AdamW  ·  FP32  ·  gradient clipping 1.0',24,'start')
text(835,1370,'Checkpoint includes model, tokenizer, config and training log.',23,'start',color=GRAY)
line(1600,1140,1600,1388,color='#d2d8de',width=1)
text(1640,1160,'Held-out validation',26,'start',bold=True)
text(1640,1212,'No shared document IDs or text',23,'start')
text(1640,1263,'Token-weighted loss',24,'start')
text(1640,1314,'Export → reload → local inference',23,'start')
text(1640,1365,'Adaptation recipe ≠ original pretraining',22,'start',color=GRAY)
text(50,1454,'Tensor grids and attention values are schematic; dimensions follow the published checkpoint configuration.',22,'start',color=GRAY)
add('</svg>')
(ROOT/'assets/model-overview.svg').write_text('\n'.join(P)+'\n',encoding='utf-8')
print('Wrote assets/model-overview.svg (scientific layout v2)')
