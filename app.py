import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import cv2
import json
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="EcoLens — AI Waste Intelligence",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --green:        #16a34a;
    --green-light:  #22c55e;
    --teal:         #0d9488;
    --emerald:      #059669;
    --red:          #dc2626;
    --text:         #0f3d22;
    --muted:        #4b5563;
    --border:       #86efac;
    --border-dark:  #4ade80;
    --shadow:       rgba(22,163,74,0.15);
    --shadow-hover: rgba(22,163,74,0.30);
    --bg:           #dcfce7;
    --bg2:          #f0fdf4;
    --card:         #ffffff;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(ellipse at 10% 20%, rgba(22,163,74,0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(13,148,136,0.10) 0%, transparent 50%),
        linear-gradient(160deg, #dcfce7 0%, #d1fae5 50%, #ccfbf1 100%) !important;
    background-attachment: fixed !important;
}

#MainMenu, footer, header, .stDeployButton { visibility:hidden!important; display:none!important; }

.block-container {
    padding: 1.2rem 2% !important;
    max-width: 100% !important;
}

/* ── Hero ── */
.hero-wrap { text-align:center; padding:0.9rem 0 0.7rem; }
.hero-badge {
    display:inline-block;
    background:linear-gradient(135deg,#14532d,#0d9488);
    color:white; font-size:0.68rem; font-weight:700;
    letter-spacing:0.13em; text-transform:uppercase;
    padding:0.32rem 1rem; border-radius:999px; margin-bottom:0.5rem;
    box-shadow:0 4px 14px rgba(20,83,45,0.35);
    transition:transform 0.2s,box-shadow 0.2s;
}
.hero-badge:hover { transform:translateY(-2px); box-shadow:0 7px 20px rgba(20,83,45,0.45); }
.hero-title {
    font-size:2.9rem; font-weight:800;
    background:linear-gradient(135deg,#14532d,#16a34a,#0d9488);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    line-height:1.1; margin:0 0 0.4rem;
}
.hero-sub { font-size:0.88rem; color:var(--muted); margin:0 auto 0.85rem; line-height:1.6; }
.hero-pills { display:flex; justify-content:center; gap:0.38rem; flex-wrap:wrap; }
.hpill {
    background:white; border:1.5px solid var(--border-dark);
    border-radius:999px; padding:0.25rem 0.72rem;
    font-size:0.7rem; font-weight:600; color:var(--text);
    box-shadow:0 2px 8px var(--shadow);
    transition:all 0.2s ease; cursor:default;
}
.hpill:hover { transform:translateY(-2px); box-shadow:0 5px 16px var(--shadow-hover); border-color:var(--green); }
.hpill b { color:var(--emerald); }

.divider {
    border:none; height:1.5px;
    background:linear-gradient(90deg,transparent,var(--border-dark),transparent);
    margin:0.6rem 0;
}

/* ── Cards ── */
.card {
    background:white;
    border:1.5px solid var(--border-dark);
    border-radius:14px; overflow:hidden;
    box-shadow:0 3px 14px var(--shadow);
    margin-bottom:0.65rem;
    transition:transform 0.22s ease,box-shadow 0.22s ease;
    animation:fadeInUp 0.3s ease both;
}
.card:hover { transform:translateY(-3px); box-shadow:0 9px 28px var(--shadow-hover); }
.card-hdr {
    padding:0.5rem 0.9rem;
    border-bottom:1.5px solid var(--border);
    font-size:0.77rem; font-weight:700; color:var(--text);
    background:linear-gradient(135deg,rgba(220,252,231,0.9),rgba(204,251,241,0.7));
    display:flex; align-items:center; gap:0.35rem;
}
.card-body { padding:0.7rem 0.9rem; }

/* ── Image display: square, centered ── */
.img-square-wrap {
    width:100%;
    aspect-ratio:1/1;
    display:flex;
    align-items:center;
    justify-content:center;
    overflow:hidden;
    border-radius:10px;
    background:#f0fdf4;
}
.img-square-wrap img {
    width:100%;
    height:100%;
    object-fit:cover;
    border-radius:10px;
    transition:transform 0.3s ease;
}
.img-square-wrap img:hover { transform:scale(1.03); }

/* Override streamlit image to fill square wrap */
[data-testid="stImage"] {
    width:100% !important;
    display:flex !important;
    justify-content:center !important;
}
[data-testid="stImage"] img {
    border-radius:10px !important;
    object-fit:cover !important;
    max-height:260px !important;
    width:100% !important;
    transition:transform 0.3s ease !important;
}
[data-testid="stImage"] img:hover { transform:scale(1.02) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border:2px dashed var(--border-dark) !important;
    border-radius:10px !important;
    background:rgba(240,253,244,0.5) !important;
    transition:border-color 0.2s,background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color:var(--green) !important;
    background:rgba(220,252,231,0.7) !important;
}

/* ── Result badge ── */
.res-badge {
    border-radius:10px; padding:0.68rem 0.9rem;
    display:flex; align-items:center; gap:0.65rem;
    border:2px solid; transition:all 0.2s;
}
.res-badge:hover { transform:translateX(3px); }
.res-emoji { font-size:1.6rem; flex-shrink:0; }
.res-name  { font-size:1.02rem; font-weight:800; }
.res-conf  { font-size:0.73rem; opacity:0.72; margin-top:0.08rem; }

/* ── Bin row ── */
.bin-row { display:grid; grid-template-columns:repeat(6,1fr); gap:0.45rem; }
.bin-item {
    border-radius:10px; padding:0.62rem 0.35rem;
    text-align:center; border:1.5px solid; position:relative;
    transition:all 0.22s ease;
}
.bin-item.ok {
    background:rgba(22,163,74,0.1);
    border-color:var(--green);
    box-shadow:0 3px 10px rgba(22,163,74,0.22);
}
.bin-item.ok:hover { transform:translateY(-4px); box-shadow:0 9px 22px rgba(22,163,74,0.32); }
.bin-item.no { background:rgba(0,0,0,0.02); border-color:rgba(0,0,0,0.08); opacity:0.38; }
.bin-item.no:hover { opacity:0.55; }
.badge-ok {
    position:absolute; top:-7px; right:-7px;
    width:18px; height:18px; border-radius:50%;
    background:var(--green); color:white;
    font-size:0.58rem; font-weight:800;
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 2px 6px rgba(22,163,74,0.45);
}
.badge-no {
    position:absolute; top:-7px; right:-7px;
    width:18px; height:18px; border-radius:50%;
    background:var(--red); color:white;
    font-size:0.58rem; font-weight:800;
    display:flex; align-items:center; justify-content:center;
}
.bin-ico  { font-size:1.3rem; display:block; margin-bottom:0.18rem; }
.bin-name { font-size:0.61rem; font-weight:700; color:var(--text); display:block; }
.bin-desc { font-size:0.54rem; color:var(--muted); display:block; margin-top:0.04rem; }

/* ── Prob bars ── */
.prob-row { margin-bottom:0.42rem; }
.prob-lbl { display:flex; justify-content:space-between; font-size:0.75rem; font-weight:600; color:var(--text); margin-bottom:0.18rem; }
.prob-trk { background:#bbf7d0; border-radius:999px; height:7px; overflow:hidden; }
.prob-fil { height:100%; border-radius:999px; background:linear-gradient(90deg,var(--green),var(--teal)); }

/* ── Tips ── */
.tip-row {
    display:flex; gap:0.5rem; align-items:flex-start;
    padding:0.45rem 0.65rem; border-radius:8px;
    background:rgba(220,252,231,0.7);
    border:1.5px solid var(--border);
    margin-bottom:0.32rem;
    font-size:0.76rem; color:var(--text); line-height:1.4;
    transition:all 0.2s ease;
}
.tip-row:hover {
    background:rgba(187,247,208,0.9);
    border-color:var(--green);
    transform:translateX(5px);
    box-shadow:0 3px 12px var(--shadow-hover);
}
.tip-ico { font-size:0.88rem; flex-shrink:0; margin-top:0.04rem; }

/* ── Kabadiwala ── */
.kabadi {
    background:linear-gradient(135deg,rgba(22,163,74,0.14),rgba(13,148,136,0.12));
    border:1.5px solid var(--border-dark); border-radius:9px;
    padding:0.5rem 0.72rem; display:flex; align-items:center; gap:0.5rem; margin-top:0.4rem;
    transition:all 0.2s;
}
.kabadi:hover { border-color:var(--green); box-shadow:0 4px 14px var(--shadow-hover); }
.kabadi-t { font-size:0.78rem; font-weight:700; color:var(--text); }
.kabadi-s { font-size:0.68rem; color:var(--muted); }

/* ── Chat bubbles ── */
.chat-scroll { max-height:240px; overflow-y:auto; display:flex; flex-direction:column; gap:0.35rem; padding:0.4rem; }
.chat-u {
    align-self:flex-end;
    background:linear-gradient(135deg,var(--green),var(--teal));
    color:white; border-radius:10px 10px 2px 10px;
    padding:0.44rem 0.75rem; max-width:80%;
    font-size:0.8rem; line-height:1.4;
    box-shadow:0 3px 10px rgba(22,163,74,0.3);
    animation:slideInRight 0.25s ease;
}
.chat-a {
    align-self:flex-start;
    background:white; border:1.5px solid var(--border-dark);
    border-radius:10px 10px 10px 2px;
    padding:0.44rem 0.75rem; max-width:88%;
    font-size:0.8rem; color:var(--text); line-height:1.5;
    box-shadow:0 2px 8px var(--shadow);
    animation:slideInLeft 0.25s ease;
}
@keyframes slideInRight { from{opacity:0;transform:translateX(15px)} to{opacity:1;transform:translateX(0)} }
@keyframes slideInLeft  { from{opacity:0;transform:translateX(-15px)} to{opacity:1;transform:translateX(0)} }
@keyframes fadeInUp     { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }

/* ── Quick Q buttons — no unnecessary gap ── */
.quick-q-row {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:0.5rem;
    margin-bottom:0.65rem;
}
.quick-q-btn {
    background:white;
    border:1.5px solid var(--border-dark);
    border-radius:9px;
    padding:0.55rem 0.7rem;
    font-family:'Outfit',sans-serif;
    font-size:0.77rem;
    font-weight:600;
    color:var(--text);
    cursor:pointer;
    text-align:center;
    transition:all 0.2s ease;
    box-shadow:0 2px 8px var(--shadow);
    width:100%;
}
.quick-q-btn:hover {
    background:rgba(220,252,231,0.95);
    border-color:var(--green);
    transform:translateY(-2px);
    box-shadow:0 5px 16px var(--shadow-hover);
}

/* ── Streamlit buttons ── */
.stButton > button {
    font-family:'Outfit',sans-serif !important;
    border-radius:8px !important;
    transition:all 0.2s ease !important;
    width:100% !important;
}
[data-testid="stButton"] button[kind="primary"] {
    background:linear-gradient(135deg,var(--green),var(--teal)) !important;
    color:white !important; border:none !important;
    font-size:0.9rem !important; font-weight:700 !important;
    padding:0.62rem 1.5rem !important;
    box-shadow:0 4px 16px rgba(22,163,74,0.35) !important;
    border-radius:10px !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 9px 24px rgba(22,163,74,0.45) !important;
    filter:brightness(1.06) !important;
}
[data-testid="stButton"] button:not([kind="primary"]) {
    background:white !important; color:var(--text) !important;
    border:1.5px solid var(--border-dark) !important;
    font-size:0.76rem !important; font-weight:600 !important;
    padding:0.5rem 0.5rem !important;
    box-shadow:0 2px 8px var(--shadow) !important;
}
[data-testid="stButton"] button:not([kind="primary"]):hover {
    background:rgba(220,252,231,0.95) !important;
    border-color:var(--green) !important;
    transform:translateY(-2px) !important;
    box-shadow:0 5px 16px var(--shadow-hover) !important;
}

/* ── Send button style for chat ── */
.send-wrap {
    display:flex; gap:0.5rem; align-items:center; margin-top:0.5rem;
}
.send-input {
    flex:1;
    border:1.5px solid var(--border-dark);
    border-radius:10px;
    padding:0.55rem 0.85rem;
    font-family:'Outfit',sans-serif;
    font-size:0.83rem;
    color:var(--text);
    background:white;
    outline:none;
    transition:border-color 0.2s,box-shadow 0.2s;
}
.send-input:focus { border-color:var(--green); box-shadow:0 0 0 3px rgba(22,163,74,0.18); }
.send-btn {
    background:linear-gradient(135deg,var(--green),var(--teal));
    color:white; border:none; border-radius:10px;
    padding:0.55rem 1.1rem;
    font-family:'Outfit',sans-serif;
    font-size:0.82rem; font-weight:700;
    cursor:pointer;
    box-shadow:0 4px 14px rgba(22,163,74,0.32);
    transition:all 0.2s ease;
    white-space:nowrap;
}
.send-btn:hover { transform:translateY(-2px); box-shadow:0 7px 20px rgba(22,163,74,0.42); filter:brightness(1.05); }

/* ── Chat input (fallback) ── */
[data-testid="stChatInput"] { position:relative!important; z-index:999!important; }
[data-testid="stChatInput"] textarea {
    font-family:'Outfit',sans-serif !important;
    font-size:0.84rem !important;
    border-radius:10px !important;
    border:1.5px solid var(--border-dark) !important;
    background:white !important;
    color:var(--text) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color:var(--green) !important;
    box-shadow:0 0 0 3px rgba(22,163,74,0.18) !important;
}

/* Spinner */
.stSpinner > div { border-top-color:var(--green) !important; }

/* Footer */
.footer { text-align:center; padding:0.7rem 0 0.2rem; font-size:0.7rem; color:var(--muted); line-height:1.9; }
</style>
""", unsafe_allow_html=True)

# ── Config ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_DIR   = BASE_DIR / "models"
CHATBOT_DIR = BASE_DIR / "src" / "chatbot_data"
TRAIN_DIR   = BASE_DIR / "data" / "train"

CLASS_NAMES  = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
IDX_TO_CLASS = {i: n for i, n in enumerate(CLASS_NAMES)}

import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"

CLASS_INFO = {
    "glass":                {"emoji":"🍶","color":"#1d4ed8","bg":"#dbeafe","bin":"blue","kabadiwala":True,
                             "tip":"Rinse and recycle — glass is 100% recyclable forever",
                             "env":"Saves energy to power a laptop for 25 mins per bottle",
                             "tips":["Remove metal lids before recycling","Rinse clean before placing in bin","Never mix broken glass with recyclables"]},
    "hazardous":            {"emoji":"⚠️","color":"#dc2626","bg":"#fee2e2","bin":"red","kabadiwala":False,
                             "tip":"Never in regular trash — authorized collection only",
                             "env":"One battery can pollute 1,000 litres of groundwater",
                             "tips":["Drop at Croma or Reliance Digital","Contact CPCB for collection drives","Never burn or bury hazardous waste"]},
    "metal":                {"emoji":"🥫","color":"#374151","bg":"#f3f4f6","bin":"blue","kabadiwala":True,
                             "tip":"Highly valuable — kabadiwala will pay for this",
                             "env":"Recycling aluminium uses 95% less energy than new production",
                             "tips":["Clean before selling to kabadiwala","Crush cans to save space","Separate aluminium from steel"]},
    "non_recyclable_trash": {"emoji":"🗑️","color":"#57534e","bg":"#f5f5f4","bin":"black","kabadiwala":False,
                             "tip":"Goes to landfill — reduce consumption first",
                             "env":"Takes hundreds of years to decompose in landfill",
                             "tips":["Choose recyclable packaging","Check if item can be repaired","Donate clothes before discarding"]},
    "organic_biodegradable":{"emoji":"🌱","color":"#15803d","bg":"#dcfce7","bin":"green","kabadiwala":False,
                             "tip":"Compost at home — makes excellent fertilizer",
                             "env":"Composting prevents methane — 25x worse than CO2",
                             "tips":["Start a compost bin at home","Use the green bin for wet waste","Try vermicomposting"]},
    "paper_cardboard":      {"emoji":"📦","color":"#b45309","bg":"#fef3c7","bin":"blue","kabadiwala":True,
                             "tip":"Flatten and recycle — kabadiwala pays Rs 10-15 per kg",
                             "env":"Recycling one tonne of paper saves 17 trees",
                             "tips":["Remove tape and staples","Keep dry — wet paper loses value","Flatten cardboard boxes"]},
    "recyclable_plastic":   {"emoji":"♻️","color":"#7c3aed","bg":"#ede9fe","bin":"blue","kabadiwala":True,
                             "tip":"Check the recycle number — types 1 and 2 are best",
                             "env":"Only 9% of all plastic ever made has been recycled",
                             "tips":["Rinse containers before recycling","No plastic bags in bins","Look for resin code 1 (PET) or 2 (HDPE)"]},
}

BINS = [
    {"id":"green",      "icon":"🟢","name":"Green Bin",  "desc":"Wet/Organic"},
    {"id":"blue",       "icon":"🔵","name":"Blue Bin",   "desc":"Dry/Recyclable"},
    {"id":"black",      "icon":"⚫","name":"Black Bin",  "desc":"General Waste"},
    {"id":"red",        "icon":"🔴","name":"Red Bin",    "desc":"Hazardous"},
    {"id":"ewaste",     "icon":"📱","name":"E-Waste",    "desc":"Electronics"},
    {"id":"kabadiwala", "icon":"🏪","name":"Kabadiwala", "desc":"Scrap Dealer"},
]

@st.cache_resource
def load_model():
    def build(n=7):
        m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        for p in m.parameters(): p.requires_grad = False
        inf = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(inf, n))
        return m
    ck = torch.load(MODEL_DIR/"best_model.pth", map_location="cpu")
    m  = build(); m.load_state_dict(ck['model_state']); m.eval()
    return m

@st.cache_resource
def load_chatbot():
    emb = SentenceTransformer('all-MiniLM-L6-v2')
    idx = faiss.read_index(str(CHATBOT_DIR/"waste_index.faiss"))
    kb  = json.load(open(CHATBOT_DIR/"knowledge_base.json"))
    cli = Groq(api_key=GROQ_API_KEY)
    return emb, idx, kb, cli

transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ToTensorV2()
])

class GradCAM:
    def __init__(self,m,l):
        self.model=m; self.grads=None; self.acts=None
        l.register_forward_hook(lambda m,i,o:setattr(self,'acts',o.detach()))
        l.register_backward_hook(lambda m,gi,go:setattr(self,'grads',go[0].detach()))
    def run(self,t):
        t=t.requires_grad_(True); self.model.zero_grad()
        out=self.model(t); idx=out.argmax(1).item()
        out[0,idx].backward(retain_graph=True)
        if self.grads is None: return None,idx,out.softmax(1)[0].detach()
        w=self.grads.mean(dim=[2,3],keepdim=True)
        cam=torch.relu((w*self.acts).sum(1)).squeeze().cpu().numpy()
        cam=(cam-cam.min())/(cam.max()+1e-8); return cam,idx,out.softmax(1)[0].detach()

def make_square(img_pil, size=280):
    """Crop center square and resize to fixed size"""
    img = img_pil.convert("RGB")
    w,h = img.size
    m   = min(w,h)
    l   = (w-m)//2; t=(h-m)//2
    img = img.crop((l,t,l+m,t+m))
    return img.resize((size,size), Image.LANCZOS)

def predict(img_pil, model):
    arr=np.array(img_pil.convert("RGB"))
    t=transform(image=arr)["image"].unsqueeze(0)
    gc=GradCAM(model,model.features[8])
    cam,pid,probs=gc.run(t)
    r=cv2.resize(arr,(224,224))
    if cam is not None:
        hm=cv2.applyColorMap(np.uint8(255*cv2.resize(cam,(224,224))),cv2.COLORMAP_JET)
        hm=cv2.cvtColor(hm,cv2.COLOR_BGR2RGB)
        ov=(0.45*hm+0.55*r).astype(np.uint8)
    else: ov=r
    return pid,probs,ov

def get_answer(q,detected,emb,idx,kb,cli):
    qv=emb.encode([q],convert_to_numpy=True); faiss.normalize_L2(qv)
    _,ids=idx.search(qv.astype(np.float32),3)
    ctx="\n\n".join([f"[{kb[i]['class']}|{kb[i]['category']}]\n{kb[i]['text']}" for i in ids[0]])
    prompt=f"""You are EcoLens AI, India waste management expert. Be concise (2-3 sentences).
Mention kabadiwala/CPCB/Swachh Bharat when relevant. Answer ONLY from context.
Detected waste: {detected}\nContext: {ctx}\nQuestion: {q}\nAnswer:"""
    r=cli.chat.completions.create(model=GROQ_MODEL,messages=[{"role":"user","content":prompt}],max_tokens=300,temperature=0.3)
    return r.choices[0].message.content

# ══════════════════════════════════
model = load_model()
emb,fidx,kb,gcli = load_chatbot()

# ── Shared column ratio — SAME for all rows ──
# 2% | 96% | 2%  → ensures everything lines up perfectly
PAD = 0.02
CTR = 0.96

# ── HERO ──────────────────────────────────────────────────────
_,hc,_ = st.columns([PAD,CTR,PAD])
with hc:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">♻️ AI-Powered Waste Intelligence</div>
        <h1 class="hero-title">EcoLens</h1>
        <p class="hero-sub">Upload any waste photo — AI classifies it, shows what it focused on, and guides eco-friendly disposal</p>
        <div class="hero-pills">
            <span class="hpill">🎯 Accuracy <b>89.6%</b></span>
            <span class="hpill">🏷️ <b>7</b> Categories</span>
            <span class="hpill">🖼️ <b>13,740</b> Images</span>
            <span class="hpill">🧠 EfficientNet-B4 + <b>Grad-CAM</b></span>
            <span class="hpill">💬 RAG <b>Chatbot</b></span>
        </div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

# ── UPLOAD BOX ────────────────────────────────────────────────
_,uc,_ = st.columns([PAD,CTR,PAD])
with uc:
    st.markdown('<div class="card"><div class="card-hdr">📸 Upload Waste Image</div><div class="card-body">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop your waste image here — JPG, JPEG or PNG",
        type=["jpg","jpeg","png"],
        label_visibility="visible"
    )
    st.markdown("</div></div>", unsafe_allow_html=True)

    if uploaded:
        if st.button("🔍 Analyse with AI", type="primary", key="analyse"):
            with st.spinner("🧠 Running AI classification..."):
                image = Image.open(uploaded)
                pid,probs,overlay = predict(image,model)
            st.session_state.update({
                'pid':pid,'probs':probs,'overlay':overlay,
                'image':image,'done':True,'messages':[]
            })

# ── IMAGE PAIR — same width as upload box ─────────────────────
if st.session_state.get('done'):
    _,ic,_ = st.columns([PAD,CTR,PAD])
    with ic:
        # Two equal columns inside the same CTR width
        img_col, gcam_col = st.columns(2, gap="medium")

        with img_col:
            st.markdown('<div class="card"><div class="card-hdr">🖼️ Uploaded Image</div><div class="card-body" style="display:flex;justify-content:center;padding:0.6rem">', unsafe_allow_html=True)
            sq = make_square(st.session_state['image'], 280)
            st.image(sq, use_container_width=False, width=280)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with gcam_col:
            st.markdown('<div class="card"><div class="card-hdr">🔥 AI Focus Map (Grad-CAM)</div><div class="card-body" style="display:flex;flex-direction:column;align-items:center;padding:0.6rem">', unsafe_allow_html=True)
            ov_sq = Image.fromarray(st.session_state['overlay'])
            ov_sq = make_square(ov_sq, 280)
            st.image(ov_sq, use_container_width=False, width=280)
            st.markdown("""<div style="font-size:0.66rem;color:#4b5563;text-align:center;margin-top:0.3rem">
                🔴 High attention &nbsp;·&nbsp; 🔵 Low attention
            </div></div></div>""", unsafe_allow_html=True)

elif uploaded and not st.session_state.get('done'):
    _,pc,_ = st.columns([PAD,CTR,PAD])
    with pc:
        st.markdown('<div class="card"><div class="card-hdr">🖼️ Preview</div><div class="card-body" style="display:flex;justify-content:center;padding:0.6rem">', unsafe_allow_html=True)
        pq = make_square(Image.open(uploaded), 280)
        st.image(pq, use_container_width=False, width=280)
        st.markdown("</div></div>", unsafe_allow_html=True)

# ── RESULTS ───────────────────────────────────────────────────
if st.session_state.get('done'):
    st.markdown(f'<div style="margin:0 2%"><div class="divider"></div></div>', unsafe_allow_html=True)

    pid        = st.session_state['pid']
    probs      = st.session_state['probs']
    pred_class = IDX_TO_CLASS[pid]
    conf       = probs[pid].item()*100
    info       = CLASS_INFO.get(pred_class,{})

    # Classification result
    _,rc,_ = st.columns([PAD,CTR,PAD])
    with rc:
        st.markdown(f"""
        <div class="card">
            <div class="card-hdr">🎯 Classification Result</div>
            <div class="card-body">
                <div class="res-badge" style="background:{info.get('bg','#dcfce7')};border-color:{info.get('color','#16a34a')}60">
                    <span class="res-emoji">{info.get('emoji','♻️')}</span>
                    <div>
                        <div class="res-name" style="color:{info.get('color','#16a34a')}">{pred_class.replace('_',' ').title()}</div>
                        <div class="res-conf" style="color:{info.get('color','#16a34a')}">{conf:.1f}% confidence &nbsp;·&nbsp; EfficientNet-B4</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Bins — 6 horizontal
    _,bc,_ = st.columns([PAD,CTR,PAD])
    with bc:
        correct = info.get('bin','black')
        kabadi  = info.get('kabadiwala',False)
        bh = '<div class="card"><div class="card-hdr">🗑️ Which Bin?</div><div class="card-body"><div class="bin-row">'
        for b in BINS:
            is_ok     = (b['id']==correct) or (b['id']=='kabadiwala' and kabadi)
            cls       = "ok" if is_ok else "no"
            badge_cls = "badge-ok" if is_ok else "badge-no"
            badge_sym = "✓" if is_ok else "✗"
            bh += f'<div class="bin-item {cls}"><div class="{badge_cls}">{badge_sym}</div><span class="bin-ico">{b["icon"]}</span><span class="bin-name">{b["name"]}</span><span class="bin-desc">{b["desc"]}</span></div>'
        bh += '</div></div></div>'
        st.markdown(bh, unsafe_allow_html=True)

    # Confidence + Tips
    _,lc,rc2,_ = st.columns([PAD,0.47,0.49,PAD])
    with lc:
        top5p,top5i = probs.topk(min(5,len(IDX_TO_CLASS)))
        h = '<div class="card"><div class="card-hdr">📊 Confidence Scores</div><div class="card-body">'
        for p,i in zip(top5p,top5i):
            c2=IDX_TO_CLASS[i.item()]; in2=CLASS_INFO.get(c2,{}); pct=p.item()*100
            h+=f'<div class="prob-row"><div class="prob-lbl"><span>{in2.get("emoji","")} {c2.replace("_"," ")}</span><span style="color:#059669;font-weight:700">{pct:.1f}%</span></div><div class="prob-trk"><div class="prob-fil" style="width:{pct}%"></div></div></div>'
        h+='</div></div>'
        st.markdown(h, unsafe_allow_html=True)

    with rc2:
        tips=[("🌿",info.get('tip','')),("🌍",info.get('env',''))]+[("✅",t) for t in info.get('tips',[])]
        h2='<div class="card"><div class="card-hdr">💡 Disposal Tips</div><div class="card-body">'
        for ico,tip in tips[:5]:
            if tip: h2+=f'<div class="tip-row"><span class="tip-ico">{ico}</span><span>{tip}</span></div>'
        if kabadi:
            h2+='<div class="kabadi"><span style="font-size:1.05rem">🏪</span><div><div class="kabadi-t">Kabadiwala Will Buy This!</div><div class="kabadi-s">Earn money while recycling — find your local scrap dealer</div></div></div>'
        h2+='</div></div>'
        st.markdown(h2, unsafe_allow_html=True)

# ── CHATBOT ───────────────────────────────────────────────────
st.markdown(f'<div style="margin:0 2%"><div class="divider"></div></div>', unsafe_allow_html=True)

_,cc,_ = st.columns([PAD,CTR,PAD])
with cc:
    if not st.session_state.get('done'):
        st.markdown("""
        <div class="card" style="text-align:center;padding:1.2rem">
            <div style="font-size:1.6rem;margin-bottom:0.4rem">🤖</div>
            <div style="font-size:0.84rem;color:#4b5563;font-weight:500">
                Upload and analyse a waste image above to unlock personalised AI guidance
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        detected = IDX_TO_CLASS[st.session_state['pid']]
        info     = CLASS_INFO.get(detected,{})

        # Quick questions — grid, no gaps
        st.markdown('<div style="font-size:0.73rem;font-weight:700;color:#0f3d22;margin-bottom:0.38rem;letter-spacing:0.02em">💬 Quick questions — click to ask:</div>', unsafe_allow_html=True)

        q1,q2,q3 = st.columns(3, gap="small")
        for col,q in zip([q1,q2,q3],[
            "How do I safely dispose of this?",
            "What is the environmental impact?",
            "Can kabadiwala take this item?"
        ]):
            with col:
                if st.button(q, key=f"qq_{q[:12]}"):
                    msgs = st.session_state.get('messages',[])
                    msgs.append({"role":"user","content":q})
                    with st.spinner("🤔 Thinking..."):
                        ans = get_answer(q,detected,emb,fidx,kb,gcli)
                    msgs.append({"role":"assistant","content":ans})
                    st.session_state['messages']=msgs
                    st.rerun()

        # Chat history
        if st.session_state.get('messages'):
            st.markdown('<div class="card" style="margin-top:0.5rem"><div class="card-body"><div class="chat-scroll">', unsafe_allow_html=True)
            for msg in st.session_state['messages']:
                role = 'chat-u' if msg['role']=='user' else 'chat-a'
                icon = '👤' if msg['role']=='user' else '🤖'
                st.markdown(f'<div class="{role}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)
            st.markdown('</div></div></div>', unsafe_allow_html=True)

        # Type own question with Send button
        st.markdown('<div style="font-size:0.72rem;font-weight:700;color:#0f3d22;margin:0.6rem 0 0.28rem;letter-spacing:0.02em">✏️ Type your own question:</div>', unsafe_allow_html=True)

        inp_col, btn_col = st.columns([0.82, 0.18], gap="small")
        with inp_col:
            user_text = st.text_input(
                "",
                placeholder="Ask anything about disposal, recycling, or environment...",
                label_visibility="collapsed",
                key="chat_input"
            )
        with btn_col:
            send = st.button("Send ➤", key="send_btn")

        if send and user_text.strip():
            msgs = st.session_state.get('messages',[])
            msgs.append({"role":"user","content":user_text.strip()})
            with st.spinner("🤔 Thinking..."):
                ans = get_answer(user_text.strip(),detected,emb,fidx,kb,gcli)
            msgs.append({"role":"assistant","content":ans})
            st.session_state['messages']=msgs
            st.rerun()

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    EcoLens — Built with <b>PyTorch</b> · <b>EfficientNet-B4</b> · <b>Grad-CAM</b> · <b>FAISS</b> · <b>Groq LLM</b> · <b>Streamlit</b><br>
    Test Accuracy <b style="color:#059669">89.6%</b> · 13,740 training images · 7 waste categories
</div>
""", unsafe_allow_html=True)
