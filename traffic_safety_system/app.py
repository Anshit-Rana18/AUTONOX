"""
AUTONOX - Streamlit front end.
Run from the project root:   streamlit run app.py
"""
import base64
import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime

from PIL import Image

import cv2
import streamlit as st
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # configs/models use relative paths

from video.overlay import DEFAULT_OPTS, OPT_ORDER, render_video

PROJECT_NAME = "AUTONOX"
TAGLINE = "AI Traffic Safety Intelligence"
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
UPLOAD, URL = "⬆  Upload a video…", "🔗  RTSP / URL…"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "img", "logo.png")        # put your logo here
try:
    ICON = Image.open(LOGO_PATH)
    with open(LOGO_PATH, "rb") as _f:
        LOGO_B64 = base64.b64encode(_f.read()).decode()
except Exception:                                            # no logo file: fall back to the emoji
    ICON, LOGO_B64 = "🛰️", None

st.set_page_config(page_title=f"{PROJECT_NAME} · Traffic Safety", page_icon=ICON, layout="wide")

# ------------------------------------------------------------------------ theme
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{--bg:#06101a;--card:rgba(255,255,255,.045);--line:rgba(160,220,255,.11);--txt:#e6f1fa;--mut:#7f9bb3;
--cy:#22d3ee;--vi:#38bdf8;--pk:#fb7185;--or:#fb923c;--gr:#34d399;--am:#fbbf24;}
.stApp,.stApp *:not([data-testid^="stIcon"]):not([class*="material"]):not([class*="Material"]){font-family:'Space Grotesk',system-ui,Segoe UI,sans-serif;}
[data-testid^="stIcon"],[class*="material-symbols"],[class*="material-icons"],span[class*="Material"]{
font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
font-feature-settings:'liga';-webkit-font-feature-settings:'liga';letter-spacing:normal !important;}
.stApp{background:radial-gradient(1100px 600px at 12% -10%,rgba(34,211,238,.15),transparent 60%),
radial-gradient(900px 600px at 95% 0%,rgba(56,189,248,.13),transparent 60%),
radial-gradient(800px 500px at 50% 110%,rgba(45,212,191,.09),transparent 60%),var(--bg);color:var(--txt);}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.4rem;max-width:1500px;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#07121d,#0b1b2b);border-right:1px solid var(--line);}
section[data-testid="stSidebar"] *{color:var(--txt);}
.sb-title{font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:var(--mut);margin:1.1rem 0 .3rem;}
.stButton>button{width:100%;border-radius:12px;border:1px solid var(--line);background:var(--card);color:var(--txt);
font-weight:600;padding:.62rem 0;transition:.2s;}
.stButton>button:hover{border-color:var(--cy);box-shadow:0 0 18px rgba(34,211,238,.28);color:#fff;}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#06b6d4,#0ea5e9 55%,#14b8a6);border:none;color:#fff;
box-shadow:0 6px 24px rgba(14,165,233,.35);}
.stButton>button[kind="primary"]:hover{filter:brightness(1.12);box-shadow:0 8px 30px rgba(20,184,166,.45);}

.hero{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:1.1rem;
padding:1.15rem 1.5rem;border-radius:20px;border:1px solid var(--line);background:var(--card);backdrop-filter:blur(14px);
position:relative;overflow:hidden;}
.hero:before{content:"";position:absolute;inset:0;background:linear-gradient(120deg,rgba(34,211,238,.11),transparent 40%,rgba(20,184,166,.10));pointer-events:none;}
.brand{display:flex;align-items:center;gap:16px;position:relative;}
.logo{width:50px;height:50px;border-radius:14px;display:grid;place-items:center;font-size:1.5rem;
background:linear-gradient(135deg,#06b6d4,#0ea5e9 60%,#14b8a6);box-shadow:0 0 28px rgba(14,165,233,.5);}
.logo.has-img{background:none;box-shadow:none;width:96px;height:96px;}
.logo img{width:100%;height:100%;object-fit:contain;filter:drop-shadow(0 0 10px rgba(34,211,238,.5));}
.name{font-size:2.05rem;font-weight:700;letter-spacing:.22em;line-height:1;
background:linear-gradient(90deg,#67e8f9,#7dd3fc 55%,#5eead4);-webkit-background-clip:text;background-clip:text;color:transparent;}
.tag{color:var(--mut);font-size:.82rem;letter-spacing:.14em;text-transform:uppercase;margin-top:.35rem;}
.pill{display:inline-flex;align-items:center;gap:9px;padding:.45rem .95rem;border-radius:999px;font-size:.78rem;font-weight:600;
letter-spacing:.12em;border:1px solid var(--line);background:rgba(0,0,0,.35);position:relative;}
.dot{width:9px;height:9px;border-radius:50%;background:var(--mut);}
.live .dot{background:var(--gr);box-shadow:0 0 0 0 rgba(52,211,153,.7);animation:pulse 1.4s infinite;}
.done .dot{background:var(--cy);box-shadow:0 0 10px var(--cy);} .stop .dot{background:var(--am);}
.err .dot{background:var(--pk);}
@keyframes pulse{70%{box-shadow:0 0 0 10px rgba(52,211,153,0);}100%{box-shadow:0 0 0 0 rgba(52,211,153,0);}}

.chips{display:flex;gap:8px;flex-wrap:wrap;margin:.2rem 0 1rem;}
.chip{padding:.28rem .7rem;border-radius:999px;font-size:.72rem;letter-spacing:.08em;border:1px solid var(--line);color:var(--mut);}
.chip.on{color:#fff;border-color:rgba(34,211,238,.5);background:rgba(34,211,238,.12);box-shadow:0 0 14px rgba(34,211,238,.18);}

.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:14px;}
@media(max-width:1100px){.grid{grid-template-columns:repeat(3,1fr);}}
.m{padding:.9rem 1rem;border-radius:16px;border:1px solid var(--line);background:var(--card);backdrop-filter:blur(12px);}
.m .l{font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:var(--mut);}
.m .v{font-family:'JetBrains Mono',monospace;font-size:1.65rem;font-weight:600;margin-top:.2rem;}
.m .v small{font-size:.75rem;color:var(--mut);font-weight:400;margin-left:4px;}
.c1 .v{color:var(--cy)}.c2 .v{color:var(--or)}.c3 .v{color:var(--pk)}.c4 .v{color:var(--gr)}.c5 .v{color:var(--vi)}.c6 .v{color:#fff}
.bar{height:5px;border-radius:5px;background:rgba(255,255,255,.07);margin:2px 0 14px;overflow:hidden;}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#22d3ee,#38bdf8,#2dd4bf);box-shadow:0 0 12px rgba(34,211,238,.6);}

.panel{border-radius:18px;border:1px solid var(--line);background:var(--card);backdrop-filter:blur(12px);padding:.9rem;}
.ptitle{font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:var(--mut);margin:.1rem .2rem .6rem;
display:flex;justify-content:space-between;}
.stage{height:640px;display:flex;align-items:center;justify-content:center;overflow:hidden;background:#02070d;
border-radius:12px;box-shadow:0 0 0 1px var(--line),0 10px 40px rgba(0,0,0,.5);}
.stage img{max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;display:block;}
.stApp video{height:640px !important;width:100% !important;object-fit:contain;background:#02070d;border-radius:12px;
box-shadow:0 0 0 1px var(--line);}
div[data-testid="stVideo"]{max-height:640px;}
.empty{display:grid;place-items:center;height:640px;border-radius:12px;color:var(--mut);text-align:center;
background:repeating-linear-gradient(0deg,rgba(160,220,255,.025) 0 1px,transparent 1px 28px),
repeating-linear-gradient(90deg,rgba(160,220,255,.025) 0 1px,transparent 1px 28px);border:1px dashed var(--line);}
.empty b{display:block;font-size:1.05rem;color:var(--txt);margin-bottom:.3rem;}
.log{max-height:560px;overflow-y:auto;padding-right:4px;}
.ev{display:flex;gap:11px;align-items:center;padding:.6rem;margin-bottom:8px;border-radius:13px;border:1px solid var(--line);
background:rgba(0,0,0,.28);}
.ev img{height:54px;width:76px;object-fit:cover;border-radius:8px;border:1px solid var(--line);}
.ev .ph{height:54px;width:76px;border-radius:8px;background:rgba(255,255,255,.05);}
.ev .b{flex:1;min-width:0;}
.badge{display:inline-block;font-size:.64rem;font-weight:700;letter-spacing:.12em;padding:.15rem .5rem;border-radius:6px;}
.badge.lane{color:#fed7aa;background:rgba(251,146,60,.18);border:1px solid rgba(251,146,60,.5);}
.badge.speed{color:#fecdd3;background:rgba(251,113,133,.18);border:1px solid rgba(251,113,133,.5);}
.ev .d{font-size:.88rem;margin-top:.3rem;} .ev .t{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--mut);}
.plate{font-family:'JetBrains Mono',monospace;font-weight:600;font-size:.78rem;padding:.22rem .55rem;border-radius:7px;white-space:nowrap;}
.plate.ok{color:#a7f3d0;background:rgba(52,211,153,.14);border:1px solid rgba(52,211,153,.5);}
.plate.rev{color:#fde68a;background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.45);}
.plate.no{color:var(--mut);border:1px dashed var(--line);}
button[data-baseweb="tab"]{font-weight:600;letter-spacing:.06em;}
button[data-baseweb="tab"][aria-selected="true"]{color:#fff !important;}
div[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,#22d3ee,#38bdf8,#2dd4bf) !important;height:3px;}
.logmeta{display:flex;gap:8px;flex-wrap:wrap;margin:.1rem 0 .8rem;}
.none{color:var(--mut);text-align:center;padding:2.2rem 0;font-size:.9rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- stream catalogue
@st.cache_data(show_spinner=False)
def discover():
    videos = sorted(p for p in glob.glob("data/videos/*") if p.lower().endswith(VIDEO_EXT))
    lane_files, speed_files = [], []
    for p in sorted(glob.glob("configs/*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if isinstance(d, dict) and "lanes" in d:
            lane_files.append(p.replace("\\", "/"))
        elif isinstance(d, dict) and "image_points" in d:
            speed_files.append(p.replace("\\", "/"))
    # optional explicit catalogue: configs/streams.yaml
    #   - name: Highway A
    #     source: data/videos/test4.mp4      (file path, rtsp://..., or 0)
    #     lanes: configs/test4.json
    #     speed: configs/speed_test4.json
    named = []
    if os.path.exists("configs/streams.yaml"):
        named = yaml.safe_load(open("configs/streams.yaml")) or []
    return [v.replace("\\", "/") for v in videos], lane_files, speed_files, named


videos, lane_files, speed_files, named = discover()
stem = lambda p: os.path.splitext(os.path.basename(p))[0]

catalogue = {}                                     # label -> dict(source, lanes, speed)
for v in videos:
    s = stem(v)
    catalogue[os.path.basename(v)] = {
        "source": v,
        "lanes": next((l for l in lane_files if stem(l) == s), None),
        "speed": next((x for x in speed_files if stem(x) in (f"speed_{s}", s)), None)}
for n in named:
    catalogue[n["name"]] = {"source": n["source"], "lanes": n.get("lanes"), "speed": n.get("speed")}
labels = list(catalogue) + [UPLOAD, URL]


# ------------------------------------------------------------------------- helpers
def pill(kind, text):
    return f'<span class="pill {kind}"><span class="dot"></span>{text}</span>'


def hero(status_html):
    logo = (f'<div class="logo has-img"><img src="data:image/png;base64,{LOGO_B64}"/></div>'
            if LOGO_B64 else '<div class="logo">🛰️</div>')
    return (f'<div class="hero"><div class="brand">{logo}<div>'
            f'<div class="name">{PROJECT_NAME}</div><div class="tag">{TAGLINE}</div></div></div>'
            f'<div>{status_html}</div></div>')


def chips(lane, speed, plates):
    c = lambda on, t: f'<span class="chip {"on" if on else ""}">{t}</span>'
    return ('<div class="chips">' + c(lane, "LANE-CHANGE DETECTION") + c(speed, "SPEED ESTIMATION")
            + c(plates, "PLATE RECOGNITION") + "</div>")


def metrics_html(s, lane, speed, plates):
    avg = f'{s["avg_speed"]:.0f}<small>km/h</small>' if s["avg_speed"] is not None else "–"
    cards = [("c1", "Vehicles in view", f'{s["vehicles_now"]}<small>/ {s["unique"]} total</small>'),
             ("c2", "Lane changes", str(s["lane_events"]) if lane else "–"),
             ("c3", "Overspeed", str(s["speed_events"]) if speed else "–"),
             ("c4", "Plates confirmed", str(s["plates"]) if plates else "–"),
             ("c5", "Avg speed", avg if speed else "–"),
             ("c6", "Processing", f'{s["fps"]:.1f}<small>fps</small>')]
    html = '<div class="grid">' + "".join(
        f'<div class="m {c}"><div class="l">{l}</div><div class="v">{v}</div></div>' for c, l, v in cards) + "</div>"
    if s["progress"] is not None:
        html += f'<div class="bar"><i style="width:{min(100, s["progress"] * 100):.1f}%"></i></div>'
    return html


def frame_html(frame, max_w=1280, max_h=960):
    """Fit any aspect ratio (landscape or portrait) into the fixed stage."""
    h, w = frame.shape[:2]
    k = min(max_w / w, max_h / h, 1.0)
    if k < 1.0:
        frame = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    return f'<div class="stage"><img src="data:image/jpeg;base64,{b64}"/></div>'


def log_html(rows, plates_on):
    if not rows:
        return '<div class="none">No violations detected yet.</div>'
    out = []
    for r in reversed(rows[-40:]):
        badge = ('<span class="badge lane">LANE CHANGE</span>' if r["kind"] == "lane"
                 else '<span class="badge speed">OVERSPEED</span>')
        img = f'<img src="data:image/jpeg;base64,{r["thumb"]}"/>' if r["thumb"] else '<div class="ph"></div>'
        if not plates_on:
            plate = ""
        elif r["plate"]:
            ok = r["plate_status"] == "CONFIRMED"
            plate = f'<span class="plate {"ok" if ok else "rev"}">{r["plate"]}{"" if ok else " ?"}</span>'
        else:
            plate = '<span class="plate no">NO PLATE</span>'
        out.append(f'<div class="ev">{img}<div class="b">{badge}<div class="d">{r["detail"]}</div>'
                   f'<div class="t">{r["id"]} · track #{r["tid"]} · t={r["ts"]:.1f}s</div></div>{plate}</div>')
    return f'<div class="log">{"".join(out)}</div>'


def empty_feed(msg_title, msg):
    return f'<div class="empty"><div><b>{msg_title}</b>{msg}</div></div>'


REC_DIR = "data/outputs"


def to_browser_mp4(raw, out):
    """OpenCV writes MPEG-4 part 2, which browsers cannot play. Re-encode to H.264."""
    exe = None
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
    if not exe:
        return None
    cmd = [exe, "-y", "-i", raw, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    try:
        r = subprocess.run(cmd, capture_output=True)
    except Exception:
        return None
    return out if r.returncode == 0 and os.path.exists(out) else None


def finalize_recording(raw):
    """raw .avi/.mp4 from OpenCV -> playable mp4. Returns (path, playable)."""
    if not raw or not os.path.exists(raw) or os.path.getsize(raw) < 1024:
        return None, False
    final = raw.replace("_raw_", "").rsplit(".", 1)[0] + ".mp4"
    out = to_browser_mp4(raw, final)
    if out:
        os.remove(raw)
        return out, True
    return raw, False


def save_overlay(path, engine):
    """Per-frame overlay data next to the clean recording, so the look can change later."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"fps": engine.src_fps, "frames": engine.overlay_frames}, f)
    except Exception:
        pass


def ensure_render(clean_path, data_path, tag, o):
    """Clean recording + overlay data + display switches -> playable video (cached)."""
    bits = "".join("1" if o[k] else "0" for k in OPT_ORDER)
    final = os.path.join(REC_DIR, f"render_{tag}_{bits}.mp4").replace("\\", "/")
    if os.path.exists(final):
        return final, True
    raw = final.replace(".mp4", "_raw_.mp4")
    with st.spinner("Rendering the video with your display options…"):
        render_video(clean_path, data_path, raw, {**DEFAULT_OPTS, **o})
        out = to_browser_mp4(raw, final)
    if out:
        os.remove(raw)
        return out, True
    return raw, False


def show_recording(path, playable, key):
    if not path or not os.path.exists(path):
        return
    if playable:
        st.video(path)
    else:
        st.warning("Recording saved, but it can't be played in the browser until an H.264 encoder is "
                   "available. Run:  pip install imageio-ffmpeg   (then run the analysis again).")
    with open(path, "rb") as f:
        st.download_button("⬇  Download this video", f.read(), os.path.basename(path), "video/mp4", key=key)


def show_annotated(path, playable, data, tag, defaults, key):
    """Recorded video with the four display switches. Falls back to plain playback for old runs."""
    if not path or not os.path.exists(path):
        return
    if not data or not tag or not os.path.exists(data):
        show_recording(path, playable, key)
        return
    d = {**DEFAULT_OPTS, **(defaults or {})}
    r1, r2 = st.columns(2), st.columns(2)
    o = {"hide_lanes": r1[0].checkbox("Hide lane colors", d["hide_lanes"], key=f"{key}_hl"),
         "hide_speed_zone": r1[1].checkbox("Hide speed zone", d["hide_speed_zone"], key=f"{key}_hz"),
         "hide_outside": r2[0].checkbox("Hide vehicles outside lanes", d["hide_outside"], key=f"{key}_ho"),
         "hide_plates": r2[1].checkbox("Hide plates", d["hide_plates"], key=f"{key}_hp")}
    out, ok = ensure_render(path, data, tag, o)
    show_recording(out, ok, key=f"{key}_dl")


def delete_entry_files(e):
    names = [e.get("path"), e.get("data")]
    if e.get("tag"):
        names += glob.glob(os.path.join(REC_DIR, f"render_{e['tag']}_*.mp4"))
    for p in names:
        try:
            if p:
                os.remove(p)
        except OSError:
            pass


LOG_FILE = os.path.join(REC_DIR, "autonox_logs.json")


def save_logs(store):
    """Write the log list to disk so it survives a restart."""
    try:
        os.makedirs(REC_DIR, exist_ok=True)
        tmp = LOG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(store, f, default=str)
        os.replace(tmp, LOG_FILE)
    except Exception:
        pass


@st.cache_resource
def get_log_store():
    """Log list shared by every browser session. Loaded from disk when the app starts."""
    store = []
    try:
        if os.path.exists(LOG_FILE):
            for e in json.load(open(LOG_FILE, encoding="utf-8")):
                if os.path.exists(e.get("path", "")):      # skip runs whose video was deleted
                    store.append(e)
    except Exception:
        pass
    return store


def render_recordings(last):
    """Player for the current run (shown under the live view)."""
    if last and last.get("rec") and last["rec"][0]:
        st.caption("Change what is drawn with the switches. The video re-renders without running detection again.")
        show_annotated(last["rec"][0], last["rec"][1], last.get("data"), last.get("tag"),
                       last.get("opts"), key=f"cur_{last.get('tag', 'x')}")
    else:
        st.caption("The recording of your run will appear here when it finishes.")


def add_log(last, status=None):
    """Save a finished or stopped run into the video logs (once)."""
    if last.get("log_id") or not last.get("rec") or not last["rec"][0]:
        return
    store = get_log_store()
    entry = {"id": (max(e["id"] for e in store) + 1) if store else 1,
             "stream": last.get("choice", "?"), "mode": last.get("mode", ""),
             "started": last.get("started", ""),
             "status": status or ("Complete" if last["status"][0] == "done" else "Stopped"),
             "path": last["rec"][0], "playable": last["rec"][1],
             "data": last.get("data"), "tag": last.get("tag"), "opts": last.get("opts"),
             "stats": last["stats"], "rows": last["rows"], "plates": last["plates"],
             "lane": last["lane"], "speed": last["speed"]}
    store.append(entry)
    save_logs(store)
    last["log_id"] = entry["id"]


def render_logs():
    store = get_log_store()
    if not store:
        st.markdown('<div class="panel">' + empty_feed(
            "No video logs yet",
            "Every analysis you run is saved here automatically,<br>and is kept on disk.")
            + "</div>", unsafe_allow_html=True)
        return
    top1, top2 = st.columns([4, 1])
    top1.caption(f"{len(store)} saved run{'s' if len(store) != 1 else ''}. "
                 f"They are saved on disk and reappear when you reopen the app.")
    if top2.button("Clear all logs", key="clear_logs"):
        for e in store:
            delete_entry_files(e)
        store.clear()
        save_logs(store)
        st.session_state.pop("log_pick", None)
        st.rerun()

    entries = list(reversed(store))
    label = {e["id"]: f'Run {e["id"]}  ·  {e["stream"]}  ·  {e["mode"]}  ·  {e["started"]}  ·  {e["status"]}'
             for e in entries}
    pick = st.selectbox("Open a log", [e["id"] for e in entries], format_func=lambda i: label[i],
                        key="log_pick", label_visibility="collapsed")
    e = next(x for x in entries if x["id"] == pick)

    chips_html = "".join(f'<span class="chip on">{t}</span>' for t in
                         (e["stream"], e["mode"], e["status"].upper(), e["started"]))
    st.markdown(f'<div class="logmeta">{chips_html}</div>', unsafe_allow_html=True)
    st.markdown(metrics_html(e["stats"], e["lane"], e["speed"], e["plates"]), unsafe_allow_html=True)
    lc, rc = st.columns([2.15, 1], gap="medium")
    with lc:
        show_annotated(e["path"], e["playable"], e.get("data"), e.get("tag"), e.get("opts"),
                       key=f"log_{e['id']}")
        if st.button("🗑  Delete this log", key=f"del_{e['id']}"):
            delete_entry_files(e)
            store.remove(e)
            save_logs(store)
            st.session_state.pop("log_pick", None)
            st.rerun()
    with rc:
        st.markdown(f'<div class="panel"><div class="ptitle"><span>Violations log</span>'
                    f'<span>{len(e["rows"])} events</span></div>{log_html(e["rows"], e["plates"])}</div>',
                    unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading plate models…")
def get_plate_models(profile):
    from autonox_engine import load_cfg, load_plate_models
    cfg = load_cfg()
    cfg["plates"]["ocr_profile"] = profile
    return load_plate_models(cfg)


# ------------------------------------------------------------------ landing page
LANDING_CSS = """
<style>
section[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none !important;}
.land{text-align:center;padding:9vh 0 2.2rem;animation:rise .9s ease both;}
.l-logo{width:min(260px,50vw);height:auto;filter:drop-shadow(0 0 28px rgba(34,211,238,.45));}
.l-emoji{font-size:5rem;}
.l-name{font-size:clamp(2.6rem,6vw,4.4rem);font-weight:700;letter-spacing:.3em;padding-left:.3em;margin-top:1.2rem;
background:linear-gradient(90deg,#67e8f9,#7dd3fc 55%,#5eead4);-webkit-background-clip:text;background-clip:text;color:transparent;}
.l-tag{color:var(--mut);letter-spacing:.2em;text-transform:uppercase;font-size:.95rem;margin-top:.9rem;}
.l-note{text-align:center;color:var(--mut);font-size:.8rem;letter-spacing:.1em;margin-top:1.1rem;}
@keyframes rise{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}
.stButton>button{font-size:1.05rem;letter-spacing:.30em;text-transform:uppercase;border-radius:1200px;padding:.95rem 3rem;
box-shadow:0 8px 32px rgba(34,211,238,.35);}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 10px 40px rgba(34,211,238,.55);}
</style>
"""

S = st.session_state
if not S.get("entered"):
    st.markdown(LANDING_CSS, unsafe_allow_html=True)
    logo = (f'<img class="l-logo" src="data:image/png;base64,{LOGO_B64}"/>' if LOGO_B64
            else '<div class="l-emoji">🛰️</div>')
    st.markdown(f'<div class="land">{logo}<div class="l-name">{PROJECT_NAME}</div>'
                f'<div class="l-tag">{TAGLINE}</div></div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([2.2, 1 , 2])
    with mid:
        if st.button(" Enter  → ", type="primary", key="enter"):
            S["entered"] = True
            st.rerun()
    n_logs = len(get_log_store())
    if n_logs:
        st.markdown(f'<div class="l-note">{n_logs} saved video log{"s" if n_logs != 1 else ""} ready</div>',
                    unsafe_allow_html=True)
    st.stop()

# ----------------------------------------------------------------------- sidebar
base_cfg = yaml.safe_load(open("configs/default.yaml"))
D = {**DEFAULT_OPTS, **(base_cfg.get("overlay") or {})}

with st.sidebar:
    st.markdown(f'<div class="name" style="font-size:1.25rem;letter-spacing:.2em">{PROJECT_NAME}</div>'
                f'<div class="tag">Control panel</div>', unsafe_allow_html=True)
    if st.button("←  Home", key="home"):
        S["entered"] = False
        st.rerun()

    st.markdown('<div class="sb-title">Video stream</div>', unsafe_allow_html=True)
    choice = st.selectbox("Stream", labels, label_visibility="collapsed")

    source = lanes_default = speed_default = None
    if choice == UPLOAD:
        up = st.file_uploader("Video file", type=[e.strip(".") for e in VIDEO_EXT], label_visibility="collapsed")
        if up is not None:
            tmp = os.path.join(tempfile.gettempdir(), f"autonox_{up.name}")
            if not os.path.exists(tmp) or os.path.getsize(tmp) != up.size:
                open(tmp, "wb").write(up.getbuffer())
            source = tmp
    elif choice == URL:
        url = st.text_input("Stream URL", placeholder="rtsp://user:pass@ip:554/stream", label_visibility="collapsed")
        source = url.strip() or None
    else:
        entry = catalogue[choice]
        source, lanes_default, speed_default = entry["source"], entry["lanes"], entry["speed"]
        if isinstance(source, str) and source.isdigit():
            source = int(source)

    st.markdown('<div class="sb-title">Detection mode</div>', unsafe_allow_html=True)
    MODES = ["Lane changes", "Speed", "Lane + Speed", "Full: Lane + Speed + Plates"]
    mode = st.radio("Mode", MODES, index=3, label_visibility="collapsed")
    do_lane = mode in (MODES[0], MODES[2], MODES[3])
    do_speed = mode in (MODES[1], MODES[2], MODES[3])
    do_plates = mode == MODES[3]
    if not do_plates:
        do_plates = st.checkbox("Also read plates", value=False)

    st.markdown('<div class="sb-title">Calibration</div>', unsafe_allow_html=True)
    key = f"{choice}"
    NONE = "(none)"
    lanes_path = speed_path = None
    if do_lane:
        opts_l = [NONE] + lane_files
        i = opts_l.index(lanes_default) if lanes_default in opts_l else 0
        sel = st.selectbox("Lane calibration", opts_l, index=i, key=f"lane_{key}")
        lanes_path = None if sel == NONE else sel
    if do_speed:
        opts_s = [NONE] + speed_files
        i = opts_s.index(speed_default) if speed_default in opts_s else 0
        sel = st.selectbox("Speed calibration", opts_s, index=i, key=f"speed_{key}")
        speed_path = None if sel == NONE else sel

    st.markdown('<div class="sb-title">Parameters</div>', unsafe_allow_html=True)
    limit = st.slider("Speed limit (km/h)", 10, 150, int(base_cfg["speed"].get("limit_kmh", 50)), 5,
                      disabled=not do_speed)
    conf = st.slider("Vehicle detection confidence", 0.10, 0.90, float(base_cfg["detector"]["conf"]), 0.05)
    profile = None
    if do_plates:
        profiles = list(base_cfg["plates"].get("ocr_profiles", {"uk": {}}))
        cur = base_cfg["plates"].get("ocr_profile", profiles[0])
        profile = st.selectbox("Plate format", profiles, index=profiles.index(cur) if cur in profiles else 0)

    st.markdown('<div class="sb-title">Display</div>', unsafe_allow_html=True)
    hide_lanes = st.checkbox("Hide lane colors", value=bool(D["hide_lanes"]), disabled=not do_lane)
    hide_zone = st.checkbox("Hide speed zone", value=bool(D["hide_speed_zone"]), disabled=not do_speed)
    hide_outside = st.checkbox("Hide vehicles outside lanes", value=bool(D["hide_outside"]), disabled=not do_lane)
    hide_plates = st.checkbox("Hide plates", value=bool(D["hide_plates"]), disabled=not do_plates)
    st.caption("These apply when you press Start. Changing one during a run restarts the page, which "
               "stops the run. After it finishes, the same switches sit under the video and re-render it.")
    if st.button("↻  Rescan files"):
        discover.clear()
        st.rerun()

    opts = {"hide_lanes": hide_lanes, "hide_speed_zone": hide_zone,
            "hide_outside": hide_outside, "hide_plates": hide_plates, "show_reason": False}

    st.markdown("<br>", unsafe_allow_html=True)
    start = st.button("▶  Start analysis", type="primary")
    c1, c2 = st.columns(2)
    with c1:
        st.button("■  Stop")    # any click reruns the app, which halts the running analysis
    with c2:
        discard = st.button("✕  Clear")     # clears the screen only; the run stays in Video logs

# --------------------------------------------------------------------- main page
S = st.session_state


def finish_pending(status=None):
    """Close out a run that was stopped part-way and save it to the video logs."""
    last = S.get("last")
    if not last:
        return
    if S.get("pending_raw"):
        time.sleep(0.8)                                      # let the stopped run close its file
        with st.spinner("Saving the partial recording…"):
            last["rec"] = finalize_recording(S["pending_raw"])
        S["pending_raw"] = None
    add_log(last, status)


if discard:
    finish_pending()                       # keep whatever was recorded in the video logs
    S["last"], S["pending_raw"] = None, None
    st.toast("Screen cleared. The run is kept in Video logs.")

hero_ph = st.empty()
chips_ph = st.empty()
tab_live, tab_logs = st.tabs(["◉   Live analysis", "▤   Video logs"])
with tab_live:
    metrics_ph = st.empty()
    left, right = st.columns([2.15, 1], gap="medium")
    with left:
        feed_ph = st.empty()
        st.markdown('<div class="ptitle" style="margin-top:1.2rem"><span>Recorded analysis</span></div>',
                    unsafe_allow_html=True)
        rec_box = st.container()
    with right:
        log_ph = st.empty()
logs_ph = tab_logs.empty()

chips_ph.markdown(chips(do_lane, do_speed, do_plates), unsafe_allow_html=True)


def paint(status, s=None, frame=None, rows=None, plates_on=False, lane=False, speed=False):
    kind, text = status
    hero_ph.markdown(hero(pill(kind, text)), unsafe_allow_html=True)
    if s is not None:
        metrics_ph.markdown(metrics_html(s, lane, speed, plates_on), unsafe_allow_html=True)
    if frame is not None:
        feed_ph.markdown(f'<div class="panel"><div class="ptitle"><span>Live analysis</span>'
                         f'<span>{choice}</span></div>{frame}</div>', unsafe_allow_html=True)
    if rows is not None:
        log_ph.markdown(f'<div class="panel"><div class="ptitle"><span>Violations log</span>'
                        f'<span>{len(rows)} events</span></div>{log_html(rows, plates_on)}</div>',
                        unsafe_allow_html=True)


def error(msg):
    paint(("err", "ERROR"))
    feed_ph.markdown(f'<div class="panel">{empty_feed("Cannot start", msg)}</div>', unsafe_allow_html=True)


if start:
    # ---- validation
    if source is None or source == "":
        error("Choose or upload a video, or enter a stream URL.")
    elif do_lane and not lanes_path:
        error("Lane detection needs a lane calibration file. Pick one in the sidebar.")
    elif do_speed and not speed_path:
        error("Speed detection needs a speed calibration file. Pick one in the sidebar.")
    else:
        try:
            from autonox_engine import AutonoxEngine, load_cfg
            cfg = load_cfg()
            cfg["speed"]["limit_kmh"] = limit
            cfg["detector"]["conf"] = conf
            cfg["overlay"] = dict(opts)
            models = None
            if do_plates:
                cfg["plates"]["ocr_profile"] = profile
                models = get_plate_models(profile)
            engine = AutonoxEngine(cfg, source, lanes_path, speed_path,
                                   do_lane, do_speed, do_plates, models)
            paint(("live", "LIVE"), engine.stats(), None, [], do_plates, do_lane, do_speed)
            feed_ph.markdown(f'<div class="panel">{empty_feed("Warming up the models…", "First frame is on its way")}</div>',
                             unsafe_allow_html=True)
            S["last"] = None
            logs_ph.caption("Video logs open when this run ends.")
            mode_label = " + ".join(n for n, on in (("Lane", do_lane), ("Speed", do_speed), ("Plates", do_plates)) if on)
            started = datetime.now().strftime("%H:%M:%S")
            os.makedirs(REC_DIR, exist_ok=True)
            tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]        # milliseconds keep names unique
            raw_path = os.path.join(REC_DIR, f"autonox_{tag}_raw_.mp4").replace("\\", "/")
            data_path = os.path.join(REC_DIR, f"autonox_{tag}.overlay.json").replace("\\", "/")
            writer = None
            S["pending_raw"] = raw_path
            t_ui = t_log = 0.0
            try:
              for frame, clean in engine.run():
                if writer is None:
                    hh, ww = clean.shape[:2]
                    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             engine.src_fps, (ww, hh))
                writer.write(clean)                      # the recording stays clean
                now = time.time()
                if now - t_ui >= 0.08:
                    t_ui = now
                    stats = engine.stats()
                    fh = frame_html(frame)               # the live view shows the chosen overlay
                    rows = None
                    if now - t_log >= 0.5:
                        t_log, rows = now, engine.rows()
                    paint(("live", "LIVE"), stats, fh, rows, do_plates, do_lane, do_speed)
                    S["last"] = {"stats": stats, "frame": fh, "rows": rows if rows is not None else (S["last"] or {}).get("rows", []),
                                 "plates": do_plates, "lane": do_lane, "speed": do_speed, "choice": choice,
                                 "mode": mode_label, "started": started,
                                 "data": data_path, "tag": tag, "opts": dict(opts),
                                 "status": ("live", "LIVE")}
            finally:
                if writer is not None:
                    writer.release()
                    save_overlay(data_path, engine)
            rows = engine.rows()                                  # final, after track merging
            stats = engine.stats()
            S["last"]["stats"], S["last"]["rows"], S["last"]["status"] = stats, rows, ("done", "COMPLETE")
            paint(("done", "COMPLETE"), stats, S["last"]["frame"], rows, do_plates, do_lane, do_speed)
            S["pending_raw"] = None
            with rec_box:
                with st.spinner("Encoding the recording so it can play in the browser…"):
                    rec = finalize_recording(raw_path)
                S["last"]["rec"] = rec
                add_log(S["last"], "Complete")
                render_recordings(S["last"])
            if rows:
                lines = ["event_id,kind,track,video_time_s,detail,plate,plate_status"]
                for r in rows:
                    lines.append(f'{r["id"]},{r["kind"]},{r["tid"]},{r["ts"]:.2f},"{r["detail"]}",{r["plate"] or ""},{r["plate_status"]}')
                with right:
                    st.download_button("⬇  Export events (CSV)", "\n".join(lines), "autonox_events.csv", "text/csv")
        except ModuleNotFoundError as ex:
            error(f"Missing module: {ex.name}. Run the app from your project root, "
                  f"and make sure the speed/ package and requirements are installed.")
        except Exception as ex:
            error(f"{type(ex).__name__}: {ex}")
else:
    last = S.get("last")
    if last:
        with rec_box:
            finish_pending("Stopped")
            render_recordings(last)
        status = last["status"] if last["status"][0] != "live" else ("stop", "STOPPED")
        paint(status, last["stats"], last["frame"], last["rows"], last["plates"], last["lane"], last["speed"])
    else:
        with rec_box:
            render_recordings(None)
        paint(("idle", "STANDBY"))
        feed_ph.markdown('<div class="panel"><div class="ptitle"><span>Live analysis</span><span>idle</span></div>'
                         + empty_feed("Ready when you are",
                                      "Pick a stream and a detection mode in the sidebar,<br>then press <b style='display:inline'>Start analysis</b>.")
                         + "</div>", unsafe_allow_html=True)
        log_ph.markdown('<div class="panel"><div class="ptitle"><span>Violations log</span><span>0 events</span></div>'
                        + log_html([], False) + "</div>", unsafe_allow_html=True)

with logs_ph.container():
    render_logs()


