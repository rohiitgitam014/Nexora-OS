#!/usr/bin/env python3
"""AgentOS: Windows-style desktop + login + 10 LangGraph agent apps (one deployable file).
Env: AGENTOS_ADMIN_PASSWORD, GEMINI_API_KEY (or run Ollama), AGENTOS_SECURE=1 behind HTTPS, PORT.

Standalone file (only --kernel mode also needs agentos.py + agentos_langgraph.py next to it):
    pip install langgraph
    ollama pull llama3.2 && ollama serve   # local model; change with OLLAMA_MODEL, server with OLLAMA_URL
    export AGENTOS_ADMIN_PASSWORD=...   # optional; otherwise a random password is printed at startup
    python agentos_apps.py              # login page at http://127.0.0.1:8000
    python agentos_apps.py --kernel     # CLI login, then run all 10 apps as AgentOS kernel processes
"""
import asyncio, getpass, hashlib, hmac, html, json, os, secrets, sys, threading, time, urllib.error, urllib.request
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypedDict
from urllib.parse import parse_qs

from langgraph.graph import StateGraph, START, END
# ---------------- LLM: local Ollama (no API key, no extra pip package) ----------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

def _ollama(prompt):
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", method="POST",
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode())
    try:
        with urllib.request.urlopen(req, timeout=300) as r: return json.load(r)["response"].strip()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Can't reach Ollama at {OLLAMA_URL}. Run `ollama serve` and `ollama pull {OLLAMA_MODEL}`. ({e.reason})")


GEMINI_KEY, GEMINI_MODEL = os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

def _gemini(prompt):
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent", method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_KEY},
                                 data=json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode())
    with urllib.request.urlopen(req, timeout=120) as r: return json.load(r)["candidates"][0]["content"]["parts"][0]["text"].strip()

def ask(prompt): return _gemini(prompt) if GEMINI_KEY else _ollama(prompt)   # set GEMINI_API_KEY to use Gemini instead of Ollama


# ---------------- 10 apps: each is a LangGraph of small agents ----------------
class _Safe(dict):
    def __missing__(self, k): return ""

def build(steps, fan=0):
    """steps = [(state_key, prompt)]. Prompts may use {q} and earlier keys.
    The first `fan` steps run in parallel and join at the next step; the rest run in a chain."""
    S = TypedDict("S", {k: str for k in ["q"] + [k for k, _ in steps]}, total=False)
    g, names = StateGraph(S), [f"{k}_agent" for k, _ in steps]
    for n, (k, p) in zip(names, steps):
        g.add_node(n, lambda s, k=k, p=p: {k: ask(p.format_map(_Safe(s)))})
    head = names[:fan] if fan > 1 else names[:1]
    for n in head: g.add_edge(START, n)
    prev = head if len(head) > 1 else head[0]
    for n in names[len(head):]: g.add_edge(prev, n); prev = n
    g.add_edge(prev, END)
    return g.compile()

APPS = {  # name: (description, graph, sample input)
    "file_organizer": ("Sort messy files into folders", build([
        ("groups", "Group these files into folders by purpose, one line per folder:\n{q}"),
        ("risks", "Flag files that should not be moved or deleted:\n{q}"),
        ("moves", "Write a move plan (file -> folder), one per line:\n{groups}")], fan=2),
     "report.pdf, IMG_2231.jpg, budget.xlsx, setup.exe, notes.txt"),
    "scheduler": ("Turn to-dos into a realistic day", build([
        ("tasks", "Extract tasks and durations from:\n{q}"),
        ("slots", "Pack them into a 9-to-6 day with breaks:\n{tasks}"),
        ("conflicts", "Point out conflicts or overload in this schedule:\n{slots}")]),
     "Gym 1h, client call 30m at 2pm, write report 3h, groceries, dentist 1h at 4pm"),
    "task_planner": ("Rank tasks and plan the top three", build([
        ("ranked", "Rank these tasks by urgency and impact:\n{q}"),
        ("plan", "Turn the top 3 into a step-by-step plan:\n{ranked}")]),
     "Renew passport, fix login bug, plan team offsite, reply to investors, clean desk"),
    "meeting_notes": ("Summary, action items and follow-up email", build([
        ("summary", "Summarize this meeting in 3 lines:\n{q}"),
        ("actions", "List action items with owners:\n{q}"),
        ("followup", "Write a short follow-up email using:\n{summary}\n{actions}")], fan=2),
     "Asha: launch slips a week. Ravi: QA needs two more days. Asha will update the client by Friday."),
    "system_doctor": ("Diagnose logs and suggest fixes", build([
        ("cause", "Name the most likely root cause of this log:\n{q}"),
        ("fix", "Give 3 fix steps for this cause:\n{cause}"),
        ("prevent", "Suggest one monitoring alert that would catch it earlier:\n{cause}")]),
     "kernel: Out of memory: Killed process 4127 (python) total-vm:8123456kB"),
    "security_auditor": ("Audit config or code for risks", build([
        ("threats", "List up to 3 security threats in:\n{q}"),
        ("perms", "List over-broad permissions or secrets in:\n{q}"),
        ("hardening", "Give a prioritized hardening checklist for:\n{threats}\n{perms}")], fan=2),
     "app.run(host='0.0.0.0', debug=True); SECRET='admin123'; chmod 777 /data"),
    "finance_advisor": ("Categorize spending and suggest a budget", build([
        ("categories", "Categorize these expenses:\n{q}"),
        ("budget", "Suggest a monthly budget from:\n{categories}"),
        ("tips", "Give 3 practical saving tips:\n{budget}")]),
     "Rent 22000, Swiggy 6500, Netflix 649, Uber 3200, SIP 10000, Amazon 4800"),
    "code_fixer": ("Explain, fix and test a snippet", build([
        ("explain", "Explain what this code does in 2 lines:\n{q}"),
        ("fix", "Rewrite it with bugs fixed:\n{q}"),
        ("tests", "Write 2 unit tests for this fixed code:\n{fix}")], fan=2),
     "def avg(xs): return sum(xs) / len(xs)"),
    "study_tutor": ("Lesson, example and quiz on any topic", build([
        ("outline", "Outline a 3-step lesson on:\n{q}"),
        ("lesson", "Teach step 1 simply, with one example:\n{outline}"),
        ("quiz", "Write 3 quiz questions on:\n{q}")]),
     "How does virtual memory work?"),
    "travel_planner": ("Places, itinerary and rough budget", build([
        ("places", "Suggest 4 places for this trip:\n{q}"),
        ("days", "Make a day-by-day itinerary from:\n{places}"),
        ("budget", "Estimate a rough budget for:\n{days}")]),
     "4 days in Coorg, 2 adults, relaxed pace, monsoon season"),
}


# ---------------- login: scrypt-hashed passwords, expiring sessions, lockout ----------------
USERS, SESSIONS, FAILS = {}, {}, {}
SESSION_TTL, MAX_FAILS, LOCK_SECS = 3600, 5, 60
SEC = "; Secure" if os.getenv("AGENTOS_SECURE") else ""   # set AGENTOS_SECURE=1 when served over HTTPS

def _hash(pw, salt): return hashlib.scrypt(pw.encode(), salt=salt, n=2**14, r=8, p=1).hex()

def add_user(name, pw):
    salt = secrets.token_bytes(16); USERS[name] = (salt, _hash(pw, salt))

def check(name, pw):
    salt, h = USERS.get(name, (b"\0" * 16, ""))          # always hash, so timing doesn't reveal valid names
    return name in USERS and hmac.compare_digest(_hash(pw, salt), h)


# ---------------- pages ----------------
CSS = """
:root{--bg:#eef1ee;--fg:#1b2a2f;--card:#fbfcfb;--mut:#5d6b6f;--acc:#0f766e;--on:#fff;--bd:#cdd6d3}
@media(prefers-color-scheme:dark){:root{--bg:#10191b;--fg:#e4ecea;--card:#172326;--mut:#93a5a7;--acc:#4fd1c0;--on:#08201d;--bd:#2b3c3f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 ui-sans-serif,system-ui,'Segoe UI',sans-serif}
main{max-width:720px;margin:0 auto;padding:clamp(1.5rem,6vw,4rem) 1.25rem}
h1{font-size:1.9rem;margin:0 0 .25rem;letter-spacing:-.02em}
p.sub{color:var(--mut);margin:0 0 1.75rem}
form{display:grid;gap:.75rem}
label{display:grid;gap:.25rem;font-weight:600;font-size:.9rem}
input,textarea{font:inherit;padding:.65rem .75rem;border:1px solid var(--bd);border-radius:6px;background:var(--card);color:var(--fg)}
input:focus-visible,textarea:focus-visible,button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
button{font:inherit;font-weight:600;padding:.65rem 1rem;border:0;border-radius:6px;background:var(--acc);color:var(--on);cursor:pointer}
button.ghost{background:none;color:var(--acc);border:1px solid var(--acc)}
.err{padding:.6rem .75rem;border-left:4px solid #c2410c;background:var(--card)}
details{border:1px solid var(--bd);border-radius:6px;background:var(--card);margin-bottom:.6rem}
summary{padding:.75rem 1rem;cursor:pointer;font-weight:600}
summary span{display:block;font-weight:400;color:var(--mut);font-size:.9rem}
details form{padding:0 1rem 1rem}
pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--bd);border-radius:6px;padding:.75rem;margin:.25rem 0 1rem}
header{display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:1.5rem}
"""

def page(title, body):
    return ("<!doctype html><html lang=en><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style><main>{body}</main></html>")

def login_page(err=""):
    e = f"<p class=err role=alert>{html.escape(err)}</p>" if err else ""
    return page("Sign in to AgentOS", f"""<h1>Sign in to AgentOS</h1>
<p class=sub>Your agent apps run after you sign in.</p>{e}
<form method=post action=/login>
<label>Username<input name=user autocomplete=username required autofocus></label>
<label>Password<input name=pw type=password autocomplete=current-password required></label>
<button>Sign in</button></form>""")

def home_page(user):
    cards = "".join(f"""<details><summary>{html.escape(n.replace('_', ' ').title())}<span>{html.escape(d)}</span></summary>
<form method=post action=/run/{n}><label>Input<textarea name=q rows=3 maxlength=2000>{html.escape(sample)}</textarea></label>
<button>Run {html.escape(n.replace('_', ' '))}</button></form></details>"""
                    for n, (d, _, sample) in APPS.items())
    return page("AgentOS apps", f"""<header><div><h1>Your agent apps</h1><p class=sub>Signed in as {html.escape(user)}</p></div>
<form method=post action=/logout><button class=ghost>Sign out</button></form></header>{cards}""")

def result_page(name, out):
    parts = "".join(f"<h2>{html.escape(k)}</h2><pre>{html.escape(str(v))}</pre>" for k, v in out.items() if k != "q")
    return page(name, f"<h1>{html.escape(name.replace('_', ' ').title())}</h1>{parts}<p><a href=/>Back to apps</a></p>")


# ---------------- server ----------------
class H(BaseHTTPRequestHandler):
    def user(self):
        t = SimpleCookie(self.headers.get("Cookie", "")).get("sid")
        s = SESSIONS.get(t.value) if t else None
        return s[0] if s and s[1] > time.time() else None

    def reply(self, body="", code=200, headers=()):
        b = body.encode()
        self.send_response(code)
        for k, v in (("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(b))),
                     ("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'"),
                     ("X-Frame-Options", "DENY"), ("Cache-Control", "no-store"), *headers):
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def go(self, headers=()): self.reply(code=303, headers=(("Location", "/"), *headers))

    def do_GET(self):
        u = self.user()
        if self.path == "/" and u: self.reply(home_page(u))
        elif self.path == "/": self.reply(login_page())
        else: self.reply(page("Not found", "<h1>Not found</h1><p><a href=/>Go home</a></p>"), 404)

    def do_POST(self):
        n = min(int(self.headers.get("Content-Length") or 0), 10_000)
        f = {k: v[0] for k, v in parse_qs(self.rfile.read(n).decode(errors="replace")).items()}
        ip, u = self.client_address[0], self.user()
        if self.path == "/login":
            cnt, ts = FAILS.get(ip, (0, 0))
            if cnt >= MAX_FAILS and time.time() - ts < LOCK_SECS:
                return self.reply(login_page("Too many attempts. Wait a minute and try again."), 429)
            if check(f.get("user", ""), f.get("pw", "")):
                FAILS.pop(ip, None); tok = secrets.token_urlsafe(32)
                SESSIONS[tok] = (f["user"], time.time() + SESSION_TTL)
                return self.go((("Set-Cookie", f"sid={tok}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_TTL}{SEC}"),))
            FAILS[ip] = (cnt + 1 if time.time() - ts < LOCK_SECS else 1, time.time())
            return self.reply(login_page("Wrong username or password."), 401)
        if not u: return self.go()
        if self.path == "/logout":
            t = SimpleCookie(self.headers.get("Cookie", "")).get("sid")
            SESSIONS.pop(t.value if t else "", None)
            return self.go((("Set-Cookie", "sid=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0" + SEC),))
        name = self.path.removeprefix("/run/")
        if self.path.startswith("/run/") and name in APPS:
            try: out = APPS[name][1].invoke({"q": f.get("q", "")[:2000]})
            except Exception as e: out = {"error": f"{type(e).__name__}: {e}"}
            return self.reply(result_page(name, out))
        self.reply(page("Not found", "<h1>Not found</h1>"), 404)

    def log_message(self, *a): pass


# ---------------- Windows-style desktop UI ----------------
# Windows 12-style desktop for AgentOS. Run AFTER the main agentos_apps cell (same notebook);
# it upgrades the running server, no restart needed. Refresh the browser tab afterwards.
ICONS = {"file_organizer": "🗂️", "scheduler": "📅", "task_planner": "✅", "meeting_notes": "📝", "system_doctor": "🩺",
         "security_auditor": "🛡️", "finance_advisor": "💰", "code_fixer": "🧑‍💻", "study_tutor": "🎓", "travel_planner": "🧳"}

UI_CSS = """
:root{--glass:rgba(30,37,54,.7);--line:rgba(255,255,255,.15);--fg:#f3f6fb;--mut:#b6c0d0;--acc:#4cc2ff}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;height:100vh;overflow:hidden;color:var(--fg);font:14px 'Segoe UI Variable','Segoe UI',system-ui,sans-serif;
background:radial-gradient(90% 70% at 15% 10%,#3b82f6 0,transparent 60%),radial-gradient(80% 70% at 88% 90%,#8b5cf6 0,transparent 55%),#0b1226}
.glass{background:var(--glass);backdrop-filter:blur(26px) saturate(1.5);-webkit-backdrop-filter:blur(26px) saturate(1.5);border:1px solid var(--line);border-radius:12px;box-shadow:0 14px 44px rgba(0,0,0,.4)}
input,textarea{font:inherit;color:var(--fg);background:rgba(255,255,255,.08);border:1px solid var(--line);border-radius:8px;padding:10px;width:100%}
input:focus-visible,textarea:focus-visible,button:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.go{border:0;border-radius:8px;padding:10px 14px;background:var(--acc);color:#04121f;font:inherit;font-weight:600;cursor:pointer}.go:disabled{opacity:.6}
.lock{display:grid;place-items:center;align-content:center;height:100vh;text-align:center}
.big{font-size:clamp(3rem,10vw,6rem);font-weight:300;line-height:1}.d{color:var(--mut)}
.card{margin-top:24px;padding:24px;width:min(340px,92vw);display:grid;gap:10px}.card h2{margin:0}.err{margin:0;color:#ffb4b4}
.av{font-size:44px;background:rgba(255,255,255,.12);border-radius:50%;width:84px;height:84px;display:grid;place-items:center;margin:0 auto}
.icons{position:absolute;inset:14px auto 70px 14px;display:grid;grid-auto-flow:column;grid-template-rows:repeat(auto-fill,92px);gap:6px;align-content:start}
.ic{width:88px;height:88px;border:0;border-radius:8px;background:none;color:var(--fg);font:inherit;cursor:pointer;padding:6px}.ic:hover{background:rgba(255,255,255,.13)}
.ic b{display:block;font-size:30px;font-weight:400}.ic span{font-size:12px}
.bar{position:fixed;left:50%;bottom:10px;transform:translateX(-50%);display:flex;gap:6px;padding:6px 10px;z-index:9999}
#run{display:flex;gap:6px}.tb{width:42px;height:42px;border:0;border-radius:8px;background:none;font-size:22px;color:var(--fg);cursor:pointer}
.tb:hover{background:rgba(255,255,255,.15)}.tb.on{box-shadow:inset 0 -3px 0 var(--acc)}
.clk{position:fixed;right:16px;bottom:14px;text-align:right;font-size:12px;z-index:9999}
#start{position:fixed;left:50%;bottom:66px;transform:translateX(-50%);width:min(560px,94vw);padding:18px;display:none;z-index:9998}#start.show{display:block}
#start h3{margin:0 0 8px}.pin{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:4px}
.foot{display:flex;justify-content:space-between;align-items:center;margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
.win{position:absolute;width:min(540px,94vw);display:flex;flex-direction:column}
.win.max{left:0!important;top:0!important;width:100%!important;height:calc(100vh - 62px);border-radius:0}
.tt{display:flex;align-items:center;gap:8px;padding:8px 10px;cursor:move;user-select:none;border-bottom:1px solid var(--line)}.tt b{flex:1;font-weight:600}
.tt button{width:30px;height:28px;border:0;border-radius:6px;background:none;color:var(--fg);cursor:pointer}.tt button:hover{background:rgba(255,255,255,.16)}.tt .x:hover{background:#e5484d}
.wb{padding:12px;display:grid;gap:10px;overflow:auto;max-height:72vh}.wb small{color:var(--mut)}
.out h4{margin:10px 0 2px;color:var(--acc);text-transform:capitalize}.out pre{white-space:pre-wrap;margin:0;font:inherit}
"""

CLOCK = ("<script>const tick=()=>{const d=new Date();document.querySelectorAll('.t').forEach(e=>e.textContent=d.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}));"
         "document.querySelectorAll('.d').forEach(e=>e.textContent=d.toLocaleDateString([],{weekday:'long',month:'long',day:'numeric'}))};tick();setInterval(tick,10000)</script>")

DESK_JS = r"""
let z=10,n=0;const $=s=>document.querySelector(s);
function sync(){const t=$('#run');t.innerHTML='';document.querySelectorAll('.win').forEach(w=>{const a=APPS[w.id.slice(2)],b=document.createElement('button');
b.className='tb'+(w.hidden?'':' on');b.textContent=a.icon;b.title=a.title;b.onclick=()=>{w.hidden=!w.hidden;if(!w.hidden)w.style.zIndex=++z;sync()};t.append(b)})}
function openApp(id){let w=document.getElementById('w-'+id);if(!w)w=mk(id);w.hidden=false;w.style.zIndex=++z;$('#start').classList.remove('show');sync()}
function mk(id){const a=APPS[id],w=document.createElement('div');w.className='win glass';w.id='w-'+id;
w.style.left=(150+(n%6)*32)+'px';w.style.top=(36+(n++%6)*32)+'px';
w.innerHTML='<div class=tt><span></span><b></b><button class=mn title=Minimize>—</button><button class=mx title=Maximize>▢</button><button class=x title=Close>✕</button></div><div class=wb><small></small><textarea rows=4 maxlength=2000></textarea><button class=go>Run agents</button><div class=out></div></div>';
w.querySelector('span').textContent=a.icon;w.querySelector('b').textContent=a.title;w.querySelector('small').textContent=a.desc;
const ta=w.querySelector('textarea'),go=w.querySelector('.go'),o=w.querySelector('.out'),tt=w.querySelector('.tt');ta.value=a.sample;
w.onpointerdown=()=>w.style.zIndex=++z;
tt.onpointerdown=e=>{if(e.target.tagName=='BUTTON'||w.classList.contains('max'))return;const dx=e.clientX-w.offsetLeft,dy=e.clientY-w.offsetTop;tt.setPointerCapture(e.pointerId);
tt.onpointermove=e=>{w.style.left=Math.max(0,e.clientX-dx)+'px';w.style.top=Math.max(0,e.clientY-dy)+'px'};tt.onpointerup=()=>tt.onpointermove=null};
tt.ondblclick=e=>{if(e.target.tagName!='BUTTON')w.classList.toggle('max')};
w.querySelector('.mx').onclick=()=>w.classList.toggle('max');w.querySelector('.mn').onclick=()=>{w.hidden=true;sync()};w.querySelector('.x').onclick=()=>{w.remove();sync()};
go.onclick=async()=>{go.disabled=true;o.textContent='Agents are working…';
try{const r=await fetch('/api/run/'+id,{method:'POST',body:JSON.stringify({q:ta.value})});if(r.status==401){location.reload();return}
const d=await r.json();o.textContent='';for(const k in d){const h=document.createElement('h4'),p=document.createElement('pre');h.textContent=k.replace(/_/g,' ');p.textContent=d[k];o.append(h,p)}}
catch(e){o.textContent='Request failed: '+e}go.disabled=false};
$('#wins').append(w);return w}
for(const id in APPS){for(const box of ['#ic','#pin']){const b=document.createElement('button'),i=document.createElement('b'),s=document.createElement('span');
b.className='ic';i.textContent=APPS[id].icon;s.textContent=APPS[id].title;b.append(i,s);b.onclick=()=>openApp(id);$(box).append(b)}}
$('#sb').onclick=e=>{e.stopPropagation();$('#start').classList.toggle('show')};
document.addEventListener('click',e=>{if(!e.target.closest('#start'))$('#start').classList.remove('show')});
"""

def ui_page(title, body):
    return ("<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{UI_CSS}</style>{body}</html>")

def login_page(err=""):
    e = f"<p class=err role=alert>{html.escape(err)}</p>" if err else ""
    return ui_page("Sign in", f"""<div class=lock><div class='t big'></div><div class=d></div>
<form method=post action=/login class='glass card'><div class=av>👤</div><h2>AgentOS</h2>{e}
<input name=user placeholder=Username autocomplete=username required autofocus>
<input name=pw type=password placeholder=Password autocomplete=current-password required>
<button class=go>Sign in →</button></form></div>{CLOCK}""")

def home_page(user):
    data = {k: {"title": k.replace("_", " ").title(), "icon": ICONS.get(k, "🤖"), "desc": d, "sample": s} for k, (d, _, s) in APPS.items()}
    js = json.dumps(data).replace("</", "<\\/")
    return ui_page("AgentOS", f"""<div class=icons id=ic></div><div id=wins></div>
<div id=start class=glass><h3>Pinned apps</h3><div class=pin id=pin></div>
<div class=foot><span>👤 {html.escape(user)}</span><form method=post action=/logout><button class=go>Sign out</button></form></div></div>
<div class='bar glass'><button class=tb id=sb title=Start>🪟</button><span id=run></span></div>
<div class=clk><div class=t></div><div class=d></div></div>
<script>const APPS={js};{DESK_JS}</script>{CLOCK}""")

def _reply(self, body="", code=200, headers=()):
    b = body.encode(); self.send_response(code)
    for k, v in (("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(b))),
                 ("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; form-action 'self'"),
                 ("X-Frame-Options", "DENY"), ("Cache-Control", "no-store"), *headers):
        self.send_header(k, v)
    self.end_headers(); self.wfile.write(b)

H._orig_post = getattr(H, "_orig_post", H.do_POST)
def _post(self):
    if not self.path.startswith("/api/run/"): return H._orig_post(self)
    name = self.path[9:]
    if not self.user(): return self.reply('{"error": "Please sign in again."}', 401)
    if name not in APPS: return self.reply('{"error": "Unknown app."}', 404)
    try:
        q = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 10_000)) or b"{}").get("q", "")[:2000]
        out = {k: v for k, v in APPS[name][1].invoke({"q": q}).items() if k != "q"}
    except Exception as e: out = {"error": f"{type(e).__name__}: {e}"}
    self.reply(json.dumps(out))
H.reply, H.do_POST = _reply, _post

# Run AFTER the windows12_ui cell, in the same notebook. Then hard-refresh the AgentOS tab (Ctrl+Shift+R).
UI_CSS += """
body{background:radial-gradient(55% 75% at 28% 42%,#0ea5e9 0,transparent 62%),radial-gradient(48% 60% at 68% 58%,#6366f1 0,transparent 62%),
radial-gradient(40% 50% at 85% 25%,rgba(34,211,238,.55) 0,transparent 60%),linear-gradient(160deg,#050b1f,#0a1a4a)}
.go{background:linear-gradient(135deg,#4cc2ff,#7c9cff)}
.lock2{height:100vh;display:grid;grid-template-rows:1fr auto 1fr;justify-items:center;text-align:center}
.lock2 .top{align-self:end;padding-bottom:4vh}
.who{width:min(320px,90vw);display:grid;gap:10px;justify-items:center}
.av2{width:104px;height:104px;border-radius:50%;display:grid;place-items:center;background:linear-gradient(145deg,#4cc2ff,#7c5cff);box-shadow:0 10px 30px rgba(0,0,0,.4)}
.brand h1{margin:0;font-size:clamp(1.9rem,5vw,2.8rem);font-weight:700;letter-spacing:.01em;background:linear-gradient(135deg,#fff,#7fd4ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.brand p{margin:6px 0 22px;color:var(--mut);font-size:clamp(.8rem,2vw,1rem);letter-spacing:.06em}
.who h2{margin:4px 0 0;font-size:24px;font-weight:600}.who input{text-align:center;border-radius:8px}
.pw{position:relative;width:100%}.pw input{padding-right:48px;text-align:left;background:rgba(255,255,255,.14)}
.pw button{position:absolute;right:4px;top:4px;bottom:4px;width:38px;border:0;border-radius:6px;background:var(--acc);color:#04121f;font-size:18px;cursor:pointer}
.hint{font-size:12px;color:var(--mut);margin:0}.sys{position:fixed;right:22px;bottom:18px;font-size:18px;letter-spacing:10px}
.ic b{width:52px;height:52px;margin:0 auto 4px;border-radius:14px;display:grid;place-items:center;font-size:26px;
background:linear-gradient(145deg,rgba(255,255,255,.3),rgba(255,255,255,.08));box-shadow:0 6px 16px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.35)}
.ic span{text-shadow:0 1px 4px rgba(0,0,0,.7)}
.bar{border-radius:16px;padding:6px 8px;background:rgba(18,24,42,.6)}
#sb{font-size:0}#sb::before{content:"";display:block;width:20px;height:20px;margin:auto;background-repeat:no-repeat;
background-image:linear-gradient(#4cc2ff,#4cc2ff),linear-gradient(#4cc2ff,#4cc2ff),linear-gradient(#4cc2ff,#4cc2ff),linear-gradient(#4cc2ff,#4cc2ff);
background-size:9px 9px;background-position:0 0,11px 0,0 11px,11px 11px}
#start{border-radius:18px;padding:22px}#start h3{font-size:15px;font-weight:600}
.win{border-radius:12px;overflow:hidden;background:rgba(26,32,50,.84)}.tt{background:rgba(255,255,255,.06)}
"""

def login_page(err=""):
    e = f"<p class=err role=alert>{html.escape(err)}</p>" if err else ""
    return ui_page("Nexora OS", f"""<div class=lock2><div class=top><div class=brand><h1>Nexora OS</h1><p>The Intelligent Agent Operating System</p></div><div class='t big'></div><div class=d></div></div>
<form method=post action=/login class=who>
<div class=av2><svg width=60 height=60 viewBox='0 0 24 24' fill=#fff><circle cx=12 cy=8 r=4.2 /><path d='M3.5 21c0-4.6 3.8-7.5 8.5-7.5s8.5 2.9 8.5 7.5z'/></svg></div>
<h2>Welcome</h2>
<input name=user value=admin aria-label=Username autocomplete=username required>
<div class=pw><input name=pw type=password placeholder=Password aria-label=Password autocomplete=current-password required autofocus><button aria-label='Sign in'>→</button></div>
{e}<p class=hint>Username: admin. Password: the one printed under the cell that started the server.</p></form><div></div>
<div class=sys>📶🔊🔋</div></div>{CLOCK}""")


# ---------------- run ----------------
def demo_kernel():
    from agentos import Kernel
    from agentos_langgraph import GraphBrain   # only --kernel mode needs your other files
    if not check(input("user: "), getpass.getpass()): sys.exit("Login failed.")
    async def main():
        k = Kernel(workers=3)
        for i, (n, (d, g, sample)) in enumerate(APPS.items(), 1):
            k.spawn(n, d, GraphBrain(g, {"q": sample}), {"memory"}, priority=i)
        await k.run()
    asyncio.run(main())

if __name__ == "__main__":
    pw = os.getenv("AGENTOS_ADMIN_PASSWORD")
    if not pw:
        pw = secrets.token_urlsafe(9); print(f"Generated admin password (username: admin): {pw}")
    add_user("admin", pw)
    if "--kernel" in sys.argv: demo_kernel()
    else:
        host = os.getenv("AGENTOS_HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
        port = int(os.getenv("PORT") or os.getenv("AGENTOS_PORT", "8000"))
        srv = ThreadingHTTPServer((host, port), H)
        if "ipykernel" in sys.modules:                      # Jupyter/Colab: serve in the background
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                from google.colab.output import eval_js
                print("Open this link in a new tab:", eval_js(f"google.colab.kernel.proxyPort({port})"))
            except ImportError: print(f"AgentOS login: http://{host}:{port}")
        else:
            print(f"AgentOS login: http://{host}:{port}"); srv.serve_forever()
