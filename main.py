import os, re, time, json, math
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import faiss
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))

# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "alive", "ready": _app_ready}), 200

@app.route("/ping")
def ping():
    return "pong", 200

# ── Frontend ──────────────────────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Vidya AI – Your Smart Tutor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;600;800&family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  /* ── RESET & BASE ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --maths-h:  220; --maths-c:  #4F8EF7;  --maths-bg:  #EEF4FF;  --maths-glow:  #a5c4fd;
    --mar-h:    140; --mar-c:    #22C97A;   --mar-bg:    #EDFAF4;  --mar-glow:    #9DECC5;
    --eng-h:    340; --eng-c:    #F7587A;   --eng-bg:    #FFF0F4;  --eng-glow:    #FFAABF;
    --dark:     #1A1D2E;
    --card:     #FFFFFF;
    --text:     #2D3252;
    --subtle:   #8A90AA;
    --radius:   20px;
    --shadow:   0 8px 32px rgba(80,100,180,.12);
    --active-color: var(--maths-c);
    --active-bg:    var(--maths-bg);
    --active-glow:  var(--maths-glow);
  }

  body {
    font-family: 'Nunito', sans-serif;
    background: #F4F6FF;
    color: var(--text);
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  /* ── STARS BACKGROUND ── */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background:
      radial-gradient(circle at 15% 20%, #d9e8ff 0%, transparent 35%),
      radial-gradient(circle at 85% 75%, #ffe0ec 0%, transparent 35%),
      radial-gradient(circle at 50% 50%, #e0fdf0 0%, transparent 50%),
      #F4F6FF;
    pointer-events: none;
  }

  /* ── HEADER ── */
  header {
    width: 100%; max-width: 900px;
    padding: 16px 24px 0;
    text-align: center;
    flex-shrink: 0;
    position: relative; z-index: 1;
    animation: fadeDown .6s ease both;
  }

  .logo-row {
    display: flex; align-items: center; justify-content: center; gap: 12px;
    margin-bottom: 6px;
  }

  .logo-icon {
    width: 52px; height: 52px; border-radius: 16px;
    background: linear-gradient(135deg, #4F8EF7, #F7587A);
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    box-shadow: 0 4px 18px rgba(80,142,247,.35);
  }

  h1 {
    font-family: 'Baloo 2', cursive;
    font-size: clamp(1.8rem, 5vw, 2.8rem);
    font-weight: 800;
    background: linear-gradient(120deg, #4F8EF7 10%, #F7587A 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .tagline {
    font-size: .9rem;
    color: var(--subtle);
    margin-bottom: 12px;
    letter-spacing: .3px;
  }

  /* ── SUBJECT TABS ── */
  .tabs {
    display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;
    margin-bottom: 12px;
    flex-shrink: 0;
    position: relative; z-index: 1;
  }

  .tab {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 26px;
    border: 2px solid transparent;
    border-radius: 50px;
    cursor: pointer;
    font-family: 'Baloo 2', cursive;
    font-size: 1rem; font-weight: 600;
    transition: all .25s ease;
    background: white;
    color: var(--subtle);
    box-shadow: 0 2px 10px rgba(0,0,0,.06);
    position: relative; overflow: hidden;
  }

  .tab .tab-emoji { font-size: 1.2rem; }

  .tab:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,.1); }

  .tab.maths.active  { background: var(--maths-bg);  border-color: var(--maths-c);  color: var(--maths-c);  box-shadow: 0 4px 18px rgba(79,142,247,.25); }
  .tab.marathi.active{ background: var(--mar-bg);    border-color: var(--mar-c);    color: var(--mar-c);    box-shadow: 0 4px 18px rgba(34,201,122,.25); }
  .tab.english.active{ background: var(--eng-bg);    border-color: var(--eng-c);    color: var(--eng-c);    box-shadow: 0 4px 18px rgba(247,88,122,.25); }

  /* ── CHAT CARD ── */
  .chat-card {
    width: calc(100% - 32px); max-width: 860px;
    background: white;
    border-radius: 28px;
    box-shadow: 0 12px 48px rgba(60,80,160,.12);
    overflow: hidden;
    display: flex; flex-direction: column;
    flex: 1;
    min-height: 0;
    position: relative; z-index: 1;
    animation: fadeUp .5s ease both;
  }

  /* card top strip */
  .card-header {
    padding: 18px 24px;
    display: flex; align-items: center; gap: 12px;
    border-bottom: 1px solid #F0F2FA;
    transition: background .35s;
  }

  .card-header.maths   { background: var(--maths-bg); }
  .card-header.marathi { background: var(--mar-bg);   }
  .card-header.english { background: var(--eng-bg);   }

  .teacher-avatar {
    width: 46px; height: 46px; border-radius: 50%;
    font-size: 24px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .maths   .teacher-avatar { background: var(--maths-bg);  box-shadow: 0 0 0 3px var(--maths-c); }
  .marathi .teacher-avatar { background: var(--mar-bg);    box-shadow: 0 0 0 3px var(--mar-c);   }
  .english .teacher-avatar { background: var(--eng-bg);    box-shadow: 0 0 0 3px var(--eng-c);   }

  .teacher-info .name {
    font-family: 'Baloo 2', cursive;
    font-weight: 700; font-size: 1rem;
  }
  .teacher-info .status {
    font-size: .78rem; color: var(--subtle);
    display: flex; align-items: center; gap: 5px;
  }
  .status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22C97A;
    animation: pulse 2s infinite;
  }

  /* ── MESSAGES AREA ── */
  .messages {
    flex: 1;
    min-height: 0;
    padding: 20px 22px;
    overflow-y: auto;
    display: flex; flex-direction: column; gap: 16px;
    scroll-behavior: smooth;
  }

  .messages::-webkit-scrollbar { width: 7px; }
  .messages::-webkit-scrollbar-track { background: #f0f2fa; border-radius: 10px; }
  .messages::-webkit-scrollbar-thumb { background: #b0b8d8; border-radius: 10px; }
  .messages::-webkit-scrollbar-thumb:hover { background: #8a90aa; }

  /* welcome */
  .welcome {
    text-align: center;
    padding: 40px 20px;
    color: var(--subtle);
    animation: fadeIn .6s ease both;
  }
  .welcome .big-emoji { font-size: 4rem; margin-bottom: 12px; }
  .welcome h2 { font-family: 'Baloo 2', cursive; font-size: 1.4rem; color: var(--text); margin-bottom: 6px; }

  /* bubbles */
  .bubble-row {
    display: flex; align-items: flex-end; gap: 10px;
    animation: fadeUp .35s ease both;
  }

  .bubble-row.user  { flex-direction: row-reverse; }

  .avatar-sm {
    width: 34px; height: 34px; border-radius: 50%;
    font-size: 18px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .bubble-row.bot  .avatar-sm { background: #EEF4FF; }
  .bubble-row.user .avatar-sm { background: #FFF0F4; }

  .bubble {
    max-width: 72%; padding: 12px 18px;
    border-radius: 20px;
    font-size: .95rem; line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .bubble-row.bot  .bubble {
    background: #F5F7FF;
    border-bottom-left-radius: 5px;
    color: var(--text);
  }

  .bubble-row.user .bubble {
    background: linear-gradient(135deg, #4F8EF7, #7BAEFF);
    color: white;
    border-bottom-right-radius: 5px;
  }

  /* subject accent for bot bubble */
  body.maths   .bubble-row.bot .bubble { border-left: 3px solid var(--maths-c); }
  body.marathi .bubble-row.bot .bubble { border-left: 3px solid var(--mar-c);   }
  body.english .bubble-row.bot .bubble { border-left: 3px solid var(--eng-c);   }

  /* typing dots */
  .typing-indicator {
    display: flex; gap: 5px; align-items: center;
    padding: 14px 18px;
    background: #F5F7FF;
    border-radius: 20px; border-bottom-left-radius: 5px;
    width: fit-content;
  }

  .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #8A90AA;
    animation: bounce 1.2s infinite;
  }
  .dot:nth-child(2) { animation-delay: .2s; }
  .dot:nth-child(3) { animation-delay: .4s; }

  /* ── INPUT AREA ── */
  .input-area {
    padding: 16px 20px;
    border-top: 1px solid #F0F2FA;
    display: flex; gap: 10px; align-items: flex-end;
    background: white;
  }

  textarea {
    flex: 1;
    padding: 12px 18px;
    border: 2px solid #E8EAF4;
    border-radius: 16px;
    font-family: 'Nunito', sans-serif;
    font-size: .95rem;
    color: var(--text);
    resize: none;
    max-height: 120px;
    outline: none;
    transition: border-color .2s;
    line-height: 1.5;
  }

  textarea:focus { border-color: #4F8EF7; }
  body.marathi textarea:focus { border-color: var(--mar-c); }
  body.english textarea:focus { border-color: var(--eng-c); }

  textarea::placeholder { color: #b0b5cc; }

  .send-btn {
    width: 48px; height: 48px;
    border: none; border-radius: 14px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    transition: transform .15s, box-shadow .15s;
    font-size: 20px;
    background: linear-gradient(135deg, #4F8EF7, #7BAEFF);
    color: white;
    box-shadow: 0 4px 14px rgba(79,142,247,.4);
  }

  body.marathi .send-btn { background: linear-gradient(135deg, #22C97A, #5FDBA0); box-shadow: 0 4px 14px rgba(34,201,122,.4); }
  body.english .send-btn { background: linear-gradient(135deg, #F7587A, #FF8FA8); box-shadow: 0 4px 14px rgba(247,88,122,.4); }

  .send-btn:hover  { transform: scale(1.08); }
  .send-btn:active { transform: scale(.96); }
  .send-btn:disabled { opacity: .5; cursor: not-allowed; transform: none; }

  /* ── MIC BUTTON ── */
  .mic-btn {
    width: 48px; height: 48px;
    border: none; border-radius: 14px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-size: 22px;
    transition: transform .15s, box-shadow .15s, background .2s;
    background: #F0F2FA;
    color: var(--subtle);
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
  }
  .mic-btn:hover { transform: scale(1.08); background: #E4E8F8; }
  .mic-btn:active { transform: scale(.96); }

  .mic-btn.listening {
    background: linear-gradient(135deg, #FF5252, #FF8A80);
    color: white;
    box-shadow: 0 4px 16px rgba(255,82,82,.5);
    animation: micPulse 1s ease-in-out infinite;
  }
  @keyframes micPulse {
    0%,100% { box-shadow: 0 4px 16px rgba(255,82,82,.5); transform: scale(1); }
    50%     { box-shadow: 0 4px 24px rgba(255,82,82,.8); transform: scale(1.06); }
  }

  /* ripple ring when listening */
  .mic-btn.listening::after {
    content: '';
    position: absolute;
    width: 48px; height: 48px;
    border-radius: 14px;
    border: 2px solid rgba(255,82,82,.6);
    animation: micRing 1s ease-out infinite;
  }
  @keyframes micRing {
    0%   { transform: scale(1); opacity: 1; }
    100% { transform: scale(1.6); opacity: 0; }
  }

  /* textarea glow when listening */
  textarea.listening {
    border-color: #FF5252 !important;
    box-shadow: 0 0 0 3px rgba(255,82,82,.15);
  }

  /* ── QUICK QUESTIONS ── */
  .quick-wrap {
    padding: 0 20px 14px;
    display: flex; gap: 8px; flex-wrap: wrap;
  }

  .quick-btn {
    padding: 6px 14px;
    border: 1.5px solid #E0E4F4;
    border-radius: 50px;
    background: white;
    font-family: 'Nunito', sans-serif;
    font-size: .82rem;
    color: var(--subtle);
    cursor: pointer;
    transition: all .2s;
  }

  .quick-btn:hover {
    border-color: var(--maths-c);
    color: var(--maths-c);
    background: var(--maths-bg);
  }
  body.marathi .quick-btn:hover { border-color: var(--mar-c); color: var(--mar-c); background: var(--mar-bg); }
  body.english .quick-btn:hover { border-color: var(--eng-c); color: var(--eng-c); background: var(--eng-bg); }

  /* ── FOOTER ── */
  footer {
    padding: 6px 0 8px;
    font-size: .72rem;
    color: var(--subtle);
    text-align: center;
    flex-shrink: 0;
    position: relative; z-index: 1;
  }

  /* ── ANIMATIONS ── */
  @keyframes fadeDown { from { opacity:0; transform: translateY(-16px); } to { opacity:1; transform: translateY(0); } }
  @keyframes fadeUp   { from { opacity:0; transform: translateY(16px);  } to { opacity:1; transform: translateY(0); } }
  @keyframes fadeIn   { from { opacity:0; } to { opacity:1; } }
  @keyframes bounce   { 0%,60%,100% { transform: translateY(0); } 30% { transform: translateY(-8px); } }
  @keyframes pulse    { 0%,100% { opacity:1; } 50% { opacity:.4; } }

  /* ── RESPONSIVE ── */
  @media(max-width: 600px) {
    .bubble { max-width: 88%; }
    h1 { font-size: 1.8rem; }
  }

  /* ── VOICE BUTTON ── */
  .bubble-wrap {
    display: flex; flex-direction: column; align-items: flex-start;
    max-width: 72%;
  }
  .bubble-wrap .bubble { max-width: 100%; }

  .voice-btn {
    display: inline-flex; align-items: center; gap: 6px;
    margin-top: 7px;
    padding: 5px 14px;
    border: 1.5px solid currentColor;
    border-radius: 20px;
    font-family: 'Nunito', sans-serif;
    font-size: .76rem; font-weight: 700;
    cursor: pointer;
    transition: all .2s ease;
    background: white;
    opacity: .75;
  }
  .voice-btn:hover { opacity: 1; transform: translateY(-1px); }
  body.maths   .voice-btn { color: var(--maths-c); }
  body.marathi .voice-btn { color: var(--mar-c);   }
  body.english .voice-btn { color: var(--eng-c);   }

  .voice-btn.speaking {
    opacity: 1; color: white !important; border-color: transparent;
    animation: voicePulse 1.1s ease-in-out infinite;
  }
  body.maths   .voice-btn.speaking { background: var(--maths-c); }
  body.marathi .voice-btn.speaking { background: var(--mar-c);   }
  body.english .voice-btn.speaking { background: var(--eng-c);   }

  @keyframes voicePulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(79,142,247,.4); }
    50%     { box-shadow: 0 0 0 6px rgba(79,142,247,0); }
  }

  .snd { display:flex; align-items:center; gap:2px; height:12px; }
  .snd b { display:block; width:3px; border-radius:2px; background:currentColor;
            animation: sndBar .75s ease-in-out infinite; }
  .snd b:nth-child(1){height:3px;  animation-delay:0s}
  .snd b:nth-child(2){height:9px;  animation-delay:.12s}
  .snd b:nth-child(3){height:12px; animation-delay:.24s}
  .snd b:nth-child(4){height:7px;  animation-delay:.36s}
  .snd b:nth-child(5){height:3px;  animation-delay:.48s}
  @keyframes sndBar { 0%,100%{ transform:scaleY(1) } 50%{ transform:scaleY(.25) } }

</style>
</head>
<body class="maths">

<!-- HEADER -->
<header>
  <div class="logo-row">
    <div class="logo-icon">🎓</div>
    <h1>Vidya AI</h1>
  </div>
  <p class="tagline">Your friendly AI tutor — Maths · Marathi · English</p>
</header>

<!-- SUBJECT TABS -->
<div class="tabs" role="tablist">
  <button class="tab maths active"   onclick="switchSubject('maths')"   aria-label="Maths">
    <span class="tab-emoji">🔢</span> Maths
  </button>
  <button class="tab marathi" onclick="switchSubject('marathi')" aria-label="Marathi">
    <span class="tab-emoji">📖</span> मराठी
  </button>
  <button class="tab english" onclick="switchSubject('english')" aria-label="English">
    <span class="tab-emoji">✏️</span> English
  </button>
</div>

<!-- CHAT CARD -->
<div class="chat-card">

  <!-- Card Header -->
  <div class="card-header maths" id="cardHeader">
    <div class="teacher-avatar" id="teacherAvatar">🧮</div>
    <div class="teacher-info">
      <div class="name" id="teacherName">Maths Teacher</div>
      <div class="status"><span class="status-dot"></span> Online &amp; ready to help!</div>
    </div>
  </div>

  <!-- Messages -->
  <div class="messages" id="messages">
    <div class="welcome" id="welcomeMsg">
      <div class="big-emoji" id="welcomeEmoji">🔢</div>
      <h2 id="welcomeTitle">Hello! I'm your Maths Teacher 👋</h2>
      <p id="welcomeText">Ask me anything about numbers, shapes, addition, subtraction — I'll explain it in a super easy way!</p>
    </div>
  </div>

  <!-- Quick Questions -->
  <div class="quick-wrap" id="quickWrap">
    <button class="quick-btn" onclick="askQuick(this)">What is addition?</button>
    <button class="quick-btn" onclick="askQuick(this)">How to multiply?</button>
    <button class="quick-btn" onclick="askQuick(this)">What are shapes?</button>
    <button class="quick-btn" onclick="askQuick(this)">Help with subtraction</button>
  </div>

  <!-- Input Area -->
  <div class="input-area">
    <button class="mic-btn" id="micBtn" onclick="toggleMic()" title="Speak your question">
      🎤
    </button>
    <textarea id="questionInput" rows="1"
      placeholder="Ask your question here… 😊"
      onkeydown="handleKey(event)"
      oninput="autoResize(this)"></textarea>
    <button class="send-btn" id="sendBtn" onclick="sendQuestion()" title="Send">
      ➤
    </button>
  </div>
</div>

<!-- FOOTER -->
<footer>Made with ❤️ for curious students · Powered by Gemini AI</footer>


<script>
  // ── CONFIG ──────────────────────────────────────────────────────────────────
  const API_URL = "/ask";   // Relative URL — works on any domain

  const SUBJECTS = {
    maths: {
      name: "Maths Teacher",
      avatar: "🧮",
      emoji: "🔢",
      title: "Hello! I'm your Maths Teacher 👋",
      text:  "Ask me anything about numbers, shapes, addition, subtraction — I'll explain it in a super easy way!",
      placeholder: "Ask your maths question… 😊",
      quick: ["What is addition?", "How to multiply?", "What are shapes?", "Help with subtraction"],
    },
    marathi: {
      name: "मराठी शिक्षक",
      avatar: "📗",
      emoji: "📖",
      title: "नमस्ते! मी तुमचा मराठी शिक्षक आहे 👋",
      text:  "मराठी व्याकरण, कविता, धड्याबद्दल काहीही विचारा — मी सोप्या भाषेत सांगेन!",
      placeholder: "तुमचा प्रश्न विचारा… 😊",
      quick: ["संज्ञा म्हणजे काय?", "विशेषण म्हणजे काय?", "काव्य म्हणजे काय?", "मराठी वाक्य कसे लिहावे?"],
    },
    english: {
      name: "English Teacher",
      avatar: "📝",
      emoji: "✏️",
      title: "Hello! I'm your English Teacher 👋",
      text:  "Ask me about grammar, reading, writing, or any English words — I'll make it fun and easy!",
      placeholder: "Ask your English question… 😊",
      quick: ["What is a noun?", "What is a verb?", "How to write a sentence?", "What are vowels?"],
    },
  };

  // ── STATE ───────────────────────────────────────────────────────────────────
  let currentSubject = "maths";
  let sessionId      = "student_" + Math.random().toString(36).slice(2, 9);
  let isLoading      = false;

  // ── SWITCH SUBJECT ──────────────────────────────────────────────────────────
  function switchSubject(subject) {
    if (subject === currentSubject) return;
    currentSubject = subject;

    // body class for color theming
    document.body.className = subject;

    // tab active states
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.tab.${subject}`).classList.add('active');

    // card header
    const hdr = document.getElementById('cardHeader');
    hdr.className = `card-header ${subject}`;

    const s = SUBJECTS[subject];
    document.getElementById('teacherAvatar').textContent = s.avatar;
    document.getElementById('teacherName').textContent   = s.name;

    // clear messages & show welcome
    const msgs = document.getElementById('messages');
    msgs.innerHTML = `
      <div class="welcome" id="welcomeMsg">
        <div class="big-emoji">${s.emoji}</div>
        <h2>${s.title}</h2>
        <p>${s.text}</p>
      </div>`;

    // quick questions
    const qw = document.getElementById('quickWrap');
    qw.innerHTML = s.quick.map(q =>
      `<button class="quick-btn" onclick="askQuick(this)">${q}</button>`
    ).join('');

    // placeholder
    document.getElementById('questionInput').placeholder = s.placeholder;
  }

  // ── SEND QUESTION ───────────────────────────────────────────────────────────
  async function sendQuestion() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    if (!question || isLoading) return;

    removeWelcome();
    appendBubble("user", question, "🙋");
    input.value = '';
    autoResize(input);

    isLoading = true;
    document.getElementById('sendBtn').disabled = true;

    // typing indicator
    const typingId = appendTyping();

    try {
      const res = await fetch(API_URL, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject:    currentSubject,
          question:   question,
          session_id: sessionId,
        }),
      });

      const data = await res.json();
      removeTyping(typingId);

      if (data.answer) {
        const avatar = SUBJECTS[currentSubject].avatar;
        appendBubble("bot", data.answer, avatar);
      } else if (data.error) {
        appendBubble("bot", "⚠️ " + data.error, "⚠️");
      }
    } catch (err) {
      removeTyping(typingId);
      appendBubble("bot",
        "😔 Oops! I could not connect to the server. Please make sure the backend is running on port 5000.",
        "⚠️");
      console.error(err);
    }

    isLoading = false;
    document.getElementById('sendBtn').disabled = false;
    input.focus();
  }

  // ── QUICK QUESTION ──────────────────────────────────────────────────────────
  function askQuick(btn) {
    document.getElementById('questionInput').value = btn.textContent;
    sendQuestion();
  }

  // ── HELPERS ─────────────────────────────────────────────────────────────────
  function removeWelcome() {
    const w = document.getElementById('welcomeMsg');
    if (w) w.remove();
  }



  let typingCounter = 0;
  function appendTyping() {
    const id   = "typing_" + (++typingCounter);
    const msgs = document.getElementById('messages');
    const row  = document.createElement('div');
    row.className = 'bubble-row bot';
    row.id        = id;
    row.innerHTML = `
      <div class="avatar-sm">${SUBJECTS[currentSubject].avatar}</div>
      <div class="typing-indicator">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </div>`;
    msgs.appendChild(row);
    msgs.scrollTop = msgs.scrollHeight;
    return id;
  }

  function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function escapeHtml(t) {
    return t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }

  function parseMarkdown(text) {
    let t = escapeHtml(text);
    t = t.replace(/^#{1,6}\\s+(.+)$/gm, "<strong style='font-size:1.05rem;display:block;margin:8px 0 4px'>$1</strong>");
    t = t.replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>");
    t = t.replace(/__(.+?)__/g, "<strong>$1</strong>");
    t = t.replace(/^[\\*\\-]\\s+(.+)$/gm, "<li style='margin:4px 0'>$1</li>");
    t = t.replace(/^\\d+\\.\\s+(.+)$/gm, "<li style='margin:4px 0'>$1</li>");
    t = t.replace(/(<li[^>]*>.*?<\\/li>)/gs, function(m) { return "<ul style='padding-left:20px;margin:6px 0'>" + m + "</ul>"; });
    t = t.replace(/\\*([^\\*]+?)\\*/g, "<em>$1</em>");
    t = t.replace(/_([^_]+?)_/g, "<em>$1</em>");
    t = t.replace(/`([^`]+?)`/g, "<code style='background:#f0f2fa;padding:2px 6px;border-radius:4px;font-size:.88em;font-family:monospace'>$1</code>");
    t = t.replace(/^---+$/gm, "<hr style='border:none;border-top:1px solid #e0e4f0;margin:10px 0'>");
    t = t.replace(/\\n\\n/g, "<br><br>");
    t = t.replace(/\\n/g, "<br>");
    return t;
  }

  function appendBubble(role, text, avatar) {
    const msgs = document.getElementById("messages");
    const row  = document.createElement("div");
    row.className = "bubble-row " + role;

    if (role === "bot") {
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.innerHTML = parseMarkdown(text);

      const plain = text.replace(/[#*_`>~]/g, "").replace(/<[^>]*>/g, "").replace(/\\s+/g, " ").trim();

      const vBtn = document.createElement("button");
      vBtn.className = "voice-btn";
      vBtn.innerHTML = "🔊 Listen";
      vBtn.title = "Click to hear the answer";
      vBtn.addEventListener("click", () => toggleVoice(vBtn, plain));

      const wrap = document.createElement("div");
      wrap.className = "bubble-wrap";
      wrap.appendChild(bubble);
      wrap.appendChild(vBtn);

      const av = document.createElement("div");
      av.className = "avatar-sm";
      av.textContent = avatar;

      row.appendChild(av);
      row.appendChild(wrap);
    } else {
      row.innerHTML = "<div class='avatar-sm'>" + avatar + "</div><div class='bubble'>" + escapeHtml(text) + "</div>";
    }
    msgs.appendChild(row);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuestion();
    }
  }

  function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }

  // ── TEXT-TO-SPEECH ───────────────────────────────────────────────────────
  let _activeBtn = null;

  const VOICE_PRIORITY = {
    maths:   ["en-IN", "en-GB", "en-US", "en"],
    english: ["en-IN", "en-GB", "en-US", "en"],
    marathi: ["mr-IN", "mr", "hi-IN", "hi", "en-IN", "en"],
  };

  function pickVoice(subject) {
    const vs   = speechSynthesis.getVoices();
    const list = VOICE_PRIORITY[subject] || VOICE_PRIORITY.english;
    const femaleHints = ["female","woman","google","zira","heera","swara","priya",
                         "raveena","neerja","veena","samantha","karen","lekha","aditi"];
    const isFemale = v =>
      femaleHints.some(h => v.name.toLowerCase().includes(h)) &&
      !["male","man","david","james","ravi","hemant"].some(h => v.name.toLowerCase().includes(h));

    for (const lang of list) {
      let v = vs.find(v => v.lang === lang && isFemale(v));           if (v) return v;
          v = vs.find(v => v.lang.startsWith(lang) && isFemale(v));   if (v) return v;
          v = vs.find(v => v.lang === lang);                           if (v) return v;
          v = vs.find(v => v.lang.startsWith(lang));                   if (v) return v;
    }
    return vs.find(v => isFemale(v)) || vs[0] || null;
  }

  // pre-load voices as soon as they are available
  function _logVoices() {
    const vs = speechSynthesis.getVoices();
    if (!vs.length) return;
    console.log("🗣 Voices:", vs.map(v => v.lang + " - " + v.name));
    console.log("🇮🇳 Marathi:", vs.filter(v=>v.lang.startsWith("mr")).map(v=>v.name).join(", ") || "None");
    console.log("🇮🇳 Hindi:",   vs.filter(v=>v.lang.startsWith("hi")).map(v=>v.name).join(", ") || "None");
  }
  speechSynthesis.onvoiceschanged = _logVoices;
  // also try immediately (Chrome sometimes loads voices sync)
  if (speechSynthesis.getVoices().length) _logVoices();

  function stopSpeech() {
    speechSynthesis.cancel();
    if (_activeBtn) { _resetBtn(_activeBtn); _activeBtn = null; }
  }

  function _resetBtn(btn) {
    btn.classList.remove("speaking");
    btn.innerHTML = "🔊 Listen";
  }

  function toggleVoice(btn, plain) {
    if (btn.classList.contains("speaking")) { stopSpeech(); return; }
    stopSpeech();

    const utt  = new SpeechSynthesisUtterance(plain);
    utt.rate   = 0.88;
    utt.pitch  = 1.05;
    utt.volume = 1;

    const speak = () => {
      const v = pickVoice(currentSubject);
      if (v) { utt.voice = v; utt.lang = v.lang; }
      _activeBtn = btn;
      btn.classList.add("speaking");
      btn.innerHTML = '<div class="snd"><b></b><b></b><b></b><b></b><b></b></div> Stop';
      utt.onend = utt.onerror = () => { _resetBtn(btn); _activeBtn = null; };
      speechSynthesis.speak(utt);
    };

    // wait for voices if not loaded yet, without overwriting onvoiceschanged
    if (speechSynthesis.getVoices().length === 0) {
      const waitVoices = setInterval(() => {
        if (speechSynthesis.getVoices().length > 0) {
          clearInterval(waitVoices);
          speak();
        }
      }, 100);
      setTimeout(() => clearInterval(waitVoices), 3000); // give up after 3s
    } else {
      speak();
    }
  }

  // stop speech on subject switch
  const _origSwitch = switchSubject;
  switchSubject = function(subject) { stopSpeech(); _origSwitch(subject); };


  // ── SPEECH TO TEXT (Mic Input) ───────────────────────────────────────────
  let _recognition  = null;
  let _isListening  = false;

  // Language for recognition per subject
  const MIC_LANG = {
    maths:   "en-IN",
    english: "en-IN",
    marathi: "mr-IN",   // will fallback to hi-IN if not supported
  };

  function toggleMic() {
    if (_isListening) {
      stopMic();
    } else {
      startMic();
    }
  }

  function startMic() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Sorry! Your browser does not support speech input.\\nPlease use Google Chrome.");
      return;
    }

    _recognition = new SpeechRecognition();
    _recognition.lang        = MIC_LANG[currentSubject] || "en-IN";
    _recognition.interimResults = true;   // show words as you speak
    _recognition.continuous     = false;
    _recognition.maxAlternatives = 1;

    const input    = document.getElementById("questionInput");
    const micBtn   = document.getElementById("micBtn");
    let finalText  = "";

    _recognition.onstart = () => {
      _isListening = true;
      micBtn.classList.add("listening");
      micBtn.innerHTML = "⏹";
      micBtn.title = "Click to stop";
      input.classList.add("listening");
      input.placeholder = "🎤 Listening… speak now!";
    };

    _recognition.onresult = (e) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) finalText += t;
        else interim += t;
      }
      // show live preview in textarea
      input.value = finalText + interim;
      autoResize(input);
    };

    _recognition.onend = () => {
      _isListening = false;
      micBtn.classList.remove("listening");
      micBtn.innerHTML = "🎤";
      micBtn.title = "Speak your question";
      input.classList.remove("listening");
      input.placeholder = (SUBJECTS[currentSubject] && SUBJECTS[currentSubject].placeholder)
                          || "Ask your question here… 😊";
      // just fill the box — student presses ➤ to send
      if (finalText.trim()) {
        input.value = finalText.trim();
        autoResize(input);
        input.focus();   // focus so student can edit or press Enter
      }
    };

    _recognition.onerror = (e) => {
      _isListening = false;
      micBtn.classList.remove("listening");
      micBtn.innerHTML = "🎤";
      input.classList.remove("listening");
      input.placeholder = "Ask your question here… 😊";
      if (e.error === "not-allowed") {
        alert("Microphone access denied.\\nPlease allow microphone permission in your browser.");
      } else if (e.error === "no-speech") {
        input.placeholder = "No speech heard. Try again… 😊";
        setTimeout(() => {
          input.placeholder = SUBJECTS[currentSubject].placeholder || "Ask your question here… 😊";
        }, 2000);
      }
    };

    _recognition.start();
  }

  function stopMic() {
    if (_recognition) {
      _recognition.stop();
      _recognition = null;
    }
  }

  // stop mic when switching subject (language changes)
  const _origSwitchWithMic = switchSubject;
  switchSubject = function(subject) {
    stopMic();
    _origSwitchWithMic(subject);
  };

</script>
</body>
</html>"""

@app.route("/")
def index():
    return HTML_PAGE, 200, {"Content-Type": "text/html; charset=utf-8"}


# ══════════════════════════════════════════════════════════════════════════════
#  RAG — Google Embeddings + FAISS (pre-built indexes loaded from disk)
# ══════════════════════════════════════════════════════════════════════════════

def get_embedding(text):
    """Get single embedding from Google API."""
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(x*x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0
    return dot / (mag_a * mag_b)

def semantic_search(query, faiss_index, chunks, k=8):
    """Search using FAISS."""
    q_emb = get_embedding(query)
    if q_emb is None:
        print(f"⚠️  Embedding failed! Falling back to keyword search")
        return keyword_search(query, chunks, k)

    q_vec = np.array([q_emb], dtype=np.float32)
    faiss.normalize_L2(q_vec)
    distances, indices = faiss_index.search(q_vec, k)
    
    selected = [chunks[i] for i in indices[0] if i < len(chunks)]
    print(f"🔍 Semantic search: query='{query[:40]}...' | k={k} | found={len(selected)} | distances={distances[0][:3]}")
    return selected

def keyword_search(query, chunks, k=8):
    """Fallback keyword search."""
    keywords = query.lower().split()
    scored = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(chunk_lower.count(kw) for kw in keywords)
        if query.lower() in chunk_lower:
            score += 100
        if score > 0:
            scored.append((score, i))
    scored.sort(reverse=True)
    return [chunks[i] for _, i in scored[:k]]

def ask_gemini(question, context, system_prompt, history_text=""):
    """Call Gemini with context."""
    prompt = f"""System: {system_prompt}
"""
    if history_text:
        prompt += f"Previous conversation:\n{history_text}\n"

    prompt += f"""
Context from textbook:
{context}

Student Question: {question}

Answer:"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
                max_output_tokens=1000,
            )
        )
        return response.text.strip()
    except Exception as e:
        raise e


# ══════════════════════════════════════════════════════════════════════════════
#  SUBJECT CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SUBJECTS_CONFIG = {
    "maths": {
        "index":  "maths_faiss.index",
        "chunks": "maths_chunks.json",
        "system": """You are a kind and friendly primary school teacher teaching Mathematics.
Rules:
- Always answer in very simple English.
- Speak politely and encourage the student.
- Explain step by step with small numbers and easy examples.
- Show calculations clearly.
- Make maths easy and fun for a 2nd standard student.
- If the answer is not in the context, politely say I dont know."""
    },
    "marathi": {
        "index":  "marathi_faiss.index",
        "chunks": "marathi_chunks.json",
        "system": """तुम्ही एक प्रेमळ प्राथमिक शाळेचे मराठी शिक्षक आहात.
नियम:
- नेहमी मराठी भाषेत देवनागरी लिपीत उत्तर द्या.
- विद्यार्थ्याशी प्रेमाने बोला.
- खूप सोप्या शब्दांत पायरी-पायरीने समजावून सांगा.
- २रीच्या विद्यार्थ्याला समजेल असे बोला.
- जर माहिती नसेल तर नम्रपणे सांगा मला माहीत नाही."""
    },
    "english": {
        "index":  "english_faiss.index",
        "chunks": "english_chunks.json",
        "system": """You are a kind and friendly primary school teacher teaching English.
Rules:
- Always answer in simple English with very easy words.
- Speak politely and encourage the student.
- Explain step by step with small examples.
- Explain so a 2nd standard student can understand.
- If the answer is not in the context, politely say I dont know."""
    }
}

conversation_history = {}


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP — load pre-built FAISS indexes from disk (like hospital project)
# ══════════════════════════════════════════════════════════════════════════════
faiss_indexes = {}
chunks_store  = {}
_app_ready    = False

def _load_all():
    global faiss_indexes, chunks_store, _app_ready
    try:
        print("\n📚 Loading pre-built FAISS indexes...")
        for subject, cfg in SUBJECTS_CONFIG.items():
            idx_path    = cfg["index"]
            chunks_path = cfg["chunks"]

            if not os.path.exists(idx_path) or not os.path.exists(chunks_path):
                print(f"⚠️  Missing index files for {subject} — skipping")
                continue

            index = faiss.read_index(idx_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            faiss_indexes[subject] = index
            chunks_store[subject]  = chunks
            print(f"  ✅ {subject}: {index.ntotal} vectors, {len(chunks)} chunks")

        _app_ready = True
        print("🎉 All indexes loaded!\n")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        import traceback; traceback.print_exc()

import threading
_load_all()

# ══════════════════════════════════════════════════════════════════════════════
#  RATE LIMITING
# ══════════════════════════════════════════════════════════════════════════════
LIMITS = {"rpm_max": 10, "rpd_max": 100, "tpm_max": 200000}
state  = {"minute_requests": 0, "minute_tokens": 0, "day_requests": 0,
          "minute_window_start": time.time(), "day_window_start": time.time()}
lock   = threading.Lock()

def estimate_tokens(text): return max(1, len(text) // 4)

def reset_windows():
    now = time.time()
    if now - state["minute_window_start"] >= 60:
        state["minute_requests"] = state["minute_tokens"] = 0
        state["minute_window_start"] = now
    if now - state["day_window_start"] >= 86400:
        state["day_requests"] = 0
        state["day_window_start"] = now

def check_and_update(question, context):
    with lock:
        reset_windows()
        est = estimate_tokens(question + context)
        print(f"📊 Token estimate: question={len(question)} chars | context={len(context)} chars | total_est={est} tokens")
        if state["day_requests"] >= LIMITS["rpd_max"]:
            rem = 86400 - (time.time() - state["day_window_start"])
            return False, f"Daily limit reached. Resets in {int(rem//3600)}h {int((rem%3600)//60)}m. 😊"
        if state["minute_requests"] >= LIMITS["rpm_max"]:
            rem = 60 - (time.time() - state["minute_window_start"])
            return False, f"Too many requests! Please wait {int(rem)+1} seconds. 😊"
        if state["minute_tokens"] + est > LIMITS["tpm_max"]:
            rem = 60 - (time.time() - state["minute_window_start"])
            return False, f"Token limit reached. Please wait {int(rem)+1} seconds. 😊"
        state["minute_requests"] += 1
        state["minute_tokens"]   += est
        state["day_requests"]    += 1
        return True, ""


# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/ask", methods=["POST"])
def ask():
    if not _app_ready:
        return jsonify({"error": "⏳ Server is warming up! Please wait 30 seconds and try again."}), 503

    data     = request.get_json()
    subject  = data.get("subject", "").lower()
    question = data.get("question", "").strip()
    session  = data.get("session_id", "default")

    if not question:                 return jsonify({"error": "Empty question"}), 400
    if subject not in faiss_indexes: return jsonify({"error": "Unknown subject"}), 400

    chunks  = semantic_search(question, faiss_indexes[subject], chunks_store[subject], k=4)
    context = "\n\n".join(chunks)

    allowed, reason = check_and_update(question, context)
    if not allowed: return jsonify({"error": reason}), 429

    history      = conversation_history.get(session + subject, [])
    history_text = "".join(f"Student: {h['q']}\nTeacher: {h['a']}\n\n" for h in history[-4:])
    system       = SUBJECTS_CONFIG[subject]["system"]

    try:
        answer = ask_gemini(question, context, system, history_text)
        conversation_history.setdefault(session + subject, []).append({"q": question, "a": answer})
        return jsonify({"answer": answer})
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            delay = re.search(r'retry in (\d+)', err)
            wait  = f" Try again in {delay.group(1)} seconds." if delay else ""
            return jsonify({"error": f"⏳ API quota reached.{wait} Come back tomorrow!"}), 429
        return jsonify({"error": f"Something went wrong: {err[:200]}"}), 500

@app.route("/ready")
def ready():
    return jsonify({"ready": _app_ready, "message": "Warming up..." if not _app_ready else "All good!"})

@app.route("/status")
def status():
    with lock:
        reset_windows()
        return jsonify({
            "minute_requests": f"{state['minute_requests']} / {LIMITS['rpm_max']}",
            "minute_tokens":   f"{state['minute_tokens']} / {LIMITS['tpm_max']}",
            "day_requests":    f"{state['day_requests']} / {LIMITS['rpd_max']}",
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
