// SchoolAI user lab - talks to /api/* when a backend is present,
// falls back to demo mode so the page works standalone (e.g. GitHub Pages).

const API = "";

async function api(path, body) {
  try {
    const res = await fetch(API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) throw new Error("bad status");
    return await res.json();
  } catch (e) {
    return null; // demo mode
  }
}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    const group = btn.parentElement;
    group.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const wrap = group.parentElement;
    wrap.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    wrap.querySelector("#panel-" + btn.dataset.tab).classList.add("active");
  });
});

// ---------- metrics ----------
async function loadMetrics() {
  let m = null;
  try {
    const res = await fetch(API + "/api/metrics");
    if (res.ok) m = await res.json();
  } catch (e) {}
  if (!m) {
    m = { cv: { accuracy: 0.68 }, dl: { accuracy: 0.84 }, nlp: { accuracy: 0.64 },
          ml: { accuracy: 0.95 }, ds: { mae: 0.72 } };
  }
  const pct = (v) => (v * 100).toFixed(1) + "%";
  document.getElementById("m-cv").textContent = m.cv && m.cv.accuracy != null ? pct(m.cv.accuracy) : "-";
  document.getElementById("m-dl").textContent = m.dl && m.dl.accuracy != null ? pct(m.dl.accuracy) : "-";
  document.getElementById("m-nlp").textContent = m.nlp && m.nlp.accuracy != null ? pct(m.nlp.accuracy) : "-";
  document.getElementById("m-ml").textContent = m.ml && m.ml.accuracy != null ? pct(m.ml.accuracy) : "-";
  document.getElementById("m-ds").textContent = m.ds && m.ds.mae != null ? "$" + Number(m.ds.mae).toFixed(2) : "-";
}
loadMetrics();

// ---------- chat ----------
const chat = document.getElementById("chat");
document.getElementById("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (!text) return;
  addMsg(text, "user");
  input.value = "";
  const data = await api("/api/chat", { message: text });
  const reply = data && data.reply
    ? data.reply
    : "Demo mode: I would route this to the matching domain. Start the backend (python app.py) for live predictions. Try a tab below: sentiment text, a digit, an image, features, or a tip request.";
  setTimeout(() => addMsg(reply, "bot"), 350);
});

function addMsg(text, who) {
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

// ---------- helpers ----------
function show(id, html) {
  const el = document.getElementById(id);
  el.innerHTML = html;
  el.classList.add("show");
}
function fileToCanvas(file, canvasId) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const c = document.getElementById(canvasId);
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, c.width, c.height);
      const s = Math.min(c.width, c.height);
      ctx.drawImage(img, (c.width - s) / 2, (c.height - s) / 2, s, s);
      URL.revokeObjectURL(img.src);
      resolve();
    };
    img.src = URL.createObjectURL(file);
  });
}

// ---------- DS ----------
async function predictDS() {
  const body = {
    total_bill: parseFloat(document.getElementById("ds-bill").value),
    size: parseInt(document.getElementById("ds-size").value),
    day: parseInt(document.getElementById("ds-day").value),
    time: parseInt(document.getElementById("ds-time").value),
  };
  const r = await api("/api/predict/ds", body);
  const tip = r && r.tip != null ? r.tip : body.total_bill * 0.18;
  show("ds-result",
    'Predicted tip: <span class="big">$' + Number(tip).toFixed(2) + "</span>" +
    (r ? "" : " <span class=\"hint\">(demo estimate)</span>"));
}

// ---------- ML ----------
async function loadMLExample(admin) {
  let row = null;
  try {
    const res = await fetch(API + "/api/example/ml");
    if (res.ok) row = (await res.json()).features;
  } catch (e) {}
  if (!row) row = [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189];
  const text = row.join(",");
  const el = document.getElementById(admin ? "ml-in" : "ml-features");
  if (el) el.value = text;
}

async function predictML() {
  const raw = document.getElementById("ml-features").value.trim();
  const features = raw.split(",").map((v) => parseFloat(v.trim())).filter((v) => !isNaN(v));
  if (features.length !== 30) {
    show("ml-result", '<span class="bad">Please provide exactly 30 numeric features.</span>');
    return;
  }
  const r = await api("/api/predict/ml", { features });
  const label = r && r.label != null ? r.label : 0;
  const conf = r && r.confidence != null ? " (" + (r.confidence * 100).toFixed(1) + "%)" : " (demo)";
  show("ml-result",
    'Prediction: <span class="big">' + (label === 1 ? "Malignant" : "Benign") + "</span>" + conf);
}

// ---------- DL ----------
const dlCanvas = document.getElementById("dl-canvas");
const dlCtx = dlCanvas.getContext("2d");
dlCtx.fillStyle = "#000";
dlCtx.fillRect(0, 0, dlCanvas.width, dlCanvas.height);
let drawing = false;
dlCanvas.addEventListener("mousedown", () => (drawing = true));
dlCanvas.addEventListener("mouseup", () => (drawing = false));
dlCanvas.addEventListener("mouseleave", () => (drawing = false));
dlCanvas.addEventListener("mousemove", (e) => {
  if (!drawing) return;
  const r = dlCanvas.getBoundingClientRect();
  dlCtx.fillStyle = "#fff";
  dlCtx.beginPath();
  dlCtx.arc(e.clientX - r.left, e.clientY - r.top, 12, 0, Math.PI * 2);
  dlCtx.fill();
});
function clearCanvas() {
  dlCtx.fillStyle = "#000";
  dlCtx.fillRect(0, 0, dlCanvas.width, dlCanvas.height);
}
document.getElementById("dl-file").addEventListener("change", async (e) => {
  if (e.target.files[0]) await fileToCanvas(e.target.files[0], "dl-canvas");
});
function canvasPixels(canvas) {
  const c = document.createElement("canvas");
  c.width = 28; c.height = 28;
  const x = c.getContext("2d");
  x.fillStyle = "#000";
  x.fillRect(0, 0, 28, 28);
  x.drawImage(canvas, 0, 0, 28, 28);
  const d = x.getImageData(0, 0, 28, 28).data;
  const out = [];
  for (let i = 0; i < d.length; i += 4) out.push(d[i] / 255);
  return out;
}
async function predictDL() {
  const r = await api("/api/predict/dl", { pixels: canvasPixels(dlCanvas) });
  const label = r && r.label != null ? r.label : Math.floor(Math.random() * 10);
  const conf = r && r.confidence != null ? " (" + (r.confidence * 100).toFixed(1) + "%)" : " (demo)";
  show("dl-result", 'Digit: <span class="big">' + label + "</span>" + conf);
}

// ---------- NLP ----------
async function predictNLP() {
  const text = document.getElementById("nlp-text").value.trim();
  if (!text) { show("nlp-result", '<span class="bad">Enter some text first.</span>'); return; }
  const r = await api("/api/predict/nlp", { text });
  const label = r && r.label != null ? r.label : 1;
  const conf = r && r.confidence != null ? " (" + (r.confidence * 100).toFixed(1) + "%)" : " (demo)";
  show("nlp-result",
    'Sentiment: <span class="big">' + (label === 1 ? "Positive" : "Negative") + "</span>" + conf);
}

// ---------- CV ----------
document.getElementById("cv-file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const prev = document.getElementById("cv-preview");
  prev.src = URL.createObjectURL(f);
  prev.style.display = "block";
});
async function predictCV() {
  const f = document.getElementById("cv-file").files[0];
  if (!f) { show("cv-result", '<span class="bad">Choose an image first.</span>'); return; }
  const c = document.createElement("canvas");
  c.width = 28; c.height = 28;
  const img = new Image();
  img.onload = async () => {
    const x = c.getContext("2d");
    x.fillStyle = "#000";
    x.fillRect(0, 0, 28, 28);
    const s = Math.min(28, 28);
    x.drawImage(img, 0, 0, 28, 28);
    const d = x.getImageData(0, 0, 28, 28).data;
    const pixels = [];
    for (let i = 0; i < d.length; i += 4) pixels.push(d[i] / 255);
    const r = await api("/api/predict/cv", { pixels });
    const classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"];
    const idx = r && r.label != null ? r.label : Math.floor(Math.random() * 10);
    const conf = r && r.confidence != null ? " (" + (r.confidence * 100).toFixed(1) + "%)" : " (demo)";
    show("cv-result", 'Class: <span class="big">' + classes[idx] + "</span>" + conf);
  };
  img.src = URL.createObjectURL(f);
}
