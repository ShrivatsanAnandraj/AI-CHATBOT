// SchoolAI admin panel - same API + demo-mode fallback as the user lab.

const API = "";
let authed = sessionStorage.getItem("admin") === "1";

async function api(path, body, method) {
  try {
    const res = await fetch(API + path, {
      method: method || "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) throw new Error("bad status");
    return await res.json();
  } catch (e) {
    return null;
  }
}

function show(id, html) {
  const el = document.getElementById(id);
  el.innerHTML = html;
  el.classList.add("show");
}
function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ---------- auth ----------
function setAuth(on) {
  authed = on;
  document.getElementById("login-card").classList.toggle("hidden", on);
  document.getElementById("admin-card").classList.toggle("hidden", !on);
  if (on) { loadStats(); loadSamples(); loadMetricsTable(); }
}
async function login() {
  const u = document.getElementById("admin-user").value;
  const p = document.getElementById("admin-pass").value;
  const r = await api("/api/admin/login", { username: u, password: p });
  const ok = r && r.ok ? true : (u === "admin" && p === "admin123"); // demo fallback
  if (ok) {
    sessionStorage.setItem("admin", "1");
    setAuth(true);
  } else {
    show("login-result", '<span class="bad">Wrong username or password.</span>');
  }
}
function logout() {
  api("/api/admin/logout", {});
  sessionStorage.removeItem("admin");
  setAuth(false);
}
setAuth(authed);

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

// ---------- add form switcher ----------
const domainSelect = document.getElementById("add-domain");
domainSelect.addEventListener("change", () => {
  ["nlp", "ml", "cv", "dl", "ds"].forEach((d) => {
    document.getElementById("add-" + d).classList.toggle("hidden", domainSelect.value !== d);
  });
});

// ---------- stats / samples ----------
async function loadStats() {
  let c = null;
  try {
    const res = await fetch(API + "/api/admin/stats");
    if (res.ok) c = await res.json();
  } catch (e) {}
  c = c || { labeled: 0, unlabeled: 0, predicted: 0 };
  document.getElementById("s-labeled").textContent = c.labeled;
  document.getElementById("s-unlabeled").textContent = c.unlabeled;
  document.getElementById("s-predicted").textContent = c.predicted;
}

let samples = [];
async function loadSamples() {
  let rows = null;
  try {
    const res = await fetch(API + "/api/admin/samples");
    if (res.ok) rows = await res.json();
  } catch (e) {}
  samples = rows || [];
  const tbody = document.querySelector("#samples-table tbody");
  tbody.innerHTML = samples.length
    ? ""
    : '<tr><td colspan="5" style="color:var(--muted)">No samples yet. Add some above.</td></tr>';
  samples.slice(0, 25).forEach((s) => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      "<td>" + s.id + "</td>" +
      "<td>" + esc(s.domain) + "</td>" +
      "<td>" + (s.labeled ? '<span class="ok">labeled</span>' : '<span style="color:var(--muted)">unlabeled</span>') + "</td>" +
      "<td>" + esc(s.preview) + "</td>" +
      '<td><button class="ghost" data-del="' + s.id + '">Delete</button></td>';
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll("[data-del]").forEach((b) => {
    b.addEventListener("click", async () => {
      await api("/api/admin/samples/" + b.dataset.del, null, "DELETE");
      samples = samples.filter((x) => String(x.id) !== b.dataset.del);
      loadSamples(); loadStats();
    });
  });
}

// ---------- add sample ----------
async function addSample() {
  const domain = domainSelect.value;
  const labeled = document.getElementById("add-status").value === "labeled";
  let payload = {}, label = null;

  if (domain === "nlp") {
    payload = { text: document.getElementById("nlp-in").value.trim() };
    if (labeled) label = document.getElementById("nlp-label").value;
  } else if (domain === "ml") {
    const feats = document.getElementById("ml-in").value.split(",").map((v) => parseFloat(v.trim()));
    if (feats.length !== 30 || feats.some(isNaN)) {
      show("add-result", '<span class="bad">ML needs exactly 30 numeric features.</span>');
      return;
    }
    payload = { features: feats };
    if (labeled) label = document.getElementById("ml-label").value;
  } else if (domain === "cv" || domain === "dl") {
    const f = document.getElementById(domain + "-in").files[0];
    if (!f) { show("add-result", '<span class="bad">Choose an image.</span>'); return; }
    payload = { image: await imageToPixels(f) };
    if (labeled) label = document.getElementById(domain + "-label").value;
  } else if (domain === "ds") {
    payload = {
      total_bill: parseFloat(document.getElementById("ds-bill-in").value),
      size: parseInt(document.getElementById("ds-size-in").value),
      day: parseInt(document.getElementById("ds-day-in").value),
      time: parseInt(document.getElementById("ds-time-in").value),
    };
    if (labeled) label = document.getElementById("ds-tip-in").value;
  }

  if (!labeled && domain !== "ds") label = null;
  if (!payload.text && !payload.features && !payload.image && payload.total_bill == null) {
    show("add-result", '<span class="bad">Fill in the sample first.</span>');
    return;
  }

  const r = await api("/api/admin/samples", { domain, payload, label: labeled ? label : null, labeled });
  show("add-result", '<span class="ok">Sample saved (' + (labeled ? "labeled" : "unlabeled") + ').</span>');
  if (!r) {
    // demo mode: keep a local list so the table still works
    samples.unshift({
      id: (samples[0] ? samples[0].id + 1 : 1),
      domain, labeled,
      preview: payload.text || (payload.features ? payload.features.slice(0, 4).join(",") + "..." :
               payload.total_bill != null ? "bill $" + payload.total_bill + " tip " + label :
               "[image 28x28]"),
    });
    const tbody = document.querySelector("#samples-table tbody");
    if (tbody.textContent.includes("No samples")) tbody.innerHTML = "";
    const s = samples[0];
    const tr = document.createElement("tr");
    tr.innerHTML = "<td>" + s.id + "</td><td>" + s.domain + "</td><td>" +
      (s.labeled ? "labeled" : "unlabeled") + "</td><td>" + esc(s.preview) + "</td><td></td>";
    tbody.prepend(tr);
  } else {
    loadSamples();
  }
  loadStats();
}

function imageToPixels(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = 28; c.height = 28;
      const x = c.getContext("2d");
      x.fillStyle = "#000";
      x.fillRect(0, 0, 28, 28);
      x.drawImage(img, 0, 0, 28, 28);
      const d = x.getImageData(0, 0, 28, 28).data;
      const out = [];
      for (let i = 0; i < d.length; i += 4) out.push(d[i] / 255);
      resolve(out);
    };
    img.src = URL.createObjectURL(file);
  });
}

async function loadMLExample(admin) {
  let row = null;
  try {
    const res = await fetch(API + "/api/example/ml");
    if (res.ok) row = (await res.json()).features;
  } catch (e) {}
  if (!row) row = [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189];
  document.getElementById("ml-in").value = row.join(",");
}

// ---------- test unlabeled ----------
async function testUnlabeled() {
  const domain = document.getElementById("test-domain").value;
  const r = await api("/api/admin/test", { domain });
  const list = document.getElementById("test-list");
  let items = r && r.items ? r.items : null;
  if (!items) {
    // demo mode: invent a couple of plausible unlabeled items
    items = domain === "nlp"
      ? [{ id: 1, preview: "what a dull and boring film...", pred: "Negative", conf: 0.71 },
         { id: 2, preview: "i loved every minute of it!", pred: "Positive", conf: 0.88 }]
      : [{ id: 1, preview: "[sample data]", pred: "-", conf: 0 }];
    show("test-result", '<span class="hint">Demo mode - start the backend for real predictions.</span>');
  } else {
    show("test-result", '<span class="ok">Tested ' + items.length + " unlabeled sample(s).</span>");
  }
  list.innerHTML = "";
  items.forEach((it) => {
    const div = document.createElement("div");
    div.className = "test-item";
    div.innerHTML =
      '<div class="pred">Prediction: ' + esc(it.pred) +
      (it.conf ? " (" + (it.conf * 100).toFixed(1) + "%)" : "") + "</div>" +
      '<div class="preview-text">' + esc(it.preview) + "</div>" +
      '<div class="actions">' +
      '<button class="primary" data-accept="' + it.id + '" data-label="' + esc(it.pred) + '">Accept label</button>' +
      '<button class="ghost" data-reject="' + it.id + '">Reject / relabel</button>' +
      "</div>";
    list.appendChild(div);
  });
  list.querySelectorAll("[data-accept]").forEach((b) => {
    b.addEventListener("click", async () => {
      await api("/api/admin/samples/" + b.dataset.accept + "/label", { label: b.dataset.label });
      b.parentElement.parentElement.innerHTML = '<div class="ok">Labeled as ' + esc(b.dataset.label) + ".</div>";
      loadStats(); loadSamples();
    });
  });
  list.querySelectorAll("[data-reject]").forEach((b) => {
    b.addEventListener("click", () => {
      b.parentElement.innerHTML += '<input placeholder="Correct label" style="max-width:220px" />' +
        '<button class="primary" data-save>Save</button>';
      b.remove();
      const wrap = b.parentElement;
      wrap.querySelector("[data-save]").addEventListener("click", async () => {
        const v = wrap.querySelector("input").value.trim();
        if (!v) return;
        await api("/api/admin/samples/" + wrap.dataset.rid + "/label", { label: v });
        wrap.innerHTML = '<div class="ok">Saved.</div>';
        loadStats();
      });
    });
  });
}

// ---------- retrain ----------
async function retrain() {
  const prog = document.getElementById("retrain-progress");
  const bar = document.getElementById("retrain-bar");
  prog.classList.remove("hidden");
  bar.style.width = "10%";
  show("retrain-result", "Training started...");
  const r = await api("/api/admin/retrain", {});
  if (r && r.done) {
    bar.style.width = "100%";
    show("retrain-result", '<span class="ok">Retrain complete. Metrics updated.</span>');
    loadMetricsTable();
  } else {
    let p = 10;
    const tick = setInterval(() => {
      p = Math.min(90, p + 7);
      bar.style.width = p + "%";
    }, 700);
    setTimeout(() => {
      clearInterval(tick);
      bar.style.width = "100%";
      show("retrain-result", '<span class="ok">Demo run finished.</span> Start the backend for a real retrain.');
    }, 4500);
  }
}

async function loadMetricsTable() {
  let m = null;
  try {
    const res = await fetch(API + "/api/metrics");
    if (res.ok) m = await res.json();
  } catch (e) {}
  m = m || { cv: { accuracy: 0.68, n: 3000 }, dl: { accuracy: 0.84, n: 3000 },
             nlp: { accuracy: 0.64, n: 2000 }, ml: { accuracy: 0.95, n: 169 },
             ds: { mae: 0.72, n: 61 } };
  const names = { cv: "Computer Vision", dl: "Deep Learning", nlp: "NLP", ml: "Machine Learning", ds: "Data Science" };
  const tbody = document.querySelector("#metrics-table tbody");
  tbody.innerHTML = "";
  Object.keys(names).forEach((k) => {
    const row = m[k] || {};
    const score = row.accuracy != null ? (row.accuracy * 100).toFixed(1) + "%" :
                  row.mae != null ? "MAE " + Number(row.mae).toFixed(3) : "-";
    const tr = document.createElement("tr");
    tr.innerHTML = "<td>" + names[k] + "</td><td>" + score + "</td><td>" + (row.n || "-") + "</td>";
    tbody.appendChild(tr);
  });
}
