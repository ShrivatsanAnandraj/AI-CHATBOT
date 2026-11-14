// Shri AI admin panel - real backend calls (password gate on server).

const API = "";
let authed = sessionStorage.getItem("admin") === "1";

function show(id, html) {
  const el = document.getElementById(id);
  el.innerHTML = html;
  el.classList.add("show");
}
function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

async function api(path, body, method) {
  const res = await fetch(API + path, {
    method: method || "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (e) {}
  if (!res.ok && data && data.error === "unauthorized") {
    setAuth(false);
    throw new Error("unauthorized");
  }
  if (!res.ok) throw new Error((data && data.error) || "request failed");
  return data;
}

// ---------- auth ----------
function setAuth(on) {
  authed = on;
  document.getElementById("login-card").classList.toggle("hidden", on);
  document.getElementById("admin-card").classList.toggle("hidden", !on);
  if (on) { loadStats(); loadSamples(); loadMetricsTable(); }
}
async function login() {
  const p = document.getElementById("admin-pass").value;
  try {
    await api("/api/admin/login", { password: p });
    sessionStorage.setItem("admin", "1");
    setAuth(true);
  } catch (e) {
    show("login-result", '<span class="bad">Wrong password.</span>');
  }
}
function logout() {
  api("/api/admin/logout", {}).catch(() => {});
  sessionStorage.removeItem("admin");
  setAuth(false);
}
document.getElementById("admin-pass").addEventListener("keydown", (e) => {
  if (e.key === "Enter") login();
});
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
  try {
    const c = await api("/api/admin/stats", null, "GET");
    document.getElementById("s-labeled").textContent = c.labeled;
    document.getElementById("s-unlabeled").textContent = c.unlabeled;
    document.getElementById("s-predicted").textContent = c.predicted;
  } catch (e) {}
}

async function loadSamples() {
  try {
    const samples = await api("/api/admin/samples", null, "GET");
    const tbody = document.querySelector("#samples-table tbody");
    tbody.innerHTML = samples.length
      ? ""
      : '<tr><td colspan="5" style="color:var(--muted)">No samples yet. Add some above.</td></tr>';
    samples.slice(0, 30).forEach((s) => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + s.id + "</td>" +
        "<td>" + esc(s.domain) + "</td>" +
        "<td>" + (s.labeled ? '<span class="ok">labeled: ' + esc(s.label ?? "-") + "</span>"
                            : '<span style="color:var(--muted)">unlabeled</span>') + "</td>" +
        "<td>" + esc(s.preview) + "</td>" +
        '<td><button class="ghost" data-del="' + s.id + '">Delete</button></td>';
      tbody.appendChild(tr);
    });
    tbody.querySelectorAll("[data-del]").forEach((b) => {
      b.addEventListener("click", async () => {
        await api("/api/admin/samples/" + b.dataset.del, null, "DELETE");
        loadSamples(); loadStats();
      });
    });
  } catch (e) {}
}

// ---------- add sample ----------
function imageToPixels(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = 28; c.height = 28;
      const x = c.getContext("2d");
      x.fillStyle = "#000";
      x.fillRect(0, 0, 28, 28);
      const s = Math.min(28, 28);
      x.drawImage(img, (28 - 28) / 2, 0, 28, 28);
      const d = x.getImageData(0, 0, 28, 28).data;
      const out = [];
      for (let i = 0; i < d.length; i += 4) out.push(d[i] / 255);
      resolve(out);
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

async function addSample() {
  const domain = domainSelect.value;
  const labeled = document.getElementById("add-status").value === "labeled";
  let payload = {}, label = null, target = null;

  try {
    if (domain === "nlp") {
      payload = { text: document.getElementById("nlp-in").value.trim() };
      if (labeled) label = document.getElementById("nlp-label").value;
      if (!payload.text) throw new Error("Enter text.");
    } else if (domain === "ml") {
      const feats = document.getElementById("ml-in").value.split(",").map((v) => parseFloat(v.trim()));
      if (feats.length !== 30 || feats.some(isNaN)) throw new Error("ML needs exactly 30 numeric features.");
      payload = { features: feats };
      if (labeled) label = document.getElementById("ml-label").value;
    } else if (domain === "cv" || domain === "dl") {
      const f = document.getElementById(domain + "-in").files[0];
      if (!f) throw new Error("Choose an image.");
      payload = { pixels: await imageToPixels(f) };
      if (labeled) label = document.getElementById(domain + "-label").value;
    } else if (domain === "ds") {
      payload = {
        total_bill: parseFloat(document.getElementById("ds-bill-in").value),
        size: parseInt(document.getElementById("ds-size-in").value),
        day: parseInt(document.getElementById("ds-day-in").value),
        time: parseInt(document.getElementById("ds-time-in").value),
      };
      if (labeled) target = parseFloat(document.getElementById("ds-tip-in").value);
    }

    await api("/api/admin/samples", { domain, payload, labeled, label, target });
    show("add-result", '<span class="ok">Sample saved (' + (labeled ? "labeled" : "unlabeled") + ').</span>');
    loadSamples();
    loadStats();
  } catch (e) {
    show("add-result", '<span class="bad">' + esc(e.message) + "</span>");
  }
}

async function loadMLExample() {
  try {
    const r = await fetch("/api/example/ml");
    if (r.ok) {
      const row = (await r.json()).features;
      document.getElementById("ml-in").value = row.join(",");
      return;
    }
  } catch (e) {}
  document.getElementById("ml-in").value =
    "17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189";
}

// ---------- test unlabeled ----------
async function testUnlabeled() {
  const domain = document.getElementById("test-domain").value;
  const list = document.getElementById("test-list");
  try {
    const r = await api("/api/admin/test", { domain });
    show("test-result", '<span class="ok">Tested ' + r.items.length + " unlabeled sample(s).</span>");
    list.innerHTML = "";
    if (!r.items.length) {
      list.innerHTML = '<p class="hint">No unlabeled samples for this domain.</p>';
      return;
    }
    r.items.forEach((it) => {
      const div = document.createElement("div");
      div.className = "test-item";
      div.dataset.id = it.id;
      div.innerHTML =
        '<div class="pred">Prediction: ' + esc(it.pred) +
        (it.conf != null ? " (" + (it.conf * 100).toFixed(1) + "%)" : "") + "</div>" +
        '<div class="preview-text">' + esc(it.preview) + "</div>" +
        '<div class="actions">' +
        '<button class="primary" data-accept>Accept label</button>' +
        '<button class="ghost" data-reject>Reject / relabel</button>' +
        "</div>";
      list.appendChild(div);
    });
    list.querySelectorAll("[data-accept]").forEach((b) => {
      b.addEventListener("click", async () => {
        const item = b.closest(".test-item");
        const pred = item.querySelector(".pred").textContent.replace("Prediction: ", "").split(" (")[0];
        await api("/api/admin/samples/" + item.dataset.id + "/label", { label: pred });
        item.innerHTML = '<div class="ok">Labeled as ' + esc(pred) + ".</div>";
        loadStats(); loadSamples();
      });
    });
    list.querySelectorAll("[data-reject]").forEach((b) => {
      b.addEventListener("click", () => {
        const item = b.closest(".test-item");
        const actions = item.querySelector(".actions");
        actions.innerHTML =
          '<input placeholder="Correct label" style="max-width:220px;margin-top:6px" />' +
          '<button class="primary" data-save>Save</button>';
        actions.querySelector("[data-save]").addEventListener("click", async () => {
          const v = actions.querySelector("input").value.trim();
          if (!v) return;
          await api("/api/admin/samples/" + item.dataset.id + "/label", { label: v });
          actions.innerHTML = '<div class="ok">Saved.</div>';
          loadStats(); loadSamples();
        });
      });
    });
  } catch (e) {
    show("test-result", '<span class="bad">' + esc(e.message) + "</span>");
  }
}

// ---------- retrain ----------
let pollTimer = null;
async function retrain() {
  const prog = document.getElementById("retrain-progress");
  const bar = document.getElementById("retrain-bar");
  prog.classList.remove("hidden");
  bar.style.width = "5%";
  show("retrain-result", "Training started...");
  try {
    await api("/api/admin/retrain", {});
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const s = await api("/api/admin/train/status", null, "GET");
        if (s.total) {
          bar.style.width = Math.round((s.step / s.total) * 100) + "%";
          show("retrain-result", "Training " + esc(s.head) + " - step " + s.step + " / " + s.total);
        }
        if (s.error) {
          clearInterval(pollTimer);
          show("retrain-result", '<span class="bad">Error: ' + esc(s.error) + "</span>");
        } else if (s.done) {
          clearInterval(pollTimer);
          bar.style.width = "100%";
          show("retrain-result", '<span class="ok">Retrain complete. Metrics updated.</span>');
          loadMetricsTable();
        }
      } catch (e) {}
    }, 1500);
  } catch (e) {
    show("retrain-result", '<span class="bad">' + esc(e.message) + "</span>");
  }
}

async function loadMetricsTable() {
  let m = {};
  try { m = await fetch("/api/metrics").then((r) => r.json()); } catch (e) {}
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
