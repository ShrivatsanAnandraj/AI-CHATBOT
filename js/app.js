// Shri AI - chat frontend. Talks to the Flask backend at the same origin.

const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");

function addMsg(text, who) {
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function typing() {
  const div = document.createElement("div");
  div.className = "msg bot typing";
  div.innerHTML = "<span></span><span></span><span></span>";
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function send(text) {
  if (!text.trim()) return;
  addMsg(text, "user");
  input.value = "";
  const t = typing();
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    t.remove();
    addMsg(data.reply || "No reply.", "bot");
  } catch (e) {
    t.remove();
    if (location.protocol === "file:") {
      addMsg("You opened this as a file. Run the server and open http://127.0.0.1:5000/ instead: python app.py", "bot");
    } else {
      addMsg("Cannot reach the backend. Start it with: python app.py", "bot");
    }
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  send(input.value);
});

document.getElementById("chips").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) send(e.target.dataset.q);
});
