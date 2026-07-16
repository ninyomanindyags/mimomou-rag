const API_BASE = ""; // kosong = server yang sama (FastAPI serve frontend juga)
// Kalau frontend dijalankan terpisah (misal Live Server), isi:
// const API_BASE = "http://127.0.0.1:8000";

const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const modeSelect = document.getElementById("mode-select");
const newChatBtn = document.getElementById("new-chat-btn");

let sessionId = localStorage.getItem("mou_session_id") || null;

function addMessage(text, role) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  chatWindow.appendChild(el);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return el;
}

function addTypingIndicator() {
  const el = document.createElement("div");
  el.className = "msg typing";
  el.innerHTML = "<span></span><span></span><span></span>";
  chatWindow.appendChild(el);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return el;
}

async function startSession({ silent = false } = {}) {
  chatWindow.innerHTML = "";
  const res = await fetch(`${API_BASE}/api/session`, { method: "POST" });
  const data = await res.json();
  sessionId = data.session_id;
  localStorage.setItem("mou_session_id", sessionId);
  if (!silent) addMessage(data.opening_message, "bot");
}

async function sendMessage(message) {
  addMessage(message, "user");
  chatInput.value = "";
  chatInput.style.height = "auto";
  sendBtn.disabled = true;

  const typingEl = addTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message,
        mode: modeSelect.value,
      }),
    });

    if (!res.ok) {
      throw new Error(`Server error ${res.status}`);
    }

    const data = await res.json();
    typingEl.remove();
    addMessage(data.answer, "bot");
  } catch (err) {
    typingEl.remove();
    addMessage(
      "Gagal terhubung ke server mou. Pastikan backend (uvicorn) sedang berjalan.",
      "error"
    );
    console.error(err);
  } finally {
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  sendMessage(message);
});

// Kirim pesan dengan Enter, baris baru dengan Shift+Enter.
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

// Auto-grow textarea.
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = `${chatInput.scrollHeight}px`;
});

newChatBtn.addEventListener("click", async () => {
  if (sessionId) {
    await fetch(`${API_BASE}/api/session/${sessionId}`, { method: "DELETE" });
  }
  await startSession();
});

// Init saat halaman dibuka.
startSession();
