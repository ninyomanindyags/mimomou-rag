// =========================================================================
// KONFIGURASI
// =========================================================================
const API_BASE = ""; // kosong = server yang sama (FastAPI serve frontend juga)
// Kalau frontend dijalankan terpisah (misal Live Server), isi:
// const API_BASE = "http://127.0.0.1:8000";

// =========================================================================
// REFERENSI ELEMEN DOM
// =========================================================================
const landingScreen = document.getElementById("landing-screen");
const chatScreen = document.getElementById("chat-screen");
const startChatBtn = document.getElementById("start-chat-btn");

const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const modeToggle = document.getElementById("mode-toggle");
const modeButtons = modeToggle.querySelectorAll(".mode-btn");
const newChatBtn = document.getElementById("new-chat-btn");

// =========================================================================
// STATE
// =========================================================================
// sessionId dipertahankan di localStorage supaya percakapan tetap ada
// walau halaman di-refresh.
let sessionId = localStorage.getItem("mou_session_id") || null;

// currentMode menyimpan pilihan metode retrieval yang sedang aktif
// ("SCG" atau "Baseline"). Ini yang dikirim ke backend tiap chat,
// jadi backend tahu pipeline mana yang harus dipakai untuk menjawab.
let currentMode =
  modeToggle.querySelector(".mode-btn.active")?.dataset.mode || "SCG";

// =========================================================================
// TOGGLE MODE (SCG vs Baseline)
// =========================================================================
// Klik salah satu pill akan mengaktifkannya secara visual (class "active"
// + aria-selected untuk aksesibilitas) dan memperbarui currentMode.
// Tidak memanggil API apa pun di sini — mode baru baru dipakai saat
// pesan berikutnya dikirim.
modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    modeButtons.forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    currentMode = btn.dataset.mode;
  });
});

// =========================================================================
// RENDER MARKDOWN RINGAN (khusus jawaban bot)
// =========================================================================
// Jawaban dari LLM sering ngandung markdown sederhana kayak **teks tebal**.
// Dua langkah:
//   1) escapeHtml -- escape karakter <, >, & dulu SEBELUM apa pun, supaya
//      teks apa adanya dari LLM tidak pernah dianggap tag HTML beneran
//      (kalau ini dilewat, bisa jadi celah XSS atau bikin layout rusak).
//   2) baru sesudah aman, ganti pola **teks** jadi <strong>teks</strong>.
// Urutannya wajib escape dulu baru markdown, kalau kebalik tag <strong>
// yang kita buat sendiri bakal ikut ke-escape jadi teks biasa.
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(text) {
  const escaped = escapeHtml(text);
  // **teks** -> <strong>teks</strong> (non-greedy, biar **a** **b** kebaca
  // sebagai 2 bold terpisah, bukan 1 bold raksasa dari ** pertama ke ** terakhir)
  return escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

// =========================================================================
// RENDER PESAN
// =========================================================================
// Menambahkan satu bubble pesan ke chat window.
// role: "user" | "bot" | "error"
// Khusus role "bot", teks diproses lewat renderMarkdown supaya **bold**
// dari jawaban LLM benar-benar tampil tebal, bukan literal tanda bintang.
// Pesan user/error tetap pakai textContent (plain, tidak perlu markdown,
// dan lebih aman karena tidak pernah lewat innerHTML).
function addMessage(text, role) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  if (role === "bot") {
    el.innerHTML = renderMarkdown(text);
  } else {
    el.textContent = text;
  }
  chatWindow.appendChild(el);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return el;
}

// Bubble "..." animasi saat menunggu jawaban dari backend.
// Dikembalikan sebagai elemen supaya bisa di-remove begitu jawaban datang.
function addTypingIndicator() {
  const el = document.createElement("div");
  el.className = "msg typing";
  el.innerHTML = "<span></span><span></span><span></span>";
  chatWindow.appendChild(el);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return el;
}

// =========================================================================
// NAVIGASI ANTAR SCREEN (landing <-> chat)
// =========================================================================
function showChatScreen() {
  landingScreen.classList.add("hidden");
  chatScreen.classList.remove("hidden");
}

function showLandingScreen() {
  chatScreen.classList.add("hidden");
  landingScreen.classList.remove("hidden");
}

// =========================================================================
// SESSION LIFECYCLE
// =========================================================================
// Membuat session baru di backend (POST /api/session), menyimpan
// session_id, lalu menampilkan opening message dari bot.
// silent=true dipakai kalau kita tidak mau menampilkan opening message
// (tidak dipakai saat ini, tapi disediakan untuk fleksibilitas).
async function startSession({ silent = false } = {}) {
  chatWindow.innerHTML = "";
  const res = await fetch(`${API_BASE}/api/session`, { method: "POST" });
  const data = await res.json();
  sessionId = data.session_id;
  localStorage.setItem("mou_session_id", sessionId);
  if (!silent) addMessage(data.opening_message, "bot");
}

// =========================================================================
// KIRIM PESAN
// =========================================================================
// Alur: render pesan user -> tampilkan typing indicator -> POST /api/chat
// dengan mode yang sedang aktif -> render jawaban bot, atau pesan error
// kalau request gagal (mis. backend belum jalan).
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
        mode: currentMode, // "SCG" atau "Baseline", dari toggle di header
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

// =========================================================================
// EVENT LISTENERS — FORM & INPUT
// =========================================================================
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

// Auto-grow textarea mengikuti panjang teks yang diketik.
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = `${chatInput.scrollHeight}px`;
});

// =========================================================================
// EVENT LISTENERS — MULAI CHAT (dari landing screen)
// =========================================================================
// Tombol besar di landing screen: pindah ke chat screen lalu buat session.
startChatBtn.addEventListener("click", async () => {
  showChatScreen();
  await startSession();
});

// =========================================================================
// EVENT LISTENERS — CHAT BARU (dari dalam chat screen)
// =========================================================================
// Menghapus session lama di backend (kalau ada), lalu membuat session baru.
// Tetap di chat screen — tidak balik ke landing.
newChatBtn.addEventListener("click", async () => {
  if (sessionId) {
    await fetch(`${API_BASE}/api/session/${sessionId}`, { method: "DELETE" });
  }
  await startSession();
});

// =========================================================================
// INIT — dijalankan begitu halaman dibuka
// =========================================================================
// Mulai dari landing screen. Session baru dibuat begitu user pencet
// "Mulai Chat Baru" (lihat listener di atas), bukan otomatis saat load.
showLandingScreen();