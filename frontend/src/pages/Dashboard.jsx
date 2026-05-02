import { useState, useEffect, useRef, useCallback } from "react";
import api from "../lib/api";

const FRUIT_EMOJI = {
  banana: "🍌", apple: "🍎", mango: "🥭", orange: "🍊",
  tomato: "🍅", strawberry: "🍓", grape: "🍇", peach: "🍑",
  kiwi: "🥝", pear: "🍐", avocado: "🥑", cherry: "🍒",
  corn: "🌽", pepper: "🫑", physalis: "🟠", cactus: "🌵",
  zucchini: "🥒",
};

const FRUITS = [
  { key: "banana", label: "Banana" },
  { key: "orange", label: "Orange" },
  { key: "tomato", label: "Tomato" },
  { key: "avocado", label: "Avocado" },
  { key: "cherry", label: "Cherry" },
  { key: "corn", label: "Corn" },
  { key: "pepper", label: "Pepper" },
  { key: "physalis", label: "Physalis" },
  { key: "zucchini", label: "Zucchini" },
  { key: "cactus", label: "Cactus" },
];

const GRADE_TIPS = {
  ripe: "Perfect to eat right now! 🎉",
  unripe: "Give it a few more days to ripen.",
  overripe: "Best used in smoothies or cooking.",
};

const TECH_STACK = [
  { icon: "🧠", name: "ONNX Runtime", desc: "Runs 10 custom-trained fruit ripeness models" },
  { icon: "⚡", name: "FastAPI", desc: "High-performance Python backend API" },
  { icon: "⚛️", name: "React + Vite", desc: "Fast, modern frontend with glassmorphism UI" },
  { icon: "🐳", name: "Docker", desc: "Containerized backend for consistent deployment" },
  { icon: "🚀", name: "Render", desc: "Cloud hosting for the AI backend" },
  { icon: "▲", name: "Vercel", desc: "Edge-deployed frontend for global speed" },
];

const HOW_IT_WORKS = [
  { step: "1", icon: "📸", title: "Upload a Photo", desc: "Take or upload a clear photo of any supported fruit." },
  { step: "2", icon: "🤖", title: "AI Detection", desc: "Our models analyze color, texture, and visual patterns." },
  { step: "3", icon: "📊", title: "Ripeness Grade", desc: "Get instant results: Unripe, Ripe, or Overripe with confidence scores." },
  { step: "4", icon: "📋", title: "Track History", desc: "Every scan is saved locally so you can track freshness over time." },
];

const LOCAL_HISTORY_KEY = "fruitsense_history";

function ConfBar({ label, value, color }) {
  const [width, setWidth] = useState(0);
  useEffect(() => { setTimeout(() => setWidth(value), 100); }, [value]);
  return (
    <div className="prob-row">
      <div className="prob-label">
        <span style={{ textTransform: "capitalize" }}>{label}</span>
        <span style={{ fontWeight: 600, color }}>{value.toFixed(1)}%</span>
      </div>
      <div className="conf-bar-bg">
        <div className="conf-bar-fill" style={{ width: `${width}%`, background: color }} />
      </div>
    </div>
  );
}

export default function Dashboard({ theme, toggleTheme }) {
  const [tab, setTab] = useState("scan");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [dragOver, setDragOver] = useState(false);
  const [listening, setListening] = useState(false);
  const [voiceText, setVoiceText] = useState("");
  const [error, setError] = useState("");
  const [selectedFruit, setSelectedFruit] = useState(null);
  const fileRef = useRef();
  const recognitionRef = useRef(null);

  useEffect(() => {
    loadLocalHistory();
    initVoice();
  }, []);

  const loadLocalHistory = () => {
    try {
      const stored = localStorage.getItem(LOCAL_HISTORY_KEY);
      if (stored) setHistory(JSON.parse(stored));
    } catch {}
  };

  const saveLocalHistory = (entry) => {
    try {
      const stored = localStorage.getItem(LOCAL_HISTORY_KEY);
      const existing = stored ? JSON.parse(stored) : [];
      const updated = [entry, ...existing].slice(0, 50);
      localStorage.setItem(LOCAL_HISTORY_KEY, JSON.stringify(updated));
      setHistory(updated);
    } catch {}
  };

  const clearHistory = () => {
    localStorage.removeItem(LOCAL_HISTORY_KEY);
    setHistory([]);
  };

  const initVoice = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    const rec = new SR();
    rec.continuous = false;
    rec.interimResults = false;
    rec.lang = "en-US";
    rec.onresult = (e) => {
      const t = e.results[0][0].transcript.toLowerCase();
      setVoiceText(t);
      setListening(false);
      if (/(check|scan|analyze|detect|upload|fruit)/i.test(t)) fileRef.current?.click();
    };
    rec.onend = () => setListening(false);
    rec.onerror = () => setListening(false);
    recognitionRef.current = rec;
  };

  const toggleVoice = () => {
    if (!recognitionRef.current) { setError("Voice not supported. Try Chrome."); return; }
    if (listening) { recognitionRef.current.stop(); setListening(false); }
    else { setVoiceText(""); recognitionRef.current.start(); setListening(true); }
  };

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) { setError("Please upload a valid image."); return; }
    setError(""); setResult(null);
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const handleDetect = async () => {
    if (!image) { setError("Please upload a fruit image first."); return; }
    setScanning(true); setError(""); setResult(null);
    const form = new FormData();
    form.append("file", image);
    try {
      let res;
      if (selectedFruit) {
        res = await api.post(`/predict?fruit=${selectedFruit}`, form);
        if (!res.data.fruit) res.data.fruit = selectedFruit;
      } else {
        res = await api.post("/detect", form);
      }
      setResult(res.data);
      if (!res.data.error && res.data.fruit) {
        saveLocalHistory({
          id: Date.now().toString(),
          fruit: res.data.fruit,
          grade: res.data.grade,
          confidence: res.data.confidence,
          timestamp: new Date().toISOString(),
        });
      }
    } catch {
      setError("Detection failed. Check your connection.");
    } finally {
      setScanning(false);
    }
  };

  const gradeColor = (g) => ({ ripe: "#4ade80", unripe: "#facc15", overripe: "#f87171" }[g] || "var(--text)");
  const formatDate = (iso) => new Date(iso).toLocaleDateString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

  return (
    <div style={{ minHeight: "100vh", paddingBottom: 60 }}>
      <div className="orb-bg">
        <div className="orb orb-1" /><div className="orb orb-2" /><div className="orb orb-3" />
      </div>

      {/* Nav */}
      <nav className="nav">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 22 }}>🍃</span>
          <span className="font-display" style={{ fontWeight: 800, fontSize: 18, background: "linear-gradient(135deg, var(--accent), var(--accent2))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            FruitSense AI
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{theme === "dark" ? "🌙" : "☀️"}</span>
          <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme" />
        </div>
      </nav>

      {/* Main */}
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "96px 20px 0" }}>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 24, flexWrap: "wrap" }}>
          <button className={`tab-btn ${tab === "scan" ? "active" : ""}`} onClick={() => setTab("scan")}>🔍 Detect</button>
          <button className={`tab-btn ${tab === "history" ? "active" : ""}`} onClick={() => setTab("history")}>📋 History ({history.length})</button>
          <button className={`tab-btn ${tab === "about" ? "active" : ""}`} onClick={() => setTab("about")}>ℹ️ About</button>
        </div>

        {/* ── SCAN TAB ── */}
        {tab === "scan" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Fruit selector grid */}
            <div className="glass" style={{ padding: 20 }}>
              <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 12 }}>
                Select fruit <span style={{ color: "var(--accent)" }}>(optional)</span> — or leave blank for auto-detect
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))", gap: 8 }}>
                {FRUITS.map(f => (
                  <button key={f.key} onClick={() => setSelectedFruit(selectedFruit === f.key ? null : f.key)}
                    style={{
                      background: selectedFruit === f.key ? "var(--accent-glow)" : "var(--surface)",
                      border: selectedFruit === f.key ? "1px solid var(--accent)" : "1px solid var(--border)",
                      borderRadius: 12, padding: "10px 6px", cursor: "pointer",
                      display: "flex", flexDirection: "column", alignItems: "center", gap: 4, transition: "all 0.15s",
                    }}>
                    <span style={{ fontSize: 24 }}>{FRUIT_EMOJI[f.key]}</span>
                    <span style={{ fontSize: 11, color: selectedFruit === f.key ? "var(--accent)" : "var(--text-muted)", fontWeight: selectedFruit === f.key ? 700 : 400 }}>
                      {f.label}
                    </span>
                  </button>
                ))}
              </div>
              <p style={{ fontSize: 12, marginTop: 10, textAlign: "center", color: selectedFruit ? "var(--accent)" : "var(--text-dim)" }}>
                {selectedFruit ? `✓ ${selectedFruit.toUpperCase()} selected — ripeness model active` : "🤖 Auto-detect mode — AI will identify the fruit"}
              </p>
            </div>

            {/* Upload zone */}
            <div className={`upload-zone ${dragOver ? "drag-over" : ""}`}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}>
              <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
                onChange={e => handleFile(e.target.files[0])} />
              {preview ? (
                <div style={{ position: "relative", display: "inline-block" }}>
                  {scanning && <div className="scan-frame" />}
                  {scanning && <div className="scan-line" />}
                  <img src={preview} className={scanning ? "" : "img-sharpen"}
                    style={{ width: 200, height: 200, objectFit: "cover", borderRadius: 16, display: "block" }} alt="fruit" />
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: 48, marginBottom: 12 }}>📸</div>
                  <p className="font-display" style={{ fontWeight: 700, fontSize: 16 }}>Drop a fruit photo here</p>
                  <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 4 }}>or click to browse · JPG, PNG, WEBP</p>
                </div>
              )}
            </div>

            {/* Action row */}
            <div style={{ display: "flex", gap: 12 }}>
              <button className={`voice-btn ${listening ? "listening" : ""}`} onClick={toggleVoice} title="Say 'scan fruit'">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
              </button>
              <button className="btn" onClick={handleDetect} disabled={scanning || !image} style={{ flex: 1 }}>
                {scanning
                  ? <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}><span className="spinner" />Analyzing...</span>
                  : selectedFruit ? `🔬 Check ${selectedFruit} Ripeness` : "🤖 Auto Detect Fruit"}
              </button>
            </div>

            {listening && (
              <div className="alert" style={{ background: "rgba(248,113,113,0.1)", border: "1px solid rgba(248,113,113,0.2)", color: "#f87171", display: "flex", alignItems: "center", gap: 8 }}>
                🎙️ Listening... say "scan fruit" or "check this"
              </div>
            )}
            {voiceText && !listening && (
              <div style={{ fontSize: 13, color: "var(--text-muted)", textAlign: "center" }}>
                🗣️ Heard: "<em>{voiceText}</em>"
              </div>
            )}

            {error && <div className="alert alert-error">{error}</div>}

            {/* Result */}
            {result && !result.error && (
              <div className="glass anim-pop" style={{ padding: 24 }}>
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 16 }}>
                  <div>
                    <div style={{ fontSize: 48 }}>{FRUIT_EMOJI[result.fruit] || "🍽️"}</div>
                    <h2 className="font-display" style={{ fontSize: 24, fontWeight: 800, textTransform: "capitalize", marginTop: 4 }}>{result.fruit}</h2>
                  </div>
                  <span className={`grade-badge grade-${result.grade}`}>{result.grade}</span>
                </div>
                <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 16 }}>{GRADE_TIPS[result.grade]}</p>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {Object.entries(result.all_probs || {}).map(([label, val]) => (
                    <ConfBar key={label} label={label} value={val} color={gradeColor(label)} />
                  ))}
                </div>
                {result.auto_detected && (
                  <p style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 12, textAlign: "right" }}>✨ Auto-detected · {result.confidence}% confident</p>
                )}
                {result.warning && (
                  <div className="alert" style={{ background: "rgba(250,204,21,0.1)", border: "1px solid rgba(250,204,21,0.2)", color: "#facc15", marginTop: 12, fontSize: 13 }}>
                    ⚠️ {result.warning}
                  </div>
                )}
              </div>
            )}
            {result?.error && <div className="alert alert-error">{result.error}</div>}
          </div>
        )}

        {/* ── HISTORY TAB ── */}
        {tab === "history" && (
          <div className="glass anim-fade-up" style={{ padding: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
              <h2 className="font-display" style={{ fontWeight: 700, fontSize: 18 }}>Scan History</h2>
              {history.length > 0 && (
                <button className="btn btn-ghost" onClick={clearHistory}
                  style={{ width: "auto", padding: "8px 16px", fontSize: 13, color: "var(--danger)", borderColor: "rgba(248,113,113,0.3)" }}>
                  Clear all
                </button>
              )}
            </div>
            {history.length === 0 ? (
              <div style={{ textAlign: "center", padding: "40px 0", color: "var(--text-muted)" }}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>🍽️</div>
                <p>No scans yet. Detect a fruit to get started!</p>
              </div>
            ) : (
              history.map((s, i) => (
                <div key={s.id || i} className="history-item" style={{ animationDelay: `${i * 0.05}s` }}>
                  <div style={{ fontSize: 32 }}>{FRUIT_EMOJI[s.fruit] || "🍽️"}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span className="font-display" style={{ fontWeight: 700, fontSize: 15, textTransform: "capitalize" }}>{s.fruit}</span>
                      <span className={`grade-badge grade-${s.grade}`} style={{ padding: "2px 10px", fontSize: 11 }}>{s.grade}</span>
                    </div>
                    <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>{formatDate(s.timestamp)}</div>
                  </div>
                  <span style={{ fontSize: 13, fontWeight: 600, color: gradeColor(s.grade) }}>{s.confidence}%</span>
                </div>
              ))
            )}
          </div>
        )}

        {/* ── ABOUT TAB ── */}
        {tab === "about" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Hero */}
            <div className="glass anim-fade-up" style={{ padding: 32, textAlign: "center" }}>
              <div style={{ fontSize: 56, marginBottom: 12 }}>🍃</div>
              <h1 className="font-display" style={{ fontSize: 28, fontWeight: 800, background: "linear-gradient(135deg, var(--accent), var(--accent2))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", marginBottom: 12 }}>
                FruitSense AI
              </h1>
              <p style={{ color: "var(--text-muted)", fontSize: 15, lineHeight: 1.7, maxWidth: 480, margin: "0 auto" }}>
                An AI-powered fruit ripeness detection system trained on 10 different fruits.
                Upload any fruit photo and get instant ripeness analysis — no expertise needed.
              </p>
              <div style={{ display: "flex", justifyContent: "center", gap: 12, marginTop: 20, flexWrap: "wrap" }}>
                <span className="grade-badge grade-ripe">10 Fruit Models</span>
                <span className="grade-badge grade-unripe">Real-time AI</span>
                <span className="grade-badge" style={{ background: "rgba(34,211,238,0.15)", color: "#22d3ee", border: "1px solid rgba(34,211,238,0.3)" }}>Open Source</span>
              </div>
            </div>

            {/* How it works */}
            <div className="glass anim-fade-up" style={{ padding: 24 }}>
              <h2 className="font-display" style={{ fontWeight: 700, fontSize: 18, marginBottom: 20 }}>How It Works</h2>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                {HOW_IT_WORKS.map(s => (
                  <div key={s.step} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 16, padding: 16 }}>
                    <div style={{ fontSize: 28, marginBottom: 8 }}>{s.icon}</div>
                    <div className="font-display" style={{ fontWeight: 700, fontSize: 14, marginBottom: 4 }}>{s.title}</div>
                    <div style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.5 }}>{s.desc}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Supported fruits */}
            <div className="glass anim-fade-up" style={{ padding: 24 }}>
              <h2 className="font-display" style={{ fontWeight: 700, fontSize: 18, marginBottom: 16 }}>Supported Fruits</h2>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                {FRUITS.map(f => (
                  <div key={f.key} style={{ display: "flex", alignItems: "center", gap: 6, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 999, padding: "6px 14px" }}>
                    <span>{FRUIT_EMOJI[f.key]}</span>
                    <span style={{ fontSize: 13, fontWeight: 500 }}>{f.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Tech stack */}
            <div className="glass anim-fade-up" style={{ padding: 24 }}>
              <h2 className="font-display" style={{ fontWeight: 700, fontSize: 18, marginBottom: 16 }}>Tech Stack</h2>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {TECH_STACK.map(t => (
                  <div key={t.name} style={{ display: "flex", alignItems: "center", gap: 14, padding: "12px 0", borderBottom: "1px solid var(--border)" }}>
                    <div style={{ fontSize: 28, width: 40, textAlign: "center", flexShrink: 0 }}>{t.icon}</div>
                    <div>
                      <div className="font-display" style={{ fontWeight: 700, fontSize: 14 }}>{t.name}</div>
                      <div style={{ fontSize: 13, color: "var(--text-muted)" }}>{t.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Footer */}
            <div style={{ textAlign: "center", padding: "16px 0", color: "var(--text-dim)", fontSize: 13 }}>
              Built with ❤️ · FruitSense AI v2.0
            </div>
          </div>
        )}
      </div>
    </div>
  );
}