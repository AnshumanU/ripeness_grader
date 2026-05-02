import { useState, useEffect, useRef, useCallback } from "react";
import api from "../lib/api";

const FRUIT_EMOJI = {
  banana: "🍌", apple: "🍎", mango: "🥭", orange: "🍊",
  tomato: "🍅", strawberry: "🍓", grape: "🍇", peach: "🍑",
  kiwi: "🥝", pear: "🍐", avocado: "🥑", cherry: "🍒",
  corn: "🌽", pepper: "🫑", physalis: "🟠", cactus: "🌵",
  zucchini: "🥒",
};

const GRADE_TIPS = {
  ripe: "Perfect to eat right now! 🎉",
  unripe: "Give it a few more days to ripen.",
  overripe: "Best used in smoothies or cooking.",
};

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
  const fileRef = useRef();
  const recognitionRef = useRef(null);

  useEffect(() => {
    loadLocalHistory();
    initVoice();
  }, []);

  // ── Local history ─────────────────────────────────────────────────────────
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

  // ── Voice ─────────────────────────────────────────────────────────────────
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

  // ── File handling ─────────────────────────────────────────────────────────
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

  // ── Detect ────────────────────────────────────────────────────────────────
  const handleDetect = async () => {
    if (!image) { setError("Please upload a fruit image first."); return; }
    setScanning(true); setError(""); setResult(null);
    const form = new FormData();
    form.append("file", image);
    try {
      const res = await api.post("/detect", form);
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
        <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
          {["scan", "history"].map(t => (
            <button key={t} className={`tab-btn ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
              {t === "scan" ? "🔍 Detect" : `📋 History (${history.length})`}
            </button>
          ))}
        </div>

        {/* ── SCAN TAB ── */}
        {tab === "scan" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Upload zone */}
            <div
              className={`upload-zone ${dragOver ? "drag-over" : ""}`}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
            >
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
                  : "🤖 Auto Detect Fruit"}
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
                    <h2 className="font-display" style={{ fontSize: 24, fontWeight: 800, textTransform: "capitalize", marginTop: 4 }}>
                      {result.fruit}
                    </h2>
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
                  <p style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 12, textAlign: "right" }}>
                    ✨ Auto-detected · {result.confidence}% confident
                  </p>
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
      </div>
    </div>
  );
}