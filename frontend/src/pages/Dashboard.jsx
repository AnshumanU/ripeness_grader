import { useState, useRef } from "react";
import api from "../lib/api";

const FRUIT_EMOJI = {
  banana: "🍌", apple: "🍎", mango: "🥭", orange: "🍊",
  tomato: "🍅", strawberry: "🍓", grape: "🍇", peach: "🍑",
  kiwi: "🥝", pear: "🍐", avocado: "🥑", lemon: "🍋",
  cherry: "🍒", corn: "🌽", pepper: "🌶️", physalis: "🫐",
  zucchini: "🥒", cactus: "🌵",
};

const AVAILABLE_FRUITS = [
  "banana", "orange", "tomato", "avocado", "cherry",
  "corn", "pepper", "physalis", "zucchini", "cactus"
];

const GRADE_COLORS = {
  ripe:     { bg: "rgba(74,222,128,0.15)", color: "#4ade80", border: "rgba(74,222,128,0.3)" },
  unripe:   { bg: "rgba(250,204,21,0.15)", color: "#facc15", border: "rgba(250,204,21,0.3)" },
  overripe: { bg: "rgba(248,113,113,0.15)", color: "#f87171", border: "rgba(248,113,113,0.3)" },
};

const BAR_COLORS = {
  ripe: "#4ade80",
  unripe: "#facc15",
  overripe: "#f87171",
};

export default function Dashboard() {
  const [tab, setTab] = useState("detector");
  const [selectedFruit, setSelectedFruit] = useState(null);
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef();

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) {
      setError("Please upload a valid image file.");
      return;
    }
    setError("");
    setResult(null);
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleDetect = async () => {
    if (!image) { setError("Please upload a fruit image first."); return; }
    if (!selectedFruit) { setError("Please select a fruit type first."); return; }

    setScanning(true);
    setError("");
    setResult(null);

    const form = new FormData();
    form.append("file", image);

    try {
      const res = await api.post(`/predict?fruit=${selectedFruit}`, form);
      console.log("API response:", res.data);
      setResult(res.data);
    } catch (e) {
      console.error("API error:", e);
      setError("Detection failed. Make sure the backend is running.");
    } finally {
      setScanning(false);
    }
  };

  // grade comes from result.grade (FastAPI) — already a percentage value
  const grade = result?.grade;
  const gradeStyle = GRADE_COLORS[grade] || GRADE_COLORS.ripe;

  return (
    <div style={{ minHeight: "100vh", background: "#060810", color: "#f0f2f8", padding: "40px 20px" }}>
      <div style={{ maxWidth: 680, margin: "0 auto" }}>

        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>🍃</div>
          <h1 style={{
            fontSize: 28, fontWeight: 800,
            background: "linear-gradient(135deg, #4ade80, #22d3ee)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            margin: 0, letterSpacing: "-0.02em"
          }}>FruitSense</h1>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
          {["detector", "about"].map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: "8px 20px", borderRadius: 10, fontFamily: "inherit",
              border: tab === t ? "1px solid rgba(255,255,255,0.15)" : "none",
              background: tab === t ? "rgba(255,255,255,0.07)" : "transparent",
              color: tab === t ? "#f0f2f8" : "rgba(240,242,248,0.45)",
              fontWeight: 600, fontSize: 14, cursor: "pointer",
            }}>
              {t === "detector" ? "🔍 Detect" : "ℹ️ About"}
            </button>
          ))}
        </div>

        {/* ── DETECTOR TAB ── */}
        {tab === "detector" && (
          <div>
            {/* Fruit Selector */}
            <div style={{
              background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 24, padding: 24, marginBottom: 24, backdropFilter: "blur(24px)"
            }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, color: "#f0f2f8" }}>
                Select Fruit Type
              </h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(75px, 1fr))", gap: 12 }}>
                {AVAILABLE_FRUITS.map(fruit => (
                  <button key={fruit} onClick={() => { setSelectedFruit(fruit); setResult(null); setError(""); }}
                    style={{
                      padding: 12, borderRadius: 14, cursor: "pointer", fontFamily: "inherit",
                      border: selectedFruit === fruit ? "2px solid #4ade80" : "1px solid rgba(255,255,255,0.08)",
                      background: selectedFruit === fruit ? "rgba(74,222,128,0.2)" : "rgba(255,255,255,0.04)",
                      color: "#f0f2f8", display: "flex", flexDirection: "column",
                      alignItems: "center", gap: 6, fontSize: 12, fontWeight: 600,
                      textTransform: "capitalize", transition: "all 0.2s",
                    }}>
                    <span style={{ fontSize: 24 }}>{FRUIT_EMOJI[fruit] || "🍽️"}</span>
                    {fruit}
                  </button>
                ))}
              </div>
              {selectedFruit && (
                <div style={{ marginTop: 12, fontSize: 13, color: "#4ade80", textAlign: "center" }}>
                  ✓ {selectedFruit.toUpperCase()} selected
                </div>
              )}
            </div>

            {/* Upload Zone */}
            <div
              style={{
                border: `2px dashed ${dragOver ? "#4ade80" : "rgba(255,255,255,0.15)"}`,
                borderRadius: 20, padding: 48, textAlign: "center", cursor: "pointer",
                marginBottom: 24, background: dragOver ? "rgba(74,222,128,0.1)" : "transparent",
                transition: "all 0.2s",
              }}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
            >
              <input ref={fileRef} type="file" accept="image/*"
                style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
              {preview ? (
                <div style={{ position: "relative", display: "inline-block" }}>
                  {scanning && (
                    <div style={{
                      position: "absolute", left: 0, right: 0, height: 2,
                      background: "linear-gradient(90deg, transparent, #4ade80, transparent)",
                      animation: "scan-line 1.5s ease-in-out infinite",
                      boxShadow: "0 0 12px #4ade80"
                    }} />
                  )}
                  <img src={preview} alt="fruit" style={{
                    width: 200, height: 200, objectFit: "cover", borderRadius: 16, display: "block",
                    animation: scanning ? "none" : "sharpening 0.9s ease-out forwards"
                  }} />
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: 48, marginBottom: 12 }}>📸</div>
                  <p style={{ fontWeight: 700, fontSize: 16, color: "#f0f2f8", margin: "0 0 4px" }}>
                    Drop a fruit photo here
                  </p>
                  <p style={{ fontSize: 13, color: "rgba(240,242,248,0.45)", margin: 0 }}>
                    or click to browse · JPG, PNG, WEBP
                  </p>
                </div>
              )}
            </div>

            {/* Error */}
            {error && (
              <div style={{
                background: "rgba(248,113,113,0.1)", border: "1px solid rgba(248,113,113,0.2)",
                color: "#f87171", padding: 12, borderRadius: 12, fontSize: 14, marginBottom: 16,
              }}>
                {error}
              </div>
            )}

            {/* Detect Button */}
            <button onClick={handleDetect} disabled={scanning || !image || !selectedFruit}
              style={{
                width: "100%", padding: 14, border: "none", borderRadius: 14, fontFamily: "inherit",
                background: scanning || !image || !selectedFruit ? "rgba(74,222,128,0.5)" : "#4ade80",
                color: "#000", fontWeight: 700, fontSize: 15, marginBottom: 24,
                cursor: scanning || !image || !selectedFruit ? "not-allowed" : "pointer",
                opacity: scanning || !image || !selectedFruit ? 0.6 : 1, transition: "all 0.15s",
              }}>
              {scanning ? (
                <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                  <span style={{
                    display: "inline-block", width: 16, height: 16,
                    border: "2px solid rgba(0,0,0,0.3)", borderTopColor: "#000",
                    borderRadius: "50%", animation: "spin 0.7s linear infinite"
                  }} />
                  Analyzing...
                </span>
              ) : "🤖 Detect Ripeness"}
            </button>

            {/* Result Card */}
            {result && !result.error && (
              <div style={{
                background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 24, padding: 24, backdropFilter: "blur(24px)",
                animation: "pop-in 0.4s cubic-bezier(0.34,1.56,0.64,1)"
              }}>
                {/* Header */}
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 16 }}>
                  <div>
                    <div style={{ fontSize: 48, marginBottom: 8 }}>{FRUIT_EMOJI[selectedFruit] || "🍽️"}</div>
                    <h2 style={{ fontSize: 24, fontWeight: 800, textTransform: "capitalize", margin: 0, color: "#f0f2f8" }}>
                      {selectedFruit}
                    </h2>
                  </div>
                  {/* Grade badge */}
                  <span style={{
                    display: "inline-flex", alignItems: "center", padding: "6px 16px",
                    borderRadius: 999, fontWeight: 700, fontSize: 13,
                    letterSpacing: "0.05em", textTransform: "uppercase",
                    background: gradeStyle.bg, color: gradeStyle.color,
                    border: `1px solid ${gradeStyle.border}`
                  }}>
                    {grade}
                  </span>
                </div>

                {/* Confidence */}
                <p style={{ fontSize: 14, color: "rgba(240,242,248,0.45)", marginBottom: 16 }}>
                  Confidence: <strong style={{ color: "#f0f2f8" }}>{result.confidence.toFixed(1)}%</strong>
                </p>

                {/* Probability bars — all_probs values are already percentages (e.g. 67.9) */}
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {Object.entries(result.all_probs || {}).map(([label, val]) => (
                    <div key={label}>
                      <div style={{
                        display: "flex", justifyContent: "space-between",
                        fontSize: 12, color: "rgba(240,242,248,0.45)", marginBottom: 4
                      }}>
                        <span style={{ textTransform: "capitalize" }}>{label}</span>
                        <span style={{ fontWeight: 600, color: BAR_COLORS[label] || "#fff" }}>
                          {val.toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ height: 6, background: "rgba(255,255,255,0.08)", borderRadius: 999, overflow: "hidden" }}>
                        <div style={{
                          height: "100%", borderRadius: 999,
                          background: BAR_COLORS[label] || "#888",
                          width: `${val}%`,  /* val is already a percentage */
                          transition: "width 1s cubic-bezier(0.16,1,0.3,1)"
                        }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* API error */}
            {result?.error && (
              <div style={{
                background: "rgba(248,113,113,0.1)", border: "1px solid rgba(248,113,113,0.2)",
                color: "#f87171", padding: 12, borderRadius: 12, fontSize: 14, textAlign: "center"
              }}>
                {result.error}
              </div>
            )}
          </div>
        )}

        {/* ── ABOUT TAB ── */}
        {tab === "about" && (
          <div style={{
            background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 24, padding: 24, backdropFilter: "blur(24px)",
            animation: "fade-up 0.3s ease-out"
          }}>
            <h2 style={{ fontSize: 22, fontWeight: 800, marginBottom: 8, color: "#f0f2f8" }}>
              About FruitSense AI
            </h2>
            <p style={{ fontSize: 14, color: "rgba(240,242,248,0.45)", marginBottom: 24 }}>
              Advanced AI-powered fruit ripeness detection system
            </p>

            {[
              { title: "📊 Dataset", color: "#4ade80", items: ["Fruits-360 Dataset","100x100 pixel images","Multiple fruit categories","High-quality labeled data"] },
              { title: "🧠 Model", color: "#22d3ee", items: ["MobileNetV3 Small","Transfer learning on ImageNet","ONNX format for fast inference","Real-time predictions"] },
              { title: "🎯 Training", color: "#f97316", items: ["6 epochs · AdamW optimizer","Cosine Annealing scheduler","Label smoothing (0.1)","Data augmentation: flip, rotation, jitter"] },
            ].map(section => (
              <div key={section.title} style={{ marginBottom: 24, paddingBottom: 24, borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12, color: section.color }}>{section.title}</h3>
                {section.items.map(item => (
                  <div key={item} style={{ display: "flex", gap: 8, fontSize: 12, marginBottom: 6 }}>
                    <span style={{ color: "#4ade80", fontWeight: 700 }}>✓</span>
                    <span style={{ color: "rgba(240,242,248,0.45)" }}>{item}</span>
                  </div>
                ))}
              </div>
            ))}

            {/* Supported fruits */}
            <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12, color: "#f0f2f8" }}>🍎 Supported Fruits</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(80px, 1fr))", gap: 10 }}>
              {AVAILABLE_FRUITS.map(fruit => (
                <div key={fruit} style={{
                  background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: 12, padding: 10, textAlign: "center",
                  fontSize: 12, color: "rgba(240,242,248,0.45)"
                }}>
                  <div style={{ fontSize: 18, marginBottom: 4 }}>{FRUIT_EMOJI[fruit] || "🍽️"}</div>
                  <div style={{ textTransform: "capitalize", fontWeight: 500 }}>{fruit}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes sharpening {
          0%   { filter: blur(20px) brightness(0.6); transform: scale(1.05); }
          60%  { filter: blur(4px)  brightness(0.9); transform: scale(1.02); }
          100% { filter: blur(0)    brightness(1);   transform: scale(1);    }
        }
        @keyframes scan-line {
          0%   { top: 0;    opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        @keyframes fade-up {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0);    }
        }
        @keyframes pop-in {
          from { opacity: 0; transform: scale(0.9); }
          to   { opacity: 1; transform: scale(1);   }
        }
      `}</style>
    </div>
  );
}