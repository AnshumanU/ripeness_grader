import { useState, useRef } from "react";
import api from "../lib/api";

const FRUIT_EMOJI = {
  banana: "🍌", apple: "🍎", mango: "🥭", orange: "🍊",
  tomato: "🍅", strawberry: "🍓", grape: "🍇", peach: "🍑",
  kiwi: "🥝", pear: "🍐", avocado: "🥑", lemon: "🍋",
};

const AVAILABLE_FRUITS = [
  "banana", "apple", "mango", "orange", "tomato", "strawberry",
  "grape", "peach", "kiwi", "pear", "avocado", "lemon"
];

export default function Dashboard() {
  const [tab, setTab] = useState("detector"); // detector | about
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
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleDetect = async () => {
    if (!image) { setError("Please upload a fruit image first."); return; }
    if (!selectedFruit) { setError("Please select a fruit type first."); return; }

    setScanning(true);
    setError("");
    setResult(null);

    const form = new FormData();
    form.append("file", image);
    form.append("fruit", selectedFruit);

    try {
      const res = await api.post(`/predict?fruit=${selectedFruit}`, form);
      setResult(res.data);
    } catch {
      setError("Detection failed. Make sure the backend is running.");
    } finally {
      setScanning(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "#060810", color: "#f0f2f8", padding: "40px 20px" }}>
      {/* Main container */}
      <div style={{ maxWidth: 680, margin: "0 auto" }}>
        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>🍃</div>
          <h1 style={{ fontSize: 28, fontWeight: 800, background: "linear-gradient(135deg, #4ade80, #22d3ee)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", margin: 0, letterSpacing: "-0.02em" }}>
            FruitSense
          </h1>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
          <button
            onClick={() => setTab("detector")}
            style={{
              padding: "8px 20px",
              borderRadius: 10,
              border: tab === "detector" ? "1px solid rgba(255, 255, 255, 0.15)" : "none",
              background: tab === "detector" ? "rgba(255, 255, 255, 0.07)" : "transparent",
              color: tab === "detector" ? "#f0f2f8" : "rgba(240, 242, 248, 0.45)",
              fontWeight: 600,
              fontSize: 14,
              cursor: "pointer",
              transition: "all 0.2s",
              fontFamily: "inherit"
            }}
          >
            🔍 Detect
          </button>
          <button
            onClick={() => setTab("about")}
            style={{
              padding: "8px 20px",
              borderRadius: 10,
              border: tab === "about" ? "1px solid rgba(255, 255, 255, 0.15)" : "none",
              background: tab === "about" ? "rgba(255, 255, 255, 0.07)" : "transparent",
              color: tab === "about" ? "#f0f2f8" : "rgba(240, 242, 248, 0.45)",
              fontWeight: 600,
              fontSize: 14,
              cursor: "pointer",
              transition: "all 0.2s",
              fontFamily: "inherit"
            }}
          >
            ℹ️ About
          </button>
        </div>

        {/* ─── DETECTOR TAB ─── */}
        {tab === "detector" && (
          <div>
            {/* Fruit Selector Card */}
            <div style={{
              background: "rgba(255, 255, 255, 0.04)",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              borderRadius: 24,
              padding: 24,
              marginBottom: 24,
              backdropFilter: "blur(24px)"
            }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, margin: 0, marginBottom: 16, color: "#f0f2f8" }}>
                Select Fruit Type
              </h3>
              
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(75px, 1fr))",
                gap: 12
              }}>
                {AVAILABLE_FRUITS.map(fruit => (
                  <button
                    key={fruit}
                    onClick={() => setSelectedFruit(fruit)}
                    style={{
                      padding: 12,
                      borderRadius: 14,
                      border: selectedFruit === fruit ? "2px solid #4ade80" : "1px solid rgba(255, 255, 255, 0.08)",
                      background: selectedFruit === fruit ? "rgba(74, 222, 128, 0.2)" : "rgba(255, 255, 255, 0.04)",
                      color: "#f0f2f8",
                      cursor: "pointer",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: 6,
                      transition: "all 0.2s",
                      fontSize: 12,
                      fontWeight: 600,
                      textTransform: "capitalize",
                      fontFamily: "inherit"
                    }}
                    onMouseEnter={(e) => {
                      if (selectedFruit !== fruit) {
                        e.target.style.borderColor = "rgba(255, 255, 255, 0.15)";
                        e.target.style.background = "rgba(255, 255, 255, 0.07)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (selectedFruit !== fruit) {
                        e.target.style.borderColor = "rgba(255, 255, 255, 0.08)";
                        e.target.style.background = "rgba(255, 255, 255, 0.04)";
                      }
                    }}
                  >
                    <span style={{ fontSize: 24 }}>{FRUIT_EMOJI[fruit]}</span>
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
                border: "2px dashed rgba(255, 255, 255, 0.15)",
                borderRadius: 20,
                padding: 48,
                textAlign: "center",
                cursor: "pointer",
                marginBottom: 24,
                background: dragOver ? "rgba(74, 222, 128, 0.1)" : "transparent",
                transition: "all 0.2s",
                borderColor: dragOver ? "#4ade80" : "rgba(255, 255, 255, 0.15)"
              }}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
            >
              <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
                onChange={(e) => handleFile(e.target.files[0])} />

              {preview ? (
                <div style={{ position: "relative", display: "inline-block" }}>
                  {scanning && (
                    <>
                      <div style={{
                        position: "absolute",
                        inset: 0,
                        borderRadius: 16,
                        animation: "scan-pulse 1.5s ease-in-out infinite",
                        boxShadow: "0 0 0 0 rgba(74, 222, 128, 0.2)"
                      }} />
                      <div style={{
                        position: "absolute",
                        left: 0,
                        right: 0,
                        height: 2,
                        background: "linear-gradient(90deg, transparent, #4ade80, transparent)",
                        animation: "scan-line 1.5s ease-in-out infinite",
                        boxShadow: "0 0 12px #4ade80"
                      }} />
                    </>
                  )}
                  <img
                    src={preview}
                    style={{
                      width: 200,
                      height: 200,
                      objectFit: "cover",
                      borderRadius: 16,
                      display: "block",
                      animation: scanning ? "none" : "sharpening 0.9s ease-out forwards"
                    }}
                    alt="fruit"
                  />
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: 48, marginBottom: 12 }}>📸</div>
                  <p style={{ fontWeight: 700, fontSize: 16, color: "#f0f2f8", margin: 0, marginBottom: 4 }}>Drop a fruit photo here</p>
                  <p style={{ fontSize: 13, color: "rgba(240, 242, 248, 0.45)", margin: 0 }}>or click to browse · JPG, PNG, WEBP</p>
                </div>
              )}
            </div>

            {/* Error message */}
            {error && (
              <div style={{
                background: "rgba(248, 113, 113, 0.1)",
                border: "1px solid rgba(248, 113, 113, 0.2)",
                color: "#f87171",
                padding: 12,
                borderRadius: 12,
                fontSize: 14,
                marginBottom: 16,
                animation: "fade-up 0.3s ease-out"
              }}>
                {error}
              </div>
            )}

            {/* Detect Button */}
            <button
              onClick={handleDetect}
              disabled={scanning || !image || !selectedFruit}
              style={{
                width: "100%",
                padding: 14,
                background: scanning || !image || !selectedFruit ? "rgba(74, 222, 128, 0.5)" : "#4ade80",
                color: "#000",
                fontWeight: 700,
                fontSize: 15,
                border: "none",
                borderRadius: 14,
                cursor: scanning || !image || !selectedFruit ? "not-allowed" : "pointer",
                transition: "all 0.15s",
                marginBottom: 24,
                opacity: scanning || !image || !selectedFruit ? 0.6 : 1
              }}
              onMouseEnter={(e) => {
                if (!scanning && image && selectedFruit) {
                  e.target.style.transform = "translateY(-2px)";
                  e.target.style.boxShadow = "0 8px 24px rgba(74, 222, 128, 0.3)";
                }
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "none";
              }}
            >
              {scanning ? (
                <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                  <span style={{
                    display: "inline-block",
                    width: 16,
                    height: 16,
                    border: "2px solid rgba(0,0,0,0.3)",
                    borderTopColor: "#000",
                    borderRadius: "50%",
                    animation: "spin 0.7s linear infinite"
                  }} />
                  Analyzing...
                </span>
              ) : (
                "🤖 Auto Detect Fruit"
              )}
            </button>

            {/* Result Card */}
            {result && !result.error && (
              <div style={{
                background: "rgba(255, 255, 255, 0.04)",
                border: "1px solid rgba(255, 255, 255, 0.08)",
                borderRadius: 24,
                padding: 24,
                backdropFilter: "blur(24px)",
                animation: "pop-in 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)"
              }}>
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 16 }}>
                  <div>
                    <div style={{ fontSize: 48, marginBottom: 8 }}>{FRUIT_EMOJI[selectedFruit] || "🍽️"}</div>
                    <h2 style={{ fontSize: 24, fontWeight: 800, textTransform: "capitalize", margin: 0, color: "#f0f2f8" }}>
                      {selectedFruit}
                    </h2>
                  </div>
                  <span style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 6,
                    padding: "6px 16px",
                    borderRadius: 999,
                    fontWeight: 700,
                    fontSize: 13,
                    letterSpacing: "0.05em",
                    textTransform: "uppercase",
                    background: result.prediction === "ripe" 
                      ? "rgba(74, 222, 128, 0.15)"
                      : result.prediction === "unripe"
                      ? "rgba(250, 204, 21, 0.15)"
                      : "rgba(248, 113, 113, 0.15)",
                    color: result.prediction === "ripe"
                      ? "#4ade80"
                      : result.prediction === "unripe"
                      ? "#facc15"
                      : "#f87171",
                    border: "1px solid " + (result.prediction === "ripe"
                      ? "rgba(74, 222, 128, 0.3)"
                      : result.prediction === "unripe"
                      ? "rgba(250, 204, 21, 0.3)"
                      : "rgba(248, 113, 113, 0.3)")
                  }}>
                    {result.prediction}
                  </span>
                </div>

                <p style={{
                  fontSize: 14,
                  color: "rgba(240, 242, 248, 0.45)",
                  marginBottom: 16,
                  margin: 0,
                  marginBottom: 16
                }}>
                  Confidence: <strong>{result.confidence.toFixed(1)}%</strong>
                </p>

                {/* Probability bars */}
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {Object.entries(result.all_probs || {}).map(([label, val]) => (
                    <div key={label}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "rgba(240, 242, 248, 0.45)", marginBottom: 4 }}>
                        <span style={{ textTransform: "capitalize" }}>{label}</span>
                        <span style={{
                          fontWeight: 600,
                          color: label === "ripe" ? "#4ade80" : label === "unripe" ? "#facc15" : "#f87171"
                        }}>
                          {(val * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{
                        height: 6,
                        background: "rgba(255, 255, 255, 0.08)",
                        borderRadius: 999,
                        overflow: "hidden"
                      }}>
                        <div style={{
                          height: "100%",
                          borderRadius: 999,
                          background: label === "ripe" ? "#4ade80" : label === "unripe" ? "#facc15" : "#f87171",
                          width: `${(val * 100)}%`,
                          transition: "width 1s cubic-bezier(0.16, 1, 0.3, 1)"
                        }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result?.error && (
              <div style={{
                background: "rgba(248, 113, 113, 0.1)",
                border: "1px solid rgba(248, 113, 113, 0.2)",
                color: "#f87171",
                padding: 12,
                borderRadius: 12,
                fontSize: 14,
                textAlign: "center"
              }}>
                {result.error}
              </div>
            )}
          </div>
        )}

        {/* ─── ABOUT TAB ─── */}
        {tab === "about" && (
          <div style={{
            background: "rgba(255, 255, 255, 0.04)",
            border: "1px solid rgba(255, 255, 255, 0.08)",
            borderRadius: 24,
            padding: 24,
            backdropFilter: "blur(24px)",
            animation: "fade-up 0.3s ease-out"
          }}>
            {/* About Header */}
            <div style={{ marginBottom: 24 }}>
              <h2 style={{ fontSize: 22, fontWeight: 800, margin: 0, marginBottom: 8, color: "#f0f2f8" }}>
                About FruitSense AI
              </h2>
              <p style={{ fontSize: 14, color: "rgba(240, 242, 248, 0.45)", margin: 0 }}>
                Advanced AI-powered fruit ripeness detection system
              </p>
            </div>

            {/* Dataset Section */}
            <div style={{ marginBottom: 24, paddingBottom: 24, borderBottom: "1px solid rgba(255, 255, 255, 0.08)" }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0, marginBottom: 12, color: "#4ade80" }}>
                📊 Dataset
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div>
                  <p style={{ fontSize: 13, fontWeight: 600, color: "#f0f2f8", margin: 0, marginBottom: 4 }}>
                    Fruits-360 Dataset
                  </p>
                  <p style={{ fontSize: 12, color: "rgba(240, 242, 248, 0.45)", margin: 0 }}>
                    ✓ 100x100 pixel images<br/>
                    ✓ Multiple fruit categories<br/>
                    ✓ Training & Test splits included<br/>
                    ✓ High-quality labeled data
                  </p>
                </div>
                <div style={{
                  background: "rgba(74, 222, 128, 0.1)",
                  border: "1px solid rgba(74, 222, 128, 0.2)",
                  borderRadius: 12,
                  padding: 12,
                  marginTop: 8
                }}>
                  <p style={{ fontSize: 12, color: "#4ade80", margin: 0, fontWeight: 600 }}>
                    📈 Coverage: 12+ fruit types
                  </p>
                  <p style={{ fontSize: 11, color: "rgba(74, 222, 128, 0.8)", margin: "4px 0 0 0" }}>
                    Each fruit trained with ripeness labels: Unripe, Ripe, Overripe
                  </p>
                </div>
              </div>
            </div>

            {/* Model Section */}
            <div style={{ marginBottom: 24, paddingBottom: 24, borderBottom: "1px solid rgba(255, 255, 255, 0.08)" }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0, marginBottom: 12, color: "#22d3ee" }}>
                🧠 Model Architecture
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div>
                  <p style={{ fontSize: 13, fontWeight: 600, color: "#f0f2f8", margin: 0, marginBottom: 4 }}>
                    MobileNetV3 Small
                  </p>
                  <p style={{ fontSize: 12, color: "rgba(240, 242, 248, 0.45)", margin: 0 }}>
                    ✓ Lightweight & fast inference<br/>
                    ✓ Transfer learning on ImageNet<br/>
                    ✓ Optimized for mobile devices<br/>
                    ✓ Real-time predictions
                  </p>
                </div>
                <div style={{
                  background: "rgba(34, 211, 238, 0.1)",
                  border: "1px solid rgba(34, 211, 238, 0.2)",
                  borderRadius: 12,
                  padding: 12,
                  marginTop: 8
                }}>
                  <p style={{ fontSize: 12, color: "#22d3ee", margin: 0, fontWeight: 600 }}>
                    ⚡ Model Format: ONNX
                  </p>
                  <p style={{ fontSize: 11, color: "rgba(34, 211, 238, 0.8)", margin: "4px 0 0 0" }}>
                    Cross-platform compatibility, GPU/CPU support
                  </p>
                </div>
              </div>
            </div>

            {/* Training Details */}
            <div style={{ marginBottom: 24, paddingBottom: 24, borderBottom: "1px solid rgba(255, 255, 255, 0.08)" }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0, marginBottom: 12, color: "#f97316" }}>
                🎯 Training Configuration
              </h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div style={{
                  background: "rgba(255, 255, 255, 0.04)",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  borderRadius: 12,
                  padding: 12
                }}>
                  <p style={{ fontSize: 11, color: "rgba(240, 242, 248, 0.45)", margin: 0 }}>Epochs</p>
                  <p style={{ fontSize: 14, fontWeight: 700, color: "#f0f2f8", margin: "4px 0 0 0" }}>6</p>
                </div>
                <div style={{
                  background: "rgba(255, 255, 255, 0.04)",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  borderRadius: 12,
                  padding: 12
                }}>
                  <p style={{ fontSize: 11, color: "rgba(240, 242, 248, 0.45)", margin: 0 }}>Batch Size</p>
                  <p style={{ fontSize: 14, fontWeight: 700, color: "#f0f2f8", margin: "4px 0 0 0" }}>8</p>
                </div>
                <div style={{
                  background: "rgba(255, 255, 255, 0.04)",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  borderRadius: 12,
                  padding: 12
                }}>
                  <p style={{ fontSize: 11, color: "rgba(240, 242, 248, 0.45)", margin: 0 }}>Learning Rate</p>
                  <p style={{ fontSize: 14, fontWeight: 700, color: "#f0f2f8", margin: "4px 0 0 0" }}>3e-4</p>
                </div>
                <div style={{
                  background: "rgba(255, 255, 255, 0.04)",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  borderRadius: 12,
                  padding: 12
                }}>
                  <p style={{ fontSize: 11, color: "rgba(240, 242, 248, 0.45)", margin: 0 }}>Input Size</p>
                  <p style={{ fontSize: 14, fontWeight: 700, color: "#f0f2f8", margin: "4px 0 0 0" }}>224x224</p>
                </div>
              </div>
            </div>

            {/* Optimization */}
            <div style={{ marginBottom: 24, paddingBottom: 24, borderBottom: "1px solid rgba(255, 255, 255, 0.08)" }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0, marginBottom: 12, color: "#f0f2f8" }}>
                🚀 Optimization Techniques
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ display: "flex", gap: 8, fontSize: 12 }}>
                  <span style={{ color: "#4ade80", fontWeight: 700 }}>✓</span>
                  <span style={{ color: "rgba(240, 242, 248, 0.45)" }}>Label Smoothing (0.1) for regularization</span>
                </div>
                <div style={{ display: "flex", gap: 8, fontSize: 12 }}>
                  <span style={{ color: "#4ade80", fontWeight: 700 }}>✓</span>
                  <span style={{ color: "rgba(240, 242, 248, 0.45)" }}>Cosine Annealing scheduler for learning rate decay</span>
                </div>
                <div style={{ display: "flex", gap: 8, fontSize: 12 }}>
                  <span style={{ color: "#4ade80", fontWeight: 700 }}>✓</span>
                  <span style={{ color: "rgba(240, 242, 248, 0.45)" }}>Data augmentation: rotation, flip, color jitter</span>
                </div>
                <div style={{ display: "flex", gap: 8, fontSize: 12 }}>
                  <span style={{ color: "#4ade80", fontWeight: 700 }}>✓</span>
                  <span style={{ color: "rgba(240, 242, 248, 0.45)" }}>AdamW optimizer for stable convergence</span>
                </div>
              </div>
            </div>

            {/* Supported Fruits */}
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0, marginBottom: 12, color: "#f0f2f8" }}>
                🍎 Supported Fruits
              </h3>
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
                gap: 10
              }}>
                {AVAILABLE_FRUITS.map(fruit => (
                  <div
                    key={fruit}
                    style={{
                      background: "rgba(255, 255, 255, 0.04)",
                      border: "1px solid rgba(255, 255, 255, 0.08)",
                      borderRadius: 12,
                      padding: 10,
                      textAlign: "center",
                      fontSize: 12,
                      color: "rgba(240, 242, 248, 0.45)"
                    }}
                  >
                    <div style={{ fontSize: 18, marginBottom: 4 }}>{FRUIT_EMOJI[fruit]}</div>
                    <div style={{ textTransform: "capitalize", fontWeight: 500 }}>{fruit}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Framework & Tools */}
            <div style={{
              background: "rgba(255, 255, 255, 0.04)",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              borderRadius: 12,
              padding: 16
            }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, margin: 0, marginBottom: 12, color: "#f0f2f8" }}>
                🛠️ Technology Stack
              </h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, fontSize: 12 }}>
                <div>
                  <p style={{ color: "rgba(240, 242, 248, 0.45)", margin: 0, marginBottom: 4 }}>Backend</p>
                  <p style={{ color: "#22d3ee", margin: 0, fontWeight: 600 }}>Flask, ONNX Runtime</p>
                </div>
                <div>
                  <p style={{ color: "rgba(240, 242, 248, 0.45)", margin: 0, marginBottom: 4 }}>Frontend</p>
                  <p style={{ color: "#22d3ee", margin: 0, fontWeight: 600 }}>React, Axios</p>
                </div>
                <div>
                  <p style={{ color: "rgba(240, 242, 248, 0.45)", margin: 0, marginBottom: 4 }}>ML Framework</p>
                  <p style={{ color: "#22d3ee", margin: 0, fontWeight: 600 }}>PyTorch, TorchVision</p>
                </div>
                <div>
                  <p style={{ color: "rgba(240, 242, 248, 0.45)", margin: 0, marginBottom: 4 }}>Model Export</p>
                  <p style={{ color: "#22d3ee", margin: 0, fontWeight: 600 }}>ONNX Format</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes sharpening {
          0% { filter: blur(20px) brightness(0.6); transform: scale(1.05); }
          60% { filter: blur(4px) brightness(0.9); transform: scale(1.02); }
          100% { filter: blur(0) brightness(1); transform: scale(1); }
        }
        @keyframes scan-line {
          0% { top: 0; opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        @keyframes scan-pulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.3); }
          50% { box-shadow: 0 0 0 12px transparent; }
        }
        @keyframes fade-up {
          from { opacity: 0; transform: translateY(16px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pop-in {
          from { opacity: 0; transform: scale(0.9); }
          to { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
}