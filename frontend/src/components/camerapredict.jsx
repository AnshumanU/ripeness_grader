import { useRef, useState, useCallback } from "react"
import axios from "axios"

const FRUITS = [
  "banana", "orange", "tomato", "avocado", "cherry",
  "corn", "pepper", "physalis", "zucchini", "cactus"
]
const COLORS = { ripe: "#2ecc71", overripe: "#e74c3c", unripe: "#f39c12" }

export default function CameraPredict() {
  const videoRef  = useRef(null)
  const canvasRef = useRef(null)
  const [fruit,    setFruit]    = useState("banana")
  const [active,   setActive]   = useState(false)
  const [result,   setResult]   = useState(null)
  const [error,    setError]    = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [snapshot, setSnapshot] = useState(null)

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      videoRef.current.srcObject = stream
      setActive(true)
      setResult(null)
      setError(null)
      setSnapshot(null)
    } catch (e) {
      setError("Camera access denied. Please allow camera permission in your browser.")
    }
  }

  const stopCamera = () => {
    videoRef.current?.srcObject?.getTracks().forEach(t => t.stop())
    setActive(false)
  }

  const capture = useCallback(async () => {
    const video  = videoRef.current
    const canvas = canvasRef.current
    canvas.width  = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext("2d").drawImage(video, 0, 0)
    setSnapshot(canvas.toDataURL("image/jpeg"))
    setLoading(true)
    setError(null)
    setResult(null)

    canvas.toBlob(async (blob) => {
      const form = new FormData()
      form.append("file", blob, "capture.jpg")
      form.append("fruit", fruit) // ← also send as form field

      try {
        const res = await axios.post(
          `${import.meta.env.VITE_API_URL}/predict`,
          form
        )
        if (res.data.error) {
          setError(res.data.error)
        } else {
          setResult(res.data)
        }
      } catch (e) {
        setError("Detection failed. Make sure the backend is running.")
      }
      setLoading(false)
    }, "image/jpeg")
  }, [fruit])

  // Flask returns result.prediction (not result.grade)
  const grade = result?.prediction

  return (
    <div>
      {/* Fruit selector */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 16 }}>
        {FRUITS.map(f => (
          <button key={f}
            onClick={() => { setFruit(f); setResult(null); setError(null) }}
            style={{
              padding: "6px 18px", borderRadius: 6, cursor: "pointer",
              background: fruit === f ? "#1b4332" : "#f0f0f0",
              color: fruit === f ? "#fff" : "#333",
              border: "none", fontWeight: 500, textTransform: "capitalize"
            }}>{f}</button>
        ))}
      </div>

      {/* Video */}
      <div style={{
        borderRadius: 12, overflow: "hidden", background: "#111",
        marginBottom: 16, minHeight: 240,
        display: "flex", alignItems: "center", justifyContent: "center"
      }}>
        <video ref={videoRef} autoPlay playsInline muted
          style={{ width: "100%", display: active ? "block" : "none" }} />
        {!active && <p style={{ color: "#666", margin: 0 }}>Camera is off</p>}
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 10, marginBottom: 20 }}>
        {!active
          ? <button onClick={startCamera} style={{
              flex: 1, padding: 12, borderRadius: 8, border: "none",
              background: "#2d6a4f", color: "#fff", fontSize: 15, cursor: "pointer"
            }}>Start Camera</button>
          : <>
              <button onClick={capture} disabled={loading} style={{
                flex: 2, padding: 12, borderRadius: 8, border: "none",
                background: "#2d6a4f", color: "#fff", fontSize: 15, cursor: "pointer"
              }}>{loading ? "Predicting..." : "Capture & Predict"}</button>
              <button onClick={stopCamera} style={{
                flex: 1, padding: 12, borderRadius: 8,
                border: "1px solid #ccc", background: "transparent",
                fontSize: 15, cursor: "pointer"
              }}>Stop</button>
            </>
        }
      </div>

      {/* Snapshot */}
      {snapshot && (
        <img src={snapshot} alt="snapshot"
          style={{ width: "100%", borderRadius: 10, marginBottom: 16 }} />
      )}

      {/* Error */}
      {error && (
        <div style={{
          marginTop: 10, padding: 16, borderRadius: 12,
          border: "1.5px solid #e74c3c", background: "#fff5f5"
        }}>
          <p style={{ color: "#e74c3c", fontWeight: 600, margin: "0 0 4px", fontSize: 15 }}>
            Cannot predict
          </p>
          <p style={{ color: "#c0392b", fontSize: 14, margin: 0 }}>{error}</p>
        </div>
      )}

      {/* Result — uses result.prediction not result.grade */}
      {result && (
        <div style={{ padding: 20, borderRadius: 12, border: "1px solid #eee", marginTop: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
            <span style={{
              background: COLORS[grade] || "#888", color: "#fff",
              padding: "4px 14px", borderRadius: 20, fontWeight: 600,
              textTransform: "capitalize"
            }}>
              {grade}  {/* ← result.prediction */}
            </span>
            <span style={{ color: "#666" }}>{result.confidence.toFixed(1)}% confidence</span>
          </div>

          {/* Probability bars */}
          {Object.entries(result.all_probs).map(([cls, prob]) => (
            <div key={cls} style={{ marginBottom: 10 }}>
              <div style={{
                display: "flex", justifyContent: "space-between",
                fontSize: 13, marginBottom: 4
              }}>
                <span style={{ textTransform: "capitalize" }}>{cls}</span>
                <span>{(prob * 100).toFixed(1)}%</span>
              </div>
              <div style={{ background: "#f0f0f0", borderRadius: 4, height: 10 }}>
                <div style={{
                  width: `${prob * 100}%`, height: "100%", borderRadius: 4,
                  background: COLORS[cls] || "#888", transition: "width 0.5s"
                }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
} 