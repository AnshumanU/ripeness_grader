import { useState } from "react"
import axios from "axios"

const FRUITS = [
  "banana", "orange", "tomato", "avocado", "cherry",
  "corn", "pepper", "physalis", "zucchini", "cactus"
]
const COLORS = {
  ripe: "#2ecc71",
  overripe: "#e74c3c",
  unripe: "#f39c12"
}

export default function UploadPredict() {
  const [fruit, setFruit] = useState("banana")
  const [preview, setPreview] = useState(null)
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const onFileChange = (e) => {
    const f = e.target.files[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }

  const predict = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append("file", file)
    form.append("fruit", fruit) // ← also send as form field (Flask reads request.form)

    try {
      const res = await axios.post(
        `${import.meta.env.VITE_API_URL}/predict`,
        form
      )
      console.log(res.data)
      if (res.data.error) {
        setError(res.data.error)
      } else {
        setResult(res.data)
      }
    } catch (e) {
      console.error(e)
      setError("Detection failed. Make sure the backend is running.")
    }
    setLoading(false)
  }

  // result.prediction is what Flask returns (not result.grade)
  const grade = result?.prediction

  return (
    <div>
      {/* Fruit selector */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 16 }}>
        {FRUITS.map((f) => (
          <button key={f} onClick={() => { setFruit(f); setResult(null); setError(null) }}
            style={{
              padding: "6px 18px", borderRadius: 6, cursor: "pointer",
              background: fruit === f ? "#1b4332" : "#f0f0f0",
              color: fruit === f ? "#fff" : "#333",
              border: "none", fontWeight: 500, textTransform: "capitalize",
            }}>{f}</button>
        ))}
      </div>

      {/* Upload zone */}
      <label style={{
        display: "block", border: "2px dashed #ccc", borderRadius: 12,
        padding: "32px", textAlign: "center", cursor: "pointer",
        background: "#fafafa", marginBottom: 16,
      }}>
        <input type="file" accept="image/*" onChange={onFileChange} style={{ display: "none" }} />
        {preview ? (
          <img src={preview} alt="preview"
            style={{ maxWidth: "100%", maxHeight: 300, borderRadius: 8 }} />
        ) : (
          <p style={{ color: "#aaa", margin: 0 }}>Click or drag an image here</p>
        )}
      </label>

      {/* Predict button */}
      <button onClick={predict} disabled={!file || loading}
        style={{
          width: "100%", padding: "12px", borderRadius: 8, border: "none",
          background: file ? "#2d6a4f" : "#ccc", color: "#fff",
          fontSize: 16, cursor: file ? "pointer" : "not-allowed", fontWeight: 500,
        }}>
        {loading ? "Predicting..." : "Predict Ripeness"}
      </button>

      {/* Error */}
      {error && (
        <div style={{
          marginTop: 20, padding: 16, borderRadius: 12,
          border: "1.5px solid #e74c3c", background: "#fff5f5",
        }}>
          <p style={{ color: "#e74c3c", fontWeight: 600, margin: "0 0 4px", fontSize: 15 }}>
            Cannot predict
          </p>
          <p style={{ color: "#c0392b", fontSize: 14, margin: 0 }}>{error}</p>
        </div>
      )}

      {/* Result — uses result.prediction not result.grade */}
      {result && (
        <div style={{ marginTop: 24, padding: 20, borderRadius: 12, border: "1px solid #eee" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
            <span style={{
              background: COLORS[grade] || "#888", color: "#fff",
              padding: "4px 14px", borderRadius: 20, fontWeight: 600,
              fontSize: 15, textTransform: "capitalize",
            }}>
              {grade}  {/* ← result.prediction */}
            </span>
            <span style={{ color: "#666" }}>
              {result.confidence.toFixed(1)}% confidence
            </span>
          </div>

          {/* Probability bars */}
          {Object.entries(result.all_probs).map(([cls, prob]) => (
            <div key={cls} style={{ marginBottom: 10 }}>
              <div style={{
                display: "flex", justifyContent: "space-between",
                fontSize: 13, marginBottom: 4,
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