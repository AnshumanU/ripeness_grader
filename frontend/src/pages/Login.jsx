import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import api from "../lib/api";

const FRUIT_EMOJIS = ["🍎","🍊","🍋","🍇","🍓","🥭","🍌","🍑","🥝","🍐"];

export default function Login({ theme, toggleTheme }) {
  const nav = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!email || !password) { setError("Please fill in all fields."); return; }
    setLoading(true); setError("");
    try {
      const res = await api.post("/login", { email, password });
      localStorage.setItem("token", res.data.token);
      localStorage.setItem("user", JSON.stringify(res.data.user));
      nav("/dashboard");
    } catch {
      setError("Invalid email or password.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: "24px", position: "relative" }}>
      {/* Orb background */}
      <div className="orb-bg">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      {/* Floating fruit emojis */}
      {FRUIT_EMOJIS.map((e, i) => (
        <div key={i} style={{
          position: "fixed",
          fontSize: `${16 + (i % 3) * 8}px`,
          opacity: 0.08,
          top: `${10 + (i * 9)}%`,
          left: `${(i % 2 === 0 ? 5 : 85) + (i * 2)}%`,
          animation: `orb-drift ${12 + i * 2}s ease-in-out infinite alternate`,
          animationDelay: `${-i * 1.5}s`,
          pointerEvents: "none",
          zIndex: 0,
          userSelect: "none",
        }}>{e}</div>
      ))}

      {/* Theme toggle top right */}
      <div style={{ position: "fixed", top: 20, right: 24, zIndex: 200, display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{theme === "dark" ? "🌙" : "☀️"}</span>
        <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme" />
      </div>

      {/* Card */}
      <div className="glass anim-fade-up" style={{ width: "100%", maxWidth: 420, padding: "40px 36px", position: "relative", zIndex: 1 }}>
        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div style={{ fontSize: 48, marginBottom: 8 }}>🍃</div>
          <h1 className="font-display" style={{ fontSize: 28, fontWeight: 800, letterSpacing: "-0.02em", background: "linear-gradient(135deg, var(--accent), var(--accent2))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            FruitSense AI
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 14, marginTop: 4 }}>Sign in to your account</p>
        </div>

        {error && <div className="alert alert-error" style={{ marginBottom: 16 }}>{error}</div>}

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <input
            className="input" type="email" placeholder="Email address"
            value={email} onChange={e => setEmail(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleLogin()}
          />
          <input
            className="input" type="password" placeholder="Password"
            value={password} onChange={e => setPassword(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleLogin()}
          />
          <button className="btn" onClick={handleLogin} disabled={loading} style={{ marginTop: 4 }}>
            {loading ? <span className="spinner" style={{ borderTopColor: "#000" }} /> : "Sign In"}
          </button>
        </div>

        <p style={{ textAlign: "center", marginTop: 24, fontSize: 14, color: "var(--text-muted)" }}>
          No account?{" "}
          <Link to="/register" style={{ color: "var(--accent)", fontWeight: 600, textDecoration: "none" }}>
            Create one →
          </Link>
        </p>
      </div>
    </div>
  );
}
