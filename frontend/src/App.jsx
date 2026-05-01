import { useState, useEffect } from "react";
import Dashboard from "./pages/Dashboard";
import "./index.css";

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === "dark" ? "light" : "dark");

  return (
    <Dashboard theme={theme} toggleTheme={toggleTheme} />
  );
}