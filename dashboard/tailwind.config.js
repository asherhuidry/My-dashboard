/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg:       { DEFAULT: "#060a13", secondary: "#0a1020", card: "#0c1526", hover: "#111d30", deep: "#040710", elevated: "#0e1a2e" },
        border:   { DEFAULT: "#162033", bright: "#1e3a5f", glow: "#2d5a8e" },
        accent:   { DEFAULT: "#3b82f6", dim: "#1d4ed8", glow: "#60a5fa", muted: "#2563eb" },
        positive: { DEFAULT: "#10b981", dim: "#059669", glow: "#34d399" },
        negative: { DEFAULT: "#ef4444", dim: "#dc2626", glow: "#f87171" },
        warning:  { DEFAULT: "#f59e0b", glow: "#fbbf24" },
        purple:   { DEFAULT: "#8b5cf6", glow: "#a78bfa", dim: "#7c3aed" },
        cyan:     { DEFAULT: "#06b6d4", glow: "#22d3ee" },
        text:     { DEFAULT: "#e2e8f0", secondary: "#94a3b8", muted: "#475569", bright: "#f1f5f9" },
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "'Fira Code'", "monospace"],
        sans: ["'Inter'", "system-ui", "sans-serif"],
      },
      boxShadow: {
        glow:          "0 0 20px rgba(59,130,246,0.15), 0 0 60px rgba(59,130,246,0.05)",
        "glow-lg":     "0 0 30px rgba(59,130,246,0.25), 0 0 80px rgba(59,130,246,0.08)",
        "glow-green":  "0 0 20px rgba(16,185,129,0.15)",
        "glow-red":    "0 0 20px rgba(239,68,68,0.15)",
        "glow-purple": "0 0 20px rgba(139,92,246,0.15)",
        card:          "0 4px 24px rgba(0,0,0,0.5), 0 0 1px rgba(255,255,255,0.05)",
        "card-hover":  "0 8px 40px rgba(0,0,0,0.6), 0 0 1px rgba(255,255,255,0.08)",
        elevated:      "0 12px 48px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.04)",
      },
      animation: {
        "pulse-slow":  "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "glow-pulse":  "glowPulse 2s ease-in-out infinite",
        "float":       "float 6s ease-in-out infinite",
        "fade-in":     "fadeIn 0.4s ease-out",
        "slide-up":    "slideUp 0.3s ease-out",
        "slide-right": "slideRight 0.3s ease-out",
      },
      keyframes: {
        glowPulse: {
          "0%, 100%": { boxShadow: "0 0 5px rgba(59,130,246,0.3)" },
          "50%":      { boxShadow: "0 0 20px rgba(59,130,246,0.8)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%":      { transform: "translateY(-6px)" },
        },
        fadeIn: {
          "0%":   { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%":   { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideRight: {
          "0%":   { opacity: "0", transform: "translateX(-8px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
      },
      backgroundImage: {
        "gradient-radial":  "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":   "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
    },
  },
  plugins: [],
}
