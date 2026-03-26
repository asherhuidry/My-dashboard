/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg:       { DEFAULT: "#080b14", secondary: "#0d1117", card: "#0f1624", hover: "#141d2e" },
        border:   { DEFAULT: "#1a2740", bright: "#243b5e" },
        accent:   { DEFAULT: "#3b82f6", dim: "#1d4ed8", glow: "#60a5fa" },
        positive: { DEFAULT: "#10b981", dim: "#059669", glow: "#34d399" },
        negative: { DEFAULT: "#ef4444", dim: "#dc2626", glow: "#f87171" },
        warning:  { DEFAULT: "#f59e0b", glow: "#fbbf24" },
        purple:   { DEFAULT: "#8b5cf6", glow: "#a78bfa" },
        cyan:     { DEFAULT: "#06b6d4", glow: "#22d3ee" },
        text:     { DEFAULT: "#e2e8f0", secondary: "#94a3b8", muted: "#475569" },
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "'Fira Code'", "monospace"],
        sans: ["'Inter'", "system-ui", "sans-serif"],
      },
      boxShadow: {
        glow:         "0 0 20px rgba(59,130,246,0.15)",
        "glow-green": "0 0 20px rgba(16,185,129,0.15)",
        "glow-red":   "0 0 20px rgba(239,68,68,0.15)",
        card:         "0 4px 24px rgba(0,0,0,0.4)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "glow-pulse": "glowPulse 2s ease-in-out infinite",
        "float":      "float 6s ease-in-out infinite",
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
      },
    },
  },
  plugins: [],
}

