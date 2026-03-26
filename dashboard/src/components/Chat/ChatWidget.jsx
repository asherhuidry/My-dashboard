import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { MessageSquare, X, Send, Bot, User, Sparkles, Loader2 } from 'lucide-react'
import axios from 'axios'

const SUGGESTIONS = [
  "What does the RSI tell us about AAPL right now?",
  "Explain the MACD signal in simple terms",
  "What's the difference between realized vol and ATR?",
  "How do I interpret a Bollinger Band squeeze?",
  "What makes a stock technically bullish?",
]

function Message({ msg }) {
  const isBot = msg.role === 'assistant'
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-2.5 ${isBot ? '' : 'flex-row-reverse'}`}
    >
      <div className={`w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center border ${
        isBot ? 'bg-accent/20 border-accent/40' : 'bg-purple/20 border-purple/40'
      }`}>
        {isBot
          ? <Bot size={13} className="text-accent" />
          : <User size={13} className="text-purple" />
        }
      </div>
      <div className={`max-w-[85%] rounded-xl px-3 py-2.5 text-xs leading-relaxed ${
        isBot
          ? 'bg-bg border border-border text-text'
          : 'bg-accent/15 border border-accent/25 text-text'
      }`}>
        {msg.content.split('\n').map((line, i) => (
          <span key={i}>
            {line.replace(/\*\*(.*?)\*\*/g, (_, t) => t)}
            {i < msg.content.split('\n').length - 1 && <br />}
          </span>
        ))}
        {msg.tokens && (
          <div className="mt-1 text-[9px] text-text-muted">{msg.tokens} tokens</div>
        )}
      </div>
    </motion.div>
  )
}

export default function ChatWidget({ symbol, analysisContext }) {
  const [open, setOpen]       = useState(false)
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: `Hi! I'm FinBrain AI. ${symbol ? `I can see you're looking at **${symbol}** — ask me anything about it, or any other market question.` : 'Ask me anything about markets, technical analysis, or specific stocks.'}`,
  }])
  const [input, setInput]     = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef             = useRef()
  const inputRef              = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, open])

  const send = async (text) => {
    const msg = text ?? input.trim()
    if (!msg || loading) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: msg }])
    setLoading(true)
    try {
      const { data } = await axios.post('http://localhost:8000/api/chat', {
        message: msg,
        symbol:  symbol ?? null,
        context: analysisContext ?? null,
      })
      setMessages(prev => [...prev, {
        role:    'assistant',
        content: data.reply,
        tokens:  data.tokens,
      }])
    } catch (e) {
      setMessages(prev => [...prev, {
        role:    'assistant',
        content: 'Sorry, I encountered an error. Make sure the API is running and your ANTHROPIC_API_KEY is set.',
      }])
    }
    setLoading(false)
    setTimeout(() => inputRef.current?.focus(), 100)
  }

  return (
    <>
      {/* Floating button */}
      <motion.button
        onClick={() => setOpen(o => !o)}
        className="fixed bottom-6 right-6 z-50 w-13 h-13 rounded-full bg-accent border border-accent/60 shadow-glow flex items-center justify-center hover:bg-accent/80 transition-colors"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {open
          ? <X size={18} className="text-white" />
          : <div className="relative">
              <MessageSquare size={18} className="text-white" />
              <span className="absolute -top-1 -right-1 w-2 h-2 bg-positive rounded-full" />
            </div>
        }
      </motion.button>

      {/* Chat panel */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1,    y: 0  }}
            exit={{   opacity: 0, scale: 0.95, y: 20  }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="fixed bottom-24 right-6 z-50 w-80 md:w-96 h-[520px] flex flex-col glass-bright rounded-2xl border border-border shadow-card overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center gap-2.5 px-4 py-3 border-b border-border bg-bg-secondary/80">
              <div className="w-7 h-7 rounded-full bg-accent/20 border border-accent/40 flex items-center justify-center animate-glow">
                <Sparkles size={12} className="text-accent" />
              </div>
              <div>
                <div className="text-xs font-semibold text-text">FinBrain AI</div>
                <div className="text-[9px] text-positive flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-positive inline-block" /> Online
                </div>
              </div>
              {symbol && (
                <span className="ml-auto text-[10px] text-accent bg-accent/10 border border-accent/20 px-2 py-0.5 rounded">
                  {symbol}
                </span>
              )}
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-3 space-y-3">
              {messages.map((m, i) => <Message key={i} msg={m} />)}
              {loading && (
                <div className="flex gap-2.5">
                  <div className="w-7 h-7 rounded-full bg-accent/20 border border-accent/40 flex items-center justify-center">
                    <Bot size={13} className="text-accent" />
                  </div>
                  <div className="bg-bg border border-border rounded-xl px-3 py-2.5 flex items-center gap-2">
                    <Loader2 size={12} className="text-accent animate-spin" />
                    <span className="text-[10px] text-text-muted">Thinking…</span>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            {/* Suggestions (when empty-ish) */}
            {messages.length <= 1 && (
              <div className="px-3 pb-2 space-y-1">
                {SUGGESTIONS.slice(0,3).map((s,i) => (
                  <button key={i} onClick={() => send(s)}
                    className="w-full text-left text-[10px] text-text-secondary px-2.5 py-1.5 rounded-lg hover:bg-bg-hover border border-border hover:border-border-bright transition-colors">
                    {s}
                  </button>
                ))}
              </div>
            )}

            {/* Input */}
            <div className="p-3 border-t border-border">
              <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-bg border border-border focus-within:border-accent/50 transition-colors">
                <input
                  ref={inputRef}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
                  placeholder="Ask about markets, indicators…"
                  className="flex-1 bg-transparent text-xs text-text placeholder-text-muted outline-none"
                  disabled={loading}
                />
                <button
                  onClick={() => send()}
                  disabled={!input.trim() || loading}
                  className="w-6 h-6 rounded-lg bg-accent/20 border border-accent/30 flex items-center justify-center hover:bg-accent/30 transition-colors disabled:opacity-40"
                >
                  <Send size={10} className="text-accent" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
