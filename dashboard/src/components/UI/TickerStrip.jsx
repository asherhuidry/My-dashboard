import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchSearch } from '../../lib/api'
import axios from 'axios'

const FALLBACK_SYMS = [
  "AAPL","MSFT","NVDA","TSLA","GOOGL","AMZN","META","JPM","SPY","QQQ",
  "BTC-USD","ETH-USD","SOL-USD","BNB-USD","GLD"
]

function Tick({ item }) {
  if (!item?.price) return null
  const up    = item.change_pct >= 0
  const label = item.label ?? item.symbol?.replace('-USD','').replace('=X','')
  return (
    <div className="flex items-center gap-2 px-4 border-r border-border/50 flex-shrink-0">
      <span className="text-xs font-medium text-text-secondary">{label}</span>
      <span className="ticker-value text-xs text-text font-semibold">${Number(item.price).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: item.price > 100 ? 2 : 4 })}</span>
      <span className={`text-[10px] font-medium ${up ? 'text-positive' : 'text-negative'}`}>
        {up ? '▲' : '▼'} {Math.abs(item.change_pct ?? 0).toFixed(2)}%
      </span>
    </div>
  )
}

export default function TickerStrip() {
  const [prices, setPrices] = useState([])
  const wsRef   = useRef(null)
  const trackRef = useRef(null)
  const animRef  = useRef(null)
  const posRef   = useRef(0)

  // WebSocket connection
  useEffect(() => {
    let ws
    const connect = () => {
      try {
        ws = new WebSocket('ws://localhost:8000/ws/prices')
        ws.onopen    = () => { wsRef.current = ws }
        ws.onmessage = (e) => {
          const msg = JSON.parse(e.data)
          if (msg.type === 'prices' && msg.data?.length) {
            setPrices(msg.data.filter(p => p.price))
          }
        }
        ws.onerror = () => {}
        ws.onclose = () => {
          // REST fallback
          axios.get('http://localhost:8000/api/live-prices')
            .then(r => { if (r.data?.data?.length) setPrices(r.data.data.filter(p => p.price)) })
            .catch(() => {})
        }
      } catch (e) {}
    }
    connect()
    // Also poll REST every 60s as backup
    const interval = setInterval(() => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        axios.get('http://localhost:8000/api/live-prices')
          .then(r => { if (r.data?.data?.length) setPrices(r.data.data.filter(p => p.price)) })
          .catch(() => {})
      }
    }, 60_000)
    return () => {
      ws?.close()
      clearInterval(interval)
    }
  }, [])

  // Marquee animation
  useEffect(() => {
    if (!trackRef.current || prices.length === 0) return
    const el  = trackRef.current
    const speed = 0.4 // px per frame

    const animate = () => {
      posRef.current -= speed
      const halfWidth = el.scrollWidth / 2
      if (Math.abs(posRef.current) >= halfWidth) posRef.current = 0
      el.style.transform = `translateX(${posRef.current}px)`
      animRef.current = requestAnimationFrame(animate)
    }

    animRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animRef.current)
  }, [prices])

  if (prices.length === 0) return null

  // Duplicate for seamless loop
  const doubled = [...prices, ...prices]

  return (
    <div className="h-8 bg-bg-secondary border-b border-border overflow-hidden flex items-center relative">
      {/* Left fade */}
      <div className="absolute left-0 top-0 w-8 h-full bg-gradient-to-r from-bg-secondary to-transparent z-10 pointer-events-none" />
      {/* Right fade */}
      <div className="absolute right-0 top-0 w-8 h-full bg-gradient-to-l from-bg-secondary to-transparent z-10 pointer-events-none" />

      <div ref={trackRef} className="flex items-center will-change-transform">
        {doubled.map((item, i) => <Tick key={i} item={item} />)}
      </div>
    </div>
  )
}
