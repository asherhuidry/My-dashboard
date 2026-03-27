import { useEffect, useRef, useState } from 'react'
import {
  createChart, ColorType, CrosshairMode, LineStyle,
  CandlestickSeries, HistogramSeries, LineSeries,
} from 'lightweight-charts'

const BG = 'transparent'

function makeTheme(container) {
  return {
    layout: {
      background: { type: ColorType.Solid, color: BG },
      textColor: '#64748b',
      fontSize: 11,
    },
    grid: {
      vertLines: { color: 'rgba(26,39,64,0.6)', style: LineStyle.Dashed },
      horzLines: { color: 'rgba(26,39,64,0.6)', style: LineStyle.Dashed },
    },
    crosshair: {
      mode: CrosshairMode.Normal,
      vertLine: { color: '#3b82f6', width: 1, labelBackgroundColor: '#0f1624' },
      horzLine: { color: '#3b82f6', width: 1, labelBackgroundColor: '#0f1624' },
    },
    rightPriceScale: { borderColor: 'rgba(26,39,64,0.8)', scaleMargins: { top: 0.08, bottom: 0.18 } },
    timeScale: { borderColor: 'rgba(26,39,64,0.8)', timeVisible: true, secondsVisible: false },
    width:  container.clientWidth,
    height: container.clientHeight,
  }
}

const OVERLAY_STYLES = {
  bb_upper:  { color: 'rgba(139,92,246,0.6)',  lineWidth: 1, lineStyle: LineStyle.Dashed },
  bb_lower:  { color: 'rgba(139,92,246,0.6)',  lineWidth: 1, lineStyle: LineStyle.Dashed },
  bb_mid:    { color: 'rgba(139,92,246,0.25)', lineWidth: 1, lineStyle: LineStyle.Dotted },
  ema_9:     { color: '#f59e0b', lineWidth: 1 },
  ema_21:    { color: '#3b82f6', lineWidth: 1 },
  ema_50:    { color: '#10b981', lineWidth: 1 },
  ema_200:   { color: '#ef4444', lineWidth: 2 },
}

export default function PriceChart({ candles, indicators, overlays = [] }) {
  const containerRef = useRef()
  const chartRef     = useRef()
  const [hovered, setHovered] = useState(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el || !candles?.length) return

    const chart = createChart(el, makeTheme(el))
    chartRef.current = chart

    // Candlestick series
    const candle = chart.addSeries(CandlestickSeries, {
      upColor:         '#10b981',
      downColor:       '#ef4444',
      borderUpColor:   '#10b981',
      borderDownColor: '#ef4444',
      wickUpColor:     '#10b981',
      wickDownColor:   '#ef4444',
    })
    candle.setData(candles)

    // Volume
    const vol = chart.addSeries(HistogramSeries, {
      priceFormat:  { type: 'volume' },
      priceScaleId: 'vol',
    })
    chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } })
    vol.setData(candles.map(c => ({
      time:  c.time,
      value: c.volume,
      color: c.close >= c.open ? 'rgba(16,185,129,0.25)' : 'rgba(239,68,68,0.25)',
    })))

    // Overlays
    if (indicators) {
      overlays.forEach(key => {
        const style = OVERLAY_STYLES[key]
        const data  = indicators[key]
        if (!style || !data?.length) return
        const s = chart.addSeries(LineSeries, style)
        s.setData(data)
      })
    }

    // Crosshair hover
    chart.subscribeCrosshairMove(p => {
      const price = p?.seriesData?.get(candle)
      setHovered(price ? { ...price, time: p.time } : null)
    })

    // Resize
    const ro = new ResizeObserver(() => {
      if (el) chart.applyOptions({ width: el.clientWidth, height: el.clientHeight })
    })
    ro.observe(el)
    chart.timeScale().fitContent()

    return () => { ro.disconnect(); chart.remove() }
  }, [candles, JSON.stringify(overlays)])

  const h  = hovered
  const up = h ? h.close >= h.open : null

  return (
    <div className="relative w-full h-full">
      {h && (
        <div className="absolute top-3 left-3 z-10 flex items-center gap-3 ticker-value text-[11px] bg-bg/80 backdrop-blur px-2 py-1 rounded-lg border border-border">
          {['open','high','low','close'].map(k => (
            <span key={k} className={up ? 'text-positive' : 'text-negative'}>
              {k[0].toUpperCase()} {h[k]?.toFixed(2)}
            </span>
          ))}
        </div>
      )}
      <div ref={containerRef} className="w-full h-full" />
    </div>
  )
}
