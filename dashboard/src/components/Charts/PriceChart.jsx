import { useEffect, useRef, useState } from 'react'
import { createChart, CrosshairMode, LineStyle } from 'lightweight-charts'

const THEME = {
  layout:     { background: { color: 'transparent' }, textColor: '#94a3b8' },
  grid:       { vertLines: { color: '#1a2740', style: LineStyle.Dashed }, horzLines: { color: '#1a2740', style: LineStyle.Dashed } },
  crosshair:  { mode: CrosshairMode.Normal, vertLine: { color: '#3b82f6', labelBackgroundColor: '#0f1624' }, horzLine: { color: '#3b82f6', labelBackgroundColor: '#0f1624' } },
  rightPriceScale: { borderColor: '#1a2740' },
  timeScale:  { borderColor: '#1a2740', timeVisible: true, secondsVisible: false },
}

export default function PriceChart({ candles, indicators, overlays = [] }) {
  const containerRef = useRef()
  const chartRef     = useRef()
  const seriesRef    = useRef({})
  const [hoveredPrice, setHoveredPrice] = useState(null)

  useEffect(() => {
    if (!containerRef.current || !candles?.length) return

    const chart = createChart(containerRef.current, {
      ...THEME,
      width:  containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
    })
    chartRef.current = chart

    // Main candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor:          '#10b981',
      downColor:        '#ef4444',
      borderUpColor:    '#10b981',
      borderDownColor:  '#ef4444',
      wickUpColor:      '#10b981',
      wickDownColor:    '#ef4444',
    })
    candleSeries.setData(candles)
    seriesRef.current.candle = candleSeries

    // Volume histogram
    const volSeries = chart.addHistogramSeries({
      priceFormat:    { type: 'volume' },
      priceScaleId:   'vol',
      color:          '#1a2740',
      scaleMargins:   { top: 0.8, bottom: 0 },
    })
    const volData = candles.map(c => ({
      time:  c.time,
      value: c.volume,
      color: c.close >= c.open ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)',
    }))
    volSeries.setData(volData)
    seriesRef.current.volume = volSeries

    // Overlay series
    const overlayDefs = {
      bb_upper:  { color: 'rgba(139,92,246,0.5)', lineWidth: 1, lineStyle: LineStyle.Dashed,  title: 'BB Upper' },
      bb_lower:  { color: 'rgba(139,92,246,0.5)', lineWidth: 1, lineStyle: LineStyle.Dashed,  title: 'BB Lower' },
      bb_mid:    { color: 'rgba(139,92,246,0.3)', lineWidth: 1, lineStyle: LineStyle.Dotted,  title: 'BB Mid'   },
      ema_9:     { color: '#f59e0b',              lineWidth: 1, title: 'EMA 9'    },
      ema_21:    { color: '#3b82f6',              lineWidth: 1, title: 'EMA 21'   },
      ema_50:    { color: '#10b981',              lineWidth: 1, title: 'EMA 50'   },
      ema_200:   { color: '#ef4444',              lineWidth: 2, title: 'EMA 200'  },
    }

    if (indicators) {
      overlays.forEach(key => {
        const def  = overlayDefs[key]
        const data = indicators[key]
        if (!def || !data?.length) return
        const s = chart.addLineSeries({ color: def.color, lineWidth: def.lineWidth ?? 1, lineStyle: def.lineStyle ?? LineStyle.Solid, title: def.title })
        s.setData(data)
        seriesRef.current[key] = s
      })
    }

    // Crosshair price display
    chart.subscribeCrosshairMove(param => {
      if (param.time) {
        const price = param.seriesData.get(candleSeries)
        if (price) setHoveredPrice({ ...price, time: param.time })
      } else {
        setHoveredPrice(null)
      }
    })

    // Resize observer
    const ro = new ResizeObserver(() => {
      chart.applyOptions({ width: containerRef.current?.clientWidth })
    })
    ro.observe(containerRef.current)

    chart.timeScale().fitContent()

    return () => { ro.disconnect(); chart.remove() }
  }, [candles, JSON.stringify(overlays)])

  const p = hoveredPrice
  const isUp = p ? p.close >= p.open : null

  return (
    <div className="relative h-full">
      {p && (
        <div className="absolute top-2 left-3 z-10 flex items-center gap-3 text-xs ticker-value">
          <span className={isUp ? 'text-positive' : 'text-negative'}>O {p.open?.toFixed(2)}</span>
          <span className={isUp ? 'text-positive' : 'text-negative'}>H {p.high?.toFixed(2)}</span>
          <span className={isUp ? 'text-positive' : 'text-negative'}>L {p.low?.toFixed(2)}</span>
          <span className={isUp ? 'text-positive' : 'text-negative'}>C {p.close?.toFixed(2)}</span>
        </div>
      )}
      <div ref={containerRef} className="w-full h-full" />
    </div>
  )
}
