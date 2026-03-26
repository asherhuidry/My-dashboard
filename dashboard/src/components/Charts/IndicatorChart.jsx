import { useEffect, useRef } from 'react'
import { createChart, LineStyle } from 'lightweight-charts'

const THEME = {
  layout:     { background: { color: 'transparent' }, textColor: '#94a3b8' },
  grid:       { vertLines: { color: '#1a2740', style: LineStyle.Dashed }, horzLines: { color: '#1a2740', style: LineStyle.Dashed } },
  crosshair:  { vertLine: { color: '#3b82f6', labelBackgroundColor: '#0f1624' }, horzLine: { color: '#3b82f6', labelBackgroundColor: '#0f1624' } },
  rightPriceScale: { borderColor: '#1a2740' },
  timeScale:  { borderColor: '#1a2740', timeVisible: true },
}

// RSI Panel
export function RSIChart({ data14, data28 }) {
  const ref = useRef()
  useEffect(() => {
    if (!ref.current || !data14?.length) return
    const chart = createChart(ref.current, { ...THEME, width: ref.current.clientWidth, height: ref.current.clientHeight })

    // Overbought/oversold bands
    const ob = chart.addLineSeries({ color: 'rgba(239,68,68,0.3)',  lineWidth: 1, lineStyle: LineStyle.Dashed, title: '70' })
    const os = chart.addLineSeries({ color: 'rgba(16,185,129,0.3)', lineWidth: 1, lineStyle: LineStyle.Dashed, title: '30' })
    const mid= chart.addLineSeries({ color: 'rgba(148,163,184,0.2)',lineWidth: 1, lineStyle: LineStyle.Dotted, title: '50' })
    const rsi14 = data14.map(d => d.time)
    ob.setData(rsi14.map(t => ({ time: t, value: 70 })))
    os.setData(rsi14.map(t => ({ time: t, value: 30 })))
    mid.setData(rsi14.map(t => ({ time: t, value: 50 })))

    const s14 = chart.addLineSeries({ color: '#3b82f6', lineWidth: 2, title: 'RSI 14' })
    s14.setData(data14)
    if (data28?.length) {
      const s28 = chart.addLineSeries({ color: '#8b5cf6', lineWidth: 1, lineStyle: LineStyle.Dashed, title: 'RSI 28' })
      s28.setData(data28)
    }

    const ro = new ResizeObserver(() => chart.applyOptions({ width: ref.current?.clientWidth }))
    ro.observe(ref.current)
    chart.timeScale().fitContent()
    return () => { ro.disconnect(); chart.remove() }
  }, [data14, data28])
  return <div ref={ref} className="w-full h-full" />
}

// MACD Panel
export function MACDChart({ macd, signal, hist }) {
  const ref = useRef()
  useEffect(() => {
    if (!ref.current || !macd?.length) return
    const chart = createChart(ref.current, { ...THEME, width: ref.current.clientWidth, height: ref.current.clientHeight })

    if (hist?.length) {
      const histSeries = chart.addHistogramSeries({
        color: '#3b82f6',
        priceLineVisible: false,
      })
      histSeries.setData(hist.map(d => ({
        time:  d.time,
        value: d.value,
        color: d.value >= 0 ? 'rgba(16,185,129,0.6)' : 'rgba(239,68,68,0.6)',
      })))
    }

    const macdSeries   = chart.addLineSeries({ color: '#3b82f6', lineWidth: 2, title: 'MACD' })
    const signalSeries = chart.addLineSeries({ color: '#f59e0b', lineWidth: 1, lineStyle: LineStyle.Dashed, title: 'Signal' })
    macdSeries.setData(macd)
    if (signal?.length) signalSeries.setData(signal)

    const ro = new ResizeObserver(() => chart.applyOptions({ width: ref.current?.clientWidth }))
    ro.observe(ref.current)
    chart.timeScale().fitContent()
    return () => { ro.disconnect(); chart.remove() }
  }, [macd, signal, hist])
  return <div ref={ref} className="w-full h-full" />
}

// Stochastic Panel
export function StochasticChart({ k, d }) {
  const ref = useRef()
  useEffect(() => {
    if (!ref.current || !k?.length) return
    const chart = createChart(ref.current, { ...THEME, width: ref.current.clientWidth, height: ref.current.clientHeight })

    const times = k.map(d => d.time)
    const ob = chart.addLineSeries({ color: 'rgba(239,68,68,0.3)',  lineWidth: 1, lineStyle: LineStyle.Dashed })
    const os = chart.addLineSeries({ color: 'rgba(16,185,129,0.3)', lineWidth: 1, lineStyle: LineStyle.Dashed })
    ob.setData(times.map(t => ({ time: t, value: 80 })))
    os.setData(times.map(t => ({ time: t, value: 20 })))

    const kSeries = chart.addLineSeries({ color: '#06b6d4', lineWidth: 2, title: '%K' })
    kSeries.setData(k)
    if (d?.length) {
      const dSeries = chart.addLineSeries({ color: '#f59e0b', lineWidth: 1, lineStyle: LineStyle.Dashed, title: '%D' })
      dSeries.setData(d)
    }
    const ro = new ResizeObserver(() => chart.applyOptions({ width: ref.current?.clientWidth }))
    ro.observe(ref.current)
    chart.timeScale().fitContent()
    return () => { ro.disconnect(); chart.remove() }
  }, [k, d])
  return <div ref={ref} className="w-full h-full" />
}
