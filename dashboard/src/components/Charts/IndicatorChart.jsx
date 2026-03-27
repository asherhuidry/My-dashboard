import { useEffect, useRef } from 'react'
import {
  createChart, ColorType, LineStyle,
  LineSeries, HistogramSeries,
} from 'lightweight-charts'

function baseTheme(el) {
  return {
    layout: {
      background: { type: ColorType.Solid, color: 'transparent' },
      textColor:  '#64748b',
      fontSize:   10,
    },
    grid: {
      vertLines: { color: 'rgba(26,39,64,0.4)', style: LineStyle.Dashed },
      horzLines: { color: 'rgba(26,39,64,0.4)', style: LineStyle.Dashed },
    },
    crosshair: {
      vertLine: { color: '#3b82f6', labelBackgroundColor: '#0f1624' },
      horzLine: { color: '#3b82f6', labelBackgroundColor: '#0f1624' },
    },
    rightPriceScale: { borderColor: 'rgba(26,39,64,0.6)' },
    timeScale:       { borderColor: 'rgba(26,39,64,0.6)', timeVisible: true },
    width:  el.clientWidth,
    height: el.clientHeight,
  }
}

function useChart(ref, builder) {
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const chart = createChart(el, baseTheme(el))
    builder(chart)
    const ro = new ResizeObserver(() => {
      if (el) chart.applyOptions({ width: el.clientWidth, height: el.clientHeight })
    })
    ro.observe(el)
    chart.timeScale().fitContent()
    return () => { ro.disconnect(); chart.remove() }
  })
}

// ── RSI ────────────────────────────────────────────────────────────────────

export function RSIChart({ data14, data28 }) {
  const ref = useRef()
  useEffect(() => {
    const el = ref.current
    if (!el || !data14?.length) return
    const chart = createChart(el, baseTheme(el))

    const times = data14.map(d => d.time)
    const band = (val, color) => {
      const s = chart.addSeries(LineSeries, { color, lineWidth: 1, lineStyle: LineStyle.Dashed, lastValueVisible: false, priceLineVisible: false })
      s.setData(times.map(t => ({ time: t, value: val })))
    }
    band(70, 'rgba(239,68,68,0.35)')
    band(50, 'rgba(148,163,184,0.2)')
    band(30, 'rgba(16,185,129,0.35)')

    const rsi14 = chart.addSeries(LineSeries, { color: '#3b82f6', lineWidth: 2, lastValueVisible: true, title: 'RSI 14' })
    rsi14.setData(data14)
    if (data28?.length) {
      const rsi28 = chart.addSeries(LineSeries, { color: '#8b5cf6', lineWidth: 1, lineStyle: LineStyle.Dashed, lastValueVisible: true, title: 'RSI 28' })
      rsi28.setData(data28)
    }

    const ro = new ResizeObserver(() => { if (el) chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }) })
    ro.observe(el)
    chart.timeScale().fitContent()
    return () => { ro.disconnect(); chart.remove() }
  }, [data14, data28])

  return <div ref={ref} className="w-full h-full" />
}

// ── MACD ───────────────────────────────────────────────────────────────────

export function MACDChart({ macd, signal, hist }) {
  const ref = useRef()
  useEffect(() => {
    const el = ref.current
    if (!el || !macd?.length) return
    const chart = createChart(el, baseTheme(el))

    if (hist?.length) {
      const h = chart.addSeries(HistogramSeries, { lastValueVisible: false, priceLineVisible: false })
      h.setData(hist.map(d => ({
        time:  d.time,
        value: d.value ?? 0,
        color: (d.value ?? 0) >= 0 ? 'rgba(16,185,129,0.55)' : 'rgba(239,68,68,0.55)',
      })))
    }
    const m = chart.addSeries(LineSeries, { color: '#3b82f6', lineWidth: 2, title: 'MACD', lastValueVisible: true })
    m.setData(macd)
    if (signal?.length) {
      const s = chart.addSeries(LineSeries, { color: '#f59e0b', lineWidth: 1, lineStyle: LineStyle.Dashed, title: 'Signal', lastValueVisible: true })
      s.setData(signal)
    }

    const ro = new ResizeObserver(() => { if (el) chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }) })
    ro.observe(el)
    chart.timeScale().fitContent()
    return () => { ro.disconnect(); chart.remove() }
  }, [macd, signal, hist])

  return <div ref={ref} className="w-full h-full" />
}

// ── Stochastic ─────────────────────────────────────────────────────────────

export function StochasticChart({ k, d }) {
  const ref = useRef()
  useEffect(() => {
    const el = ref.current
    if (!el || !k?.length) return
    const chart = createChart(el, baseTheme(el))

    const times = k.map(x => x.time)
    const band = (val, color) => {
      const s = chart.addSeries(LineSeries, { color, lineWidth: 1, lineStyle: LineStyle.Dashed, lastValueVisible: false, priceLineVisible: false })
      s.setData(times.map(t => ({ time: t, value: val })))
    }
    band(80, 'rgba(239,68,68,0.35)')
    band(20, 'rgba(16,185,129,0.35)')

    const sk = chart.addSeries(LineSeries, { color: '#06b6d4', lineWidth: 2, title: '%K', lastValueVisible: true })
    sk.setData(k)
    if (d?.length) {
      const sd = chart.addSeries(LineSeries, { color: '#f59e0b', lineWidth: 1, lineStyle: LineStyle.Dashed, title: '%D', lastValueVisible: true })
      sd.setData(d)
    }

    const ro = new ResizeObserver(() => { if (el) chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }) })
    ro.observe(el)
    chart.timeScale().fitContent()
    return () => { ro.disconnect(); chart.remove() }
  }, [k, d])

  return <div ref={ref} className="w-full h-full" />
}

// ── OBV ────────────────────────────────────────────────────────────────────

export function OBVChart({ data }) {
  const ref = useRef()
  useEffect(() => {
    const el = ref.current
    if (!el || !data?.length) return
    const chart = createChart(el, baseTheme(el))
    const s = chart.addSeries(LineSeries, { color: '#10b981', lineWidth: 2, title: 'OBV', lastValueVisible: true })
    s.setData(data)
    const ro = new ResizeObserver(() => { if (el) chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }) })
    ro.observe(el)
    chart.timeScale().fitContent()
    return () => { ro.disconnect(); chart.remove() }
  }, [data])
  return <div ref={ref} className="w-full h-full" />
}
