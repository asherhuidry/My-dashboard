import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

const GROUPS = {
  'Returns':    ['ret_1d','ret_5d','ret_10d','ret_21d','ret_63d','log_ret_1d','log_ret_5d'],
  'Volatility': ['realized_vol_5d','realized_vol_21d','garman_klass_vol','atr_14','atr_ratio'],
  'Momentum':   ['rsi_14','rsi_28','macd','macd_signal','macd_hist','stoch_k','stoch_d','roc_10','roc_21','williams_r_14'],
  'Trend':      ['bb_width','bb_pct','adx_14','ema_9_21_cross','ema_21_50_cross'],
  'Volume':     ['obv_ema_21','vwap_21','volume_ratio_10','cmf_20'],
  'Regime':     ['rolling_sharpe_21','rolling_sharpe_63','max_drawdown_63','rolling_vol_21'],
  'Calendar':   ['day_of_week','month','quarter','is_month_end','fomc_window','earnings_season'],
}

function normalize(val, key) {
  if (val == null) return 0
  // Normalize to [-1, 1] range for color mapping
  const ranges = {
    rsi_14: [0,100], rsi_28: [0,100], stoch_k: [0,100], stoch_d: [0,100],
    bb_pct: [0,1], adx_14: [0,60], day_of_week: [0,4], month: [1,12],
  }
  const r = ranges[key]
  if (r) return ((val - r[0]) / (r[1] - r[0])) * 2 - 1
  if (['ema_9_21_cross','ema_21_50_cross','is_month_end','fomc_window','earnings_season'].includes(key)) return val * 2 - 1
  return Math.tanh(val) // sigmoid-like for unbounded
}

export default function FeatureHeatmap({ features }) {
  const ref = useRef()

  useEffect(() => {
    if (!ref.current || !features) return
    const el = ref.current
    d3.select(el).selectAll('*').remove()

    const entries = Object.entries(GROUPS).flatMap(([group, keys]) =>
      keys.filter(k => features[k] != null).map(k => ({
        group, key: k, val: features[k], norm: normalize(features[k], k)
      }))
    )

    const groups = [...new Set(entries.map(e => e.group))]
    const cellW  = 56
    const cellH  = 44
    const labelW = 90
    const maxCols = Math.max(...groups.map(g => entries.filter(e => e.group === g).length))
    const W = labelW + maxCols * cellW + 16
    const H = groups.length * (cellH + 4) + 32

    const svg = d3.select(el).append('svg')
      .attr('width', '100%')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')

    const color = d3.scaleSequential(d3.interpolateRgbBasis([
      '#ef4444', '#7f1d1d', '#1a2740', '#064e3b', '#10b981'
    ])).domain([-1, 1])

    groups.forEach((group, gi) => {
      const row = entries.filter(e => e.group === group)
      const y   = gi * (cellH + 4) + 16

      // Group label
      svg.append('text')
        .attr('x', 0).attr('y', y + cellH / 2 + 4)
        .attr('fill', '#64748b').attr('font-size', 9)
        .attr('font-family', 'Inter, sans-serif')
        .attr('font-weight', '600')
        .attr('text-transform', 'uppercase')
        .text(group.toUpperCase())

      row.forEach((entry, ci) => {
        const x = labelW + ci * cellW
        const g = svg.append('g').attr('transform', `translate(${x},${y})`)

        // Cell bg
        g.append('rect')
          .attr('width', cellW - 2).attr('height', cellH)
          .attr('rx', 4)
          .attr('fill', color(entry.norm))
          .attr('opacity', 0.85)

        // Key label
        g.append('text')
          .attr('x', (cellW-2)/2).attr('y', 14)
          .attr('text-anchor', 'middle')
          .attr('fill', '#e2e8f0').attr('font-size', 7.5)
          .attr('font-family', 'JetBrains Mono, monospace')
          .text(entry.key.replace(/_/g, ' ').slice(0,10))

        // Value
        g.append('text')
          .attr('x', (cellW-2)/2).attr('y', 30)
          .attr('text-anchor', 'middle')
          .attr('fill', '#f8fafc').attr('font-size', 9)
          .attr('font-weight', '600')
          .attr('font-family', 'JetBrains Mono, monospace')
          .text(Math.abs(entry.val) < 0.01
            ? entry.val.toExponential(1)
            : entry.val > 1000 ? entry.val.toExponential(1)
            : entry.val.toFixed(2))

        // Tooltip
        g.append('title').text(`${entry.key}: ${entry.val}`)
      })
    })
  }, [features])

  return <div ref={ref} className="w-full overflow-x-auto" />
}
