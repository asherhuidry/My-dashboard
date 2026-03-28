import { useRef, useEffect, useCallback, useMemo, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

const EDGE_STYLE = {
  SENSITIVE_TO:    { dash: null,   opacity: 'cc' },
  CORRELATED_WITH: { dash: [4, 3], opacity: 'aa' },
  BELONGS_TO:      { dash: [2, 2], opacity: '88' },
  HAS_FEATURES:    { dash: [6, 2], opacity: '88' },
}

export default function ForceGraph({ nodes, edges, onNodeClick, onNodeRightClick }) {
  const fgRef     = useRef()
  const [tooltip, setTooltip] = useState(null)
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })

  const graphData = useMemo(() => ({
    nodes: nodes.map(n => ({ ...n, id: n.id })),
    links: edges.map(e => ({ source: e.source, target: e.target, ...e })),
  }), [nodes, edges])

  const paintNode = useCallback((node, ctx, globalScale) => {
    if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) return

    const degree = node.degree ?? 0
    const baseR  = Math.max(degree * 0.8 + 3, 3.5)
    const r      = baseR / globalScale * 1.5 + 2
    const label  = node.label ?? node.id
    const col    = node.color ?? '#3b82f6'

    // Outer glow — stronger for high-degree nodes
    const glowR    = r * (degree > 5 ? 3 : 2.2)
    const glowAlpha = degree > 5 ? '50' : '30'
    const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, glowR)
    grad.addColorStop(0, col + glowAlpha)
    grad.addColorStop(1, 'transparent')
    ctx.beginPath()
    ctx.arc(node.x, node.y, glowR, 0, 2 * Math.PI)
    ctx.fillStyle = grad
    ctx.fill()

    // Circle fill
    ctx.beginPath()
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
    ctx.fillStyle = col + 'dd'
    ctx.fill()

    // Ring — thicker for hub nodes
    ctx.strokeStyle = col
    ctx.lineWidth   = (degree > 8 ? 2 : 1) / globalScale
    ctx.stroke()

    // Inner highlight dot for high-degree nodes
    if (degree > 10) {
      ctx.beginPath()
      ctx.arc(node.x, node.y, r * 0.3, 0, 2 * Math.PI)
      ctx.fillStyle = '#ffffff60'
      ctx.fill()
    }

    // Label
    if (globalScale > 0.6) {
      const fontSize = Math.max(10 / globalScale, 2.5)
      ctx.font = `${degree > 5 ? 'bold ' : ''}${fontSize}px JetBrains Mono, monospace`
      ctx.fillStyle = '#e2e8f0'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      const text = label.length > 10 ? label.slice(0, 9) + '\u2026' : label
      ctx.fillText(text, node.x, node.y + r + fontSize + 1)
    }
  }, [])

  const paintLink = useCallback((link, ctx, globalScale) => {
    if (!Number.isFinite(link.source.x) || !Number.isFinite(link.target.x)) return

    const col   = link.color ?? '#1a2740'
    const width = Math.max((link.width ?? 1) / globalScale, 0.3)
    const style = EDGE_STYLE[link.label] ?? { dash: null, opacity: '99' }

    ctx.beginPath()
    if (style.dash) ctx.setLineDash(style.dash.map(d => d / globalScale))
    else ctx.setLineDash([])

    ctx.moveTo(link.source.x, link.source.y)
    ctx.lineTo(link.target.x, link.target.y)
    ctx.strokeStyle = col + style.opacity
    ctx.lineWidth   = width
    ctx.stroke()
    ctx.setLineDash([])

    // Edge label when zoomed in
    if (link.label && globalScale > 1.2) {
      const mx = (link.source.x + link.target.x) / 2
      const my = (link.source.y + link.target.y) / 2
      const fontSize = Math.max(7 / globalScale, 2.5)
      ctx.font = `${fontSize}px JetBrains Mono, monospace`
      ctx.fillStyle = '#94a3b8'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(link.label, mx, my)
    }

    // Beta / correlation value when zoomed further
    if (globalScale > 2) {
      const val = link.beta ?? link.correlation ?? link.value
      if (val != null) {
        const mx = (link.source.x + link.target.x) / 2
        const my = (link.source.y + link.target.y) / 2
        const fontSize = Math.max(5.5 / globalScale, 2)
        ctx.font = `${fontSize}px JetBrains Mono, monospace`
        ctx.fillStyle = '#64748b'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        const prefix = link.beta != null ? '\u03B2=' : 'r='
        ctx.fillText(`${prefix}${Number(val).toFixed(2)}`, mx, my + fontSize + 1)
      }
    }
  }, [])

  const handleHover = useCallback(node => setTooltip(node ?? null), [])
  const modeReplace = useCallback(() => 'replace', [])

  const forcesConfigured = useRef(false)
  useEffect(() => {
    if (fgRef.current && !forcesConfigured.current) {
      fgRef.current.d3Force('charge').strength(-140)
      fgRef.current.d3Force('link').distance(l => {
        const w = l.width ?? 1
        return Math.max(100 - w * 12, 40)
      })
      forcesConfigured.current = true
    }
  })

  return (
    <div
      className="relative w-full h-full rounded-xl overflow-hidden bg-bg"
      onMouseMove={e => setMousePos({ x: e.clientX, y: e.clientY })}
      onContextMenu={e => e.preventDefault()}
    >
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        backgroundColor="transparent"
        nodeCanvasObject={paintNode}
        nodeCanvasObjectMode={modeReplace}
        linkCanvasObject={paintLink}
        linkCanvasObjectMode={modeReplace}
        onNodeClick={onNodeClick}
        onNodeRightClick={onNodeRightClick}
        onNodeHover={handleHover}
        cooldownTicks={120}
        enableZoomInteraction
        enablePanInteraction
        minZoom={0.2}
        maxZoom={8}
      />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 pointer-events-none text-xs glass-bright rounded-lg px-3 py-2 border border-border max-w-[220px]"
          style={{ left: mousePos.x + 14, top: mousePos.y - 10 }}
        >
          <div className="font-semibold text-text">{tooltip.label ?? tooltip.id}</div>
          {tooltip.type  && <div className="text-text-muted text-[10px]">{tooltip.type}</div>}
          {tooltip.class && <div className="text-text-muted text-[10px] capitalize">{tooltip.class}</div>}
          {tooltip.degree != null && <div className="text-text-muted text-[10px]">{tooltip.degree} connections</div>}
          {tooltip.price != null && <div className="ticker-value text-accent">${tooltip.price}</div>}
          {tooltip.ret_1d != null && (
            <div className={tooltip.ret_1d >= 0 ? 'text-positive' : 'text-negative'}>
              {tooltip.ret_1d >= 0 ? '+' : ''}{tooltip.ret_1d}% today
            </div>
          )}
          {tooltip.value != null && <div className="text-text-muted">corr: {tooltip.value}</div>}
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 right-4 glass rounded-lg px-3 py-2 text-[10px] space-y-1">
        {[...new Set(nodes.map(n => n.class ?? n.type))].filter(Boolean).slice(0,6).map(cls => (
          <div key={cls} className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full" style={{ background: nodes.find(n => (n.class ?? n.type) === cls)?.color ?? '#6b7280' }} />
            <span className="text-text-muted capitalize">{cls}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
