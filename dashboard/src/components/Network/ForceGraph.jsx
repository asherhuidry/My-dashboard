import { useRef, useEffect, useCallback, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { motion } from 'framer-motion'

export default function ForceGraph({ nodes, edges, onNodeClick, title }) {
  const fgRef     = useRef()
  const [tooltip, setTooltip] = useState(null)
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })

  const graphData = {
    nodes: nodes.map(n => ({ ...n, id: n.id })),
    links: edges.map(e => ({ source: e.source, target: e.target, ...e })),
  }

  const paintNode = useCallback((node, ctx, globalScale) => {
    if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) return

    const r     = (node.val ?? 6) / globalScale * 1.5 + 3
    const label = node.label ?? node.id
    const col   = node.color ?? '#3b82f6'

    // Glow
    const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, r * 2.5)
    grad.addColorStop(0, col + '40')
    grad.addColorStop(1, 'transparent')
    ctx.beginPath()
    ctx.arc(node.x, node.y, r * 2.5, 0, 2 * Math.PI)
    ctx.fillStyle = grad
    ctx.fill()

    // Circle
    ctx.beginPath()
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
    ctx.fillStyle = col + 'cc'
    ctx.fill()
    ctx.strokeStyle = col
    ctx.lineWidth   = 1 / globalScale
    ctx.stroke()

    // Label (only when zoomed in enough)
    if (globalScale > 0.8) {
      const fontSize = Math.max(10 / globalScale, 3)
      ctx.font = `${fontSize}px JetBrains Mono, monospace`
      ctx.fillStyle = '#e2e8f0'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(label.length > 8 ? label.slice(0,7)+'…' : label, node.x, node.y + r + fontSize + 1)
    }
  }, [])

  const paintLink = useCallback((link, ctx, globalScale) => {
    if (!Number.isFinite(link.source.x) || !Number.isFinite(link.target.x)) return

    const col   = link.color ?? '#1a2740'
    const width = (link.width ?? 1) / globalScale
    ctx.beginPath()
    ctx.moveTo(link.source.x, link.source.y)
    ctx.lineTo(link.target.x, link.target.y)
    ctx.strokeStyle = col + '99'
    ctx.lineWidth   = width
    ctx.stroke()

    // Edge label when zoomed in (knowledge graph edges have a label field)
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
  }, [])

  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge').strength(-120)
      fgRef.current.d3Force('link').distance(80)
    }
  }, [nodes, edges])

  return (
    <div
      className="relative w-full h-full rounded-xl overflow-hidden bg-bg"
      onMouseMove={e => setMousePos({ x: e.clientX, y: e.clientY })}
    >
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        backgroundColor="transparent"
        nodeCanvasObject={paintNode}
        nodeCanvasObjectMode={() => 'replace'}
        linkCanvasObject={paintLink}
        linkCanvasObjectMode={() => 'replace'}
        onNodeClick={node => onNodeClick?.(node)}
        onNodeHover={node => setTooltip(node ?? null)}
        cooldownTicks={120}
        enableZoomInteraction
        enablePanInteraction
        minZoom={0.2}
        maxZoom={6}
      />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 pointer-events-none text-xs glass-bright rounded-lg px-3 py-2 border border-border"
          style={{ left: mousePos.x + 14, top: mousePos.y - 10 }}
        >
          <div className="font-semibold text-text">{tooltip.label ?? tooltip.id}</div>
          {tooltip.type  && <div className="text-text-muted">{tooltip.type}</div>}
          {tooltip.price && <div className="ticker-value text-accent">${tooltip.price}</div>}
          {tooltip.ret_1d != null && (
            <div className={tooltip.ret_1d >= 0 ? 'text-positive' : 'text-negative'}>
              {tooltip.ret_1d >= 0 ? '+' : ''}{tooltip.ret_1d}% today
            </div>
          )}
          {tooltip.value != null && <div className="text-text-muted">corr: {tooltip.value}</div>}
          {tooltip.class && <div className="text-text-muted capitalize">{tooltip.class}</div>}
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
