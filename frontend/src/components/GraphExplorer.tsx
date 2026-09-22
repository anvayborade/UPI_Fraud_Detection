import { Network, UsersRound } from 'lucide-react'
import type { DemoScenario, GraphNode } from '../types'
import { GlassCard } from './GlassCard'
import { money } from '../lib/format'

const nodeTone: Record<GraphNode['type'], string> = { target: '#ff3d81', payer: '#79f7ff', recipient: '#9b8cff', merchant: '#c8ff61' }

export function GraphExplorer({ scenario }: { scenario?: DemoScenario }) {
  if (!scenario || scenario.graph.nodes.length === 0) return <GlassCard className="empty-state"><Network/><h3>No graph snapshot</h3><p>Export frontend data after training to render transaction-time recipient neighbourhoods.</p></GlassCard>
  const { nodes, edges } = scenario.graph
  const width = 820, height = 520, cx = width / 2, cy = height / 2
  const target = nodes.find(n => n.type === 'target') ?? nodes[0]
  const others = nodes.filter(n => n.id !== target.id).slice(0, 30)
  const positions = new Map<string, {x:number;y:number}>()
  positions.set(target.id, { x: cx, y: cy })
  others.forEach((node, i) => {
    const ring = i < 14 ? 1 : 2
    const ringIndex = ring === 1 ? i : i - 14
    const ringCount = ring === 1 ? Math.min(14, others.length) : Math.max(1, others.length - 14)
    const angle = (Math.PI * 2 * ringIndex) / ringCount - Math.PI / 2
    const radius = ring === 1 ? 155 : 225
    positions.set(node.id, { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius })
  })
  const visibleEdges = edges.filter(e => positions.has(e.source) && positions.has(e.target))
  return (
    <GlassCard className="graph-card">
      <div className="section-heading"><div><span className="eyebrow">RECIPIENT GRAPH</span><h2>Prior-only neighbourhood</h2></div><div className="graph-legend"><span><i style={{background:nodeTone.target}}/>Target</span><span><i style={{background:nodeTone.payer}}/>Payer</span><span><i style={{background:nodeTone.merchant}}/>Merchant</span></div></div>
      <div className="graph-wrap">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Recipient graph">
          <defs><filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
          {visibleEdges.map(edge => {
            const a = positions.get(edge.source)!, b = positions.get(edge.target)!
            const strokeWidth = 0.7 + Math.min(5, Math.log10(edge.amount + 10))
            return <g key={edge.id}><line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="graph-edge" strokeWidth={strokeWidth}><title>{`${edge.count} txn · ${money(edge.amount)}`}</title></line></g>
          })}
          {[target, ...others].map(node => {
            const p = positions.get(node.id)!
            const radius = node.type === 'target' ? 22 : 11
            return <g key={node.id} className="graph-node"><circle cx={p.x} cy={p.y} r={radius + 7} fill={nodeTone[node.type]} opacity="0.09"/><circle cx={p.x} cy={p.y} r={radius} fill={nodeTone[node.type]} opacity="0.94" filter={node.type === 'target' ? 'url(#glow)' : undefined}/><title>{node.id}</title>{node.type === 'target' && <text x={p.x} y={p.y + 42} textAnchor="middle">{node.id.length > 26 ? `${node.id.slice(0,23)}…` : node.id}</text>}</g>
          })}
        </svg>
        <div className="graph-stats"><div><UsersRound/><span>Visible entities</span><strong>{Math.min(nodes.length,31)}</strong></div><div><Network/><span>Visible relations</span><strong>{visibleEdges.length}</strong></div></div>
      </div>
      <p className="microcopy">This visual uses only transactions at or before the selected transaction timestamp, matching the project’s temporal-leakage guardrail.</p>
    </GlassCard>
  )
}
