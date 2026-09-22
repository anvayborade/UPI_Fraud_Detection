import { Clock3, MapPin, Smartphone, Sparkles } from 'lucide-react'
import type { DemoScenario } from '../types'
import { GlassCard } from './GlassCard'
import { money, pct, titleize } from '../lib/format'

export function TransactionTimeline({ scenario }: { scenario?: DemoScenario }) {
  if (!scenario || scenario.timeline.length === 0) return <GlassCard className="empty-state"><Clock3/><h3>No temporal history</h3><p>Run the frontend data exporter to create a prior-only payer timeline.</p></GlassCard>
  return (
    <GlassCard className="timeline-card">
      <div className="section-heading"><div><span className="eyebrow">SEQUENCE VIEW</span><h2>What happened before the decision</h2></div><span className="mode-chip">{scenario.timeline.length} prior events</span></div>
      <div className="timeline">
        {scenario.timeline.map((event, i) => (
          <div className={`timeline-event ${event.selected ? 'selected' : ''}`} key={event.transaction_id}>
            <div className="timeline-rail"><span>{i + 1}</span></div>
            <div className="timeline-content">
              <div className="timeline-top"><strong>{money(event.amount)}</strong><time>{new Date(event.timestamp).toLocaleString('en-IN', {dateStyle:'medium', timeStyle:'short'})}</time></div>
              <div className="timeline-title">{titleize(event.scenario)}</div>
              <div className="timeline-tags">
                {event.tags.length ? event.tags.map(tag => <span key={tag}><Smartphone size={12}/>{tag}</span>) : <span className="quiet"><Sparkles size={12}/>No high-impact session flag</span>}
                {event.fraud_probability != null && <span className="risk-tag">risk {pct(event.fraud_probability)}</span>}
                {event.selected && <span className="selected-tag"><MapPin size={12}/>Selected transaction</span>}
              </div>
            </div>
          </div>
        ))}
      </div>
    </GlassCard>
  )
}
