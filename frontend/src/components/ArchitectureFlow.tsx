import { ArrowDown, BrainCircuit, Database, GitMerge, Network, RadioTower, Scale, Server, Shield, TimerReset } from 'lucide-react'
import { GlassCard } from './GlassCard'

const experts = [
  ['Transaction LightGBM','Fast row-level risk'], ['Legitimate novelty','Explains unusual-but-valid behaviour'], ['Transformer','Sequence takeover / social engineering'],
  ['HGT + temporal graph','Relationships, recipient trust, mule structure'], ['Graph anomaly','Unseen network patterns'], ['Complaint intelligence','Delayed recipient reputation'],
  ['Journey & device','Context, proximity, session integrity'], ['Rules','Deterministic hard constraints'],
]

export function ArchitectureFlow() {
  return <div className="architecture-page">
    <GlassCard className="architecture-hero"><span className="eyebrow">DUAL-SPEED ARCHITECTURE</span><h2>Fraud evidence and legitimate explanations meet before policy acts</h2><p>Heavy sequence and graph intelligence refreshes asynchronously into Redis. The authorisation path uses fast tabular models plus those time-appropriate cached expert states, then a learned fusion layer resolves disagreement.</p></GlassCard>
    <div className="architecture-flow">
      <div className="arch-stage"><div className="arch-icon"><RadioTower/></div><span>Kafka</span><strong>Transaction events</strong><small>Streaming ingress</small></div><ArrowDown className="arch-arrow"/>
      <div className="arch-split">
        <GlassCard className="arch-path fast"><TimerReset/><span>FAST PATH</span><strong>Rules + LightGBM + context</strong><small>Synchronous pre-authorisation scoring</small></GlassCard>
        <GlassCard className="arch-path slow"><Network/><span>NEAR-REAL-TIME PATH</span><strong>Transformer + HGT + TGN + anomaly</strong><small>Refreshes entity intelligence</small></GlassCard>
      </div><ArrowDown className="arch-arrow"/>
      <div className="arch-stage wide"><Database/><span>Redis feature / snapshot layer</span><strong>Current entity state + transaction-time snapshots</strong><small>Low-latency lookup, explicit availability</small></div><ArrowDown className="arch-arrow"/>
      <div className="expert-cloud">{experts.map(([name,desc])=><div className="expert-chip" key={name}><BrainCircuit/><div><strong>{name}</strong><span>{desc}</span></div></div>)}</div><ArrowDown className="arch-arrow"/>
      <div className="arch-stage fusion"><GitMerge/><span>Mixture-of-Experts fusion</span><strong>Learned reconciliation of contradictory signals</strong><small>Final fraud probability + uncertainty + fraud-type scores</small></div><ArrowDown className="arch-arrow"/>
      <div className="arch-stage policy"><Scale/><span>Policy engine</span><strong>ALLOW · WARN · CONFIRM · STEP_UP · HOLD · BLOCK</strong><small>Operational action is separate from model probability</small></div><ArrowDown className="arch-arrow"/>
      <div className="arch-split bottom"><GlassCard className="arch-end"><Server/><span>FastAPI</span><strong>Decision API</strong></GlassCard><GlassCard className="arch-end"><Shield/><span>React console</span><strong>Explainable presentation layer</strong></GlassCard></div>
    </div>
  </div>
}
