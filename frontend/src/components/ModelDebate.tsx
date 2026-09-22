import { ArrowRight, BrainCircuit, Check, Gavel, Shield, TriangleAlert, X } from 'lucide-react'
import type { AppMode, FraudPrediction } from '../types'
import { pct, titleize } from '../lib/format'
import { GlassCard } from './GlassCard'
import { ScoreBar } from './ScoreBar'

const protectiveKeys = new Set(['legitimate_novelty', 'graph_merchant_legitimacy', 'journey_plausibility'])
const labels: Record<string, string> = {
  transaction_fraud: 'Transaction expert', statistical_anomaly: 'Anomaly expert', device_session_risk: 'Device/session expert',
  sequence_account_takeover: 'Takeover sequence expert', sequence_social_engineering: 'Social-engineering sequence', graph_mule: 'Mule graph expert',
  graph_merchant_legitimacy: 'Merchant trust expert', graph_anomaly: 'Graph anomaly expert', complaint_intelligence: 'Complaint intelligence',
  journey_plausibility: 'Journey expert', legitimate_novelty: 'Legitimate-context expert', rule_risk: 'Rules engine',
}

function stance(key: string, value: number) {
  if (protectiveKeys.has(key)) return value >= .70 ? 'supports' : value <= .30 ? 'weak' : 'neutral'
  return value >= .70 ? 'warns' : value <= .25 ? 'clears' : 'neutral'
}

export function ModelDebate({ decision, mode }: { decision?: FraudPrediction; mode: AppMode }) {
  if (!decision) return <GlassCard className="empty-state"><BrainCircuit/><h3>No scored transaction yet</h3><p>Run an edge case to see the experts disagree, support one another, and reach a fused decision.</p></GlassCard>
  const entries = Object.entries(decision.model_scores).filter(([, value]) => typeof value === 'number')
  const riskEntries = entries.filter(([key]) => !protectiveKeys.has(key))
  const protectiveEntries = entries.filter(([key]) => protectiveKeys.has(key))
  return (
    <div className="debate-layout">
      <GlassCard className="debate-panel">
        <div className="section-heading"><div><span className="eyebrow">MODEL DEBATE</span><h2>What each specialist is saying</h2></div><span className="mode-chip">{mode}</span></div>
        <div className="debate-columns">
          <div><h3 className="column-title danger-text"><TriangleAlert size={16}/> Suspicious evidence</h3>{riskEntries.map(([key, value]) => <ScoreBar key={key} name={labels[key] ?? key} value={value}/>)}</div>
          <div><h3 className="column-title safe-text"><Shield size={16}/> Protective evidence</h3>{protectiveEntries.map(([key, value]) => <ScoreBar key={key} name={labels[key] ?? key} value={value} protective/>)}</div>
        </div>
      </GlassCard>
      <GlassCard className="verdict-panel" glow={decision.recommended_action === 'ALLOW' ? 'safe' : 'danger'}>
        <Gavel size={24}/><span className="eyebrow">FUSION VERDICT</span><strong>{decision.recommended_action}</strong>
        <p>Final fraud risk <b>{pct(decision.fraud_probability)}</b> after reconciling specialist evidence, missing-state signals and contradictions.</p>
        <div className="reason-stack">
          {decision.reason_codes.map((code) => <div key={code}><Check size={14}/><span>{titleize(code)}</span></div>)}
        </div>
        {decision.fraud_types.length > 0 && <div className="fraud-tags">{decision.fraud_types.map(type => <span key={type}><X size={12}/>{titleize(type)}</span>)}</div>}
        <div className="fusion-arrow"><span>Experts</span><ArrowRight size={16}/><span>MoE fusion</span><ArrowRight size={16}/><span>Policy</span></div>
      </GlassCard>
    </div>
  )
}
