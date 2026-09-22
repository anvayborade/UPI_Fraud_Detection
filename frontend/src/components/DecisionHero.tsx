import { CheckCircle2, Clock, Fingerprint, ShieldAlert, ShieldCheck, TriangleAlert, XCircle } from 'lucide-react'
import type { AppMode, DemoScenario, FraudPrediction } from '../types'
import { actionTone, money, pct } from '../lib/format'
import { GlassCard } from './GlassCard'
import { RiskOrb } from './RiskOrb'

const actionIcon = (action: string) => {
  const a = action.toUpperCase()
  if (a === 'ALLOW') return ShieldCheck
  if (a === 'BLOCK') return XCircle
  if (a === 'HOLD') return Clock
  if (a === 'STEP_UP') return Fingerprint
  if (a === 'WARN') return TriangleAlert
  return CheckCircle2
}

export function DecisionHero({ scenario, decision, mode, busy, onScore, onPrecheck }: {
  scenario?: DemoScenario
  decision?: FraudPrediction
  mode: AppMode
  busy: boolean
  onScore: () => void
  onPrecheck: () => void
}) {
  const risk = decision?.fraud_probability ?? scenario?.offline?.fraud_probability ?? 0
  const action = decision?.recommended_action ?? scenario?.offline?.recommended_action ?? 'READY'
  const ActionIcon = actionIcon(action)
  return (
    <GlassCard className="decision-hero" glow={actionTone(action)}>
      <div className="hero-copy">
        <div className="eyebrow-row"><span className="eyebrow">CURRENT DECISION</span><span className={`action-pill ${actionTone(action)}`}><ActionIcon size={14}/>{action}</span></div>
        <h1>{scenario?.label ?? 'Select an edge case'}</h1>
        <p>{scenario?.description ?? 'Choose a held-out scenario to replay it against the live FastAPI scoring service.'}</p>
        {scenario && <div className="hero-meta">
          <span>{money(scenario.amount)}</span><i/> <span>{scenario.source_dataset}</span><i/> <span>{scenario.recipient_profile}</span>
        </div>}
        {decision && mode === 'executive' && <div className="plain-language-callout">
          <ShieldAlert size={17}/>
          <span>{decision.recommended_action === 'ALLOW'
            ? 'The payment is unusual only where the surrounding context can explain it; protective signals outweigh fraud evidence.'
            : `The system found enough combined evidence to recommend ${decision.recommended_action.toLowerCase()} rather than relying on one suspicious feature.`}</span>
        </div>}
        <div className="hero-actions">
          <button className="button primary" disabled={!scenario || busy} onClick={onScore}>{busy ? 'Scoring…' : 'Run full decision'}</button>
          <button className="button ghost" disabled={!scenario || busy} onClick={onPrecheck}>Recipient precheck</button>
        </div>
      </div>
      <div className="hero-orb-wrap">
        <RiskOrb value={risk}/>
        {decision && <div className="orb-substats"><span>Legitimate context <b>{pct(decision.legitimate_novelty_probability)}</b></span><span>Uncertainty <b>{pct(decision.uncertainty)}</b></span><span>Latency <b>{decision.latency_ms.toFixed(1)} ms</b></span></div>}
      </div>
    </GlassCard>
  )
}
