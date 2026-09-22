import { FlaskConical, RotateCcw, Zap } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { DemoScenario, FraudPrediction, UPIEvent } from '../types'
import { GlassCard } from './GlassCard'
import { money, pct } from '../lib/format'

const boolFields = [
  ['device_known', 'Known device'], ['qr_verified', 'QR verified'], ['collect_request', 'Collect request'],
  ['recent_pin_reset', 'Recent PIN reset'], ['app_reregistered', 'App re-registered'], ['remote_access_indicator', 'Remote access'],
  ['overlay_indicator', 'Overlay detected'], ['play_integrity_ok', 'Play Integrity OK'], ['is_new_payee', 'New payee'],
] as const

export function WhatIfPanel({ scenario, onRun, result, busy }: { scenario?: DemoScenario; onRun: (payload: UPIEvent) => void; result?: FraudPrediction; busy: boolean }) {
  const base = scenario?.payload
  const [draft, setDraft] = useState<UPIEvent>({})
  useEffect(() => setDraft(base ? { ...base } : {}), [base])
  const changed = useMemo(() => base ? Object.keys(draft).filter(key => draft[key] !== base[key]) : [], [draft, base])
  if (!scenario) return null
  return (
    <GlassCard className="whatif-card">
      <div className="section-heading"><div><span className="eyebrow">COUNTERFACTUAL LAB</span><h2>What-if stress test</h2></div><FlaskConical/></div>
      <p className="microcopy prominent">Changes affect the fast-path request while the historical graph/sequence snapshot stays anchored to <b>{scenario.reference_transaction_id}</b>. This is a stress test, not a new held-out evaluation row.</p>
      <div className="whatif-grid">
        <label className="field"><span>Amount · {money(Number(draft.amount ?? 0))}</span><input type="range" min="100" max="100000" step="500" value={Number(draft.amount ?? 1000)} onChange={e => setDraft({...draft, amount:Number(e.target.value)})}/></label>
        <label className="field"><span>Complaint intelligence input · {pct(Number(draft.recipient_complaint_score ?? 0))}</span><input type="range" min="0" max="1" step="0.05" value={Number(draft.recipient_complaint_score ?? 0)} onChange={e => setDraft({...draft, recipient_complaint_score:Number(e.target.value)})}/></label>
      </div>
      <div className="toggle-grid">{boolFields.map(([key,label]) => <button key={key} className={`mini-toggle ${Boolean(draft[key]) ? 'on' : ''}`} onClick={() => setDraft({...draft,[key]:!Boolean(draft[key])})}><span>{label}</span><i/></button>)}</div>
      <div className="whatif-actions"><button className="button primary" disabled={busy} onClick={() => onRun({...draft, transaction_id:`WHATIF-${Date.now()}`, simulation_mode:true})}><Zap size={15}/>{busy?'Running…':'Re-score what-if'}</button><button className="button ghost" onClick={() => setDraft(base ? {...base}: {})}><RotateCcw size={15}/>Reset</button><span>{changed.length} field{changed.length === 1 ? '' : 's'} changed</span></div>
      {result && <div className="whatif-result"><span>New risk <b>{pct(result.fraud_probability)}</b></span><span>Action <b>{result.recommended_action}</b></span><span>Uncertainty <b>{pct(result.uncertainty)}</b></span></div>}
    </GlassCard>
  )
}
