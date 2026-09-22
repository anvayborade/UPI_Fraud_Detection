import { Ambulance, BriefcaseBusiness, CarFront, Hotel, Landmark, QrCode, Repeat2, ShieldAlert, Smartphone, Store, UserRoundCheck, UsersRound } from 'lucide-react'
import type { DemoScenario } from '../types'
import { GlassCard } from './GlassCard'
import { money, pct } from '../lib/format'

const icons: Record<string, typeof CarFront> = {
  auto: CarFront, hospital: Ambulance, qr_scam: QrCode, account_takeover: Smartphone, mule: UsersRound,
  collect_scam: ShieldAlert, remote_access: Smartphone, legitimate_split: Repeat2, new_merchant: Store,
  gig_worker: BriefcaseBusiness, travel_hotel: Hotel, familiar: UserRoundCheck,
}

export function EdgeCaseGallery({ scenarios, selectedKey, onSelect }: { scenarios: DemoScenario[]; selectedKey?: string; onSelect: (scenario: DemoScenario) => void }) {
  return (
    <div className="edge-gallery">
      {scenarios.map(scenario => {
        const Icon = icons[scenario.key] ?? Landmark
        const risk = Number(scenario.offline?.fraud_probability ?? 0)
        return <button key={scenario.key} className={`edge-card ${scenario.kind} ${selectedKey === scenario.key ? 'selected' : ''}`} onClick={() => onSelect(scenario)}>
          <div className="edge-card-top"><span className="edge-icon"><Icon size={19}/></span><span className={`kind-pill ${scenario.kind}`}>{scenario.kind === 'fraud' ? 'FRAUD CHALLENGE' : 'LEGITIMATE EDGE'}</span></div>
          <h3>{scenario.label}</h3><p>{scenario.description}</p>
          <div className="edge-card-meta"><span>{money(scenario.amount)}</span><span>{scenario.source_dataset}</span></div>
          <div className="edge-risk"><span>Offline fused risk</span><strong>{pct(risk)}</strong></div>
        </button>
      })}
    </div>
  )
}
