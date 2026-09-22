import type { CSSProperties } from 'react'
import { pct, riskTone } from '../lib/format'

export function RiskOrb({ value, label = 'Final fraud risk', size = 'large' }: { value: number; label?: string; size?: 'small' | 'large' }) {
  const tone = riskTone(value)
  const degrees = Math.max(3, Math.min(360, value * 360))
  return (
    <div className={`risk-orb ${size} ${tone}`} style={{ '--risk-angle': `${degrees}deg` } as CSSProperties}>
      <div className="risk-orb-core">
        <span className="eyebrow">{label}</span>
        <strong>{pct(value, value < 0.01 ? 2 : 1)}</strong>
        <span className="risk-caption">{tone === 'safe' ? 'Low risk' : tone === 'warning' ? 'Elevated' : 'Critical'}</span>
      </div>
    </div>
  )
}
