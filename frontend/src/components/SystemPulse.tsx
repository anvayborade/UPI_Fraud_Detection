import { Activity, BrainCircuit, Database, RadioTower } from 'lucide-react'
import type { HealthState } from '../types'
import { API_URL } from '../lib/api'

function Dot({ ok }: { ok?: boolean }) { return <span className={`status-dot ${ok ? 'online' : 'offline'}`} /> }

export function SystemPulse({ health, online }: { health: HealthState; online: boolean }) {
  return (
    <div className="system-pulse glass-inline">
      <div className="pulse-item"><Activity size={14}/><Dot ok={online}/><span>API</span></div>
      <div className="pulse-item"><Database size={14}/><Dot ok={health.redis}/><span>Redis</span></div>
      <div className="pulse-item"><BrainCircuit size={14}/><Dot ok={health.fusion_model_loaded}/><span>Fusion</span></div>
      <div className="pulse-item dim"><RadioTower size={14}/><span>{API_URL.replace('http://', '')}</span></div>
    </div>
  )
}
