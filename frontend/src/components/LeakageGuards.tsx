import { CheckCircle2, ClockArrowDown, DatabaseZap, Fingerprint, History, ShieldCheck, Split, TestTube2 } from 'lucide-react'
import { GlassCard } from './GlassCard'

const guards = [
  { icon:Split, title:'Chronological split', text:'Train, validation and test are separated in time rather than randomly mixing future behaviour into the past.' },
  { icon:ClockArrowDown, title:'Prior-only graph state', text:'Temporal graph scores are produced before the current event is appended to history.' },
  { icon:History, title:'Delayed complaints', text:'A complaint can influence a transaction only when the complaint timestamp is already available.' },
  { icon:DatabaseZap, title:'Transaction-time Redis snapshot', text:'Historical demos request snapshot:txn:<id> instead of borrowing the recipient’s latest cache state.' },
  { icon:Fingerprint, title:'Stable recipient identity', text:'A recipient keeps one profile/archetype; compromise transitions move forward chronologically rather than oscillating row by row.' },
  { icon:TestTube2, title:'Validation-only calibration', text:'Fraud-type thresholds are selected on validation data; the test set is reserved for final measurement.' },
  { icon:ShieldCheck, title:'Missing ≠ zero risk', text:'Fusion sees availability flags so a missing graph/cache expert is not silently interpreted as a clean zero-risk signal.' },
  { icon:CheckCircle2, title:'Contradictory fusion examples', text:'The gate is exposed to plausible journeys with fraud, noisy graph anomalies with legitimacy, and scams before complaints arrive.' },
]

export function LeakageGuards() {
  return <div className="guard-grid">{guards.map(({icon:Icon,title,text})=><GlassCard className="guard-card" key={title}><div className="guard-icon"><Icon size={20}/></div><div><h3>{title}</h3><p>{text}</p></div><CheckCircle2 className="guard-check" size={17}/></GlassCard>)}</div>
}
