import { MessageSquareWarning, Sparkles } from 'lucide-react'
import { useState } from 'react'
import { GlassCard } from './GlassCard'
import { submitComplaint } from '../lib/api'

export function ComplaintLab({ defaultPayee }: { defaultPayee?: string }) {
  const [payee, setPayee] = useState(defaultPayee ?? 'A000001')
  const [text, setText] = useState('A caller said I would receive a refund and asked me to scan a QR code urgently. Money was debited after I entered my UPI PIN.')
  const [useLlm, setUseLlm] = useState(false)
  const [result, setResult] = useState<Record<string, unknown>>()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const run = async () => { setBusy(true); setError(''); try { setResult(await submitComplaint(payee,text,useLlm)) } catch(e){ setError(e instanceof Error?e.message:String(e)) } finally{setBusy(false)} }
  return <div className="complaint-layout"><GlassCard className="complaint-form"><div className="section-heading"><div><span className="eyebrow">COMPLAINT INTELLIGENCE</span><h2>Turn narrative reports into recipient evidence</h2></div><MessageSquareWarning/></div><label className="text-field"><span>Recipient ID</span><input value={payee} onChange={e=>setPayee(e.target.value)}/></label><label className="text-field"><span>Complaint narrative</span><textarea rows={8} value={text} onChange={e=>setText(e.target.value)}/></label><label className="checkbox-line"><input type="checkbox" checked={useLlm} onChange={e=>setUseLlm(e.target.checked)}/><span>Use configured external LLM extractor</span></label><button className="button primary" onClick={run} disabled={busy || text.length<5}><Sparkles size={15}/>{busy?'Extracting…':'Extract intelligence'}</button>{error&&<div className="error-box">{error}</div>}</GlassCard><GlassCard className="complaint-result"><span className="eyebrow">STRUCTURED OUTPUT</span>{result?<pre>{JSON.stringify(result,null,2)}</pre>:<div className="empty-mini"><Sparkles/><p>Scam type, entities, urgency and recipient complaint signals will appear here.</p></div>}</GlassCard></div>
}
