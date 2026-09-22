import { pct, titleize } from '../lib/format'

export function ScoreBar({ name, value, protective = false, subtle = false }: { name: string; value: number; protective?: boolean; subtle?: boolean }) {
  const barValue = Math.max(0, Math.min(100, value * 100))
  return (
    <div className={`score-row ${protective ? 'protective' : 'risk'} ${subtle ? 'subtle' : ''}`}>
      <div className="score-row-head"><span>{titleize(name)}</span><strong>{pct(value)}</strong></div>
      <div className="score-track"><span style={{ width: `${barValue}%` }} /></div>
    </div>
  )
}
