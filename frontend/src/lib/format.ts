export const pct = (value: number | undefined | null, digits = 1) =>
  value == null || Number.isNaN(value) ? '—' : `${(value * 100).toFixed(digits)}%`

export const money = (value: number | undefined | null) =>
  value == null || Number.isNaN(value)
    ? '—'
    : new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(value)

export const compact = (value: number | undefined | null) =>
  value == null || Number.isNaN(value) ? '—' : new Intl.NumberFormat('en-IN', { notation: 'compact', maximumFractionDigits: 1 }).format(value)

export const titleize = (value: string) =>
  value.replace(/^is_/, '').replaceAll('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase())

export const riskTone = (risk: number) => {
  if (risk >= 0.82) return 'danger'
  if (risk >= 0.42) return 'warning'
  return 'safe'
}

export const actionTone = (action: string) => {
  const normalized = action.toUpperCase()
  if (normalized === 'BLOCK') return 'danger'
  if (normalized === 'HOLD' || normalized === 'STEP_UP') return 'warning'
  if (normalized === 'WARN' || normalized === 'CONFIRM') return 'caution'
  return 'safe'
}
