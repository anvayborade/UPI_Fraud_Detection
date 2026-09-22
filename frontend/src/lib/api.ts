import type { FraudPrediction, HealthState, UPIEvent } from '../types'

export const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

async function parseError(response: Response) {
  try {
    const body = await response.json()
    return body?.detail || JSON.stringify(body)
  } catch {
    return response.text()
  }
}

export async function getHealth(): Promise<HealthState> {
  const response = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(4000) })
  if (!response.ok) throw new Error(await parseError(response))
  return response.json()
}

export async function scoreEvent(payload: UPIEvent): Promise<FraudPrediction> {
  const response = await fetch(`${API_URL}/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(30000),
  })
  if (!response.ok) throw new Error(await parseError(response))
  return response.json()
}

export async function precheckEvent(payload: UPIEvent): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_URL}/precheck`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(30000),
  })
  if (!response.ok) throw new Error(await parseError(response))
  return response.json()
}

export async function submitComplaint(payee_id: string, text: string, use_llm: boolean) {
  const response = await fetch(`${API_URL}/complaint`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ payee_id, text, use_llm }),
    signal: AbortSignal.timeout(30000),
  })
  if (!response.ok) throw new Error(await parseError(response))
  return response.json()
}
