export type AppMode = 'executive' | 'technical'
export type ViewId = 'command' | 'edge-cases' | 'explain' | 'graph' | 'timeline' | 'performance' | 'trust' | 'architecture' | 'complaints'

export type ModelScores = {
  rule_risk: number
  statistical_anomaly: number
  transaction_fraud: number
  legitimate_novelty: number
  device_session_risk: number
  sequence_account_takeover: number
  sequence_social_engineering: number
  graph_mule: number
  graph_merchant_legitimacy: number
  graph_anomaly: number
  complaint_intelligence: number
  journey_plausibility: number
  [key: string]: number
}

export type FraudPrediction = {
  transaction_id: string
  fraud_probability: number
  legitimate_novelty_probability: number
  uncertainty: number
  fraud_types: string[]
  recommended_action: string
  reason_codes: string[]
  model_scores: ModelScores
  latency_ms: number
}

export type UPIEvent = Record<string, string | number | boolean | null | undefined>

export type GraphNode = {
  id: string
  label: string
  type: 'target' | 'payer' | 'recipient' | 'merchant'
  risk?: number
}

export type GraphEdge = {
  id: string
  source: string
  target: string
  amount: number
  count: number
}

export type TimelineEvent = {
  transaction_id: string
  timestamp: string
  amount: number
  scenario: string
  action?: string
  fraud_probability?: number
  tags: string[]
  selected: boolean
}

export type DemoScenario = {
  key: string
  label: string
  short_label: string
  scenario: string
  kind: 'legitimate' | 'fraud'
  category: string
  description: string
  source_dataset: string
  recipient_profile: string
  reference_transaction_id: string
  payer_id: string
  payee_id: string
  amount: number
  payload: UPIEvent
  offline?: Partial<FraudPrediction>
  graph: { nodes: GraphNode[]; edges: GraphEdge[] }
  timeline: TimelineEvent[]
}

export type HealthState = {
  status?: string
  redis?: boolean
  transaction_model_loaded?: boolean
  context_model_loaded?: boolean
  fusion_model_loaded?: boolean
}

export type MetricsData = {
  transaction_pr_auc?: number
  transaction_roc_auc?: number
  context_pr_auc?: number
  final_pr_auc?: number
  final_roc_auc?: number
  legitimate_novelty_false_positive_rate_at_05?: number
  fraud_type_thresholds?: Record<string, number>
  fraud_type_test_metrics?: Record<string, { pr_auc: number; prevalence: number }>
}

export type EvaluationData = {
  summary?: Record<string, number | string>
  by_source?: Record<string, Record<string, number>>
  by_scenario?: Record<string, { rows: number; fraud_rate: number; mean_risk: number; mean_uncertainty: number; actions: Record<string, number> }>
  action_distribution?: Record<string, number>
}
