import { AlertTriangle, BarChart3, Gauge, ShieldCheck, Target } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { EvaluationData, MetricsData } from '../types'
import { GlassCard } from './GlassCard'
import { compact, pct, titleize } from '../lib/format'

const chartColors = ['#75f7ff', '#9b8cff', '#c8ff61', '#ff6f91', '#ffb86b', '#5df2b6', '#e8e2ff']

export function PerformanceDashboard({ metrics, evaluation }: { metrics?: MetricsData; evaluation?: EvaluationData }) {
  if (!metrics) return <GlassCard className="empty-state"><BarChart3/><h3>Metrics unavailable</h3><p>Run the frontend data exporter after training so the UI can read models/metrics.json.</p></GlassCard>
  const fraudTypes = Object.entries(metrics.fraud_type_test_metrics ?? {}).map(([key, value]) => ({ name:titleize(key), pr_auc:value.pr_auc, prevalence:value.prevalence }))
  const sources = Object.entries(evaluation?.by_source ?? {}).map(([name, values]) => ({ name, pr_auc:Number(values.pr_auc ?? 0), roc_auc:Number(values.roc_auc ?? 0), rows:Number(values.rows ?? 0) }))
  const scenarios = Object.entries(evaluation?.by_scenario ?? {}).map(([name, values]) => {
    const allowed = values.actions?.ALLOW ?? 0
    const intervention = values.rows ? 1 - allowed / values.rows : 0
    return { name:titleize(name), rows:values.rows, fraud_rate:values.fraud_rate, mean_risk:values.mean_risk, intervention }
  })
  return (
    <div className="performance-stack">
      <div className="metric-strip">
        <GlassCard className="metric-card"><span><Target size={16}/>Final PR-AUC</span><strong>{metrics.final_pr_auc?.toFixed(4) ?? '—'}</strong><small>Primary imbalanced-class ranking metric</small></GlassCard>
        <GlassCard className="metric-card"><span><Gauge size={16}/>Final ROC-AUC</span><strong>{metrics.final_roc_auc?.toFixed(4) ?? '—'}</strong><small>Overall ranking separation</small></GlassCard>
        <GlassCard className="metric-card"><span><ShieldCheck size={16}/>Novelty FPR</span><strong>{pct(metrics.legitimate_novelty_false_positive_rate_at_05,2)}</strong><small>At probability threshold 0.50</small></GlassCard>
        <GlassCard className="metric-card"><span><BarChart3 size={16}/>Transaction PR-AUC</span><strong>{metrics.transaction_pr_auc?.toFixed(4) ?? '—'}</strong><small>Fast tabular expert alone</small></GlassCard>
      </div>

      <div className="two-column">
        <GlassCard className="chart-card"><div className="section-heading"><div><span className="eyebrow">FRAUD-TYPE HEADS</span><h2>What the system identifies well</h2></div></div><div className="chart-box"><ResponsiveContainer width="100%" height={340}><BarChart data={fraudTypes} layout="vertical" margin={{left:24,right:24}}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.06)"/><XAxis type="number" domain={[0,1]} tick={{fill:'#8c91a8',fontSize:11}}/><YAxis dataKey="name" type="category" width={145} tick={{fill:'#cdd0df',fontSize:11}}/><Tooltip cursor={{fill:'rgba(255,255,255,.03)'}} contentStyle={{background:'#10121b',border:'1px solid rgba(255,255,255,.12)',borderRadius:14}} formatter={(value)=>Number(value).toFixed(4)}/><Bar dataKey="pr_auc" radius={[0,8,8,0]}>{fraudTypes.map((_,i)=><Cell key={i} fill={chartColors[i%chartColors.length]}/>)}</Bar></BarChart></ResponsiveContainer></div><div className="warning-callout"><AlertTriangle size={15}/><span>Transaction splitting is deliberately shown as a weakness rather than hidden behind the aggregate score.</span></div></GlassCard>
        <GlassCard className="chart-card"><div className="section-heading"><div><span className="eyebrow">DATASET ROBUSTNESS</span><h2>Performance by source</h2></div></div><div className="source-cards">{sources.map(source=><div className="source-card" key={source.name}><div><span>{source.name}</span><small>{compact(source.rows)} test rows</small></div><strong>{source.pr_auc.toFixed(4)}</strong><em>PR-AUC</em><div className="thin-track"><i style={{width:`${source.pr_auc*100}%`}}/></div><small>ROC-AUC {source.roc_auc.toFixed(4)}</small></div>)}</div><p className="microcopy">The source gap is useful evidence: very high aggregate performance should not be presented without showing that PaySim is materially harder than BankSim.</p></GlassCard>
      </div>

      <GlassCard className="scenario-matrix"><div className="section-heading"><div><span className="eyebrow">EDGE-CASE OPERATIONS</span><h2>Scenario behaviour on held-out test rows</h2></div></div><div className="scenario-table"><div className="scenario-table-head"><span>Scenario</span><span>Rows</span><span>Fraud prevalence</span><span>Mean risk</span><span>Intervention</span></div>{scenarios.map(s=><div className="scenario-table-row" key={s.name}><strong>{s.name}</strong><span>{s.rows}</span><span>{pct(s.fraud_rate)}</span><span>{pct(s.mean_risk)}</span><span>{pct(s.intervention)}</span></div>)}</div></GlassCard>
    </div>
  )
}
