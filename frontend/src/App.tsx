import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity, ArrowUpRight, Braces, ChevronRight, CircleHelp, Code2, Menu, RefreshCw,
  ShieldCheck, Sparkles, X,
} from 'lucide-react'
import type { AppMode, DemoScenario, EvaluationData, FraudPrediction, HealthState, MetricsData, UPIEvent, ViewId } from './types'
import { getHealth, precheckEvent, scoreEvent } from './lib/api'
import { money, pct, titleize } from './lib/format'
import { Sidebar } from './components/Sidebar'
import { SystemPulse } from './components/SystemPulse'
import { DecisionHero } from './components/DecisionHero'
import { ModelDebate } from './components/ModelDebate'
import { EdgeCaseGallery } from './components/EdgeCaseGallery'
import { WhatIfPanel } from './components/WhatIfPanel'
import { GraphExplorer } from './components/GraphExplorer'
import { TransactionTimeline } from './components/TransactionTimeline'
import { PerformanceDashboard } from './components/PerformanceDashboard'
import { LeakageGuards } from './components/LeakageGuards'
import { ArchitectureFlow } from './components/ArchitectureFlow'
import { ComplaintLab } from './components/ComplaintLab'
import { GlassCard } from './components/GlassCard'
import { ScoreBar } from './components/ScoreBar'

const viewTitles: Record<ViewId, [string, string]> = {
  command: ['Command Center', 'One decision, every layer visible.'],
  'edge-cases': ['Edge-Case Gallery', 'Stress-test fraud and legitimate novelty without hiding ambiguity.'],
  explain: ['Model Debate', 'See why specialists disagree before fusion and policy decide.'],
  graph: ['Recipient Graph', 'Inspect the transaction-time relational neighbourhood.'],
  timeline: ['Transaction Timeline', 'See the behavioural sequence leading into the selected payment.'],
  performance: ['Model Performance', 'Aggregate quality, source gaps, fraud-type heads and scenario operations.'],
  trust: ['Leakage Guards', 'The controls that make a high synthetic score more defensible.'],
  architecture: ['System Architecture', 'Fast path, near-real-time intelligence and explainable fusion.'],
  complaints: ['Complaint Lab', 'Convert narrative fraud reports into structured recipient intelligence.'],
}

const protective = new Set(['legitimate_novelty', 'graph_merchant_legitimacy', 'journey_plausibility'])

type RecentDecision = Record<string, string | number | null>

async function fetchJson<T>(path: string, fallback: T): Promise<T> {
  try {
    const response = await fetch(path, { cache: 'no-store' })
    if (!response.ok) return fallback
    return await response.json()
  } catch {
    return fallback
  }
}

function Drawer({ title, children, onClose }: { title: string; children: ReactNode; onClose: () => void }) {
  return <motion.div className="drawer-backdrop" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} onClick={onClose}>
    <motion.aside className="drawer" initial={{x:420}} animate={{x:0}} exit={{x:420}} transition={{type:'spring',damping:28,stiffness:260}} onClick={e=>e.stopPropagation()}>
      <div className="drawer-head"><div><span className="eyebrow">LIVE API</span><h3>{title}</h3></div><button className="icon-button" onClick={onClose}><X size={18}/></button></div>{children}
    </motion.aside>
  </motion.div>
}

function RawTechnical({ decision, scenario }: { decision?: FraudPrediction; scenario?: DemoScenario }) {
  if (!decision || !scenario) return null
  return <div className="technical-grid">
    <GlassCard><span className="eyebrow">REQUEST SNAPSHOT</span><pre className="code-block">{JSON.stringify(scenario.payload,null,2)}</pre></GlassCard>
    <GlassCard><span className="eyebrow">RAW DECISION</span><pre className="code-block">{JSON.stringify(decision,null,2)}</pre></GlassCard>
  </div>
}

function RecentFeed({ rows }: { rows: RecentDecision[] }) {
  return <GlassCard className="recent-feed"><div className="section-heading"><div><span className="eyebrow">HELD-OUT DECISION FEED</span><h2>Recent offline-scored transactions</h2></div><span className="mode-chip">{rows.length} exported</span></div>
    <div className="feed-table"><div className="feed-head"><span>Transaction</span><span>Scenario</span><span>Amount</span><span>Risk</span><span>Action</span></div>{rows.slice().reverse().slice(0,12).map((row,i)=><div className="feed-row" key={`${row.transaction_id}-${i}`}><code>{String(row.transaction_id).slice(0,24)}</code><span>{titleize(String(row.scenario??'unknown'))}</span><strong>{money(Number(row.amount??0))}</strong><span>{pct(Number(row.final_fraud_probability??0))}</span><em className={`tiny-action ${String(row.recommended_action).toLowerCase()}`}>{String(row.recommended_action)}</em></div>)}</div>
  </GlassCard>
}

export default function App() {
  const [view, setView] = useState<ViewId>('command')
  const [mode, setMode] = useState<AppMode>('executive')
  const [collapsed, setCollapsed] = useState(false)
  const [scenarios, setScenarios] = useState<DemoScenario[]>([])
  const [selectedKey, setSelectedKey] = useState('hospital')
  const [decision, setDecision] = useState<FraudPrediction>()
  const [whatIfResult, setWhatIfResult] = useState<FraudPrediction>()
  const [precheck, setPrecheck] = useState<Record<string,unknown>>()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [health, setHealth] = useState<HealthState>({})
  const [online, setOnline] = useState(false)
  const [metrics, setMetrics] = useState<MetricsData>()
  const [evaluation, setEvaluation] = useState<EvaluationData>()
  const [recent, setRecent] = useState<RecentDecision[]>([])

  const selected = useMemo(() => scenarios.find(s=>s.key===selectedKey) ?? scenarios[0], [scenarios, selectedKey])
  const displayDecision = decision ?? (selected?.offline as FraudPrediction | undefined)

  useEffect(() => {
    Promise.all([
      fetchJson<DemoScenario[]>('/demo_scenarios.json', []),
      fetchJson<MetricsData>('/metrics.json', {}),
      fetchJson<EvaluationData>('/evaluation_report.json', {}),
      fetchJson<RecentDecision[]>('/recent_decisions.json', []),
    ]).then(([demoData, metricData, evaluationData, recentData]) => {
      setScenarios(demoData); setMetrics(metricData); setEvaluation(evaluationData); setRecent(recentData)
      if (demoData.length && !demoData.some(s=>s.key===selectedKey)) setSelectedKey(demoData[0].key)
    })
  }, [])

  useEffect(() => {
    const check = async () => { try { const h = await getHealth(); setHealth(h); setOnline(h.status === 'ok' || h.status === 'degraded') } catch { setOnline(false); setHealth({}) } }
    check(); const timer = window.setInterval(check, 10000); return () => window.clearInterval(timer)
  }, [])

  useEffect(() => { setDecision(undefined); setWhatIfResult(undefined); setError('') }, [selectedKey])

  async function runScore(payload?: UPIEvent) {
    if (!selected) return
    setBusy(true); setError('')
    try { const result = await scoreEvent(payload ?? selected.payload); if (payload) setWhatIfResult(result); else setDecision(result) }
    catch(e){ setError(e instanceof Error?e.message:String(e)) } finally { setBusy(false) }
  }

  async function runPrecheck() {
    if (!selected) return
    setBusy(true); setError('')
    try { setPrecheck(await precheckEvent(selected.payload)); setDrawerOpen(true) }
    catch(e){ setError(e instanceof Error?e.message:String(e)) } finally { setBusy(false) }
  }

  function chooseScenario(scenario: DemoScenario) { setSelectedKey(scenario.key); if(view==='edge-cases') window.scrollTo({top:0,behavior:'smooth'}) }

  const title = viewTitles[view]

  return <div className="app-shell">
    <div className="ambient ambient-one"/><div className="ambient ambient-two"/><div className="noise-layer"/>
    <Sidebar active={view} onChange={setView} collapsed={collapsed} onToggle={()=>setCollapsed(v=>!v)}/>
    <main className={`main-shell ${collapsed?'wide':''}`}>
      <header className="topbar">
        <div className="topbar-title"><button className="mobile-menu" onClick={()=>setCollapsed(v=>!v)}><Menu size={18}/></button><div><span className="eyebrow">UPI FRAUD INTENT FIREWALL</span><h2>{title[0]}</h2><p>{title[1]}</p></div></div>
        <div className="topbar-actions"><SystemPulse health={health} online={online}/><div className="mode-switch"><button className={mode==='executive'?'active':''} onClick={()=>setMode('executive')}><Sparkles size={14}/>Executive</button><button className={mode==='technical'?'active':''} onClick={()=>setMode('technical')}><Code2 size={14}/>Technical</button></div></div>
      </header>

      {error && <div className="error-banner"><CircleHelp size={17}/><span>{error}</span><button onClick={()=>setError('')}><X size={15}/></button></div>}
      {!scenarios.length && <div className="setup-banner"><Braces size={20}/><div><strong>Live demo data has not been exported yet.</strong><span>From the frontend folder run <code>npm run sync-data</code>, then refresh. Performance pages already use the bundled latest metrics.</span></div></div>}

      <AnimatePresence mode="wait">
        <motion.div key={view} className="page" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} exit={{opacity:0,y:-6}} transition={{duration:.24}}>
          {view==='command' && <>
            <DecisionHero scenario={selected} decision={displayDecision} mode={mode} busy={busy} onScore={()=>runScore()} onPrecheck={runPrecheck}/>
            <div className="command-grid">
              <GlassCard className="scenario-picker"><div className="section-heading compact"><div><span className="eyebrow">DEMO QUICK PICK</span><h2>Edge cases</h2></div><button className="text-button" onClick={()=>setView('edge-cases')}>View all <ChevronRight size={14}/></button></div><div className="quick-picks">{scenarios.slice(0,6).map(s=><button key={s.key} className={`${s.kind} ${s.key===selectedKey?'active':''}`} onClick={()=>chooseScenario(s)}><span>{s.short_label}</span><small>{s.category}</small></button>)}</div></GlassCard>
              <GlassCard className="why-card"><span className="eyebrow">WHY THIS PROJECT IS DIFFERENT</span><h2>Unusual does not automatically mean fraudulent.</h2><p>The architecture asks two questions in parallel: <b>“How suspicious is this?”</b> and <b>“Is there a convincing legitimate explanation?”</b></p><button className="text-button" onClick={()=>setView('architecture')}>Explore the architecture <ArrowUpRight size={14}/></button></GlassCard>
            </div>
            <ModelDebate decision={displayDecision} mode={mode}/>
            <WhatIfPanel scenario={selected} onRun={runScore} result={whatIfResult} busy={busy}/>
            <RecentFeed rows={recent}/>
            {mode==='technical' && <RawTechnical decision={displayDecision} scenario={selected}/>}          
          </>}

          {view==='edge-cases' && <>
            {selected && <DecisionHero scenario={selected} decision={displayDecision} mode={mode} busy={busy} onScore={()=>runScore()} onPrecheck={runPrecheck}/>} 
            <div className="section-intro"><span className="eyebrow">12 DEMONSTRATIONS · NO RETRAINING REQUIRED</span><h2>Fraud challenges and hard legitimate negatives</h2><p>The five original manifest scenarios are preserved; additional held-out scenarios are selected by the frontend data exporter from the existing scored test set.</p></div>
            <EdgeCaseGallery scenarios={scenarios} selectedKey={selectedKey} onSelect={chooseScenario}/>
            <WhatIfPanel scenario={selected} onRun={runScore} result={whatIfResult} busy={busy}/>
          </>}

          {view==='explain' && <><ModelDebate decision={displayDecision} mode={mode}/>{displayDecision && <GlassCard className="expert-detail"><div className="section-heading"><div><span className="eyebrow">ALL EXPERT SCORES</span><h2>Risk evidence versus protective context</h2></div></div><div className="expert-score-grid">{Object.entries(displayDecision.model_scores).map(([key,value])=><ScoreBar key={key} name={key} value={value} protective={protective.has(key)}/>)}</div></GlassCard>}{mode==='technical'&&<RawTechnical decision={displayDecision} scenario={selected}/>}</>}
          {view==='graph' && <><div className="context-selector">{scenarios.map(s=><button key={s.key} className={s.key===selectedKey?'active':''} onClick={()=>chooseScenario(s)}>{s.short_label}</button>)}</div><GraphExplorer scenario={selected}/></>}
          {view==='timeline' && <><div className="context-selector">{scenarios.map(s=><button key={s.key} className={s.key===selectedKey?'active':''} onClick={()=>chooseScenario(s)}>{s.short_label}</button>)}</div><TransactionTimeline scenario={selected}/></>}
          {view==='performance' && <PerformanceDashboard metrics={metrics} evaluation={evaluation}/>}          
          {view==='trust' && <><div className="section-intro"><span className="eyebrow">TRUST LAYER</span><h2>High performance is only useful when the evaluation is honest.</h2><p>These controls were added specifically to reduce synthetic shortcuts, recipient contamination and future-information leakage.</p></div><LeakageGuards/><GlassCard className="limitation-card"><ShieldCheck/><div><span className="eyebrow">IMPORTANT LIMITATION</span><h3>Excellent held-out synthetic performance is not a claim of real-bank UPI accuracy.</h3><p>The interface intentionally surfaces source gaps and weak specialists so the prototype stays explainable rather than becoming a 99%-score showcase.</p></div></GlassCard></>}
          {view==='architecture' && <ArchitectureFlow/>}
          {view==='complaints' && <ComplaintLab defaultPayee={selected?.payee_id}/>}          
        </motion.div>
      </AnimatePresence>
    </main>

    <AnimatePresence>{drawerOpen && <Drawer title="Recipient precheck" onClose={()=>setDrawerOpen(false)}><pre className="code-block drawer-code">{JSON.stringify(precheck,null,2)}</pre></Drawer>}</AnimatePresence>
  </div>
}
