import {
  Activity, Boxes, BrainCircuit, ChartNoAxesCombined, Clock3, GitBranch, LayoutDashboard,
  MessageSquareWarning, Network, ShieldCheck, Sparkles, TestTubeDiagonal,
} from 'lucide-react'
import type { ViewId } from '../types'

const items: Array<{ id: ViewId; label: string; icon: typeof Activity; section?: string }> = [
  { id: 'command', label: 'Command Center', icon: LayoutDashboard, section: 'Operate' },
  { id: 'edge-cases', label: 'Edge-Case Gallery', icon: TestTubeDiagonal },
  { id: 'explain', label: 'Model Debate', icon: BrainCircuit, section: 'Explain' },
  { id: 'graph', label: 'Recipient Graph', icon: Network },
  { id: 'timeline', label: 'Transaction Timeline', icon: Clock3 },
  { id: 'performance', label: 'Performance', icon: ChartNoAxesCombined, section: 'Validate' },
  { id: 'trust', label: 'Leakage Guards', icon: ShieldCheck },
  { id: 'architecture', label: 'Architecture', icon: GitBranch, section: 'System' },
  { id: 'complaints', label: 'Complaint Lab', icon: MessageSquareWarning },
]

export function Sidebar({ active, onChange, collapsed, onToggle }: { active: ViewId; onChange: (id: ViewId) => void; collapsed: boolean; onToggle: () => void }) {
  let lastSection = ''
  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <button className="brand" onClick={onToggle} aria-label="Toggle navigation">
        <div className="brand-mark"><Sparkles size={18}/></div>
        {!collapsed && <div><strong>UPI / FI</strong><span>Intent Firewall</span></div>}
      </button>
      <nav>
        {items.map((item) => {
          const showSection = item.section && item.section !== lastSection
          if (item.section) lastSection = item.section
          const Icon = item.icon
          return (
            <div key={item.id}>
              {showSection && !collapsed && <div className="nav-section">{item.section}</div>}
              <button className={`nav-item ${active === item.id ? 'active' : ''}`} onClick={() => onChange(item.id)} title={collapsed ? item.label : undefined}>
                <Icon size={18}/>{!collapsed && <span>{item.label}</span>}
              </button>
            </div>
          )
        })}
      </nav>
      <div className="sidebar-foot">
        <div className="mini-stack"><Boxes size={15}/>{!collapsed && <span>Multi-expert</span>}</div>
        <div className="mini-stack"><Activity size={15}/>{!collapsed && <span>Time-aware</span>}</div>
      </div>
    </aside>
  )
}
