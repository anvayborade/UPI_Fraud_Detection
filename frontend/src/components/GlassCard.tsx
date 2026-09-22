import type { ReactNode } from 'react'

export function GlassCard({ children, className = '', glow = '' }: { children: ReactNode; className?: string; glow?: string }) {
  return <section className={`glass-card ${glow ? `glow-${glow}` : ''} ${className}`}>{children}</section>
}
