import { Activity, Server } from 'lucide-react';

export default function Header({ status }) {
  const isLive = status === 'live';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', paddingBottom: '16px', borderBottom: '1px solid var(--border)', marginBottom: '16px' }}>
      <div className="brand" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div className="mark" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', padding: '6px 10px' }}>
          <Activity size={16} /> TOP
        </div>
        <div>
          <h1 style={{ fontSize: '18px', margin: 0 }}>Tactical Console</h1>
          <p style={{ fontSize: '11px', margin: 0, color: 'var(--text-dim)' }}>Live decision support</p>
        </div>
      </div>
      <div className="status" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px' }}>
        <Server size={12} />
        <span className={`dot ${isLive ? 'live' : 'down'}`}></span>
        <span>{isLive ? 'Backend connected' : 'Backend offline'}</span>
      </div>
    </div>
  );
}
