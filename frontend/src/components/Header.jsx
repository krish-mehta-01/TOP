import { Activity, Server } from 'lucide-react';

export default function Header({ status }) {
  const isLive = status === 'live';
  return (
    <header>
      <div className="brand">
        <div className="mark" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={18} /> TOP
        </div>
        <div>
          <h1>Tactical Console</h1>
          <p>Live timeout decision support — 28 retrained models, one recommendation</p>
        </div>
      </div>
      <div className="status">
        <Server size={14} />
        <span className={`dot ${isLive ? 'live' : 'down'}`}></span>
        <span>{isLive ? 'backend connected' : 'backend unreachable — start: uvicorn app:app --port 8000'}</span>
      </div>
    </header>
  );
}
