const BallIcon = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="#b91c1c" stroke="#7f1d1d" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{transform: 'rotate(25deg)'}}>
    <circle cx="12" cy="12" r="10"></circle>
    <path d="M11 2v20" stroke="#fecaca" strokeWidth="1.5" strokeDasharray="2 2"></path>
    <path d="M13 2v20" stroke="#fecaca" strokeWidth="1.5" strokeDasharray="2 2"></path>
    <path d="M9 3a10 10 0 0 0 0 18" stroke="#991b1b" strokeWidth="1"></path>
    <path d="M15 3a10 10 0 0 1 0 18" stroke="#991b1b" strokeWidth="1"></path>
  </svg>
);

const BatIcon = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="#d97706" stroke="#78350f" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{transform: 'rotate(25deg)'}}>
    <path d="M11 2h2v5h-2z" fill="#334155" stroke="#1e293b"></path>
    <path d="M11 3h2" stroke="#475569"></path>
    <path d="M11 4.5h2" stroke="#475569"></path>
    <path d="M11 6h2" stroke="#475569"></path>
    <path d="M8.5 7h7v13c0 1.5-1.5 2-3.5 2s-3.5-0.5-3.5-2z" fill="#f59e0b"></path>
    <path d="M10 7v13" stroke="#b45309" strokeWidth="1"></path>
    <path d="M12 7v13" stroke="#b45309" strokeWidth="1"></path>
    <path d="M14 7v13" stroke="#b45309" strokeWidth="1"></path>
  </svg>
);

export default function RoleToggle({ role, setRole }) {
  return (
    <div className="role-toggle">
      <button 
        data-role="bowling" 
        className={role === 'bowling' ? 'active' : ''}
        onClick={() => setRole('bowling')}
      >
        <BallIcon size={16} /> Bowling
      </button>
      <button 
        data-role="batting" 
        className={role === 'batting' ? 'active' : ''}
        onClick={() => setRole('batting')}
      >
        <BatIcon size={16} /> Batting
      </button>
    </div>
  );
}
