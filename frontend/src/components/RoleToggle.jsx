import { Crosshair, Shield } from 'lucide-react';

export default function RoleToggle({ role, setRole }) {
  return (
    <div className="role-toggle">
      <button 
        data-role="bowling" 
        className={role === 'bowling' ? 'active' : ''}
        onClick={() => setRole('bowling')}
      >
        <Crosshair size={16} /> Bowling
      </button>
      <button 
        data-role="batting" 
        className={role === 'batting' ? 'active' : ''}
        onClick={() => setRole('batting')}
      >
        <Shield size={16} /> Batting
      </button>
    </div>
  );
}
