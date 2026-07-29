import { Trophy } from 'lucide-react';
import { motion } from 'framer-motion';

export default function LiveScoreBanner({ balls, matchState }) {
  let runs = 0;
  let wickets = 0;
  let legalBalls = 0;

  balls.forEach(b => {
    runs += b.batter_runs + b.wides + b.noballs + b.byes + b.legbyes;
    if (b.is_wicket === 1) wickets++;
    if (b.wides === 0 && b.noballs === 0) legalBalls++;
  });

  const overs = Math.floor(legalBalls / 6);
  const extraBalls = legalBalls % 6;
  const oversStr = `${overs}.${extraBalls}`;
  const crr = legalBalls > 0 ? ((runs / legalBalls) * 6).toFixed(2) : '0.00';

  return (
    <motion.div 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="panel"
      style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', background: 'linear-gradient(90deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.7) 100%)' }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <Trophy size={28} className="icon-accent" />
        <div>
          <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
            {matchState.venue} • Innings {matchState.innings}
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ color: 'var(--text-light)' }}>{matchState.batting_team}</span>
            <span style={{ color: 'var(--accent-primary)' }}>{runs}/{wickets}</span>
            <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>({oversStr})</span>
          </div>
        </div>
      </div>
      
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', textAlign: 'right' }}>
        <div>
          <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>CRR</div>
          <div style={{ fontSize: '1.25rem', fontWeight: '600', color: 'var(--text-light)' }}>{crr}</div>
        </div>
        <div style={{ paddingLeft: '1rem', borderLeft: '1px solid rgba(255,255,255,0.1)' }}>
          <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>vs</div>
          <div style={{ fontSize: '1.25rem', fontWeight: '600', color: 'var(--text-light)' }}>{matchState.bowling_team}</div>
        </div>
      </div>
    </motion.div>
  );
}
