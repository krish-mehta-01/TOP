import { motion, AnimatePresence } from 'framer-motion';

export default function OverTimeline({ balls }) {
  if (!balls || balls.length === 0) return null;

  // Take the last 18 balls for the timeline
  const recentBalls = balls.slice(-18);

  const getBallBadge = (b, idx) => {
    const isWicket = b.is_wicket === 1;
    const isBoundary = b.batter_runs === 4 || b.batter_runs === 6;
    const isDot = b.batter_runs === 0 && b.wides === 0 && b.noballs === 0 && b.byes === 0 && b.legbyes === 0 && !isWicket;
    const totalRuns = b.batter_runs + b.wides + b.noballs + b.byes + b.legbyes;
    
    let bg = 'rgba(255, 255, 255, 0.05)';
    let color = 'var(--text-muted)';
    let border = '1px solid rgba(255,255,255,0.1)';
    
    if (isWicket) {
      bg = 'rgba(239, 68, 68, 0.15)';
      color = '#ef4444';
      border = '1px solid rgba(239, 68, 68, 0.3)';
    } else if (isBoundary) {
      bg = 'rgba(16, 185, 129, 0.15)';
      color = '#10b981';
      border = '1px solid rgba(16, 185, 129, 0.3)';
    } else if (!isDot) {
      color = 'var(--text-light)';
    }

    let label = totalRuns.toString();
    if (isWicket) label = 'W';
    else if (b.wides > 0) label = `${b.wides}Wd`;
    else if (b.noballs > 0) label = `${b.noballs}Nb`;
    else if (isDot) label = '•';

    return (
      <motion.div
        key={`${b.over}-${b.ball_in_over}-${idx}`}
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0, opacity: 0 }}
        style={{
          width: '2.5rem',
          height: '2.5rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '50%',
          background: bg,
          border: border,
          color: color,
          fontWeight: 'bold',
          fontSize: '0.875rem',
          flexShrink: 0
        }}
        title={`${b.over}.${b.ball_in_over} - ${b.bowler} to ${b.batter}`}
      >
        {label}
      </motion.div>
    );
  };

  return (
    <div className="panel" style={{ marginBottom: '1.5rem', padding: '1rem' }}>
      <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '0.75rem', letterSpacing: '0.05em' }}>
        Recent Sequence
      </div>
      <div style={{ display: 'flex', gap: '0.5rem', overflowX: 'auto', paddingBottom: '0.5rem', scrollbarWidth: 'thin' }}>
        <AnimatePresence>
          {recentBalls.map((b, idx) => getBallBadge(b, idx))}
        </AnimatePresence>
      </div>
    </div>
  );
}
