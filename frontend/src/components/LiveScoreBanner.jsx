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

  // Get active players from the last ball logged
  const lastBall = balls.length > 0 ? balls[balls.length - 1] : null;
  const striker = lastBall ? lastBall.batter : "Waiting...";
  const nonStriker = lastBall ? lastBall.non_striker : "Waiting...";
  const bowler = matchState.current_bowler || "Waiting...";

  return (
    <div className="dashboard-header">
      <div style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
        <div>
          <div style={{ fontSize: '12px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            {matchState.venue} • INN {matchState.innings}
          </div>
          <div style={{ fontSize: '24px', fontWeight: '800', marginTop: '4px', letterSpacing: '-0.02em' }}>
            {matchState.batting_team.toUpperCase()}
          </div>
        </div>

        <div className="clinical-score">
          <span className="runs">{runs}</span>
          <span className="slash">/</span>
          <span className="wickets">{wickets}</span>
        </div>

        <div className="clinical-overs">
          <div style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Overs</div>
          <div className="val">{oversStr}</div>
        </div>

        <div className="clinical-overs">
          <div style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>CRR</div>
          <div className="val">{crr}</div>
        </div>
      </div>
      
      <div className="active-players">
        <div className="player-box">
          <div className="lbl">BATTERS</div>
          <div className="name active">{striker} *</div>
          <div className="name">{nonStriker}</div>
        </div>
        <div className="player-box">
          <div className="lbl">BOWLER</div>
          <div className="name active" style={{color: 'var(--amber-bright)'}}>{bowler}</div>
        </div>
      </div>
    </div>
  );
}
