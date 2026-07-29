import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceDot } from 'recharts';

export default function MomentumChart({ balls, role }) {
  if (!balls || balls.length === 0) return null;

  const data = [];
  let totalRuns = 0;
  
  balls.forEach((b, i) => {
    const extras = b.wides + b.noballs + b.byes + b.legbyes;
    totalRuns += b.batter_runs + extras;
    
    data.push({
      ball: `${b.over}.${b.ball_in_over}`,
      runs: totalRuns,
      isWicket: b.is_wicket === 1,
      batter: b.batter,
      bowler: b.bowler
    });
  });

  const color = role === 'bowling' ? '#10b981' : '#f59e0b';

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      return (
        <div style={{
          background: 'rgba(24, 24, 27, 0.8)',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(255,255,255,0.1)',
          padding: '8px 12px',
          borderRadius: '8px',
          fontSize: '12px',
          color: '#fafafa'
        }}>
          <div style={{fontFamily: 'monospace', color: '#a1a1aa', marginBottom: '4px'}}>Over {d.ball}</div>
          <div>Score: <strong style={{color: color}}>{d.runs}</strong></div>
          {d.isWicket && <div style={{color: '#ef4444', fontWeight: 600, marginTop: '4px'}}>Wicket!</div>}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="chart-container">
      <div style={{fontSize: '12px', color: '#a1a1aa', marginBottom: '12px', fontWeight: 600}}>Match Momentum</div>
      <div style={{ width: '100%', height: '150px' }}>
        <ResponsiveContainer>
          <AreaChart data={data} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="colorRuns" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={color} stopOpacity={0.4}/>
                <stop offset="95%" stopColor={color} stopOpacity={0}/>
              </linearGradient>
            </defs>
            <XAxis dataKey="ball" stroke="#3f3f46" fontSize={10} tickMargin={8} />
            <YAxis stroke="#3f3f46" fontSize={10} />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1, strokeDasharray: '3 3' }} />
            <Area type="monotone" dataKey="runs" stroke={color} strokeWidth={2} fillOpacity={1} fill="url(#colorRuns)" />
            
            {data.map((d, i) => (
              d.isWicket ? 
                <ReferenceDot key={`dot-${i}`} x={d.ball} y={d.runs} r={4} fill="#ef4444" stroke="#000" strokeWidth={2} /> 
              : null
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
