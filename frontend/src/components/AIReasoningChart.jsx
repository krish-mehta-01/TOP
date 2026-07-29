import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function AIReasoningChart({ recommendation }) {
  if (!recommendation || !recommendation.contributions) return null;

  // Extract contributions for the chosen action
  const chosenAction = recommendation.chosen[0];
  const contributions = recommendation.contributions[chosenAction] || [];
  
  if (contributions.length === 0) return null;

  // Format data for chart
  const data = contributions.map(c => ({
    name: c.model_id,
    label: c.label,
    value: c.contribution
  })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 5); // top 5 signals

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      return (
        <div style={{
          background: 'rgba(24, 24, 27, 0.9)',
          border: '1px solid rgba(255,255,255,0.1)',
          padding: '0.75rem',
          borderRadius: '8px',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
          maxWidth: '200px'
        }}>
          <p style={{ margin: 0, fontSize: '0.75rem', color: 'var(--text-muted)' }}>{d.name}</p>
          <p style={{ margin: '0.25rem 0', fontWeight: 'bold', color: 'var(--text-light)', fontSize: '0.875rem' }}>{d.label}</p>
          <p style={{ margin: 0, fontWeight: 'bold', color: d.value > 0 ? '#10b981' : '#ef4444' }}>
            {d.value > 0 ? '+' : ''}{d.value.toFixed(2)}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div style={{ marginTop: '1.5rem', paddingTop: '1.5rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '1rem', letterSpacing: '0.05em' }}>
        AI Signal Breakdown
      </div>
      <div style={{ height: 180, width: '100%' }}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="name" width={40} tick={{ fill: 'var(--text-muted)', fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#10b981' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
