import React from 'react';

const ModelDiagnostics = ({ recommendation }) => {
  if (!recommendation || !recommendation.raw_model_outputs) return null;

  const raw = recommendation.raw_model_outputs;
  const contribs = recommendation.contributions;

  return (
    <div style={{ marginTop: '24px', backgroundColor: 'var(--surface)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border)' }}>
      <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', color: 'var(--primary)' }}>RESEARCH DIAGNOSTICS (PhD DATA)</h3>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        {/* Raw Model Outputs */}
        <div>
          <h4 style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px' }}>RAW MODEL PREDICTIONS</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {Object.entries(raw).map(([modelId, value]) => (
              <div key={modelId} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px', backgroundColor: 'var(--bg)', borderRadius: '6px' }}>
                <span style={{ fontWeight: 'bold', fontSize: '13px' }}>{modelId}</span>
                <span style={{ fontFamily: 'monospace', color: 'var(--text-muted)' }}>
                  {typeof value === 'number' ? value.toFixed(3) : value}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Action Contributions */}
        <div>
          <h4 style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px' }}>ACTION CONTRIBUTIONS</h4>
          {Object.entries(contribs).slice(0, 3).map(([action, factors]) => (
            <div key={action} style={{ marginBottom: '16px' }}>
              <div style={{ fontSize: '13px', fontWeight: 'bold', marginBottom: '6px', color: 'white' }}>{action.replace(/_/g, ' ').toUpperCase()}</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {factors.map((f, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', padding: '4px 8px', backgroundColor: 'var(--bg)', borderRadius: '4px', borderLeft: `3px solid ${f.contribution > 0 ? 'var(--success)' : 'var(--danger)'}` }}>
                    <span style={{ color: 'var(--text-muted)' }}>{f.model_id}: {f.label}</span>
                    <span style={{ fontWeight: 'bold', color: f.contribution > 0 ? 'var(--success)' : 'var(--danger)' }}>
                      {f.contribution > 0 ? '+' : ''}{f.contribution.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelDiagnostics;
