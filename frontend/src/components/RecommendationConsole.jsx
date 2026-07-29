import { Cpu, ChevronRight, AlertTriangle } from 'lucide-react';
import AIReasoningChart from './AIReasoningChart';

const ACTION_LABELS = {
  change_bowler: "Change the bowler", maintain_current_bowler: "Continue with current bowler",
  bring_spinner: "Bring on a spinner", bowl_yorkers_death: "Bowl yorkers / death lines",
  use_variations: "Use variations", defensive_field: "Set a defensive field",
  attacking_field: "Set an attacking field", target_matchup: "Target the matchup",
  accelerate: "Accelerate the scoring rate", rotate_strike: "Rotate strike",
  defend_wicket: "Play cautiously, protect the wicket", target_boundaries: "Target boundaries",
  build_partnership: "Build the partnership",
};
function actionLabel(a){ return ACTION_LABELS[a] || a; }

function confidenceFromText(text){
  const m = text?.match(/Confidence:\s*(\d+)%/);
  return m ? parseInt(m[1]) : 0;
}

function scoreToPercent(score){
  const clamped = Math.max(-2.5, Math.min(2.5, score));
  return (Math.abs(clamped) / 2.5) * 48;
}

export default function RecommendationConsole({ error, recommendation, role }) {
  if (error) {
    return (
      <div className="panel">
        <div className="panel-head"><Cpu size={16}/> <h2>Decision Engine Output</h2></div>
        <div>
          <div className="err-box"><AlertTriangle size={18}/> {error}</div>
        </div>
      </div>
    );
  }

  if (!recommendation) {
    return (
      <div className="panel">
        <div className="panel-head"><Cpu size={16}/> <h2>Decision Engine Output</h2></div>
        <div>
          <div className="rec-empty">
            <Cpu size={48} className="icon" strokeWidth={1} />
            <p>Log at least one ball, then request a recommendation. The engine runs every active model for this role, normalizes and aggregates their signals, checks hard rules, and returns the highest-scoring unblocked action.</p>
          </div>
        </div>
      </div>
    );
  }

  const { text, phase, chosen, audit, contributions } = recommendation;
  const conf = confidenceFromText(text);

  return (
    <div className="panel">
      <div className="panel-head"><Cpu size={16}/> <h2>Decision Engine Output</h2></div>
      <div>
        <div className={`rec-headline ${role}`}>
          <div className="eyebrow">{role} · {phase} overs</div>
          <div className="action">{chosen ? actionLabel(chosen[0]) : "No confident recommendation"}</div>
          {chosen && (
            <div className="rec-confidence">
              <div className="conf-bar"><div className="conf-fill" style={{width: `${conf}%`}}></div></div>
              <div className="conf-num">{conf}%</div>
            </div>
          )}
          <div className="rec-note">{text}</div>
        </div>
        <div className="signal-readout">
          {audit.map((row, i) => {
            const pct = scoreToPercent(row.score);
            const isPos = row.score >= 0;
            return (
              <div key={i} className={`readout-row ${row.chosen ? 'chosen' : ''}`}>
                <div className="label">{actionLabel(row.action)}</div>
                <div className="readout-track">
                  <div className="readout-axis"></div>
                  <div className={`readout-fill ${isPos ? 'pos' : 'neg'}`} style={{width: `${pct}%`, [isPos ? 'left' : 'right']: '50%'}}></div>
                </div>
                <div className="score" title={`Raw score: ${row.score}`}>
                  {row.blocked_by ? <span className="blocked-tag" title={`Blocked by: ${row.blocked_by}`}>blocked</span> : row.score.toFixed(2)}
                </div>
              </div>
            );
          })}
        </div>

        {chosen && contributions[chosen[0]] && (
          <div className="contrib-list">
            <AIReasoningChart recommendation={recommendation} />
            
            <div className="subhead" style={{margin:'24px 0 12px', padding:'0 12px'}}><ChevronRight size={14}/> Top Drivers</div>
            {contributions[chosen[0]].slice(0, 6).map((c, i) => {
              const cls = c.contribution >= 0 ? 'pos' : 'neg';
              return (
                <div key={i} className="item">
                  <span className="model">{c.model_id} — {c.label}</span>
                  <span className={`val ${cls}`}>{c.contribution >= 0 ? '+' : ''}{c.contribution.toFixed(3)}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
