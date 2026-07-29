import { useState } from 'react';
import MomentumChart from './MomentumChart';
import OverTimeline from './OverTimeline';
import { Plus, Play, Trash2, MapPin, Users, Target } from 'lucide-react';
import rosterData from '../data/roster.json';

export default function ScorecardPanel({ 
  role, 
  balls, 
  onAddBall, 
  onDeleteBall, 
  onClear, 
  onLoadSample, 
  onRecommend,
  isRecommending,
  matchState,
  setMatchState
}) {
  const [ballInput, setBallInput] = useState({
    b_over: 0,
    b_ball: 1,
    b_batter: 'V Kohli',
    b_nonstriker: 'F du Plessis',
    b_bowler: 'JJ Bumrah',
    b_runs: 0,
    b_wide: 0,
    b_nb: 0,
    b_bye: 0,
    b_legbye: 0,
    b_wicket: 0
  });

  const handleInputChange = (e) => {
    const { id, value, type } = e.target;
    setBallInput(prev => ({
      ...prev,
      [id]: type === 'number' ? parseInt(value) || 0 : value
    }));
  };

  const handleMatchStateChange = (e) => {
    const { id, value } = e.target;
    setMatchState(prev => ({ ...prev, [id]: value }));
  };

  const handleAddBall = () => {
    onAddBall(ballInput);
    
    const isLegal = ballInput.b_wide === 0 && ballInput.b_nb === 0;
    let nextOver = ballInput.b_over;
    let nextBall = ballInput.b_ball + (isLegal ? 1 : 0);
    if (nextBall > 6) {
      nextOver += 1;
      nextBall = 1;
    }
    
    setBallInput(prev => ({
      ...prev,
      b_over: nextOver,
      b_ball: nextBall,
      b_wicket: 0,
      b_runs: 0,
      b_wide: 0,
      b_nb: 0,
      b_bye: 0,
      b_legbye: 0
    }));
    
    setMatchState(prev => ({ ...prev, current_bowler: ballInput.b_bowler }));
  };

  const totalRuns = balls.reduce((s,b) => s + b.batter_runs + b.wides + b.noballs + b.byes + b.legbyes, 0);
  const wkts = balls.reduce((s,b) => s + b.is_wicket, 0);
  const legalBalls = balls.filter(b => b.wides === 0 && b.noballs === 0).length;
  const overs = Math.floor(legalBalls/6) + "." + (legalBalls % 6);
  const rr = legalBalls > 0 ? (totalRuns / (legalBalls/6)).toFixed(2) : "0.00";

  return (
    <div className="panel">
      <div className="panel-head">
        <Target size={16} />
        <h2>Live Scorecard Input</h2>
      </div>
      <div className="panel-body">

        <div className="field-row">
          <div className="field">
            <label><MapPin size={12} style={{display:'inline', marginBottom:'-2px'}}/> Venue</label>
            <select id="venue" value={matchState.venue} onChange={handleMatchStateChange}>
              {rosterData.venues.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div className="field"><label>Innings</label>
            <select id="innings" value={matchState.innings} onChange={handleMatchStateChange}>
              <option value="1">1st</option>
              <option value="2">2nd</option>
            </select>
          </div>
        </div>
        <div className="field-row">
          <div className="field">
            <label><Users size={12} style={{display:'inline', marginBottom:'-2px'}}/> Batting team</label>
            <select id="batting_team" value={matchState.batting_team} onChange={handleMatchStateChange}>
              {rosterData.teams.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className="field">
            <label><Users size={12} style={{display:'inline', marginBottom:'-2px'}}/> Bowling team</label>
            <select id="bowling_team" value={matchState.bowling_team} onChange={handleMatchStateChange}>
              {rosterData.teams.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        </div>
        <div className="field" style={{ display: role === 'bowling' ? 'flex' : 'none' }}>
          <label>Current bowler</label>
          <select id="current_bowler" value={matchState.current_bowler} onChange={handleMatchStateChange}>
            {rosterData.roster[matchState.bowling_team]?.bowlers.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        <div className="subhead"><Play size={14} /> Add a ball</div>
        <div className="ball-form">
          <div className="field"><label>Over</label><input type="number" id="b_over" min="0" max="19" value={ballInput.b_over} onChange={handleInputChange} /></div>
          <div className="field"><label>Ball</label><input type="number" id="b_ball" min="1" max="9" value={ballInput.b_ball} onChange={handleInputChange} /></div>
          <div className="field wide-field">
            <label>Batter</label>
            <select id="b_batter" value={ballInput.b_batter} onChange={handleInputChange}>
              <option value="">-- Select --</option>
              {rosterData.roster[matchState.batting_team]?.batters.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div className="field wide-field">
            <label>Non-striker</label>
            <select id="b_nonstriker" value={ballInput.b_nonstriker} onChange={handleInputChange}>
              <option value="">-- Select --</option>
              {rosterData.roster[matchState.batting_team]?.batters.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div className="field wide-field">
            <label>Bowler</label>
            <select id="b_bowler" value={ballInput.b_bowler} onChange={handleInputChange}>
              <option value="">-- Select --</option>
              {rosterData.roster[matchState.bowling_team]?.bowlers.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div className="field"><label>Runs</label><input type="number" id="b_runs" min="0" max="6" value={ballInput.b_runs} onChange={handleInputChange} /></div>
          <div className="field"><label>Wide</label><input type="number" id="b_wide" min="0" value={ballInput.b_wide} onChange={handleInputChange} /></div>
          <div className="field"><label>No-ball</label><input type="number" id="b_nb" min="0" value={ballInput.b_nb} onChange={handleInputChange} /></div>
          <div className="field"><label>Bye</label><input type="number" id="b_bye" min="0" value={ballInput.b_bye} onChange={handleInputChange} /></div>
          <div className="field"><label>Leg-bye</label><input type="number" id="b_legbye" min="0" value={ballInput.b_legbye} onChange={handleInputChange} /></div>
          <div className="field"><label>Wicket?</label>
            <select id="b_wicket" value={ballInput.b_wicket} onChange={(e) => setBallInput(prev => ({...prev, b_wicket: parseInt(e.target.value)}))}>
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </div>
          <div className="field"><button className="btn wide" onClick={handleAddBall}><Plus size={14}/> Add</button></div>
        </div>

        <div className="log-table-wrap">
          <table>
            <thead><tr><th>Ov.Bl</th><th>Batter</th><th>Bowler</th><th>Runs</th><th>Extras</th><th>Wkt</th><th></th></tr></thead>
            <tbody>
              {balls.length === 0 ? (
                <tr><td colSpan="7" className="empty-log">No balls logged yet — add the innings so far, then request a recommendation.</td></tr>
              ) : (
                balls.map((b, i) => {
                  const extras = b.wides + b.noballs + b.byes + b.legbyes;
                  return (
                    <tr key={i} className={b.is_wicket ? 'wicket' : ''}>
                      <td>{b.over}.{b.ball_in_over}</td>
                      <td>{b.batter}</td>
                      <td>{b.bowler}</td>
                      <td>{b.batter_runs}</td>
                      <td>{extras || '–'}</td>
                      <td>{b.is_wicket ? 'W' : '–'}</td>
                      <td><button className="del-btn" onClick={() => onDeleteBall(i)}><Trash2 size={14}/></button></td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>

        {balls.length > 0 && (
          <>
            <div className="state-strip">
              <div className="state-cell"><div className="k">Score</div><div className="v">{totalRuns}-{wkts}</div></div>
              <div className="state-cell"><div className="k">Overs</div><div className="v">{overs}</div></div>
              <div className="state-cell"><div className="k">Run rate</div><div className="v">{rr}</div></div>
              <div className="state-cell"><div className="k">Balls logged</div><div className="v">{balls.length}</div></div>
            </div>
            
            <OverTimeline balls={balls} />
            <MomentumChart balls={balls} role={role} />
          </>
        )}

        <div style={{ marginTop: '24px', display: 'flex', gap: '10px' }}>
          <button className="btn ghost" onClick={onLoadSample}>Load sample innings</button>
          <button className="btn ghost" onClick={onClear}>Clear</button>
        </div>

        <button 
          className={`btn big wide primary ${role}`}
          style={{ marginTop: '14px' }} 
          onClick={onRecommend}
          disabled={isRecommending}
        >
          {isRecommending ? 'Running models…' : 'Get Recommendation'}
        </button>
      </div>
    </div>
  );
}
