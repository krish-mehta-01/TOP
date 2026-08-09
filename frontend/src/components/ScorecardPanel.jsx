import { useState, useEffect } from 'react';
import MomentumChart from './MomentumChart';
import OverTimeline from './OverTimeline';
import { Plus, Play, Trash2, MapPin, Users, Target, Crosshair, AlertTriangle, Undo2 } from 'lucide-react';
import rosterData from '../data/roster.json';

const PITCH_ZONES = [
  { id: 'short-off', length: 'Short', line: 'Outside Off', top: '25%', left: '0%' },
  { id: 'short-mid', length: 'Short', line: 'Stumps', top: '25%', left: '33.3%' },
  { id: 'short-leg', length: 'Short', line: 'Leg Side', top: '25%', left: '66.6%' },
  { id: 'good-off', length: 'Good', line: 'Outside Off', top: '50%', left: '0%' },
  { id: 'good-mid', length: 'Good', line: 'Stumps', top: '50%', left: '33.3%' },
  { id: 'good-leg', length: 'Good', line: 'Leg Side', top: '50%', left: '66.6%' },
  { id: 'full-off', length: 'Full', line: 'Outside Off', top: '75%', left: '0%' },
  { id: 'full-mid', length: 'Full', line: 'Stumps', top: '75%', left: '33.3%' },
  { id: 'full-leg', length: 'Full', line: 'Leg Side', top: '75%', left: '66.6%' },
];

const WAGON_ZONES = [
  { id: 'third-man', label: '3rd Man', transform: 'rotate(0deg) skew(45deg)' },
  { id: 'point', label: 'Point', transform: 'rotate(45deg) skew(45deg)' },
  { id: 'cover', label: 'Cover', transform: 'rotate(90deg) skew(45deg)' },
  { id: 'long-off', label: 'Long Off', transform: 'rotate(135deg) skew(45deg)' },
  { id: 'long-on', label: 'Long On', transform: 'rotate(180deg) skew(45deg)' },
  { id: 'mid-wicket', label: 'Mid Wicket', transform: 'rotate(225deg) skew(45deg)' },
  { id: 'square-leg', label: 'Square Leg', transform: 'rotate(270deg) skew(45deg)' },
  { id: 'fine-leg', label: 'Fine Leg', transform: 'rotate(315deg) skew(45deg)' },
];

export default function ScorecardPanel({ 
  role, balls, onAddBall, onDeleteBall, onClear, onLoadSample, onRecommend, isRecommending, matchState, setMatchState
}) {
  const [ballInput, setBallInput] = useState({
    b_batter: 'V Kohli', b_nonstriker: 'F du Plessis', b_bowler: 'JJ Bumrah',
  });
  
  const [activeSpatial, setActiveSpatial] = useState({ length: null, line: null, wagon: null });
  const [modals, setModals] = useState({ wicket: false, extras: null });
  const [dismissalForm, setDismissalForm] = useState({ type: 'Caught', fielder: '', nextBatter: '', whoGotOut: 'striker', runsCompleted: 0 });
  
  const handleMatchStateChange = (e) => {
    const { id, value } = e.target;
    setMatchState(prev => ({ ...prev, [id]: value }));
  };

  const handleInputChange = (e) => {
    const { id, value } = e.target;
    setBallInput(prev => ({ ...prev, [id]: value }));
  };

  const logTacticalBall = (runs = 0, isWicket = false, extrasType = null, extraRuns = 0) => {
    const safeRuns = Number(runs) || 0;
    const safeExtraRuns = Number(extraRuns) || 0;
    const legalBalls = balls.filter(b => b.wides === 0 && b.noballs === 0).length;
    let nextOver = Math.floor(legalBalls / 6);
    
    if (nextOver >= 20) {
      alert("Innings complete! 20 overs maximum reached.");
      return;
    }

    let nextBall = (legalBalls % 6) + 1;
    if (extrasType !== null) {
      nextBall = (legalBalls % 6);
      if(nextBall === 0 && legalBalls > 0) {
        nextBall = 6;
        nextOver = nextOver - 1;
      } else if (nextBall === 0) {
        nextBall = 1;
      }
    }
    
    const payload = {
      over: nextOver,
      ball_in_over: nextBall,
      batter: ballInput.b_batter,
      non_striker: ballInput.b_nonstriker,
      bowler: matchState.current_bowler || ballInput.b_bowler,
      batter_runs: safeRuns,
      wides: extrasType === 'wd' ? 1 + safeExtraRuns : 0,
      noballs: extrasType === 'nb' ? 1 : 0,
      byes: extrasType === 'b' ? extraRuns : 0,
      legbyes: extrasType === 'lb' ? extraRuns : 0,
      is_wicket: isWicket ? 1 : 0,
      spatial: activeSpatial
    };

    onAddBall(payload);
    
    const isLegal = extrasType === null;
    const isOddRun = runs % 2 !== 0;
    const endOfOver = isLegal && nextBall === 6;
    const shouldRotate = isOddRun !== endOfOver;

    setBallInput(prev => {
      let nextStriker = prev.b_batter;
      let nextNonStriker = prev.b_nonstriker;
      
      if (isWicket && dismissalForm.nextBatter) {
        if (dismissalForm.whoGotOut === 'non_striker') {
          nextNonStriker = dismissalForm.nextBatter;
        } else {
          nextStriker = dismissalForm.nextBatter;
        }
      }

      if (shouldRotate) {
        return {
          ...prev,
          b_batter: nextNonStriker,
          b_nonstriker: nextStriker
        };
      }
      return {
        ...prev,
        b_batter: nextStriker,
        b_nonstriker: nextNonStriker
      };
    });

    setActiveSpatial({ length: null, line: null, wagon: null });
    setModals({ wicket: false, extras: null });
    setDismissalForm({ type: 'Caught', fielder: '', nextBatter: '', whoGotOut: 'striker' });

    if (endOfOver && nextOver < 20) {
      setMatchState(prev => ({ ...prev, current_bowler: '' }));
    }
  };

  const totalRuns = balls.reduce((s,b) => s + b.batter_runs + b.wides + b.noballs + b.byes + b.legbyes, 0);
  const wkts = balls.reduce((s,b) => s + b.is_wicket, 0);
  const legalBalls = balls.filter(b => b.wides === 0 && b.noballs === 0).length;
  const currentOver = Math.floor(legalBalls/6);
  const overs = currentOver + "." + (legalBalls % 6);
  const rr = legalBalls > 0 ? (totalRuns / (legalBalls/6)).toFixed(2) : "0.00";

  const isEndOfOver = legalBalls > 0 && legalBalls % 6 === 0;
  const isBowlingTimeoutWindow = isEndOfOver && (currentOver >= 6 && currentOver <= 8);
  const isBattingTimeoutWindow = isEndOfOver && (currentOver >= 13 && currentOver <= 15);
  const isTimeoutWindow = isBowlingTimeoutWindow || isBattingTimeoutWindow;
  
  const isFreeHit = balls.length > 0 && balls[balls.length - 1].noballs > 0;

  const battedPlayers = new Set();
  const bowlerOvers = {};
  let lastBowler = null;

  balls.forEach(b => {
    battedPlayers.add(b.batter);
    battedPlayers.add(b.non_striker);
    if (!bowlerOvers[b.bowler]) bowlerOvers[b.bowler] = new Set();
    bowlerOvers[b.bowler].add(b.over);
    lastBowler = b.bowler;
  });
  battedPlayers.add(ballInput.b_batter);
  battedPlayers.add(ballInput.b_nonstriker);

  const availableBatters = (rosterData.roster[matchState.batting_team]?.batters || []).filter(p => !battedPlayers.has(p));

  const isNewOver = legalBalls > 0 && legalBalls % 6 === 0;
  const availableBowlers = (rosterData.roster[matchState.bowling_team]?.bowlers || []).filter(p => {
    const oversBowled = bowlerOvers[p] ? bowlerOvers[p].size : 0;
    if (isNewOver) {
      if (p === lastBowler) return false;
      if (oversBowled >= 4) return false;
    } else {
      if (oversBowled >= 4 && p !== lastBowler) return false;
    }
    return true;
  });

  return (
    <div className="panel">
      <div className="panel-body">


        
        <div className="field-row">
          <div className="field">
            <label>Batter</label>
            <select id="b_batter" value={ballInput.b_batter} onChange={handleInputChange}>
              <option value="">-- Select --</option>
              {rosterData.roster[matchState.batting_team]?.batters.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Non-striker</label>
            <select id="b_nonstriker" value={ballInput.b_nonstriker} onChange={handleInputChange}>
              <option value="">-- Select --</option>
              {rosterData.roster[matchState.batting_team]?.batters.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
        </div>
        <div className="field">
          <label>Current bowler</label>
          <select id="current_bowler" value={matchState.current_bowler} onChange={handleMatchStateChange}>
            <option value="">-- Select New Bowler --</option>
            {availableBowlers.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        <div className="subhead" style={{marginTop: '20px'}}><Play size={14} /> Tactical Grid</div>
        
        {!matchState.current_bowler && (
            <div style={{color: '#f59e0b', fontSize: '13px', margin: '12px 0', fontWeight: 'bold', display: 'flex', gap: '6px', alignItems: 'center'}}>
                <AlertTriangle size={14} /> Please select a bowler for this over to continue logging.
            </div>
        )}

        <div style={{opacity: matchState.current_bowler ? 1 : 0.3, pointerEvents: matchState.current_bowler ? 'auto' : 'none'}}>
            {modals.extras && (
                <div style={{marginBottom: '12px', padding: '12px', border: '1px solid #ca8a04', background: 'rgba(202,138,4,0.1)', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '8px'}}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
                        <AlertTriangle size={14} color="#ca8a04" />
                        <span style={{color: '#ca8a04', fontWeight: 'bold', fontSize: '12px', textTransform: 'uppercase'}}>
                            {modals.extras === 'wd' && 'Wide (+1 Auto): Extra Wide Runs'}
                            {modals.extras === 'nb' && 'No Ball (+1 Auto): Batter Runs'}
                            {modals.extras === 'b' && 'Byes: Total Bye Runs'}
                            {modals.extras === 'lb' && 'Leg Byes: Total Leg Bye Runs'}
                        </span>
                        <button className="btn ghost" style={{marginLeft: 'auto', padding: '2px 8px'}} onClick={() => setModals({...modals, extras: null})}>Cancel</button>
                    </div>
                    <div className="tactile-grid" style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px'}}>
                        {[0,1,2,3,4,5,6].map(val => (
                            <button key={val} className="btn-run btn-gray" style={{height: '48px', fontSize: '18px', border: '1px solid rgba(255,255,255,0.1)'}} onClick={() => {
                                if (modals.extras === 'wd') logTacticalBall(0, false, 'wd', val);
                                else if (modals.extras === 'nb') logTacticalBall(val, false, 'nb', 0);
                                else if (modals.extras === 'b') logTacticalBall(0, false, 'b', val);
                                else if (modals.extras === 'lb') logTacticalBall(0, false, 'lb', val);
                            }}>+{val}</button>
                        ))}
                    </div>
                </div>
            )}

            <div className="tactile-grid" style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', marginBottom: '12px'}}>
                <button className="btn-run btn-gray" onClick={() => logTacticalBall(0)}>0</button>
                <button className="btn-run btn-blue" onClick={() => logTacticalBall(1)}>1</button>
                <button className="btn-run btn-blue" onClick={() => logTacticalBall(2)}>2</button>
                <button className="btn-run btn-blue" onClick={() => logTacticalBall(3)}>3</button>
                <button className="btn-run btn-green" onClick={() => logTacticalBall(4)}>4</button>
                <button className="btn-run btn-green" onClick={() => logTacticalBall(5)}>5</button>
                <button className="btn-run btn-green" onClick={() => logTacticalBall(6)}>6</button>
                <button className="btn-run btn-gray" 
                    style={{opacity: balls.length > 0 ? 1 : 0.4, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', fontSize: '14px'}} 
                    onClick={() => { if(balls.length > 0) onDeleteBall(balls.length - 1); }}
                    disabled={balls.length === 0}
                    title="Undo Last Ball"
                >
                    <Undo2 size={16} /> UNDO
                </button>
            </div>
            
            <div style={{display: 'flex', gap: '8px', marginBottom: '12px'}}>
                <button className="btn-run btn-gray" style={{flex: 1, height: '40px', fontSize: '14px', border: '1px solid rgba(255,255,255,0.1)'}} onClick={() => setModals({...modals, extras: 'wd'})}>WIDE</button>
                <button className="btn-run btn-gray" style={{flex: 1, height: '40px', fontSize: '14px', border: '1px solid rgba(255,255,255,0.1)'}} onClick={() => setModals({...modals, extras: 'nb'})}>NO BALL</button>
                <button className="btn-run btn-gray" style={{flex: 1, height: '40px', fontSize: '14px', border: '1px solid rgba(255,255,255,0.1)'}} onClick={() => setModals({...modals, extras: 'b'})}>BYES</button>
                <button className="btn-run btn-gray" style={{flex: 1, height: '40px', fontSize: '14px', border: '1px solid rgba(255,255,255,0.1)'}} onClick={() => setModals({...modals, extras: 'lb'})}>LEG BYES</button>
            </div>

            {!modals.wicket ? (
                <button className="btn-red" onClick={() => setModals({...modals, wicket: true})}>WICKET</button>
            ) : (
                <div className="modal-box" style={{ width: '100%', maxWidth: 'none', marginTop: '12px', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
                    <h2 style={{color: '#ef4444', marginBottom: '16px'}}>LOG DISMISSAL</h2>
                    <div className="field-row">
                        <div className="field">
                            <label>Type {isFreeHit && <span style={{color: '#f59e0b'}}>(Free Hit)</span>}</label>
                            <select 
                              value={isFreeHit ? 'Run Out' : dismissalForm.type} 
                              onChange={(e) => setDismissalForm({...dismissalForm, type: e.target.value})}
                              disabled={isFreeHit}
                            >
                                {!isFreeHit && (
                                  <>
                                    <option>Caught</option>
                                    <option>Bowled</option>
                                    <option>LBW</option>
                                  </>
                                )}
                                <option>Run Out</option>
                            </select>
                        </div>
                        <div className="field">
                            <label>Who Got Out?</label>
                            <select value={dismissalForm.whoGotOut} onChange={(e) => setDismissalForm({...dismissalForm, whoGotOut: e.target.value})}>
                                <option value="striker">Striker</option>
                                <option value="non_striker">Non-Striker</option>
                            </select>
                        </div>
                    </div>
                    <div className="field-row">
                        <div className="field" style={{marginBottom: '24px'}}>
                            <label>Next Batter In</label>
                            <select value={dismissalForm.nextBatter} onChange={(e) => setDismissalForm({...dismissalForm, nextBatter: e.target.value})}>
                                <option value="">-- Select --</option>
                                {availableBatters.map(p => <option key={p} value={p}>{p}</option>)}
                            </select>
                        </div>
                        <div className="field" style={{marginBottom: '24px'}}>
                            <label>Runs Completed</label>
                            <select value={dismissalForm.runsCompleted} onChange={(e) => setDismissalForm({...dismissalForm, runsCompleted: parseInt(e.target.value)})}>
                                {[0,1,2,3,4,5,6].map(val => <option key={val} value={val}>{val}</option>)}
                            </select>
                        </div>
                    </div>
                    <div style={{display: 'flex', gap: '12px'}}>
                        <button className="btn-run btn-red" style={{flex: 1, letterSpacing: 'normal', opacity: dismissalForm.nextBatter ? 1 : 0.5}} onClick={() => logTacticalBall(dismissalForm.runsCompleted, true)} disabled={!dismissalForm.nextBatter}>CONFIRM</button>
                        <button className="btn-run btn-gray" style={{flex: 1}} onClick={() => setModals({...modals, wicket: false})}>CANCEL</button>
                    </div>
                </div>
            )}
        </div>



        <div className="log-table-wrap" style={{marginTop: '24px'}}>
          <table>
            <thead><tr><th>Ov.Bl</th><th>Batter</th><th>Bowler</th><th>Runs</th><th>Extras</th><th>Wkt</th><th></th></tr></thead>
            <tbody>
              {balls.length === 0 ? (
                <tr><td colSpan="7" className="empty-log">No balls logged yet. Use tactical grid to log events.</td></tr>
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
          className={`btn big wide primary ${role} ${isTimeoutWindow ? 'pulse-glow' : ''}`}
          style={{ marginTop: '14px', position: 'relative', overflow: 'visible' }} 
          onClick={onRecommend}
          disabled={isRecommending || wkts >= 10 || currentOver >= 20}
        >
          {isRecommending ? 'Running models…' : 
            isTimeoutWindow ? 'STRATEGIC TIMEOUT: GENERATE PLAN' : 'Get Recommendation'}
          
          {isBowlingTimeoutWindow && <span className="timeout-badge">BOWLING TIMEOUT</span>}
          {isBattingTimeoutWindow && <span className="timeout-badge">BATTING TIMEOUT</span>}
        </button>
      </div>
    </div>
  );
}
