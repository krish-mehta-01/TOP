import { useState, useEffect } from 'react';
import RoleToggle from './components/RoleToggle';
import LiveScoreBanner from './components/LiveScoreBanner';
import ScorecardPanel from './components/ScorecardPanel';
import RecommendationConsole from './components/RecommendationConsole';
import ModelDiagnostics from "./components/ModelDiagnostics";
import './index.css';

const API_BASE = "http://localhost:8000";

function App() {
  const [role, setRole] = useState('bowling');
  const [balls, setBalls] = useState([]);
  const [backendStatus, setBackendStatus] = useState('checking');
  
  const [matchState, setMatchState] = useState({
    venue: "M Chinnaswamy Stadium, Bengaluru",
    innings: "1",
    batting_team: "Royal Challengers Bengaluru",
    bowling_team: "Mumbai Indians",
    current_bowler: "JJ Bumrah"
  });

  const [recommendation, setRecommendation] = useState(null);
  const [error, setError] = useState(null);
  const [isRecommending, setIsRecommending] = useState(false);

  useEffect(() => {
    let mounted = true;
    const checkHealth = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(2500) });
        if (r.ok && mounted) {
          setBackendStatus('live');
          return;
        }
        throw new Error();
      } catch (e) {
        if (mounted) setBackendStatus('down');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 8000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleAddBall = (ballData) => {
    setBalls(prev => [...prev, ballData]);
    setRecommendation(null);
    setError(null);
  };

  const handleDeleteBall = (index) => {
    setBalls(prev => prev.filter((_, i) => i !== index));
    setRecommendation(null);
    setError(null);
  };

  const handleRoleChange = (newRole) => {
    setRole(newRole);
    setRecommendation(null);
    setError(null);
  };

  const handleClear = () => {
    setBalls([]);
    setRecommendation(null);
    setError(null);
  };

  const handleLoadSample = () => {
    const newBalls = [];
    const bowlers3 = ["JJ Bumrah", "T Boult", "M Pathirana"];
    const runsPattern = [0,1,0,4,0,2];
    const overCount = role === "bowling" ? 3 : 12;
    for(let over=0; over<overCount; over++){
      const bowler = role === "bowling" ? "JJ Bumrah" : bowlers3[over % 3];
      for(let ball=1; ball<=6; ball++){
        newBalls.push({
          over, 
          ball_in_over: ball,
          batter: ball % 2 === 0 ? "V Kohli" : "F du Plessis",
          non_striker: ball % 2 === 0 ? "F du Plessis" : "V Kohli",
          bowler,
          batter_runs: runsPattern[ball-1],
          wides:0, noballs:0, byes:0, legbyes:0,
          is_wicket: (over === Math.floor(overCount*0.6) && ball === 3) ? 1 : 0,
        });
      }
    }
    setBalls(newBalls);
    setMatchState(prev => ({ ...prev, current_bowler: "JJ Bumrah" }));
    setRecommendation(null);
    setError(null);
  };

  const handleRecommend = async () => {
    if (balls.length === 0) {
      setError("Log at least one ball first.");
      setRecommendation(null);
      return;
    }
    setIsRecommending(true);
    setError(null);

    const payload = {
      role,
      venue: matchState.venue,
      batting_team: matchState.batting_team,
      bowling_team: matchState.bowling_team,
      innings: parseInt(matchState.innings),
      balls,
    };
    if (role === "bowling") {
      payload.bowler = matchState.current_bowler;
    }

    try {
      const r = await fetch(`${API_BASE}/api/recommend`, {
        method: "POST", 
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const data = await r.json();
      if (!r.ok) {
        setError(data.detail || "Request failed");
        setRecommendation(null);
        return;
      }
      setRecommendation(data);
    } catch (e) {
      setError(`Could not reach the backend at ${API_BASE}. Is uvicorn running?`);
      setRecommendation(null);
    } finally {
      setIsRecommending(false);
    }
  };

  return (
    <div className="app-dashboard">
      <LiveScoreBanner balls={balls} matchState={matchState} />
      
      <div className="dashboard-left">
        <div style={{display: 'flex', flexDirection: 'column', gap: '16px', marginBottom: '16px'}}>
            <RoleToggle role={role} setRole={handleRoleChange} />
        </div>
        
        <ScorecardPanel 
          role={role}
          balls={balls}
          onAddBall={handleAddBall}
          onDeleteBall={handleDeleteBall}
          onClear={handleClear}
          onLoadSample={handleLoadSample}
          onRecommend={handleRecommend}
          isRecommending={isRecommending}
          matchState={matchState}
          setMatchState={setMatchState}
        />
      </div>
      
      <div className="dashboard-right">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <RecommendationConsole 
            error={error}
            recommendation={recommendation}
            role={role}
          />
          <ModelDiagnostics recommendation={recommendation} />
        </div>
      </div>
    </div>
  );
}

export default App;
