import React, { useEffect, useState, useMemo } from 'react';

export const NeuralNetworkViz: React.FC = () => {
  // Sequence: 0=Forward Pass, 1=Loss Calculation, 2=Backprop
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
        setActiveStep(prev => (prev + 1) % 3);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const width = 600;
  const height = 300;
  const layerX = [80, 240, 400, 540]; // Input, H1, H2, Output
  
  const inputNodes = [80, 150, 220];
  const h1Nodes = [50, 100, 150, 200, 250];
  const h2Nodes = [70, 130, 190, 250];
  const outputNodes = [110, 190];

  const createConnections = (nodesLeft: number[], nodesRight: number[], xLeft: number, xRight: number) => {
    const paths = [];
    for (let i = 0; i < nodesLeft.length; i++) {
        for (let j = 0; j < nodesRight.length; j++) {
            // Randomish "weight" for visualization
            const weight = 0.2 + Math.random() * 0.8;
            paths.push({
                d: `M ${xLeft} ${nodesLeft[i]} C ${xLeft + (xRight - xLeft)/2} ${nodesLeft[i]}, ${xLeft + (xRight - xLeft)/2} ${nodesRight[j]}, ${xRight} ${nodesRight[j]}`,
                weight,
                key: `${xLeft}-${i}-${j}`
            });
        }
    }
    return paths;
  };

  const c1 = useMemo(() => createConnections(inputNodes, h1Nodes, layerX[0], layerX[1]), []);
  const c2 = useMemo(() => createConnections(h1Nodes, h2Nodes, layerX[1], layerX[2]), []);
  const c3 = useMemo(() => createConnections(h2Nodes, outputNodes, layerX[2], layerX[3]), []);

  return (
    <div className="w-full h-80 bg-slate-950 rounded-2xl border border-slate-800 flex items-center justify-center relative overflow-hidden shadow-inner select-none">
       <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
          <defs>
            <filter id="nodeGlow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <linearGradient id="forwardGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#6366f1" />
                <stop offset="100%" stopColor="#818cf8" />
            </linearGradient>
            <linearGradient id="backGrad" x1="100%" y1="0%" x2="0%" y2="0%">
                <stop offset="0%" stopColor="#f43f5e" />
                <stop offset="100%" stopColor="#fb7185" />
            </linearGradient>
          </defs>

          {/* Connections Layer */}
          {[c1, c2, c3].map((layer, lIdx) => (
              <g key={lIdx}>
                  {layer.map((conn) => (
                      <path 
                        key={conn.key} 
                        d={conn.d} 
                        stroke="#1e293b" 
                        strokeWidth={conn.weight * 2} 
                        fill="none" 
                        className="transition-all duration-1000"
                        strokeOpacity={activeStep === 0 ? 0.8 : 0.3}
                      />
                  ))}
                  {/* Active Flow */}
                  {activeStep === 0 && layer.map((conn) => (
                      <path 
                        key={`${conn.key}-active`}
                        d={conn.d} 
                        stroke="url(#forwardGrad)" 
                        strokeWidth={conn.weight * 2} 
                        fill="none" 
                        strokeDasharray="4 8"
                        className="animate-[flow_1s_linear_infinite]"
                        strokeOpacity={0.6}
                      />
                  ))}
                  {activeStep === 2 && layer.map((conn) => (
                      <path 
                        key={`${conn.key}-back`}
                        d={conn.d} 
                        stroke="url(#backGrad)" 
                        strokeWidth={conn.weight * 1.5} 
                        fill="none" 
                        strokeDasharray="4 8"
                        className="animate-[flow_1s_linear_infinite]"
                        style={{ animationDirection: 'reverse' }}
                        strokeOpacity={0.5}
                      />
                  ))}
              </g>
          ))}

          {/* Nodes Layer */}
          {[inputNodes, h1Nodes, h2Nodes, outputNodes].map((layer, lIdx) => (
              <g key={lIdx}>
                  {layer.map((y, nIdx) => (
                      <circle 
                        key={nIdx} 
                        cx={layerX[lIdx]} 
                        cy={y} 
                        r={lIdx === 0 || lIdx === 3 ? 10 : 8} 
                        fill="#0f172a" 
                        stroke={activeStep === 0 ? (lIdx === 0 ? '#6366f1' : '#334155') : activeStep === 2 ? '#f43f5e' : '#1e293b'} 
                        strokeWidth="2" 
                        filter={activeStep === 0 && lIdx === 0 ? "url(#nodeGlow)" : ""}
                      />
                  ))}
              </g>
          ))}

          {/* Legend / Status Overlay */}
          <g transform="translate(20, 20)">
             <rect width="120" height="40" rx="4" fill="#0f172a" stroke="#1e293b" />
             <text x="10" y="25" fill="#94a3b8" fontSize="10" fontWeight="bold" fontClassName="font-mono">
                {activeStep === 0 ? "FORWARD PASS..." : activeStep === 1 ? "CALC ERROR" : "BACKPROPAGATING"}
             </text>
             <circle cx="105" cy="21" r="4" fill={activeStep === 0 ? "#6366f1" : activeStep === 1 ? "#fbbf24" : "#f43f5e"} className="animate-pulse" />
          </g>
       </svg>
    </div>
  );
};