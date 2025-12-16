import React, { useEffect, useState, useMemo } from 'react';

export const NeuralNetworkViz: React.FC = () => {
  // Sequence: 0=Input->Hidden, 1=Hidden Activation, 2=Hidden->Output, 3=Output Activation
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
        setActiveStep(prev => (prev + 1) % 4);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Geometry Configuration
  const width = 600;
  const height = 300;
  const layerX = [100, 300, 500]; // X coordinates for Input, Hidden, Output
  
  // Node positions Y
  const inputNodes = [80, 150, 220];
  const hiddenNodes = [60, 120, 180, 240];
  const outputNodes = [100, 200];

  // Generate SVG Paths for connections
  const connections1 = useMemo(() => {
    const paths = [];
    for (let i = 0; i < inputNodes.length; i++) {
        for (let j = 0; j < hiddenNodes.length; j++) {
            paths.push({
                d: `M ${layerX[0]} ${inputNodes[i]} C ${layerX[0] + 100} ${inputNodes[i]}, ${layerX[1] - 100} ${hiddenNodes[j]}, ${layerX[1]} ${hiddenNodes[j]}`,
                key: `i${i}-h${j}`
            });
        }
    }
    return paths;
  }, []);

  const connections2 = useMemo(() => {
    const paths = [];
    for (let i = 0; i < hiddenNodes.length; i++) {
        for (let j = 0; j < outputNodes.length; j++) {
            paths.push({
                d: `M ${layerX[1]} ${hiddenNodes[i]} C ${layerX[1] + 100} ${hiddenNodes[i]}, ${layerX[2] - 100} ${outputNodes[j]}, ${layerX[2]} ${outputNodes[j]}`,
                key: `h${i}-o${j}`
            });
        }
    }
    return paths;
  }, []);

  return (
    <div className="w-full h-64 bg-slate-900 rounded-lg border border-slate-700 flex items-center justify-center relative overflow-hidden shadow-inner select-none">
       <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
          <defs>
            <linearGradient id="gradInput" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#6366f1" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#94a3b8" stopOpacity="0.2" />
            </linearGradient>
            <linearGradient id="gradOutput" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#94a3b8" stopOpacity="0.2" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.8" />
            </linearGradient>
          </defs>

          {/* Layer 1 Connections (Input -> Hidden) */}
          {connections1.map((path) => (
              <g key={path.key}>
                  {/* Background Line */}
                  <path d={path.d} stroke="#334155" strokeWidth="1" fill="none" />
                  {/* Active Pulse Line */}
                  <path 
                    d={path.d} 
                    stroke="url(#gradInput)" 
                    strokeWidth={activeStep === 0 ? 3 : 0} 
                    fill="none"
                    strokeDasharray="10 5"
                    className={activeStep === 0 ? "animate-[flow_0.5s_linear_infinite]" : ""}
                    style={{ opacity: activeStep === 0 ? 1 : 0, transition: 'opacity 0.2s' }}
                  />
              </g>
          ))}

          {/* Layer 2 Connections (Hidden -> Output) */}
          {connections2.map((path) => (
              <g key={path.key}>
                  <path d={path.d} stroke="#334155" strokeWidth="1" fill="none" />
                  <path 
                    d={path.d} 
                    stroke="url(#gradOutput)" 
                    strokeWidth={activeStep === 2 ? 3 : 0} 
                    fill="none"
                    strokeDasharray="10 5"
                    className={activeStep === 2 ? "animate-[flow_0.5s_linear_infinite]" : ""}
                    style={{ opacity: activeStep === 2 ? 1 : 0, transition: 'opacity 0.2s' }}
                  />
              </g>
          ))}

          {/* Input Nodes */}
          {inputNodes.map((y, i) => (
              <g key={`in-${i}`} transform={`translate(${layerX[0]}, ${y})`}>
                  <circle r="12" fill="#0f172a" stroke={activeStep === 0 || activeStep === 3 ? "#6366f1" : "#475569"} strokeWidth="2" className="transition-colors duration-300" />
                  <text x="0" y="4" textAnchor="middle" fontSize="10" fill="#cbd5e1" fontFamily="monospace">x{i+1}</text>
                  {/* Glow effect */}
                  {activeStep === 0 && <circle r="16" fill="none" stroke="#6366f1" strokeOpacity="0.5" className="animate-ping" />}
              </g>
          ))}

          {/* Hidden Nodes */}
          {hiddenNodes.map((y, i) => (
              <g key={`hid-${i}`} transform={`translate(${layerX[1]}, ${y})`}>
                  <circle 
                    r="10" 
                    fill={activeStep >= 1 && activeStep <= 2 ? "#e2e8f0" : "#0f172a"} 
                    stroke={activeStep >= 1 && activeStep <= 2 ? "#ffffff" : "#475569"} 
                    strokeWidth="2" 
                    className="transition-all duration-300" 
                  />
                  {activeStep === 1 && <circle r="14" fill="none" stroke="#e2e8f0" strokeOpacity="0.5" className="animate-ping" />}
              </g>
          ))}

          {/* Output Nodes */}
          {outputNodes.map((y, i) => (
              <g key={`out-${i}`} transform={`translate(${layerX[2]}, ${y})`}>
                  <circle r="12" fill="#0f172a" stroke={activeStep === 3 ? "#10b981" : "#475569"} strokeWidth="2" className="transition-colors duration-300" />
                  <text x="0" y="4" textAnchor="middle" fontSize="10" fill="#cbd5e1" fontFamily="monospace">y{i+1}</text>
                  {activeStep === 3 && <circle r="16" fill="none" stroke="#10b981" strokeOpacity="0.5" className="animate-ping" />}
              </g>
          ))}

          {/* Labels */}
          <text x={layerX[0]} y="40" textAnchor="middle" fill="#64748b" fontSize="10" fontWeight="bold" letterSpacing="2">INPUT</text>
          <text x={layerX[1]} y="30" textAnchor="middle" fill="#64748b" fontSize="10" fontWeight="bold" letterSpacing="2">HIDDEN</text>
          <text x={layerX[2]} y="40" textAnchor="middle" fill="#64748b" fontSize="10" fontWeight="bold" letterSpacing="2">OUTPUT</text>
       </svg>
    </div>
  );
};