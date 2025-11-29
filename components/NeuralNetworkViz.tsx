import React, { useEffect, useRef, useState, useLayoutEffect } from 'react';

export const NeuralNetworkViz: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  const inputRefs = useRef<(HTMLDivElement | null)[]>([]);
  const hiddenRefs = useRef<(HTMLDivElement | null)[]>([]);
  const outputRefs = useRef<(HTMLDivElement | null)[]>([]);

  // Store style objects for edges
  const [edges1, setEdges1] = useState<{style: React.CSSProperties}[]>([]);
  const [edges2, setEdges2] = useState<{style: React.CSSProperties}[]>([]);
  
  // State for activation animation sequence (0: Input, 1: Hidden, 2: Output)
  const [activeLayer, setActiveLayer] = useState<0 | 1 | 2>(0);

  // Network Topology
  const inputNodes = [1, 2, 3];
  const hiddenNodes = [1, 2, 3, 4];
  const outputNodes = [1, 2];

  // Logic to calculate CSS geometry for connecting lines
  const updateLines = () => {
    if (!containerRef.current) return;
    const containerRect = containerRef.current.getBoundingClientRect();

    const getGeometry = (n1: HTMLDivElement, n2: HTMLDivElement): React.CSSProperties => {
        const r1 = n1.getBoundingClientRect();
        const r2 = n2.getBoundingClientRect();
        
        // Calculate centers relative to container
        const x1 = r1.left - containerRect.left + r1.width / 2;
        const y1 = r1.top - containerRect.top + r1.height / 2;
        const x2 = r2.left - containerRect.left + r2.width / 2;
        const y2 = r2.top - containerRect.top + r2.height / 2;

        const length = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
        const angle = Math.atan2(y2 - y1, x2 - x1) * (180 / Math.PI);

        return {
            position: 'absolute',
            top: y1,
            left: x1,
            width: length,
            height: 1, 
            transform: `rotate(${angle}deg)`,
            transformOrigin: '0 0',
            zIndex: 0,
            transition: 'opacity 0.5s ease, height 0.3s ease',
            pointerEvents: 'none'
        };
    };

    // Edges between Input and Hidden
    const newEdges1: {style: React.CSSProperties}[] = [];
    inputRefs.current.forEach(inNode => {
        hiddenRefs.current.forEach(hidNode => {
            if(inNode && hidNode) {
                const style = getGeometry(inNode, hidNode);
                style.background = 'linear-gradient(90deg, #6366f1, #94a3b8)'; // Indigo to Slate
                newEdges1.push({ style });
            }
        });
    });
    setEdges1(newEdges1);

    // Edges between Hidden and Output
    const newEdges2: {style: React.CSSProperties}[] = [];
    hiddenRefs.current.forEach(hidNode => {
        outputRefs.current.forEach(outNode => {
            if(hidNode && outNode) {
                const style = getGeometry(hidNode, outNode);
                style.background = 'linear-gradient(90deg, #94a3b8, #10b981)'; // Slate to Emerald
                newEdges2.push({ style });
            }
        });
    });
    setEdges2(newEdges2);
  };

  // Recalculate lines on mount and resize
  useLayoutEffect(() => {
    updateLines();
    window.addEventListener('resize', updateLines);
    // Delay to allow layout to settle
    const timer = setTimeout(updateLines, 200);
    return () => {
        window.removeEventListener('resize', updateLines);
        clearTimeout(timer);
    };
  }, []);

  // Animation Loop
  useEffect(() => {
    const interval = setInterval(() => {
        setActiveLayer(prev => (prev + 1) % 3 as 0|1|2);
    }, 1200);
    return () => clearInterval(interval);
  }, []);

  return (
    <div ref={containerRef} className="w-full h-64 bg-slate-900 rounded-lg border border-slate-700 flex items-center justify-around relative overflow-hidden p-8 shadow-inner select-none">
       
       {/* --- EDGES (Pure CSS Divs) --- */}
       
       {/* Input -> Hidden Edges */}
       {edges1.map((edge, i) => (
           <div 
             key={`e1-${i}`} 
             style={{
               ...edge.style, 
               opacity: activeLayer === 0 ? 0.8 : 0.15, 
               height: activeLayer === 0 ? 2 : 1 
             }} 
           />
       ))}

       {/* Hidden -> Output Edges */}
       {edges2.map((edge, i) => (
           <div 
             key={`e2-${i}`} 
             style={{
               ...edge.style, 
               opacity: activeLayer === 1 ? 0.8 : 0.15,
               height: activeLayer === 1 ? 2 : 1
             }} 
           />
       ))}

       {/* --- NODES (Flex Items) --- */}

       {/* Input Layer */}
       <div className="flex flex-col justify-center gap-6 z-10">
        <div className="text-[10px] text-center text-slate-400 font-mono uppercase tracking-wider mb-2">Input</div>
        {inputNodes.map((n, i) => (
          <div 
            key={`in-${n}`}
            ref={el => { inputRefs.current[i] = el; }}
            className={`w-8 h-8 rounded-full border-2 transition-all duration-500 ease-in-out ${
              activeLayer === 0 
                ? 'bg-indigo-500 border-white shadow-[0_0_15px_rgba(99,102,241,0.6)] scale-110' 
                : 'bg-indigo-900/40 border-indigo-500/40'
            }`}
          />
        ))}
      </div>

      {/* Hidden Layer */}
      <div className="flex flex-col justify-center gap-4 z-10">
        <div className="text-[10px] text-center text-slate-400 font-mono uppercase tracking-wider mb-2">Hidden</div>
        {hiddenNodes.map((n, i) => (
          <div 
            key={`hid-${n}`}
            ref={el => { hiddenRefs.current[i] = el; }}
            className={`w-6 h-6 rounded-full border-2 transition-all duration-500 ease-in-out ${
              activeLayer === 1
                ? 'bg-slate-200 border-white shadow-[0_0_15px_rgba(255,255,255,0.6)] scale-110' 
                : 'bg-slate-800 border-slate-600'
            }`}
          />
        ))}
      </div>

      {/* Output Layer */}
      <div className="flex flex-col justify-center gap-8 z-10">
        <div className="text-[10px] text-center text-slate-400 font-mono uppercase tracking-wider mb-2">Output</div>
        {outputNodes.map((n, i) => (
          <div 
            key={`out-${n}`}
            ref={el => { outputRefs.current[i] = el; }}
            className={`w-8 h-8 rounded-full border-2 transition-all duration-500 ease-in-out ${
              activeLayer === 2
                ? 'bg-emerald-500 border-white shadow-[0_0_15px_rgba(16,185,129,0.6)] scale-110' 
                : 'bg-emerald-900/40 border-emerald-500/40'
            }`}
          />
        ))}
      </div>
      
      {/* Legend */}
      <div className="absolute bottom-2 right-4 text-[10px] font-mono text-slate-500 opacity-70">
        {activeLayer === 0 && "Step 1: Input Features"}
        {activeLayer === 1 && "Step 2: Hidden Processing"}
        {activeLayer === 2 && "Step 3: Prediction"}
      </div>
    </div>
  );
};