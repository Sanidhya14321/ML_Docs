import React, { useEffect, useRef, useState, useCallback } from 'react';

export const NeuralNetworkViz: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Refs for nodes to calculate positions
  const inputRefs = useRef<(HTMLDivElement | null)[]>([]);
  const hiddenRefs = useRef<(HTMLDivElement | null)[]>([]);
  const outputRefs = useRef<(HTMLDivElement | null)[]>([]);

  // State for activation animation
  const [activeLayer, setActiveLayer] = useState<0 | 1 | 2>(0);

  // Data layers
  const inputNodes = [1, 2, 3];
  const hiddenNodes = [1, 2, 3, 4];
  const outputNodes = [1, 2];

  // Animation Loop for Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let startTime = performance.now();

    const getCoords = (refs: React.MutableRefObject<(HTMLDivElement | null)[]>) => {
      return refs.current.map(el => {
        if (!el) return { x: 0, y: 0 };
        const rect = el.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        return {
          x: rect.left - containerRect.left + rect.width / 2,
          y: rect.top - containerRect.top + rect.height / 2
        };
      });
    };

    const draw = (time: number) => {
      // Handle Resize
      canvas.width = container.offsetWidth;
      canvas.height = container.offsetHeight;
      
      const inputs = getCoords(inputRefs);
      const hiddens = getCoords(hiddenRefs);
      const outputs = getCoords(outputRefs);

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Cycle time: 3000ms
      // 0-1000ms: Input -> Hidden
      // 1000-2000ms: Hidden -> Output
      // 2000-3000ms: Pause/Reset
      const duration = 3000;
      const progress = (time - startTime) % duration;
      
      // Update React State for Node Glow (roughly aligned with phases)
      if (progress < 1000) setActiveLayer(0);
      else if (progress < 2000) setActiveLayer(1);
      else setActiveLayer(2);

      // --- Helper to draw connections ---
      const drawConnections = (
        layerA: {x: number, y: number}[], 
        layerB: {x: number, y: number}[], 
        phaseStart: number, 
        phaseEnd: number,
        colorStart: string,
        colorEnd: string
      ) => {
        const phaseProgress = Math.max(0, Math.min(1, (progress - phaseStart) / (phaseEnd - phaseStart)));
        const isActivePhase = progress >= phaseStart && progress < phaseEnd;

        layerA.forEach(a => {
          layerB.forEach(b => {
            // Static Line
            const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
            grad.addColorStop(0, colorStart);
            grad.addColorStop(1, colorEnd);
            
            ctx.beginPath();
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.2;
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();

            // Animated Packet
            if (isActivePhase) {
              const px = a.x + (b.x - a.x) * phaseProgress;
              const py = a.y + (b.y - a.y) * phaseProgress;
              
              ctx.beginPath();
              ctx.fillStyle = '#fff';
              ctx.globalAlpha = 1;
              ctx.shadowBlur = 10;
              ctx.shadowColor = 'white';
              ctx.arc(px, py, 2, 0, Math.PI * 2);
              ctx.fill();
              ctx.shadowBlur = 0; // Reset
            }
          });
        });
      };

      // Draw Input -> Hidden
      drawConnections(inputs, hiddens, 0, 1000, 'rgba(99, 102, 241, 0.5)', 'rgba(148, 163, 184, 0.5)');

      // Draw Hidden -> Output
      drawConnections(hiddens, outputs, 1000, 2000, 'rgba(148, 163, 184, 0.5)', 'rgba(16, 185, 129, 0.5)');

      animationFrameId = requestAnimationFrame(draw);
    };

    animationFrameId = requestAnimationFrame(draw);

    return () => cancelAnimationFrame(animationFrameId);
  }, []);

  return (
    <div ref={containerRef} className="w-full h-64 bg-slate-900 rounded-lg border border-slate-700 flex items-center justify-around relative overflow-hidden p-8 shadow-inner">
      {/* Canvas for Gradient Lines & Particles */}
      <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none z-0" />

      {/* Input Layer */}
      <div className="flex flex-col justify-center space-y-6 z-10">
        <div className="text-xs text-center text-slate-400 mb-2 font-mono uppercase tracking-wider">Input</div>
        {inputNodes.map((n, i) => (
          <div 
            key={`in-${n}`}
            ref={el => { inputRefs.current[i] = el; }}
            className={`w-8 h-8 rounded-full border-2 transition-all duration-300 ${
              activeLayer === 0 
                ? 'bg-indigo-500 border-white shadow-[0_0_20px_rgba(99,102,241,0.8)] scale-110' 
                : 'bg-indigo-900/50 border-indigo-500/50'
            }`}
          />
        ))}
      </div>

      {/* Hidden Layer */}
      <div className="flex flex-col justify-center space-y-4 z-10">
        <div className="text-xs text-center text-slate-400 mb-2 font-mono uppercase tracking-wider">Hidden</div>
        {hiddenNodes.map((n, i) => (
          <div 
            key={`hid-${n}`}
            ref={el => { hiddenRefs.current[i] = el; }}
            className={`w-6 h-6 rounded-full border-2 transition-all duration-300 ${
              activeLayer === 1
                ? 'bg-slate-200 border-white shadow-[0_0_20px_rgba(255,255,255,0.8)] scale-110' 
                : 'bg-slate-800 border-slate-600'
            }`}
          />
        ))}
      </div>

      {/* Output Layer */}
      <div className="flex flex-col justify-center space-y-8 z-10">
        <div className="text-xs text-center text-slate-400 mb-2 font-mono uppercase tracking-wider">Output</div>
        {outputNodes.map((n, i) => (
          <div 
            key={`out-${n}`}
            ref={el => { outputRefs.current[i] = el; }}
            className={`w-8 h-8 rounded-full border-2 transition-all duration-300 ${
              activeLayer === 2
                ? 'bg-emerald-500 border-white shadow-[0_0_20px_rgba(16,185,129,0.8)] scale-110' 
                : 'bg-emerald-900/50 border-emerald-500/50'
            }`}
          />
        ))}
      </div>
      
      {/* Legend / Status Text */}
      <div className="absolute bottom-2 right-4 text-[10px] font-mono text-slate-500">
        {activeLayer === 0 && "Forward Pass: Input Features"}
        {activeLayer === 1 && "Forward Pass: Hidden Activations"}
        {activeLayer === 2 && "Forward Pass: Output Prediction"}
      </div>
    </div>
  );
};
