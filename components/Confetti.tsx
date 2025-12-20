
// A lightweight confetti implementation without external dependencies
export const triggerConfetti = () => {
    const canvas = document.createElement('canvas');
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '9999';
    document.body.appendChild(canvas);
  
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
  
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  
    const particles: any[] = [];
    const particleCount = 150;
    const colors = ['#6366f1', '#818cf8', '#10b981', '#f43f5e', '#fbbf24'];
  
    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: canvas.width / 2,
        y: canvas.height / 2,
        vx: (Math.random() - 0.5) * 15,
        vy: (Math.random() - 0.5) * 15 - 5,
        size: Math.random() * 8 + 4,
        color: colors[Math.floor(Math.random() * colors.length)],
        rotation: Math.random() * 360,
        rotationSpeed: (Math.random() - 0.5) * 10,
        opacity: 1
      });
    }
  
    let animationId: number;
  
    const animate = () => {
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
  
      let activeParticles = 0;
  
      particles.forEach(p => {
        if (p.opacity <= 0) return;
        activeParticles++;
  
        p.x += p.vx;
        p.y += p.vy;
        p.vy += 0.2; // Gravity
        p.vx *= 0.96; // Friction
        p.vy *= 0.96;
        p.rotation += p.rotationSpeed;
        p.opacity -= 0.008;
  
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate((p.rotation * Math.PI) / 180);
        ctx.globalAlpha = p.opacity;
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size);
        ctx.restore();
      });
  
      if (activeParticles > 0) {
        animationId = requestAnimationFrame(animate);
      } else {
        cancelAnimationFrame(animationId);
        document.body.removeChild(canvas);
      }
    };
  
    animate();
  };
