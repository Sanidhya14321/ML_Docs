
import React, { useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Points, PointMaterial, Float, MeshDistortMaterial } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';

function NeuralParticles() {
  const ref = useRef<THREE.Points>(null!);
  const { mouse } = useThree();
  const sphere = new Float32Array(3000 * 3);
  for (let i = 0; i < 3000; i++) {
    const r = 2;
    const theta = 2 * Math.PI * Math.random();
    const phi = Math.acos(2 * Math.random() - 1);
    sphere[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    sphere[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    sphere[i * 3 + 2] = r * Math.cos(phi);
  }

  useFrame((state, delta) => {
    ref.current.rotation.x -= delta / 15;
    ref.current.rotation.y -= delta / 20;
    ref.current.position.x = THREE.MathUtils.lerp(ref.current.position.x, mouse.x * 0.2, 0.1);
    ref.current.position.y = THREE.MathUtils.lerp(ref.current.position.y, mouse.y * 0.2, 0.1);
  });

  return (
    <Points ref={ref} positions={sphere} stride={3} frustumCulled={false}>
      <PointMaterial
        transparent
        color="#00f0ff"
        size={0.01}
        sizeAttenuation={true}
        depthWrite={false}
        opacity={0.8}
      />
    </Points>
  );
}

function CentralNode() {
  const mesh = useRef<THREE.Mesh>(null!);
  const { mouse } = useThree();
  
  useFrame((state, delta) => {
    mesh.current.rotation.x += delta * 0.5;
    mesh.current.rotation.y += delta * 0.2;
    mesh.current.position.x = THREE.MathUtils.lerp(mesh.current.position.x, mouse.x * 0.5, 0.05);
    mesh.current.position.y = THREE.MathUtils.lerp(mesh.current.position.y, mouse.y * 0.5, 0.05);
  });

  return (
    <Float speed={2} rotationIntensity={1} floatIntensity={1}>
      <mesh ref={mesh}>
        <torusKnotGeometry args={[0.4, 0.15, 128, 16]} />
        <MeshDistortMaterial
          color="#00f0ff"
          speed={2}
          distort={0.4}
          radius={1}
          wireframe
          emissive="#00f0ff"
          emissiveIntensity={2}
        />
      </mesh>
    </Float>
  );
}

export const Hero3D: React.FC = () => {
  return (
    <div className="absolute inset-0 z-0 opacity-60 pointer-events-none">
      <Canvas camera={{ position: [0, 0, 2] }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <NeuralParticles />
        <CentralNode />
        <EffectComposer>
          <Bloom luminanceThreshold={0.2} luminanceSmoothing={0.9} height={300} />
        </EffectComposer>
      </Canvas>
    </div>
  );
};
