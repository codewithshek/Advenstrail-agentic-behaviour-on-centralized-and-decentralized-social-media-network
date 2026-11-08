/**
 * AgentInTheShell Editions Winter '26 — WebGL Perlin-Noise Burn Transition Curtain
 * Implements GPU-accelerated chapter dissolve with burning ember edges between
 * Obsidian Dark (#070709) and Warm Renaissance Parchment (#f7f6f0).
 */

export function initShaderCurtain(canvasEl) {
  if (!canvasEl || !window.THREE) {
    console.warn("[ShaderCurtain] Three.js or canvasEl unavailable. Using CSS fallback.");
    return {
      setBurnProgress: () => {},
      setPointer: () => {},
      destroy: () => {}
    };
  }

  let renderer = null;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas: canvasEl,
      alpha: true,
      antialias: false,
      powerPreference: "high-performance"
    });
  } catch (err) {
    console.warn("[ShaderCurtain] WebGL initialization failed:", err);
    return {
      setBurnProgress: () => {},
      setPointer: () => {},
      destroy: () => {}
    };
  }

  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));

  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  // Simplex 2D / FBM Perlin Noise GLSL
  const vertexShader = `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position, 1.0);
    }
  `;

  const fragmentShader = `
    uniform float uProgress;
    uniform float uTime;
    uniform vec2 uPointer;
    uniform vec3 uDarkColor;
    uniform vec3 uParchmentColor;
    uniform vec3 uEmberColor;
    varying vec2 vUv;

    // Classic 2D Perlin-like pseudo-random noise
    vec2 hash(vec2 p) {
      p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
      return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
    }

    float noise(vec2 p) {
      const float K1 = 0.366025404; // (sqrt(3)-1)/2;
      const float K2 = 0.211324865; // (3-sqrt(3))/6;
      vec2 i = floor(p + (p.x + p.y) * K1);
      vec2 a = p - i + (i.x + i.y) * K2;
      vec2 o = (a.x > a.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
      vec2 b = a - o + K2;
      vec2 c = a - 1.0 + 2.0 * K2;
      vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
      vec3 n = h * h * h * h * vec3(dot(a, hash(i + 0.0)), dot(b, hash(i + o)), dot(c, hash(i + 1.0)));
      return dot(n, vec3(70.0));
    }

    float fbm(vec2 p) {
      float f = 0.0;
      float w = 0.5;
      for (int i = 0; i < 4; i++) {
        f += w * noise(p);
        p *= 2.02;
        w *= 0.5;
      }
      return f;
    }

    void main() {
      if (uProgress <= 0.001) {
        gl_FragColor = vec4(uDarkColor, 0.0);
        return;
      }
      if (uProgress >= 0.999) {
        gl_FragColor = vec4(uParchmentColor, 1.0);
        return;
      }

      // Add directional sweep from top-left to bottom-right
      vec2 uv = vUv;
      float sweep = (uv.x * 0.4 + uv.y * 0.6);
      
      // Calculate organic noise threshold
      float n = fbm(uv * 4.5 + vec2(uTime * 0.05, 0.0)) * 0.25;
      float threshold = sweep + n;

      // Burn progress range
      float edge = smoothstep(uProgress - 0.08, uProgress + 0.02, threshold);
      float emberWidth = 0.04;
      float ember = smoothstep(uProgress - emberWidth, uProgress, threshold) - 
                    smoothstep(uProgress, uProgress + emberWidth, threshold);

      // Sparkle modulation on the burning rim
      float sparkle = sin(uTime * 12.0 + (uv.x + uv.y) * 40.0) * 0.5 + 0.5;
      vec3 activeEmberColor = mix(uEmberColor, vec3(1.0, 0.85, 0.3), sparkle);

      // Blend parchment base, burning ember line, and transparent dark overlay
      vec3 color = mix(uParchmentColor, uDarkColor, edge);
      color += activeEmberColor * ember * 1.8;

      float alpha = clamp(1.0 - edge + ember, 0.0, 1.0);
      gl_FragColor = vec4(color, alpha);
    }
  `;

  const uniforms = {
    uProgress: { value: 0.0 },
    uTime: { value: 0.0 },
    uPointer: { value: new THREE.Vector2(0.5, 0.5) },
    uDarkColor: { value: new THREE.Color(0x070709) },
    uParchmentColor: { value: new THREE.Color(0xf7f6f0) },
    uEmberColor: { value: new THREE.Color(0xd4af37) }
  };

  const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms,
    transparent: true,
    depthWrite: false,
    depthTest: false
  });

  const geometry = new THREE.PlaneGeometry(2, 2);
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  let animationFrameId = null;
  let isRunning = false;
  let lastTime = performance.now();

  function render() {
    const now = performance.now();
    uniforms.uTime.value += (now - lastTime) * 0.001;
    lastTime = now;

    renderer.render(scene, camera);

    // Keep loop running if in active transition
    const p = uniforms.uProgress.value;
    if (p > 0.001 && p < 0.999) {
      animationFrameId = requestAnimationFrame(render);
      isRunning = true;
    } else {
      isRunning = false;
    }
  }

  function setBurnProgress(progress) {
    const clamped = Math.max(0, Math.min(1, progress));
    uniforms.uProgress.value = clamped;
    if (!isRunning && clamped > 0.001 && clamped < 0.999) {
      lastTime = performance.now();
      render();
    } else if (clamped <= 0.001 || clamped >= 0.999) {
      renderer.render(scene, camera);
    }
  }

  function setPointer(nx, ny) {
    uniforms.uPointer.value.set(nx, ny);
  }

  function onResize() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.render(scene, camera);
  }

  window.addEventListener("resize", onResize, { passive: true });

  return {
    setBurnProgress,
    setPointer,
    destroy() {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      window.removeEventListener("resize", onResize);
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    }
  };
}
