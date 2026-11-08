/**
 * AgentInTheShell Editions Winter '26 — Main Application Orchestrator
 * Boots WebGL Shader Curtain, Lenis + GSAP Scroll Choreography,
 * Interactive Topology Force Engine, and Live SimGym Workbench.
 */

import { initShaderCurtain } from "./shader-curtain.js";
import { initScrollChoreography } from "./scroll-choreography.js";
import { initTopologyEngine } from "./topology-engine.js";
import { initSimGym } from "./simgym-client.js";

document.addEventListener("DOMContentLoaded", () => {
  // 1. Initialize WebGL Shader Curtain
  const burnCanvas = document.getElementById("webgl-burn-canvas");
  const shaderCurtain = initShaderCurtain(burnCanvas);

  // 2. Initialize GSAP ScrollTrigger Choreography
  const choreography = initScrollChoreography({ shaderCurtain });

  // 3. Initialize Interactive Dual-Topology Canvas
  const topologyCanvas = document.getElementById("hero-topology-canvas");
  let topologyEngine = null;
  if (topologyCanvas) {
    topologyEngine = initTopologyEngine(topologyCanvas, {
      initialMode: "centralized",
      onMetricsChange: (metrics) => {
        const titleEl = document.getElementById("board-topology-title");
        if (titleEl) {
          titleEl.textContent = metrics.entropy === "0.000"
            ? "Centralized Star Hub (X / Instagram)"
            : "Decentralized Federated Mesh (ActivityPub / A2A)";
        }
      }
    });

    // Wire topology mode toggle buttons
    const modeBtns = document.querySelectorAll(".mode-toggle-btn");
    modeBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        modeBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const mode = btn.dataset.mode;
        topologyEngine.setMode(mode);
      });
    });
  }

  // 4. Initialize SimGym Interactive Workbench
  const simgymContainer = document.getElementById("simgym");
  if (simgymContainer) {
    initSimGym(simgymContainer, { apiBaseUrl: "http://127.0.0.1:8000" });
  }

  // 5. Da Vinci Wireframe Blueprint Canvas (delicate background circles)
  const daVinciCanvas = document.getElementById("davinci-canvas");
  if (daVinciCanvas) {
    const bgCtx = daVinciCanvas.getContext("2d");
    function resizeBg() {
      daVinciCanvas.width = window.innerWidth;
      daVinciCanvas.height = window.innerHeight;
    }
    resizeBg();
    window.addEventListener("resize", resizeBg, { passive: true });

    let t = 0;
    function drawDaVinci() {
      t += 0.005;
      bgCtx.clearRect(0, 0, daVinciCanvas.width, daVinciCanvas.height);
      const cx = daVinciCanvas.width * 0.72;
      const cy = daVinciCanvas.height * 0.45;

      bgCtx.strokeStyle = "rgba(212, 175, 55, 0.06)";
      bgCtx.lineWidth = 1;
      bgCtx.setLineDash([6, 6]);

      [100, 180, 290, 460, 700].forEach((r, idx) => {
        bgCtx.beginPath();
        bgCtx.arc(cx, cy, r + Math.sin(t + idx) * 3, 0, Math.PI * 2);
        bgCtx.stroke();
      });

      // Caliper coordinate crosshair
      bgCtx.beginPath();
      bgCtx.moveTo(cx - 30, cy);
      bgCtx.lineTo(cx + 30, cy);
      bgCtx.moveTo(cx, cy - 30);
      bgCtx.lineTo(cx, cy + 30);
      bgCtx.stroke();
      bgCtx.setLineDash([]);

      requestAnimationFrame(drawDaVinci);
    }
    drawDaVinci();
  }

  console.log("✅ [AgentInTheShell Editions Winter '26] System online.");
});
