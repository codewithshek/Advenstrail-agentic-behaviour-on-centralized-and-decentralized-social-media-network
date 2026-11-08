/**
 * AgentInTheShell Editions Winter '26 — Interactive Dual-Topology Force Graph
 * Renders Centralized Star Hub (H = 0.0) vs Federated ActivityPub Pods (H = 1.606)
 * with animated packet pulses, draggable nodes, and IntersectionObserver pausing.
 */

export function initTopologyEngine(canvasEl, { initialMode = "centralized", onMetricsChange = null } = {}) {
  if (!canvasEl) return { setMode: () => {}, destroy: () => {} };

  const ctx = canvasEl.getContext("2d");
  let width = (canvasEl.width = canvasEl.offsetWidth || 800);
  let height = (canvasEl.height = canvasEl.offsetHeight || 440);

  let currentMode = initialMode;
  let animId = null;
  let isVisible = true;
  let draggedNode = null;

  // Pod definitions for Federated mode
  const pods = [
    { name: "@rogue.social", color: "#f87171", cx: 0.24, cy: 0.52, r: 100 },
    { name: "@mastodon.social", color: "#a78bfa", cx: 0.58, cy: 0.42, r: 110 },
    { name: "@fosstodon.org", color: "#62e6f0", cx: 0.82, cy: 0.58, r: 90 }
  ];

  // Generate nodes based on mode
  let nodes = [];
  let edges = [];
  let packets = [];

  function buildGraph(mode) {
    nodes = [];
    edges = [];
    packets = [];

    if (mode === "centralized") {
      // Central Star Hub (@influencer_hub)
      const hub = {
        id: "hub",
        label: "@influencer_hub",
        x: width * 0.5,
        y: height * 0.5,
        vx: 0,
        vy: 0,
        r: 16,
        color: "#d4af37",
        isHub: true,
        pod: "x.com"
      };
      nodes.push(hub);

      // 16 Surrounding leaf / amplifier bot nodes
      for (let i = 0; i < 16; i++) {
        const angle = (i / 16) * Math.PI * 2;
        const dist = 130 + (i % 3) * 35;
        const leaf = {
          id: `bot_${i}`,
          label: `@bot_amp_${i + 1}`,
          x: hub.x + Math.cos(angle) * dist,
          y: hub.y + Math.sin(angle) * dist,
          vx: 0,
          vy: 0,
          r: i < 6 ? 8 : 5,
          color: i < 6 ? "#f87171" : "#c8c7be",
          isHub: false,
          pod: "x.com"
        };
        nodes.push(leaf);
        edges.push({ source: leaf, target: hub, isCross: false });
      }

      // Add packets
      for (let p = 0; p < 8; p++) {
        packets.push({
          edgeIndex: p % edges.length,
          t: Math.random(),
          speed: 0.008 + Math.random() * 0.008
        });
      }

      if (onMetricsChange) {
        onMetricsChange({ entropy: "0.000", crossRatio: "0.0%", gini: "0.584", bridge: "0.000" });
      }
    } else {
      // Federated ActivityPub Mesh
      pods.forEach((pod, pIdx) => {
        const podNodesCount = pIdx === 0 ? 5 : 7;
        const pcx = width * pod.cx;
        const pcy = height * pod.cy;

        for (let i = 0; i < podNodesCount; i++) {
          const angle = (i / podNodesCount) * Math.PI * 2;
          const dist = 30 + Math.random() * (pod.r - 40);
          const node = {
            id: `fed_${pIdx}_${i}`,
            label: `${i === 0 ? "@gateway" : "@peer"}${pod.name}`,
            x: pcx + Math.cos(angle) * dist,
            y: pcy + Math.sin(angle) * dist,
            vx: 0,
            vy: 0,
            r: i === 0 ? 10 : 5,
            color: pod.color,
            isHub: i === 0,
            pod: pod.name,
            podIndex: pIdx
          };
          nodes.push(node);
        }
      });

      // Internal pod edges
      nodes.forEach((n1, idx1) => {
        nodes.forEach((n2, idx2) => {
          if (idx2 > idx1 && n1.pod === n2.pod) {
            const dist = Math.hypot(n1.x - n2.x, n1.y - n2.y);
            if (dist < 85) {
              edges.push({ source: n1, target: n2, isCross: false });
            }
          }
        });
      });

      // Malicious Cross-Instance Bridge Edges from @rogue.social
      const rogueGateway = nodes.find((n) => n.podIndex === 0 && n.isHub) || nodes[0];
      const mastoGateway = nodes.find((n) => n.podIndex === 1 && n.isHub) || nodes[5];
      const fosstoGateway = nodes.find((n) => n.podIndex === 2 && n.isHub) || nodes[12];

      edges.push({ source: rogueGateway, target: mastoGateway, isCross: true });
      edges.push({ source: rogueGateway, target: fosstoGateway, isCross: true });

      // Cross-bridge packets
      for (let p = 0; p < 12; p++) {
        packets.push({
          edgeIndex: edges.length - 2 + (p % 2),
          t: Math.random(),
          speed: 0.007 + Math.random() * 0.006
        });
      }

      if (onMetricsChange) {
        onMetricsChange({ entropy: "1.606", crossRatio: "57.2%", gini: "0.336", bridge: "1.000" });
      }
    }
  }

  buildGraph(currentMode);

  // Drag interaction
  canvasEl.addEventListener("pointerdown", (e) => {
    const rect = canvasEl.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    draggedNode = nodes.find((n) => Math.hypot(n.x - mx, n.y - my) <= n.r + 8) || null;
  });

  window.addEventListener("pointermove", (e) => {
    if (!draggedNode) return;
    const rect = canvasEl.getBoundingClientRect();
    draggedNode.x = e.clientX - rect.left;
    draggedNode.y = e.clientY - rect.top;
  });

  window.addEventListener("pointerup", () => {
    draggedNode = null;
  });

  // Render loop
  function draw() {
    if (!isVisible) return;
    ctx.clearRect(0, 0, width, height);

    // If federated, draw convex pod hulls
    if (currentMode === "federated") {
      pods.forEach((pod) => {
        const pcx = width * pod.cx;
        const pcy = height * pod.cy;
        ctx.beginPath();
        ctx.arc(pcx, pcy, pod.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255, 255, 255, 0.02)";
        ctx.fill();
        ctx.strokeStyle = pod.color;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label
        ctx.font = "11px 'Space Grotesk', sans-serif";
        ctx.fillStyle = pod.color;
        ctx.fillText(pod.name, pcx - 40, pcy - pod.r - 8);
      });
    }

    // Draw Edges
    edges.forEach((edge) => {
      ctx.beginPath();
      ctx.moveTo(edge.source.x, edge.source.y);
      ctx.lineTo(edge.target.x, edge.target.y);
      if (edge.isCross) {
        ctx.strokeStyle = "#f87171";
        ctx.lineWidth = 2.2;
        ctx.setLineDash([5, 5]);
      } else {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
        ctx.lineWidth = 1;
        ctx.setLineDash([]);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // Draw Packets
    packets.forEach((pkt) => {
      pkt.t += pkt.speed;
      if (pkt.t > 1) pkt.t = 0;
      const edge = edges[pkt.edgeIndex];
      if (!edge) return;
      const px = edge.source.x + (edge.target.x - edge.source.x) * pkt.t;
      const py = edge.source.y + (edge.target.y - edge.source.y) * pkt.t;
      ctx.beginPath();
      ctx.arc(px, py, edge.isCross ? 3.5 : 2, 0, Math.PI * 2);
      ctx.fillStyle = edge.isCross ? "#f87171" : "#d4af37";
      ctx.shadowColor = edge.isCross ? "#f87171" : "#d4af37";
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    // Draw Nodes
    nodes.forEach((node) => {
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
      ctx.fillStyle = node.color;
      ctx.shadowColor = node.color;
      ctx.shadowBlur = node.isHub ? 12 : 4;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    animId = requestAnimationFrame(draw);
  }

  // IntersectionObserver to pause loop when offscreen
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      isVisible = entry.isIntersecting;
      if (isVisible && !animId) {
        animId = requestAnimationFrame(draw);
      } else if (!isVisible && animId) {
        cancelAnimationFrame(animId);
        animId = null;
      }
    });
  }, { threshold: 0.1 });

  observer.observe(canvasEl);
  animId = requestAnimationFrame(draw);

  function onResize() {
    width = canvasEl.width = canvasEl.offsetWidth || 800;
    height = canvasEl.height = canvasEl.offsetHeight || 440;
    buildGraph(currentMode);
  }
  window.addEventListener("resize", onResize, { passive: true });

  return {
    setMode(mode) {
      if (mode !== currentMode) {
        currentMode = mode;
        buildGraph(mode);
      }
    },
    destroy() {
      if (animId) cancelAnimationFrame(animId);
      observer.disconnect();
      window.removeEventListener("resize", onResize);
    }
  };
}
