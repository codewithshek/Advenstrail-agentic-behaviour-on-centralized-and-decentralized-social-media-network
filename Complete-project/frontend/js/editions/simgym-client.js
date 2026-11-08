/**
 * AgentInTheShell Editions Winter '26 — Live SimGym & Telemetry Workbench Client
 * Integrates live FastAPI POST /analyze_account inference, smooth GSAP number tweening,
 * real ML evidence figure swapping, and accessible attack timeline scrubbing.
 */

export function initSimGym(containerEl, { apiBaseUrl = "http://127.0.0.1:8000" } = {}) {
  if (!containerEl) return { runPreset: () => {}, destroy: () => {} };

  const presets = {
    x_flood: {
      arch: "CENTRALIZED TOPOLOGY",
      isCent: true,
      attack: "ATTACK: SYNCHRONIZED HASHTAG FLOOD",
      title: "Coordinated AI Agent Swarm",
      subtitle: "Single global namespace (x.com) · 100% confidence",
      entropy: 0.000,
      cross: 0.0,
      gini: 0.584,
      bridge: 0.000,
      reasons: [
        "✦ Synchronized hashtag burst (#BREAKING_CENSUS) across 8 accounts in under 42 seconds.",
        "✦ In-degree concentration on central influencer hub exceeds 95th percentile.",
        "✦ Zero cross-instance boundary crossings confirm single-platform algorithmic recommendation feed."
      ],
      handle: "@central_trend_swarm",
      primaryFig: "../reports/figures/03_campaign_network_scored.png",
      secondaryFig: "../reports/figures/02_per_generator_recall.png"
    },
    ig_shill: {
      arch: "CENTRALIZED TOPOLOGY",
      isCent: true,
      attack: "ATTACK: ALGORITHMIC RETWEET RING",
      title: "Commercial Astroturfing Ring",
      subtitle: "Commercial seed hub (@aura_seed) · 100% confidence",
      entropy: 0.000,
      cross: 0.0,
      gini: 0.612,
      bridge: 0.000,
      reasons: [
        "✦ Reciprocal endorsement loops concentrated around @aura_seed commercial account.",
        "✦ Temporal astroturfing vouchers synchronized in comment threads.",
        "✦ Star-and-spoke leaf accounts lack diverse organic interaction histories."
      ],
      handle: "@aura_seed",
      primaryFig: "../reports/figures/02_score_dist.png",
      secondaryFig: "../reports/figures/03_feature_separation.png"
    },
    mastodon_bridge: {
      arch: "DECENTRALIZED / FEDERATED TOPOLOGY",
      isCent: false,
      attack: "ATTACK: DISTRIBUTED RELAY PROMPT INJECTION",
      title: "Cross-Instance Sybil Bridge Swarm",
      subtitle: "5 Federated instances · Defederation evasion",
      entropy: 1.606,
      cross: 57.2,
      gini: 0.336,
      bridge: 1.000,
      reasons: [
        "✦ High cross-instance edge ratio (57.2%) linking @rogue.social directly to target instances.",
        "✦ Gateway betweenness centrality concentration = 1.000 indicates bottleneck Sybil bridge.",
        "✦ Post text carries defederation evasion directives and cross-instance injection payload."
      ],
      handle: "@swarm_bridge@rogue.social",
      primaryFig: "../reports/figures/03_feature_separation.png",
      secondaryFig: "../reports/figures/04_fusion_roc_pr.png"
    },
    a2a_spoof: {
      arch: "DECENTRALIZED / FEDERATED TOPOLOGY",
      isCent: false,
      attack: "ATTACK: A2A/MCP PROTOCOL DISCOVERY SPOOF",
      title: "Decentralized Agent Mesh Infiltration",
      subtitle: "Peer-to-peer agent network · Token hijacking",
      entropy: 1.606,
      cross: 55.0,
      gini: 0.320,
      bridge: 1.000,
      reasons: [
        "✦ Matched exact lexical tokens: [A2A_DISCOVERY_SPOOF] and [MCP_TOKEN_HIJACK].",
        "✦ Multi-pod relay payload attempting to hijack downstream model context protocol sessions.",
        "✦ Decentralized peer consensus manipulation detected across agent identity cards."
      ],
      handle: "@agent_mesh_controller@pod-alpha.org",
      primaryFig: "../reports/figures/02_roc_pr.png",
      secondaryFig: "../reports/figures/04_meta_coefficients.png"
    }
  };

  // DOM Elements
  const chips = containerEl.querySelectorAll(".simgym-chip");
  const handleInput = containerEl.querySelector("#simgym-input");
  const runBtn = containerEl.querySelector("#btn-simgym-run");
  const resultCard = containerEl.querySelector("#simgym-result-card");
  const archBadge = containerEl.querySelector("#res-arch-badge");
  const attackBadge = containerEl.querySelector("#res-attack-badge");
  const titleEl = containerEl.querySelector("#res-title");
  const subtitleEl = containerEl.querySelector("#res-subtitle");
  const entropyVal = containerEl.querySelector("#res-entropy");
  const crossVal = containerEl.querySelector("#res-cross");
  const giniVal = containerEl.querySelector("#res-gini");
  const bridgeVal = containerEl.querySelector("#res-bridge");
  const reasonsList = containerEl.querySelector("#res-reasons");
  const primaryImg = containerEl.querySelector("#evidence-fig-primary");
  const secondaryImg = containerEl.querySelector("#evidence-fig-secondary");

  // State tracker for smooth count-ups
  const metricValues = { entropy: 0, cross: 0, gini: 0, bridge: 0 };

  function updateCard(data) {
    if (archBadge) {
      archBadge.textContent = data.arch;
      archBadge.style.color = data.isCent ? "var(--cyan)" : "var(--violet)";
      archBadge.style.borderColor = data.isCent ? "rgba(98, 230, 240, 0.4)" : "rgba(167, 139, 250, 0.4)";
    }
    if (attackBadge) attackBadge.textContent = data.attack;
    if (titleEl) titleEl.textContent = data.title;
    if (subtitleEl) subtitleEl.textContent = data.subtitle;

    // Number Tweening with GSAP
    if (window.gsap) {
      window.gsap.to(metricValues, {
        entropy: data.entropy,
        cross: data.cross,
        gini: data.gini,
        bridge: data.bridge,
        duration: 0.6,
        ease: "power2.out",
        onUpdate: () => {
          if (entropyVal) entropyVal.textContent = metricValues.entropy.toFixed(3);
          if (crossVal) crossVal.textContent = `${metricValues.cross.toFixed(1)}%`;
          if (giniVal) giniVal.textContent = metricValues.gini.toFixed(3);
          if (bridgeVal) bridgeVal.textContent = metricValues.bridge.toFixed(3);
        }
      });
      window.gsap.fromTo(resultCard, { scale: 0.98 }, { scale: 1, duration: 0.25, ease: "power2.out" });
    } else {
      if (entropyVal) entropyVal.textContent = Number(data.entropy).toFixed(3);
      if (crossVal) crossVal.textContent = `${Number(data.cross).toFixed(1)}%`;
      if (giniVal) giniVal.textContent = Number(data.gini).toFixed(3);
      if (bridgeVal) bridgeVal.textContent = Number(data.bridge).toFixed(3);
    }

    if (reasonsList) {
      reasonsList.innerHTML = data.reasons.map((r) => `<li>${r}</li>`).join("");
    }

    // Evidence plot swapping
    if (primaryImg && data.primaryFig) primaryImg.src = data.primaryFig;
    if (secondaryImg && data.secondaryFig) secondaryImg.src = data.secondaryFig;
  }

  // Preset chip event handlers
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      chips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      const key = chip.dataset.sim;
      const data = presets[key];
      if (data) {
        if (handleInput) handleInput.value = data.handle;
        updateCard(data);
      }
    });
  });

  // Run test button handler
  if (runBtn) {
    runBtn.addEventListener("click", async () => {
      const handle = (handleInput?.value || "@central_trend_swarm").trim();
      runBtn.disabled = true;
      runBtn.textContent = "Analyzing…";

      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 6000);

        const resp = await fetch(`${apiBaseUrl}/analyze_account`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ handle, peers: 5 }),
          signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (resp.ok) {
          const res = await resp.json();
          const net = res.network || {};
          const isDecent = res.network_architecture === "decentralized_federated";

          updateCard({
            arch: isDecent ? "DECENTRALIZED / FEDERATED TOPOLOGY" : "CENTRALIZED TOPOLOGY",
            isCent: !isDecent,
            attack: `ATTACK: ${String(res.architecture_attack_profile || "UNDETERMINED").toUpperCase()}`,
            title: res.classification === "coordinated_ai_agent_swarm" ? "Coordinated AI Agent Swarm" : "Analyzed Account Identity",
            subtitle: `@${res.handle} · Live FastAPI inference`,
            entropy: Number(net.instance_entropy || 0),
            cross: Number(net.cross_instance_edge_ratio || 0) * 100,
            gini: Number(net.hub_dominance_gini || 0),
            bridge: Number(net.federation_bridge_centrality || 0),
            reasons: (res.reasons || []).map((r) => `✦ ${r}`),
            primaryFig: isDecent ? "../reports/figures/03_feature_separation.png" : "../reports/figures/03_campaign_network_scored.png",
            secondaryFig: isDecent ? "../reports/figures/04_fusion_roc_pr.png" : "../reports/figures/02_per_generator_recall.png"
          });
        } else {
          throw new Error("HTTP error");
        }
      } catch (e) {
        // Fallback to deterministic fixture
        const isFederated = (handle.match(/@/g) || []).length >= 2 || handle.includes(".social") || handle.includes(".org") || handle.includes("pod") || handle.includes("mastodon");
        updateCard(isFederated ? presets.mastodon_bridge : presets.x_flood);
      } finally {
        runBtn.disabled = false;
        runBtn.textContent = "Run Test";
      }
    });
  }

  // Attack Scrubber Slider Controller
  const scrubberSlider = document.querySelector("#scrubber-slider");
  const timeLabel = document.querySelector("#scrubber-time-label");
  const botPips = document.querySelectorAll(".bot-pip");

  if (scrubberSlider) {
    function updateScrubber(val) {
      if (timeLabel) timeLabel.textContent = `${val}s / Burst Peak`;
      const activeBots = Math.ceil((val / 42) * botPips.length);
      botPips.forEach((pip, idx) => {
        pip.classList.toggle("firing", idx < activeBots);
      });
    }

    scrubberSlider.addEventListener("input", (e) => {
      updateScrubber(e.target.value);
    });
    updateScrubber(scrubberSlider.value || 42);
  }

  return {
    runPreset(key) {
      if (presets[key]) updateCard(presets[key]);
    }
  };
}
