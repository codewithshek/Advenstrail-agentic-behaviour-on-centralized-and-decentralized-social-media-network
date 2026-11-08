/**
 * AgentInTheShell Editions Winter '26 — 5-Primitive GSAP ScrollTrigger Choreography Engine
 * Implements Lenis smooth scroll, 3-layer depth parallax, multi-beat pinned stages,
 * WebGL noise-burn scrubs, Flip transitions, and 3D card tilt physics.
 */

export function initScrollChoreography({ shaderCurtain }) {
  if (!window.gsap || !window.ScrollTrigger) {
    console.warn("[ScrollChoreography] GSAP or ScrollTrigger unavailable.");
    return { lenis: null, refresh: () => {}, destroy: () => {} };
  }

  gsap.registerPlugin(ScrollTrigger);
  if (window.Flip) {
    gsap.registerPlugin(Flip);
  }

  // 1. Lenis Smooth Scrolling Foundation
  let lenis = null;
  if (window.Lenis && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    lenis = new Lenis({
      lerp: 0.1,
      wheelMultiplier: 1.0,
      smoothWheel: true,
      syncTouch: false
    });

    lenis.on("scroll", ScrollTrigger.update);
    gsap.ticker.add((time) => lenis.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);
  }

  const mm = gsap.matchMedia();

  // 2. Desktop Choreography (>= 1024px)
  mm.add("(min-width: 1024px)", () => {
    // Primitive 1: 3-Layer Depth Sandwich on Hero (#foundations)
    const heroTl = gsap.timeline({
      scrollTrigger: {
        trigger: "#foundations",
        start: "top top",
        end: "bottom top",
        scrub: 0.8
      }
    });

    heroTl.to("#hero-depth-sandwich .layer-back", {
      yPercent: -14,
      scale: 1.05,
      ease: "none"
    }, 0);

    heroTl.to("#hero-depth-sandwich .layer-mid", {
      yPercent: 0,
      ease: "none"
    }, 0);

    heroTl.to("#hero-depth-sandwich .layer-front", {
      yPercent: -36,
      rotationY: 4,
      ease: "none"
    }, 0);

    // Primitive 2: Multi-Beat Pinned Scene Timeline in Chapter 02 (#centralized)
    const centPinTl = gsap.timeline({
      scrollTrigger: {
        trigger: "#centralized-pin-stage",
        start: "top top",
        end: "+=260%",
        pin: true,
        scrub: 0.8,
        anticipatePin: 1
      }
    });

    // Beat 1: Split Words & Expand Portal
    centPinTl.to("#centralized-word-left", {
      xPercent: -32,
      opacity: 0.4,
      ease: "power2.inOut"
    }, 0);

    centPinTl.to("#centralized-word-right", {
      xPercent: 32,
      opacity: 0.4,
      ease: "power2.inOut"
    }, 0);

    centPinTl.to("#centralized-portal", {
      clipPath: "inset(0% 0% 0% 0% round 16px)",
      scale: 1.02,
      ease: "power2.inOut"
    }, 0);

    // Beat 2: Fly Evidence Cards across the 3D Stage
    centPinTl.fromTo("#cent-artifact-recall", 
      { xPercent: 120, yPercent: 40, rotationZ: 8, opacity: 0 },
      { xPercent: 0, yPercent: 0, rotationZ: 0, opacity: 1, ease: "power2.out" },
      0.8
    );

    centPinTl.fromTo("#cent-artifact-scrubber",
      { xPercent: -120, yPercent: 60, rotationZ: -6, opacity: 0 },
      { xPercent: 0, yPercent: 0, rotationZ: 0, opacity: 1, ease: "power2.out" },
      1.1
    );

    // Beat 3: Telemetry Badges Elastic Pop
    centPinTl.fromTo(".cent-callout-badge",
      { scale: 0, opacity: 0 },
      { scale: 1, opacity: 1, stagger: 0.15, ease: "back.out(2)" },
      1.8
    );

    // Primitive 1 (Repeated): Chapter 03 Federated Depth Sandwich
    const fedTl = gsap.timeline({
      scrollTrigger: {
        trigger: "#federated",
        start: "top 70%",
        end: "bottom top",
        scrub: 0.8
      }
    });

    fedTl.to("#fed-depth-sandwich .layer-back", {
      yPercent: -12,
      ease: "none"
    }, 0);

    fedTl.to("#fed-depth-sandwich .layer-front", {
      yPercent: -32,
      rotationY: -4,
      ease: "none"
    }, 0);

    // Primitive 3: WebGL Perlin-Noise Burn Transition into Chapter 03
    if (shaderCurtain && shaderCurtain.setBurnProgress) {
      ScrollTrigger.create({
        trigger: "#federated",
        start: "top 90%",
        end: "top 20%",
        scrub: true,
        onUpdate: (self) => {
          shaderCurtain.setBurnProgress(self.progress);
        }
      });

      // Burn back to dark as #protocols enters
      ScrollTrigger.create({
        trigger: "#protocols",
        start: "top 80%",
        end: "top 20%",
        scrub: true,
        onUpdate: (self) => {
          shaderCurtain.setBurnProgress(1.0 - self.progress);
        }
      });
    }

    // Primitive 5: 3D Mouse Tilt & Levitation
    const tiltCards = document.querySelectorAll(".floating-artifact, .feature-card-shopify");
    tiltCards.forEach((card) => {
      const setRotX = gsap.quickTo(card, "rotationX", { duration: 0.4, ease: "power3.out" });
      const setRotY = gsap.quickTo(card, "rotationY", { duration: 0.4, ease: "power3.out" });

      card.addEventListener("pointermove", (e) => {
        const rect = card.getBoundingClientRect();
        const nx = (e.clientX - rect.left) / rect.width - 0.5;
        const ny = (e.clientY - rect.top) / rect.height - 0.5;
        setRotY(nx * 14);
        setRotX(-ny * 14);
      });

      card.addEventListener("pointerleave", () => {
        setRotX(0);
        setRotY(0);
      });
    });
  });

  // 3. Mobile Fallback (< 1024px)
  mm.add("(max-width: 1023px)", () => {
    // Smooth simple vertical parallax without heavy 300vh pins
    gsap.utils.toArray(".chapter-section").forEach((sec) => {
      gsap.from(sec.querySelector(".chapter-header-large"), {
        scrollTrigger: {
          trigger: sec,
          start: "top 85%",
          toggleActions: "play none none reverse"
        },
        y: 40,
        opacity: 0,
        duration: 0.8,
        ease: "power2.out"
      });
    });
  });

  // 4. Scrollspy Active Chapter Tracking for Sidebar & Mobile Pillbar
  const sections = document.querySelectorAll(".chapter-section, .hero-chapter");
  const desktopLinks = document.querySelectorAll(".chapter-link");
  const mobilePills = document.querySelectorAll(".mobile-pill-item");

  sections.forEach((sec) => {
    ScrollTrigger.create({
      trigger: sec,
      start: "top 45%",
      end: "bottom 45%",
      onToggle: (self) => {
        if (self.isActive) {
          const id = sec.id;
          desktopLinks.forEach((link) => {
            link.classList.toggle("active", link.getAttribute("href") === `#${id}`);
          });
          mobilePills.forEach((pill) => {
            pill.classList.toggle("active", pill.getAttribute("href") === `#${id}`);
          });
        }
      }
    });
  });

  return {
    lenis,
    refresh() {
      ScrollTrigger.refresh();
    },
    destroy() {
      mm.revert();
      if (lenis) lenis.destroy();
      ScrollTrigger.getAll().forEach((st) => st.kill());
    }
  };
}
