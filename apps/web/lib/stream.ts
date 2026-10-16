"use client";

// Tiny timeline engine for the "streaming" reveal: one clock, every element
// derives what it shows from (t - its start time). No per-element timers.

import { useEffect, useState } from "react";

export const CH = 6; // ms per character while "typing"

export function prefersReducedMotion() {
  return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/** Milliseconds since `key` changed (while `active`); stops ticking after `until`. */
export function useClock(active: boolean, key: unknown, until: number): number {
  const [t, setT] = useState(0);
  useEffect(() => {
    if (!active) {
      setT(0);
      return;
    }
    if (prefersReducedMotion()) {
      setT(Number.MAX_SAFE_INTEGER);
      return;
    }
    // a hidden tab gets no animation frames — just show the end state
    if (document.hidden) {
      setT(Number.MAX_SAFE_INTEGER);
      return;
    }
    // A timer, not requestAnimationFrame: rAF stops completely in hidden or
    // embedded views, a timer only slows down — the reveal always finishes.
    const t0 = performance.now();
    const id = window.setInterval(() => {
      const e = performance.now() - t0;
      setT(e);
      if (e >= until) window.clearInterval(id);
    }, 33);
    const onVis = () => {
      if (!document.hidden) return;
      window.clearInterval(id);
      setT(Number.MAX_SAFE_INTEGER);
    };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [active, key, until]);
  return t;
}

export const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
export const easeOut = (x: number) => 1 - Math.pow(1 - clamp01(x), 3);

/** Progress 0..1 of a segment that starts at `at` and lasts `dur` ms. */
export const prog = (t: number, at: number, dur: number) => clamp01((t - at) / Math.max(1, dur));
