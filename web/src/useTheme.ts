/**
 * Light/dark selection.
 *
 * The reader's OS preference decides which of the two the page opens in; the
 * control then switches between them explicitly. There is no third "system"
 * state: it read as a mode of its own in a two-item toggle, and a reader who
 * wants the page to follow the OS simply does not touch the control.
 *
 * The chosen theme is written to ``data-theme`` on <html>; CSS does the rest.
 * With nothing stored, no attribute is written and the ``prefers-color-scheme``
 * media query applies. The same write happens in an inline script in index.html
 * before first paint, so a dark reader never sees a flash of the light page.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export type Theme = "light" | "dark";

const STORAGE_KEY = "measure-theme";


/** How long the dusk lasts. Matches the CSS; kept here because the class has to
 *  come off again once the colours have arrived. */
const SHIFT_MS = 420;

function apply(theme: Theme, animate: boolean): void {
  const root = document.documentElement;
  if (animate) {
    // Transitions are switched on for the length of the change and no longer.
    // Left on permanently they would follow every hover and focus around the
    // page, and the point here is one moment, not a page that is always fading.
    root.classList.add("theme-shifting");
    window.setTimeout(() => root.classList.remove("theme-shifting"), SHIFT_MS);
  }
  root.setAttribute("data-theme", theme);
}

function stored(): Theme {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (raw === "light" || raw === "dark") return raw;
  // No stored choice: open in whatever the OS is asking for.
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(stored);

  // First run is the page arriving, not a change: fading in from the wrong
  // palette on load would be a flash of the other theme with extra steps.
  const settled = useRef(false);

  useEffect(() => {
    apply(theme, settled.current);
    settled.current = true;
    localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  const cycle = useCallback(() => {
    setTheme((current) => (current === "dark" ? "light" : "dark"));
  }, []);

  return { theme, setTheme, cycle };
}
