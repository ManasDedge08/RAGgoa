/**
 * Light/dark selection, with "system" as a real third state rather than an
 * absence of one.
 *
 * Two-state toggles quietly override the reader's OS setting forever after the
 * first click. Keeping "system" selectable means the page can be handed back to
 * the operating system, which is what most readers want most of the time.
 *
 * The chosen theme is written to ``data-theme`` on <html>; CSS does the rest.
 * "system" writes no attribute at all, letting the ``prefers-color-scheme``
 * media query apply. The same write happens in an inline script in index.html
 * before first paint, so a dark reader never sees a flash of the light page.
 */

import { useCallback, useEffect, useState } from "react";

export type Theme = "system" | "light" | "dark";

const STORAGE_KEY = "measure-theme";
const ORDER: Theme[] = ["system", "light", "dark"];

function apply(theme: Theme): void {
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
}

function stored(): Theme {
  const raw = localStorage.getItem(STORAGE_KEY);
  return raw === "light" || raw === "dark" ? raw : "system";
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(stored);

  useEffect(() => {
    apply(theme);
    if (theme === "system") localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  // Follow the OS while on "system": without this the page keeps whatever the
  // OS was at load time until a reload.
  useEffect(() => {
    if (theme !== "system") return;
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => apply("system");
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, [theme]);

  const cycle = useCallback(() => {
    setTheme((current) => ORDER[(ORDER.indexOf(current) + 1) % ORDER.length]);
  }, []);

  return { theme, setTheme, cycle };
}
