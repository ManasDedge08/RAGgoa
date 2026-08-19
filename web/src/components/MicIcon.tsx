/**
 * The mic glyph on the "ask out loud" button.
 *
 * Drawn rather than loaded, like the rest of the page's marks, so it takes its
 * colour from the button it sits in and follows the theme without a second
 * asset. Capsule, arc and stand — the shape reads at 15px, which is all it has.
 */
export function MicIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.1"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <rect x="9" y="2" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <path d="M12 18v3" />
    </svg>
  );
}
