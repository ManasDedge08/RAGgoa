/**
 * Die-cut stickers in the reference's own style: full colour, ink outline,
 * a thick white cut-line, one highlight, a little cel shading.
 *
 * These deliberately do NOT use the page tokens. A real sticker keeps its own
 * colours whatever it is stuck to — that contrast with the surrounding palette
 * is what makes it read as applied rather than printed, and it is why the
 * white cut-line stays white in dark mode too.
 *
 * ``paint-order: stroke fill`` builds the cut-line: the stroke paints first and
 * the fill covers its inner half, so one path gives both border and shape.
 */

interface Props {
  className?: string;
}

const INK = "#20242c";

/** Wraps a silhouette so the white cut-line hugs the whole sticker at once. */
function Cut({ children }: { children: React.ReactNode }) {
  return (
    <g stroke="#ffffff" strokeWidth="13" strokeLinejoin="round" strokeLinecap="round" paintOrder="stroke fill">
      {children}
    </g>
  );
}

/** A tall drink: ice, mint, a wheel of lime, striped straw. */
export function DrinkSticker({ className }: Props) {
  const glass = "M30 46 L38 138 Q39 148 50 148 L74 148 Q85 148 86 138 L94 46 Z";
  return (
    <svg className={className} viewBox="0 0 130 168" role="img" aria-label="A tall drink">
      <Cut>
        <path d={glass} fill="#e8f4f6" />
        <path d="M78 40 L102 6" stroke="#ffffff" strokeWidth="19" />
        <circle cx="104" cy="44" r="21" fill="#f2c236" />
        <path d="M40 30 Q26 12 44 8 Q52 18 50 32 Z" fill="#3f9c58" />
        <path d="M58 28 Q54 6 72 10 Q72 26 62 34 Z" fill="#57b56f" />
      </Cut>

      {/* Liquid, then ice on top of it. */}
      <path d="M33 74 L38 138 Q39 148 50 148 L74 148 Q85 148 86 138 L91 74 Z" fill="#d8425f" />
      <g fill="#f4fbfc" stroke={INK} strokeWidth="2.5">
        <rect x="42" y="86" width="20" height="20" rx="3" transform="rotate(-12 52 96)" />
        <rect x="63" y="100" width="19" height="19" rx="3" transform="rotate(9 72 109)" />
        <rect x="47" y="114" width="18" height="18" rx="3" transform="rotate(6 56 123)" />
      </g>
      <path d={glass} fill="none" stroke={INK} strokeWidth="3.5" />

      {/* Lime wheel: rind, flesh, segment cuts. */}
      <circle cx="104" cy="44" r="21" fill="none" stroke={INK} strokeWidth="3.5" />
      <circle cx="104" cy="44" r="14" fill="#f7d967" stroke={INK} strokeWidth="2.5" />
      {[0, 45, 90, 135].map((a) => (
        <line
          key={a}
          x1={104 + Math.cos((a * Math.PI) / 180) * 13}
          y1={44 + Math.sin((a * Math.PI) / 180) * 13}
          x2={104 - Math.cos((a * Math.PI) / 180) * 13}
          y2={44 - Math.sin((a * Math.PI) / 180) * 13}
          stroke={INK}
          strokeWidth="2"
        />
      ))}

      {/* Straw, with the stripes that make it read as a straw. */}
      <path d="M78 40 L102 6" stroke="#f0f2f4" strokeWidth="9" strokeLinecap="round" />
      <path d="M78 40 L102 6" stroke={INK} strokeWidth="3" strokeLinecap="round" fill="none" opacity="0.9" strokeDasharray="0 0" />
      <g stroke="#d8425f" strokeWidth="4.5" strokeLinecap="round">
        <path d="M82 34 L88 26" />
        <path d="M91 25 L97 17" />
      </g>

      {/* Mint. */}
      <path d="M40 30 Q26 12 44 8 Q52 18 50 32 Z" fill="#3f9c58" stroke={INK} strokeWidth="3" />
      <path d="M58 28 Q54 6 72 10 Q72 26 62 34 Z" fill="#57b56f" stroke={INK} strokeWidth="3" />
      <path d="M44 28 Q40 18 44 10" stroke={INK} strokeWidth="2" fill="none" />
      <path d="M62 32 Q62 20 68 12" stroke={INK} strokeWidth="2" fill="none" />

      <path d="M45 96 L49 132" stroke="#ffffff" strokeWidth="5" strokeLinecap="round" opacity="0.55" />
    </svg>
  );
}

/** A hibiscus. Five petals, a long style, one leaf. */
export function HibiscusSticker({ className }: Props) {
  const petal = "M0 0 Q34 -30 58 -6 Q40 26 0 16 Z";
  const petals = [0, 72, 144, 216, 288].map((a) => (
    <path key={a} d={petal} transform={`rotate(${a}) translate(6 0)`} />
  ));
  return (
    <svg className={className} viewBox="0 0 160 160" role="img" aria-label="A hibiscus flower">
      <g transform="translate(80 84)">
        <Cut>
          <g fill="#e14b4b">{petals}</g>
          <path d="M-6 -6 L44 -62" stroke="#ffffff" strokeWidth="16" />
        </Cut>
        <g fill="#e14b4b" stroke={INK} strokeWidth="3.5">
          {petals}
        </g>
        {/* Veins, drawn once per petal at its own angle. */}
        <g stroke="#b8303a" strokeWidth="2.5" fill="none">
          {[0, 72, 144, 216, 288].map((a) => (
            <path key={a} d="M6 2 Q30 -6 46 -8" transform={`rotate(${a})`} />
          ))}
        </g>
        <path d="M-4 -4 L42 -60" stroke="#f2c236" strokeWidth="6" strokeLinecap="round" />
        <circle cx="44" cy="-62" r="9" fill="#f2c236" stroke={INK} strokeWidth="3" />
        <circle cx="0" cy="0" r="11" fill="#f7d967" stroke={INK} strokeWidth="3.5" />
      </g>
    </svg>
  );
}

/** A measure of beer, in the same die-cut style as the rest. */
export function MugSticker({ className }: Props) {
  const body = "M26 44 L34 142 Q35 152 46 152 L84 152 Q95 152 96 142 L104 44 Z";
  const handle = "M104 66 Q134 66 134 96 Q134 124 100 124";
  return (
    <svg className={className} viewBox="0 0 152 172" role="img" aria-label="A measure of beer">
      <Cut>
        <path d={handle} fill="none" stroke="#ffffff" strokeWidth="30" />
        <path d={body} fill="#f0a921" />
        <path d="M22 34 Q30 16 46 24 Q56 10 72 20 Q92 12 100 30 Q100 44 62 44 Q24 44 22 34 Z" fill="#fdfaf2" />
      </Cut>
      <path d={handle} fill="none" stroke="#f0a921" strokeWidth="15" />
      <path d={handle} fill="none" stroke={INK} strokeWidth="3.5" />
      <path d={body} fill="#f0a921" />
      <path d={body} fill="none" stroke={INK} strokeWidth="3.5" />
      {/* Bubbles. */}
      <g fill="#fdd67a" opacity="0.85">
        <circle cx="52" cy="86" r="5" />
        <circle cx="72" cy="106" r="4" />
        <circle cx="58" cy="124" r="3.5" />
      </g>
      <path d="M22 34 Q30 16 46 24 Q56 10 72 20 Q92 12 100 30 Q100 44 62 44 Q24 44 22 34 Z" fill="#fdfaf2" stroke={INK} strokeWidth="3.5" />
      <path d="M40 60 L45 132" stroke="#ffffff" strokeWidth="6" strokeLinecap="round" opacity="0.6" />
    </svg>
  );
}
