/**
 * The shack: a beach band behind the masthead and a palm at each page edge.
 *
 * All of it is drawn, not loaded — inline SVG with colours from the page
 * tokens, so the scenery moves with the theme instead of sitting on top of it
 * as a foreign image. In dark mode the same shapes read as the beach at night.
 *
 * Everything here is decoration and carries aria-hidden: the page's meaning is
 * in the measurements, and a screen reader should not have to wade through
 * palm trees to reach them.
 */

/** One frond, built from a spine and leaflets stepped along its normal. */
function frond(angleDeg: number, length: number, key: string, droop: number) {
  const a = (angleDeg * Math.PI) / 180;
  const p0: [number, number] = [0, 0];
  const p1: [number, number] = [Math.cos(a) * length, Math.sin(a) * length + droop];
  const cp: [number, number] = [
    Math.cos(a) * length * 0.55,
    Math.sin(a) * length * 0.55 - droop * 0.7,
  ];

  const at = (t: number): [number, number] => {
    const u = 1 - t;
    return [
      u * u * p0[0] + 2 * u * t * cp[0] + t * t * p1[0],
      u * u * p0[1] + 2 * u * t * cp[1] + t * t * p1[1],
    ];
  };
  const tangent = (t: number): [number, number] => {
    const dx = 2 * (1 - t) * (cp[0] - p0[0]) + 2 * t * (p1[0] - cp[0]);
    const dy = 2 * (1 - t) * (cp[1] - p0[1]) + 2 * t * (p1[1] - cp[1]);
    const m = Math.hypot(dx, dy) || 1;
    return [dx / m, dy / m];
  };

  const blades: string[] = [];
  for (let i = 0; i < 12; i += 1) {
    const t = 0.12 + (i / 11) * 0.84;
    const [x, y] = at(t);
    const [tx, ty] = tangent(t);
    const len = 6 + Math.sin(t * Math.PI) * 16;
    for (const side of [1, -1]) {
      const ex = x + -ty * side * len - tx * len * 0.45;
      const ey = y + tx * side * len - ty * len * 0.45;
      blades.push(`M${x.toFixed(1)} ${y.toFixed(1)} L${ex.toFixed(1)} ${ey.toFixed(1)}`);
    }
  }
  const spine = `M0 0 Q${cp[0].toFixed(1)} ${cp[1].toFixed(1)} ${p1[0].toFixed(1)} ${p1[1].toFixed(1)}`;

  return (
    <g key={key}>
      <g fill="none" stroke="var(--palm-dark)" strokeWidth="5" strokeLinecap="round">
        {blades.map((d) => (
          <path key={d} d={d} />
        ))}
      </g>
      <path d={spine} fill="none" stroke="var(--palm-dark)" strokeWidth="4" strokeLinecap="round" />
    </g>
  );
}

/** A coconut palm. `flip` mirrors it for the opposite page edge. */
export function Palm({ flip = false, className }: { flip?: boolean; className?: string }) {
  const fronds = [
    frond(-160, 96, "a", -26),
    frond(-125, 104, "b", -16),
    frond(-88, 92, "c", -10),
    frond(-52, 106, "d", -16),
    frond(-16, 98, "e", -26),
    frond(20, 78, "f", -34),
  ];
  return (
    <svg
      className={className}
      viewBox="0 0 260 520"
      aria-hidden="true"
      focusable="false"
      style={flip ? { transform: "scaleX(-1)" } : undefined}
    >
      {/* Trunk: two strokes, the lighter one offset, so it reads as lit from the sea side. */}
      <path
        d="M96 520 Q104 380 118 268 Q126 208 138 168"
        fill="none"
        stroke="var(--trunk)"
        strokeWidth="20"
        strokeLinecap="round"
      />
      <path
        d="M96 520 Q104 380 118 268 Q126 208 138 168"
        fill="none"
        stroke="var(--trunk-lit)"
        strokeWidth="7"
        strokeLinecap="round"
        transform="translate(-5 0)"
      />
      <g transform="translate(140 162)">
        {fronds}
        <circle cx="-6" cy="12" r="9" fill="var(--coconut)" />
        <circle cx="12" cy="16" r="8" fill="var(--coconut)" />
        <circle cx="3" cy="26" r="8" fill="var(--coconut)" />
      </g>
    </svg>
  );
}

/**
 * Sky, sun, sea and sand behind the masthead.
 *
 * `preserveAspectRatio="none"` lets the band stretch to any width: the content
 * is horizontal bands and wave lines, none of which distort meaningfully.
 */
export function BeachBand({ className }: { className?: string }) {
  const waves = [0, 1, 2].map((row) => {
    const y = 194 + row * 11;
    const segments = Array.from({ length: 24 }, (_, i) => {
      const x = i * 50 + (row % 2 ? 25 : 0);
      return `M${x} ${y} q12 -7 24 0`;
    }).join(" ");
    return <path key={row} d={segments} fill="none" stroke="var(--sea-line)" strokeWidth="3" strokeLinecap="round" />;
  });

  return (
    <svg
      className={className}
      viewBox="0 0 1200 260"
      preserveAspectRatio="none"
      aria-hidden="true"
      focusable="false"
    >
      <rect x="0" y="0" width="1200" height="186" fill="var(--sky)" />
      <rect x="0" y="186" width="1200" height="46" fill="var(--sea)" />
      <rect x="0" y="232" width="1200" height="28" fill="var(--sand)" />
      {/* Surf line where sea meets sand. */}
      <path d="M0 232 h1200" stroke="var(--foam)" strokeWidth="5" fill="none" />
      {waves}
    </svg>
  );
}

/** The sun. Kept out of the band so it can sit at a fixed size, not stretched. */
export function Sun({ className }: { className?: string }) {
  const rays = Array.from({ length: 12 }, (_, i) => {
    const a = (i / 12) * Math.PI * 2;
    return (
      <line
        key={i}
        x1={50 + Math.cos(a) * 30}
        y1={50 + Math.sin(a) * 30}
        x2={50 + Math.cos(a) * 44}
        y2={50 + Math.sin(a) * 44}
        stroke="var(--turmeric)"
        strokeWidth="5"
        strokeLinecap="round"
      />
    );
  });
  return (
    <svg className={className} viewBox="0 0 100 100" aria-hidden="true" focusable="false">
      {rays}
      <circle cx="50" cy="50" r="23" fill="var(--turmeric)" />
    </svg>
  );
}
