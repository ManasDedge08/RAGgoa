import { useCallback, useRef, useState } from "react";

/**
 * Plays the spoken answer, and copes with autoplay policy.
 *
 * Browsers only allow audio that a user gesture started. The spoken answer
 * arrives seconds after the click that asked the question, by which point the
 * gesture has expired, so `play()` rejects and — if that rejection is
 * swallowed — the demo simply goes quiet with nothing on screen to explain it.
 *
 * Two defences. One audio element is reused for the session and unlocked
 * during the click that starts a turn, which satisfies the policy in Chrome
 * and Safari. If playback is still refused, `blocked` goes true and the UI
 * offers a button, turning a silent failure into one click.
 */

const SILENCE =
  "data:audio/mp3;base64,//uQxAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAACcQCA" +
  "gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgP////////////////////////////////" +
  "8AAAAATGF2YzU4LjEzAAAAAAAAAAAAAAAAJAAAAAAAAAAAAnGZTsFAAAAAAAAAAAAAAAAAAAA=";

export function useAudioPlayer() {
  const elementRef = useRef<HTMLAudioElement | null>(null);
  const pendingRef = useRef<string | null>(null);
  const [blocked, setBlocked] = useState(false);
  const [playing, setPlaying] = useState(false);

  const element = useCallback(() => {
    if (!elementRef.current) {
      const audio = new Audio();
      audio.onplaying = () => setPlaying(true);
      audio.onended = () => setPlaying(false);
      audio.onpause = () => setPlaying(false);
      elementRef.current = audio;
    }
    return elementRef.current;
  }, []);

  /** Call from a click handler, before the answer exists. */
  const unlock = useCallback(() => {
    const audio = element();
    audio.src = SILENCE;
    audio.muted = true;
    void audio
      .play()
      .then(() => {
        audio.pause();
        audio.muted = false;
        setBlocked(false);
      })
      .catch(() => {
        audio.muted = false;
      });
  }, [element]);

  const play = useCallback(
    (audioB64: string) => {
      if (!audioB64) return;
      const audio = element();
      pendingRef.current = audioB64;
      audio.src = `data:audio/mpeg;base64,${audioB64}`;
      void audio
        .play()
        .then(() => setBlocked(false))
        .catch(() => setBlocked(true));
    },
    [element],
  );

  /** Retry from a real click after the browser refused. */
  const playPending = useCallback(() => {
    const audioB64 = pendingRef.current;
    if (!audioB64) return;
    const audio = element();
    audio.src = `data:audio/mpeg;base64,${audioB64}`;
    void audio
      .play()
      .then(() => setBlocked(false))
      .catch(() => setBlocked(true));
  }, [element]);

  const stop = useCallback(() => {
    elementRef.current?.pause();
  }, []);

  return { play, playPending, unlock, stop, blocked, playing };
}
