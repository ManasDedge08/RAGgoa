import { useCallback, useRef, useState } from "react";

/**
 * Microphone capture, returning one audio blob per press-and-stop cycle.
 *
 * Two browser realities shape this:
 *
 * 1. `navigator.mediaDevices` only exists in a secure context — HTTPS, or
 *    localhost. Opening the page over plain HTTP from another machine's IP
 *    leaves it undefined, and reading `.getUserMedia` off it throws a
 *    TypeError that says nothing useful about the cause.
 * 2. Container support differs. Chrome and Firefox record WebM/Opus; Safari
 *    records MP4/AAC and rejects a WebM mimeType outright. Sarvam accepts
 *    both, so the recorder asks the browser what it can produce rather than
 *    assuming.
 */

const CANDIDATE_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/mpeg",
  "audio/ogg;codecs=opus",
];

const EXTENSIONS: Record<string, string> = {
  "audio/webm": "webm",
  "audio/mp4": "m4a",
  "audio/mpeg": "mp3",
  "audio/ogg": "ogg",
};

export interface Recording {
  blob: Blob;
  filename: string;
}

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  return CANDIDATE_TYPES.find((type) => MediaRecorder.isTypeSupported?.(type));
}

function extensionFor(mimeType: string): string {
  const base = mimeType.split(";")[0];
  return EXTENSIONS[base] ?? "webm";
}

export function useRecorder() {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const resolveRef = useRef<((value: Recording) => void) | null>(null);

  const start = useCallback(async () => {
    setError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setError(
        window.isSecureContext
          ? "This browser does not expose microphone capture."
          : "The microphone needs a secure page. Open this over https, or on the machine " +
            "running it at http://127.0.0.1:8000 — browsers block microphone access on " +
            "plain http from another address. You can still type your question.",
      );
      return;
    }

    const mimeType = pickMimeType();
    if (!mimeType) {
      setError("This browser cannot record audio in a format the server accepts.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(chunksRef.current, { type: mimeType });
        resolveRef.current?.({ blob, filename: `question.${extensionFor(mimeType)}` });
        resolveRef.current = null;
      };
      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
    } catch (err) {
      const name = err instanceof DOMException ? err.name : "";
      if (name === "NotAllowedError") {
        setError("Microphone permission was denied. Allow it for this site and try again.");
      } else if (name === "NotFoundError") {
        setError("No microphone was found on this device.");
      } else {
        setError(
          err instanceof Error ? `Microphone unavailable: ${err.message}` : "Microphone unavailable.",
        );
      }
    }
  }, []);

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder) return Promise.resolve(null);
    const done = new Promise<Recording>((resolve) => {
      resolveRef.current = resolve;
    });
    recorder.stop();
    recorderRef.current = null;
    setRecording(false);
    return done;
  }, []);

  return { recording, error, start, stop };
}
