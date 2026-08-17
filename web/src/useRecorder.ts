import { useCallback, useRef, useState } from "react";

/** Microphone capture returning one webm/opus blob per press-and-stop cycle. */
export function useRecorder() {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const resolveRef = useRef<((blob: Blob) => void) | null>(null);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        resolveRef.current?.(blob);
        resolveRef.current = null;
      };
      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
    } catch (err) {
      setError(
        err instanceof Error
          ? `Microphone unavailable: ${err.message}`
          : "Microphone unavailable.",
      );
    }
  }, []);

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder) return Promise.resolve(null);
    const done = new Promise<Blob>((resolve) => {
      resolveRef.current = resolve;
    });
    recorder.stop();
    recorderRef.current = null;
    setRecording(false);
    return done;
  }, []);

  return { recording, error, start, stop };
}
