"""
Hybrid Text-to-Speech (TTS) Service for MediScan AI.
Supports 8 Indian & International languages:
English, Telugu, Hindi, Tamil, Kannada, Malayalam, Bengali, Marathi.

Provides:
1. Online Mode: Google Neural Text-to-Speech (gTTS) with native accents.
2. Offline Mode: Local acoustic synthesis / system voice engine.
3. Hybrid Mode: Online HD preferred with transparent offline fallback.
"""

import io
import logging
import math
import struct
import wave
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("TTSService")


@dataclass
class TTSResult:
    audio_bytes: bytes
    mime_type: str  # "audio/mp3" | "audio/wav"
    engine_used: str  # "online_hd_gtts" | "offline_local_engine" | "offline_acoustic_chime"
    is_fallback: bool
    language_code: str
    error_message: Optional[str] = None


class TTSService:
    """
    Hybrid Multilingual Speech Synthesizer.
    """

    SUPPORTED_LANGUAGES = ["en", "te", "hi", "ta", "kn", "ml", "bn", "mr"]

    def __init__(self, default_timeout_sec: int = 7):
        self.timeout_sec = default_timeout_sec

    def synthesize(
        self,
        text: str,
        lang_code: str = "en",
        mode: str = "auto",  # "auto" | "online" | "offline"
    ) -> TTSResult:
        """
        Synthesize speech from text in target language.

        mode options:
        - 'auto': Try online HD first, fall back to offline if network fails.
        - 'online': Force online HD only.
        - 'offline': Force offline engine only.
        """
        if not text or not text.strip():
            empty_wav = self._generate_silence_wav()
            return TTSResult(
                audio_bytes=empty_wav,
                mime_type="audio/wav",
                engine_used="none",
                is_fallback=False,
                language_code=lang_code,
            )

        lang_code = lang_code if lang_code in self.SUPPORTED_LANGUAGES else "en"

        # 1. If user chose offline only
        if mode == "offline":
            return self._synthesize_offline(text, lang_code, is_fallback=False)

        # 2. Try online synthesis (used for 'online' and 'auto')
        online_res = self._synthesize_online(text, lang_code)
        if online_res is not None:
            return online_res

        # 3. If online was strictly forced, return error
        if mode == "online":
            return TTSResult(
                audio_bytes=self._generate_silence_wav(),
                mime_type="audio/wav",
                engine_used="online_hd_gtts",
                is_fallback=False,
                language_code=lang_code,
                error_message="Online voice service unreachable. Please check internet connection or switch to Auto mode.",
            )

        # 4. Fallback to offline mode for 'auto'
        logger.info(f"Falling back to offline TTS for language '{lang_code}'")
        return self._synthesize_offline(text, lang_code, is_fallback=True)

    def _synthesize_online(self, text: str, lang_code: str) -> Optional[TTSResult]:
        """Synthesize via gTTS."""
        try:
            from gtts import gTTS

            # gTTS natively supports all 8 language codes: en, te, hi, ta, kn, ml, bn, mr
            tts = gTTS(text=text, lang=lang_code, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            audio_data = fp.getvalue()

            if audio_data and len(audio_data) > 200:
                return TTSResult(
                    audio_bytes=audio_data,
                    mime_type="audio/mp3",
                    engine_used="online_hd_gtts",
                    is_fallback=False,
                    language_code=lang_code,
                )
        except Exception as e:
            logger.warning(f"gTTS online synthesis failed: {e}")

        return None

    def _synthesize_offline(self, text: str, lang_code: str, is_fallback: bool = False) -> TTSResult:
        """
        Offline synthesis engine.
        Attempts local pyttsx3 or system SAPI if available, otherwise creates
        a clean acoustic notification audio track.
        """
        # Try pyttsx3 if installed
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Set rate & volume
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 0.9)
            
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 500:
                with open(tmp_path, "rb") as f:
                    wav_data = f.read()
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return TTSResult(
                    audio_bytes=wav_data,
                    mime_type="audio/wav",
                    engine_used="offline_local_engine",
                    is_fallback=is_fallback,
                    language_code=lang_code,
                )
        except Exception:
            pass

        # Robust built-in acoustic synthesizer fallback (pure Python, zero dependencies)
        chime_audio = self._generate_acoustic_chime(duration_sec=2.0)
        return TTSResult(
            audio_bytes=chime_audio,
            mime_type="audio/wav",
            engine_used="offline_acoustic_chime",
            is_fallback=is_fallback,
            language_code=lang_code,
        )

    def _generate_acoustic_chime(self, duration_sec: float = 2.0, sample_rate: int = 22050) -> bytes:
        """
        Generates a pleasant 2-tone melodic medical confirmation chime (440Hz -> 880Hz)
        in pure Python without external dependencies.
        """
        num_samples = int(duration_sec * sample_rate)
        half_samples = num_samples // 2
        buf = io.BytesIO()

        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)

            samples = []
            for i in range(num_samples):
                freq = 523.25 if i < half_samples else 659.25  # C5 -> E5 harmonic
                # Envelope decay
                decay = math.exp(-3.0 * (i % half_samples) / half_samples)
                sample_val = int(32767.0 * 0.4 * decay * math.sin(2.0 * math.pi * freq * i / sample_rate))
                samples.append(struct.pack("<h", sample_val))

            wav.writeframes(b"".join(samples))

        return buf.getvalue()

    def _generate_silence_wav(self, duration_sec: float = 0.5, sample_rate: int = 16000) -> bytes:
        """Generate silent WAV file."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"\x00\x00" * int(duration_sec * sample_rate))
        return buf.getvalue()
