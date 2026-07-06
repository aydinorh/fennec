//! Voice mode for the TUI.
//!
//! `/voice on` starts mic capture via `cpal`. Audio samples
//! stream into a buffer; on `/voice off` (or after a max
//! duration) the buffer is written to a WAV file in the
//! configured voice cache directory, then handed to the
//! existing `TranscribeAudioTool` for Whisper transcription.
//! The transcribed text is then dropped into the TUI input as
//! if the user typed it.
//!
//! `/voice tts on/off` toggles whether the bot's reply is
//! synthesized via `TextToSpeechTool` and queued for playback
//! through the OS audio output (afplay on macOS, paplay on
//! Linux). Playback itself is handled by spawning the OS player
//! since `cpal` output adds significant complexity for what's a
//! one-off "play this WAV" operation.
//!
//! All of this is opt-in: the `/voice` command kicks the state
//! machine, so users who never invoke it pay no runtime cost
//! beyond the `cpal` device list at startup.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
#[cfg(feature = "audio")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// State of the voice subsystem at any moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceState {
    /// No recording, no pending transcription.
    Idle,
    /// Mic is open and samples are being buffered.
    Recording,
    /// Recording stopped; waiting for transcription to come back.
    Transcribing,
}

/// Public handle the TUI's app state holds. Cheap to clone (Arc
/// internally). Methods are sync — recording / transcription run
/// in their own thread / task and write back into shared state.
#[derive(Clone)]
pub struct VoiceController {
    inner: Arc<VoiceInner>,
}

struct VoiceInner {
    state: Mutex<VoiceState>,
    /// Buffered PCM samples while recording. Cleared on flush.
    samples: Mutex<Vec<i16>>,
    /// Active stream handle. `cpal::Stream` is `!Send`, so we
    /// keep it inside the recording thread and drop it via a
    /// stop signal.
    stop_signal: Mutex<Option<std::sync::mpsc::Sender<()>>>,
    /// When recording started, for the elapsed-time UI badge.
    started_at: Mutex<Option<Instant>>,
    /// Sample rate of the active mic stream (used by the WAV
    /// writer).
    sample_rate: Mutex<u32>,
    /// Whether `/voice tts` is on — synthesizes bot replies.
    tts_enabled: Mutex<bool>,
    /// Most recent transcription. Polled by the TUI's tick
    /// handler; dropped into the input box when present, then
    /// cleared.
    pending_transcription: Mutex<Option<String>>,
    /// Last error message (recording failure, no mic, etc.).
    /// Surfaced as a transient status by the tick handler.
    pending_error: Mutex<Option<String>>,
    /// WAV path waiting for transcription. Set by stop_recording;
    /// taken by the run_tui polling task.
    pending_wav: Mutex<Option<std::path::PathBuf>>,
}

impl VoiceController {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(VoiceInner {
                state: Mutex::new(VoiceState::Idle),
                samples: Mutex::new(Vec::new()),
                stop_signal: Mutex::new(None),
                started_at: Mutex::new(None),
                sample_rate: Mutex::new(16_000),
                tts_enabled: Mutex::new(false),
                pending_transcription: Mutex::new(None),
                pending_error: Mutex::new(None),
                pending_wav: Mutex::new(None),
            }),
        }
    }

    /// Pop the WAV path waiting for transcription. Returns
    /// `None` if there's nothing to transcribe.
    pub fn take_pending_wav(&self) -> Option<std::path::PathBuf> {
        self.inner.pending_wav.lock().unwrap().take()
    }

    pub fn state(&self) -> VoiceState {
        *self.inner.state.lock().unwrap()
    }

    pub fn elapsed_secs(&self) -> Option<u64> {
        self.inner
            .started_at
            .lock()
            .unwrap()
            .map(|t| t.elapsed().as_secs())
    }

    pub fn tts_enabled(&self) -> bool {
        *self.inner.tts_enabled.lock().unwrap()
    }

    pub fn set_tts(&self, enabled: bool) {
        *self.inner.tts_enabled.lock().unwrap() = enabled;
    }

    /// Pop the most recent transcription. Returns `None` if no
    /// transcription is pending. The TUI calls this from its
    /// tick handler and, when `Some(text)`, drops `text` into
    /// the input box.
    pub fn take_transcription(&self) -> Option<String> {
        self.inner.pending_transcription.lock().unwrap().take()
    }

    pub fn take_error(&self) -> Option<String> {
        self.inner.pending_error.lock().unwrap().take()
    }

    /// Begin mic capture. No-op if already recording. Errors are
    /// stashed in `pending_error` for the tick handler to surface
    /// (rather than propagated, since the calling slash-command
    /// dispatch path is sync and shouldn't fail its outcome on
    /// audio backend errors).
    pub fn start_recording(&self) {
        let mut state = self.inner.state.lock().unwrap();
        if *state == VoiceState::Recording {
            return;
        }
        match self.spawn_capture_thread() {
            Ok(()) => {
                *state = VoiceState::Recording;
                *self.inner.started_at.lock().unwrap() = Some(Instant::now());
            }
            Err(e) => {
                *self.inner.pending_error.lock().unwrap() =
                    Some(format!("mic open failed: {e}"));
            }
        }
    }

    #[cfg(feature = "audio")]
    fn spawn_capture_thread(&self) -> Result<()> {
        let inner = Arc::clone(&self.inner);
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        std::thread::Builder::new()
            .name("fennec-voice-capture".into())
            .spawn(move || {
                if let Err(e) = run_capture(inner.clone(), stop_rx) {
                    *inner.pending_error.lock().unwrap() = Some(format!("capture: {e}"));
                    *inner.state.lock().unwrap() = VoiceState::Idle;
                }
            })?;
        *self.inner.stop_signal.lock().unwrap() = Some(stop_tx);
        Ok(())
    }

    /// Built without the `audio` cargo feature: mic capture is
    /// unavailable, surface a clear error instead of failing to link
    /// against ALSA/CoreAudio. TTS playback and file transcription
    /// are unaffected (they don't go through cpal).
    #[cfg(not(feature = "audio"))]
    fn spawn_capture_thread(&self) -> Result<()> {
        anyhow::bail!(
            "this build has no microphone support (cargo feature `audio` disabled). \
             Rebuild with default features — on Linux servers install libasound2-dev first."
        )
    }

    /// Stop mic capture, write the buffered samples to a WAV
    /// file, and queue it for transcription via the run_tui
    /// polling task (which calls TranscribeAudioTool against
    /// the file and then `deliver_transcription`). Returns the
    /// WAV path on success — also stashed in `pending_wav` for
    /// the polling task to consume.
    pub fn stop_recording(&self, output_dir: &std::path::Path) -> Result<std::path::PathBuf> {
        // Signal the capture thread to stop and let it close the
        // stream cleanly.
        if let Some(tx) = self.inner.stop_signal.lock().unwrap().take() {
            let _ = tx.send(());
        }
        // Brief wait for the capture thread to flush and exit.
        for _ in 0..20 {
            if *self.inner.state.lock().unwrap() != VoiceState::Recording {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let samples: Vec<i16> = std::mem::take(&mut *self.inner.samples.lock().unwrap());
        let sample_rate = *self.inner.sample_rate.lock().unwrap();
        *self.inner.started_at.lock().unwrap() = None;

        if samples.is_empty() {
            *self.inner.state.lock().unwrap() = VoiceState::Idle;
            return Err(anyhow!("no audio captured"));
        }

        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("create voice cache dir: {}", output_dir.display()))?;
        let path = output_dir.join(format!(
            "voice-{}.wav",
            chrono::Local::now().format("%Y%m%d-%H%M%S")
        ));

        write_wav(&path, &samples, sample_rate)?;
        *self.inner.state.lock().unwrap() = VoiceState::Transcribing;
        *self.inner.pending_wav.lock().unwrap() = Some(path.clone());
        Ok(path)
    }

    /// Stash the transcription text — the TUI's tick handler
    /// will drop it into the input on the next frame.
    pub fn deliver_transcription(&self, text: String) {
        *self.inner.pending_transcription.lock().unwrap() = Some(text);
        *self.inner.state.lock().unwrap() = VoiceState::Idle;
    }

    /// Mark a failed transcription so the user sees an error
    /// rather than a silent reset.
    pub fn deliver_error(&self, message: String) {
        *self.inner.pending_error.lock().unwrap() = Some(message);
        *self.inner.state.lock().unwrap() = VoiceState::Idle;
    }
}

impl Default for VoiceController {
    fn default() -> Self {
        Self::new()
    }
}

/// Serialize buffered PCM samples to a 16-bit mono WAV.
#[cfg(feature = "audio")]
fn write_wav(path: &std::path::Path, samples: &[i16], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("open WAV writer: {}", path.display()))?;
    for s in samples {
        writer.write_sample(*s)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Unreachable in practice without `audio` (recording can never
/// start, so `samples` is always empty and `stop_recording` errors
/// before calling this) — present so the module compiles.
#[cfg(not(feature = "audio"))]
fn write_wav(_path: &std::path::Path, _samples: &[i16], _sample_rate: u32) -> Result<()> {
    anyhow::bail!("this build has no audio support (cargo feature `audio` disabled)")
}

/// Run the cpal capture loop until a stop signal arrives. Buffers
/// samples into `inner.samples`. Sample format conversions
/// (i16, i32, f32) handled inline so we accept whatever the
/// default input device exposes.
#[cfg(feature = "audio")]
fn run_capture(
    inner: Arc<VoiceInner>,
    stop_rx: std::sync::mpsc::Receiver<()>,
) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device"))?;
    let config = device
        .default_input_config()
        .context("query default input config")?;
    let sample_format = config.sample_format();
    let stream_config: cpal::StreamConfig = config.clone().into();
    *inner.sample_rate.lock().unwrap() = stream_config.sample_rate.0;

    let err_fn = |e: cpal::StreamError| {
        tracing::warn!("voice capture stream error: {e}");
    };
    let inner_for_data = Arc::clone(&inner);

    let stream: cpal::Stream = match sample_format {
        cpal::SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                inner_for_data.samples.lock().unwrap().extend_from_slice(data);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buf = inner_for_data.samples.lock().unwrap();
                buf.extend(data.iter().map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16));
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I32 => device.build_input_stream(
            &stream_config,
            move |data: &[i32], _: &cpal::InputCallbackInfo| {
                let mut buf = inner_for_data.samples.lock().unwrap();
                buf.extend(data.iter().map(|s| (s >> 16) as i16));
            },
            err_fn,
            None,
        )?,
        other => {
            anyhow::bail!("unsupported sample format from mic: {other:?}");
        }
    };
    stream.play().context("start input stream")?;
    // Block until the stop signal arrives.
    let _ = stop_rx.recv();
    drop(stream);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_controller_starts_idle() {
        let v = VoiceController::new();
        assert_eq!(v.state(), VoiceState::Idle);
        assert!(v.elapsed_secs().is_none());
        assert!(!v.tts_enabled());
    }

    #[test]
    fn tts_toggle_round_trip() {
        let v = VoiceController::new();
        v.set_tts(true);
        assert!(v.tts_enabled());
        v.set_tts(false);
        assert!(!v.tts_enabled());
    }

    #[test]
    fn deliver_transcription_then_take_clears() {
        let v = VoiceController::new();
        v.deliver_transcription("hello".into());
        assert_eq!(v.take_transcription().as_deref(), Some("hello"));
        assert!(v.take_transcription().is_none());
    }

    #[test]
    fn deliver_error_then_take_clears() {
        let v = VoiceController::new();
        v.deliver_error("mic busy".into());
        assert_eq!(v.take_error().as_deref(), Some("mic busy"));
        assert!(v.take_error().is_none());
    }
}
