// kaleidomo-core/src/video_sink.rs

/// Error type returned by [`VideoFrameSink`] implementations. Boxed with
/// `Send + Sync` so a concrete implementation living outside this crate
/// (e.g. an FFmpeg-sidecar-backed sink in the Tauri application) can carry
/// native process/I/O error info across an `async`/thread boundary without
/// `kaleidomo-core` needing to know anything about that error type.
pub type VideoSinkError = Box<dyn std::error::Error + Send + Sync>;

/// Consumer of rendered RGBA video frames.
///
/// `kaleidomo-core` renders frames (CPU or GPU) and hands each one to a
/// `VideoFrameSink` without knowing or caring how they get encoded, muxed,
/// or written to disk. That responsibility belongs entirely to whatever
/// embeds this crate (e.g. `src-tauri`, which implements this trait on top
/// of the bundled FFmpeg sidecar) — this boundary is what keeps
/// `kaleidomo-core` free of any Tauri dependency.
pub trait VideoFrameSink {
    /// Called once per rendered frame with exactly `width * height * 4`
    /// tightly-packed RGBA8 bytes (row-major, no row padding). `width` and
    /// `height` are whatever the sink was constructed with by its caller;
    /// `kaleidomo-core` does not validate this itself, since it has no
    /// independent notion of the sink's expected dimensions — validation is
    /// the sink implementation's responsibility.
    fn write_rgba_frame(&mut self, rgba: &[u8]) -> Result<(), VideoSinkError>;

    /// Called exactly once after the last frame has been written (including
    /// any repeated still-ending frames). Implementations should
    /// flush/finalize/close out here (e.g. close ffmpeg's stdin, wait for it
    /// to exit, and move a temporary output file into its final place).
    fn finish(&mut self) -> Result<(), VideoSinkError>;
}