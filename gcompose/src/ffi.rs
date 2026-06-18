//! Safe-ish Rust wrappers over the C engine shims (vendored from MojoMedia/ffi).
//!
//! Phase 0 surface: open a media file, decode one frame letterboxed into an RGBA8
//! buffer, close. The C side (fpx_decode.c) owns all the FFmpeg complexity.

use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_longlong, c_void};

extern "C" {
    fn fpx_open(path: *const c_char) -> *mut c_void;
    fn fpx_decode_frame_letterbox(
        h: *mut c_void,
        frame_index: c_int,
        out: *mut u8,
        ow: c_int,
        oh: c_int,
    ) -> c_int;
    fn fpx_close(h: *mut c_void);

    // OpenCL compute shim (fpx_gpu.c). Fixed working resolution GVW x GVH.
    fn fpx_gpu_init() -> c_int;
    fn fpx_gpu_upload_u8(slot: c_int, rgba8: *const u8) -> c_int;
    fn fpx_gpu_track1(tt: c_int, t: f32, param: f32);
    fn fpx_gpu_pip(op: f32, px: f32, py: f32, pw: f32, ph: f32);
    fn fpx_gpu_grade(bright: f32, contrast: f32, sat: f32);
    fn fpx_gpu_look(kind: c_int, amt: f32, lut_n: c_int) -> c_int;
    fn fpx_gpu_download_u8(final_is_look: c_int, out: *mut u8);
    fn fpx_gpu_download_f32(final_is_look: c_int, out: *mut f32);
    fn fpx_gpu_finish();

    // Encode/mux shim (fpx_encode.c). RGBA f32 [0,1] frames -> mp4. Call order mirrors
    // MojoMedia main_editor.mojo: open -> config_video[ -> config_audio] -> start ->
    // (video_frame_f32 per frame in pts order)[ -> audio_samples_f32 ] -> finish -> close.
    fn fpx_enc_open(url: *const c_char) -> *mut c_void;
    fn fpx_enc_config_video(
        h: *mut c_void,
        codec_name: *const c_char,
        in_w: c_int,
        in_h: c_int,
        width: c_int,
        height: c_int,
        fps_num: c_int,
        fps_den: c_int,
        bit_rate: c_longlong,
    ) -> c_int;
    fn fpx_enc_config_audio(
        h: *mut c_void,
        codec_name: *const c_char,
        channels: c_int,
        sample_rate: c_int,
        bit_rate: c_longlong,
    ) -> c_int;
    fn fpx_enc_start(h: *mut c_void) -> c_int;
    fn fpx_enc_video_frame_f32(
        h: *mut c_void,
        rgba_f32: *const f32,
        in_w: c_int,
        in_h: c_int,
        ts_sec: c_double,
    ) -> c_int;
    fn fpx_enc_audio_samples_f32(h: *mut c_void, input: *const f32, nb: c_int) -> c_int;
    fn fpx_enc_finish(h: *mut c_void) -> c_int;
    fn fpx_enc_close(h: *mut c_void);

    // Audio shims (fpx_aread.c).
    //   fpx_audio_envelope: whole-track peak envelope into out[nbuckets] (0..1).
    //   fpx_decode_audio_range: decode [start,start+dur) -> interleaved f32 (out_ch).
    fn fpx_audio_envelope(path: *const c_char, nbuckets: c_int, out: *mut f32) -> c_int;
    fn fpx_decode_audio_range(
        path: *const c_char,
        start_sec: c_double,
        dur_sec: c_double,
        out_sr: c_int,
        out_ch: c_int,
        out: *mut f32,
        cap: c_int,
    ) -> c_int;
}

/// The OpenCL shim's fixed working resolution (matches GVW/GVH in fpx_gpu.c).
pub const GVW: usize = 1280;
pub const GVH: usize = 856;

/// Handle to the OpenCL compute pipeline. `init()` compiles the kernels once.
pub struct Gpu {
    _priv: (),
}

impl Gpu {
    /// Initialize OpenCL + compile kernels. Returns None if no usable device / build fails.
    pub fn init() -> Option<Gpu> {
        eprintln!("[gpu] init...");
        let rc = unsafe { fpx_gpu_init() };
        eprintln!("[gpu] init rc={rc}");
        if rc == 0 {
            Some(Gpu { _priv: () })
        } else {
            eprintln!("fpx_gpu_init failed: {rc}");
            None
        }
    }

    /// Upload an RGBA8 GVW×GVH frame to a slot (0=base/V1, 1=over/V2, 2=transition partner).
    pub fn upload(&self, slot: i32, rgba: &[u8]) {
        debug_assert_eq!(rgba.len(), GVW * GVH * 4);
        unsafe { fpx_gpu_upload_u8(slot as c_int, rgba.as_ptr()) };
    }

    /// Run the on-device pipeline (no transition → PiP composite of slot1 over slot0 → grade →
    /// look) and download the result as an RGBA8 GVW×GVH buffer.
    pub fn compose(
        &self,
        op: f32,
        px: f32,
        py: f32,
        pw: f32,
        ph: f32,
        bright: f32,
        contrast: f32,
        sat: f32,
    ) -> Vec<u8> {
        let mut out = vec![0u8; GVW * GVH * 4];
        unsafe {
            fpx_gpu_track1(-1, 0.0, 4.0); // no transition: copy base (slot 0)
            fpx_gpu_pip(op, px, py, pw, ph); // composite slot 1 over, into the PiP rect
            fpx_gpu_grade(bright, contrast, sat);
            let fin = fpx_gpu_look(0, 0.0, 0); // look kind 0 = none
            fpx_gpu_download_u8(fin, out.as_mut_ptr());
            fpx_gpu_finish();
        }
        out
    }

    /// Same pipeline as `compose`, but downloads the result as RGBA **f32** in [0,1] — the
    /// exact buffer `Encoder::video_frame` (fpx_enc_video_frame_f32) expects. Mirrors
    /// MojoMedia's render loop, which feeds the encoder via `fpx_gpu_download_f32`.
    pub fn compose_f32(
        &self,
        op: f32,
        px: f32,
        py: f32,
        pw: f32,
        ph: f32,
        bright: f32,
        contrast: f32,
        sat: f32,
    ) -> Vec<f32> {
        let mut out = vec![0f32; GVW * GVH * 4];
        unsafe {
            fpx_gpu_track1(-1, 0.0, 4.0); // no transition: copy base (slot 0)
            fpx_gpu_pip(op, px, py, pw, ph); // composite slot 1 over, into the PiP rect
            fpx_gpu_grade(bright, contrast, sat);
            let fin = fpx_gpu_look(0, 0.0, 0); // look kind 0 = none
            fpx_gpu_download_f32(fin, out.as_mut_ptr());
            fpx_gpu_finish();
        }
        out
    }
}

/// A live encoder/muxer over the fpx_encode.c shim. Configured for video (and optionally
/// audio), then fed RGBA f32 frames in pts order, then finished. Closes on drop.
///
/// Call order (enforced by the type's method sequence, mirroring MojoMedia render):
///   `Encoder::open` -> `config_video` -> [`config_audio`] -> `start`
///   -> `video_frame(..)*` [ -> `audio_samples(..)` ] -> `finish` -> (drop closes).
pub struct Encoder {
    h: *mut c_void,
    in_w: usize,
    in_h: usize,
}

impl Encoder {
    /// Allocate the output container for `url` (e.g. "/tmp/out.mp4"). None on failure.
    pub fn open(url: &str) -> Option<Encoder> {
        let c = CString::new(url).ok()?;
        let h = unsafe { fpx_enc_open(c.as_ptr()) };
        if h.is_null() {
            None
        } else {
            Some(Encoder { h, in_w: 0, in_h: 0 })
        }
    }

    /// Configure the video stream. `in_w/in_h` = source RGBA dims fed per frame; `width/height`
    /// = encoded dims (usually equal). Returns true on success (stream index >= 0).
    pub fn config_video(
        &mut self,
        codec: &str,
        in_w: usize,
        in_h: usize,
        width: usize,
        height: usize,
        fps_num: i32,
        fps_den: i32,
        bit_rate: i64,
    ) -> bool {
        let c = match CString::new(codec) {
            Ok(c) => c,
            Err(_) => return false,
        };
        let rc = unsafe {
            fpx_enc_config_video(
                self.h,
                c.as_ptr(),
                in_w as c_int,
                in_h as c_int,
                width as c_int,
                height as c_int,
                fps_num as c_int,
                fps_den as c_int,
                bit_rate as c_longlong,
            )
        };
        if rc >= 0 {
            self.in_w = in_w;
            self.in_h = in_h;
            true
        } else {
            false
        }
    }

    /// Configure the audio stream (interleaved float input). Returns true on success.
    pub fn config_audio(&mut self, codec: &str, channels: i32, sample_rate: i32, bit_rate: i64) -> bool {
        let c = match CString::new(codec) {
            Ok(c) => c,
            Err(_) => return false,
        };
        let rc = unsafe {
            fpx_enc_config_audio(
                self.h,
                c.as_ptr(),
                channels as c_int,
                sample_rate as c_int,
                bit_rate as c_longlong,
            )
        };
        rc >= 0
    }

    /// Write the container header. Must be called after config and before any frame. true=ok.
    pub fn start(&mut self) -> bool {
        unsafe { fpx_enc_start(self.h) >= 0 }
    }

    /// Encode one RGBA f32 [0,1] frame (`in_w*in_h*4` floats) at timestamp `ts_sec`. true=ok.
    pub fn video_frame(&mut self, rgba_f32: &[f32], ts_sec: f64) -> bool {
        debug_assert_eq!(rgba_f32.len(), self.in_w * self.in_h * 4);
        let rc = unsafe {
            fpx_enc_video_frame_f32(
                self.h,
                rgba_f32.as_ptr(),
                self.in_w as c_int,
                self.in_h as c_int,
                ts_sec as c_double,
            )
        };
        rc >= 0
    }

    /// Feed `nb` interleaved-float samples-per-channel (`samples.len() == nb*channels`). true=ok.
    ///
    /// Wired by the AUDIO serve command (program audio): the worker decodes each clip's audio
    /// range with `decode_audio_range` (2ch @ 48k) and feeds the interleaved floats here, passing
    /// `nb = floats / channels` (mirrors MojoMedia's `fpx_enc_audio_samples_f32(e, audmix,
    /// prog_floats // 2)`).
    pub fn audio_samples(&mut self, samples: &[f32], nb: usize) -> bool {
        if nb == 0 {
            return true; // nothing to feed is not a failure (empty clip range).
        }
        let rc = unsafe { fpx_enc_audio_samples_f32(self.h, samples.as_ptr(), nb as c_int) };
        rc >= 0
    }

    /// Flush encoders + write the trailer. Call exactly once before drop. true=ok.
    pub fn finish(&mut self) -> bool {
        unsafe { fpx_enc_finish(self.h) >= 0 }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe { fpx_enc_close(self.h) };
    }
}

/// Whole-track peak-amplitude envelope: `buckets` peaks in [0,1] across the file's audio.
/// Returns None if the file has no audio / can't be read.
pub fn audio_envelope(path: &str, buckets: usize) -> Option<Vec<f32>> {
    if buckets == 0 {
        return None;
    }
    let c = CString::new(path).ok()?;
    let mut out = vec![0f32; buckets];
    let rc = unsafe { fpx_audio_envelope(c.as_ptr(), buckets as c_int, out.as_mut_ptr()) };
    // C returns nbuckets on success, 0 if the file has no audio stream, negative on error.
    if rc as usize == buckets {
        Some(out)
    } else {
        None
    }
}

/// Decode `[start_sec, start_sec+dur_sec)` of `path`'s audio -> interleaved f32 (`out_ch`),
/// resampled to `out_sr`. Returns the decoded samples (length = floats written), or an empty
/// Vec if the file has no audio. None only on a hard error.
///
/// Wired by the AUDIO serve command (program audio): the render path decodes each clip's source
/// range with this and feeds the floats to `Encoder::audio_samples`. The C side
/// (`fpx_decode_audio_range`) returns the number of FLOATS written (frames * out_ch), 0 when the
/// file has no audio stream, negative on a hard error — mirrored here.
pub fn decode_audio_range(
    path: &str,
    start_sec: f64,
    dur_sec: f64,
    out_sr: i32,
    out_ch: i32,
    cap: usize,
) -> Option<Vec<f32>> {
    if cap == 0 {
        return Some(Vec::new());
    }
    // Guard the usize -> c_int narrowing (finding #4): the C contract takes `cap` as a c_int, so a
    // `cap` above c_int::MAX would wrap to a negative/small value and either be rejected by C's
    // `cap <= 0` guard or silently truncate the decoded audio. Callers already clamp (see
    // audio_feed's CAP_MAX), but clamp here too so this wrapper is sound for ANY caller. We shrink
    // the requested cap to the largest value the c_int can carry rather than over-allocating.
    let cap = cap.min(c_int::MAX as usize);
    let c = CString::new(path).ok()?;
    let mut buf = vec![0f32; cap];
    let rc = unsafe {
        fpx_decode_audio_range(
            c.as_ptr(),
            start_sec as c_double,
            dur_sec as c_double,
            out_sr as c_int,
            out_ch as c_int,
            buf.as_mut_ptr(),
            cap as c_int,
        )
    };
    if rc < 0 {
        return None; // hard error (open/decode/resample failure)
    }
    // rc == 0 means "no audio stream" -> an empty Vec (caller skips the clip, doesn't abort).
    buf.truncate(rc as usize);
    Some(buf)
}

/// An open media decoder handle. Closes on drop.
///
/// Holds a raw `*mut c_void` (the C decoder handle). `Decoder` is NOT `Hash`/`Eq` itself, but
/// it is fine to store in a `HashMap<String, Decoder>` keyed by the media path — that is how the
/// persistent serve loop caches one open handle per file and reuses it for repeated frames.
pub struct Decoder {
    h: *mut c_void,
}

impl Decoder {
    /// Open `path`. Returns None if the file can't be opened.
    pub fn open(path: &str) -> Option<Decoder> {
        let c = CString::new(path).ok()?;
        let h = unsafe { fpx_open(c.as_ptr()) };
        if h.is_null() {
            None
        } else {
            Some(Decoder { h })
        }
    }

    /// Decode `frame_index` letterboxed into a fresh `w*h*4` RGBA8 buffer.
    /// Returns the pixel buffer, or None on decode failure.
    pub fn decode_rgba(&mut self, frame_index: i32, w: usize, h: usize) -> Option<Vec<u8>> {
        let mut buf = vec![0u8; w * h * 4];
        let rc = unsafe {
            fpx_decode_frame_letterbox(self.h, frame_index, buf.as_mut_ptr(), w as c_int, h as c_int)
        };
        if rc >= 0 {
            Some(buf)
        } else {
            None
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe { fpx_close(self.h) };
    }
}
