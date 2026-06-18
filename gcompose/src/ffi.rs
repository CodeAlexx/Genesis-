//! Safe-ish Rust wrappers over the C engine shims (vendored from MojoMedia/ffi).
//!
//! Phase 0 surface: open a media file, decode one frame letterboxed into an RGBA8
//! buffer, close. The C side (fpx_decode.c) owns all the FFmpeg complexity.

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

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
    fn fpx_gpu_finish();
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
