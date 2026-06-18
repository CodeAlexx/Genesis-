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
}

/// An open media decoder handle. Closes on drop.
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
