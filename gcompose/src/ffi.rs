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
    // Per-clip grade (Triad-B P1): grades the PiP-composite buffer (INB) IN PLACE before the program
    // grade, so a later fpx_gpu_grade stacks on top (documented "per-clip first, then program" order).
    // A neutral grade (0/1/1) is a no-op. Run between fpx_gpu_pip and fpx_gpu_grade.
    fn fpx_gpu_grade_clip(bright: f32, contrast: f32, sat: f32);
    // P2 TRANSFORM (Shotcut-parity): rotate (degrees) + uniform scale the BASE frame about its
    // center, bilinear sample. Runs RIGHT AFTER fpx_gpu_track1 (before pip). Identity at
    // rot_deg=0,scale=1 (skipped, zero cost). Uses a device scratch copy (cannot read+write in place).
    fn fpx_gpu_transform(rot_deg: f32, scale: f32);
    // P2 LGG (3-way color wheels): per-channel out=clamp01(pow(clamp01(in*gain+lift),1/gamma)) IN
    // PLACE on the grade-result buffer (OUTB), AFTER fpx_gpu_grade, BEFORE fpx_gpu_look. Identity at
    // lift 0 / gamma 1 / gain 1 (skipped). White balance is folded into the gains by the UI.
    fn fpx_gpu_lgg(
        lr: f32, lg: f32, lb: f32,
        gar: f32, gag: f32, gab: f32,
        gnr: f32, gng: f32, gnb: f32,
    );
    // P2 BLUR: separable gaussian (2 passes via device scratch), IN PLACE on OUTB. radius=ceil(2*sigma)
    // capped at 32; sigma<=0 => no-op. Runs AFTER fpx_gpu_lgg, BEFORE fpx_gpu_look.
    fn fpx_gpu_blur(sigma: f32);
    // look kind: 0=none (final=OUTB), 1=VHS, 2=LUT3D (both → final=LOOKB). amt = mix 0..1; lut_n =
    // the LUT grid N (cube root of the uploaded 3D LUT, only read when kind==2). Returns 1 when the
    // final composed frame lives in the LOOK buffer (kind 1/2), 0 when it stays in OUTB (kind 0).
    fn fpx_gpu_look(kind: c_int, amt: f32, lut_n: c_int) -> c_int;
    // Upload a 3D LUT (N*N*N*3 interleaved RGB floats in [0,1]) to the device for the LUT3D look.
    // `nfloats` must equal N*N*N*3 and be <= the shim's MAXLUTF capacity (33^3*3). Returns 0 on
    // success, negative on a bad arg / CL write failure.
    fn fpx_gpu_upload_lut(lut: *const f32, nfloats: c_int) -> c_int;
    fn fpx_gpu_download_u8(final_is_look: c_int, out: *mut u8);
    fn fpx_gpu_download_f32(final_is_look: c_int, out: *mut f32);
    fn fpx_gpu_finish();

    // Scope kernels (fpx_gpu.c) — run on the LAST composed GPU buffer (g_buf[OUTB] when
    // final_is_look==0, g_buf[LOOKB] when 1). These read the persistent on-device frame buffer; no
    // re-compose happens here, so the caller MUST have composed the wanted frame first (a PREVIEW)
    // and must NOT have run any other compose in between.
    //   fpx_gpu_histogram   -> 768 ints: R 0..255, G 256..511, B 512..767 (NOT a rendered image).
    //   fpx_gpu_waveform    -> 256*256*4 RGBA8 luma-waveform image.
    //   fpx_gpu_vectorscope -> 256*256*4 RGBA8 U/V vectorscope image.
    fn fpx_gpu_histogram(final_is_look: c_int, out_hist: *mut c_int);
    fn fpx_gpu_waveform(final_is_look: c_int, out: *mut u8);
    fn fpx_gpu_vectorscope(final_is_look: c_int, out: *mut u8);
    //   fpx_gpu_parade -> 256*256*4 RGBA8 RGB-parade image (3 side-by-side per-channel column
    //   waveforms, R|G|B). Triad-B P1 scope kind 3.
    fn fpx_gpu_parade(final_is_look: c_int, out: *mut u8);

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
    // Constant-quality (CRF) rate control (Triad-B P1 export controls). Call AFTER config_video and
    // BEFORE start when the export uses rate_mode=1 (constant quality). Re-opens the video codec with
    // the crf private option set (x264/x265) or a global_quality/qscale fallback (mpeg4). Returns 0
    // on success, negative on error (best-effort: a failure leaves the bitrate config in place).
    fn fpx_enc_set_quality(h: *mut c_void, crf: c_int) -> c_int;
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

    // Audio FILTER shim (fpx_audio.c). Applies an arbitrary libavfilter chain (volume/pan/aeq/
    // anequalizer/acompressor/agate/loudnorm/...) to interleaved-float audio. P3 Triad-B: the
    // per-clip AudioFx chain is applied to a decoded clip range BEFORE the gain+offset mix.
    //   fpx_au_apply(sr, ch, chain, in, nb, out, out_cap):
    //     `in` is `nb` interleaved-float samples-per-channel (ch channels); `chain` is a libavfilter
    //     chain string with NO spaces (commas between filters, '=' / ':' inside). Writes up to
    //     out_cap floats into `out`; returns OUTPUT samples-per-channel (>= 0), or negative on error.
    //     A filter that changes the sample count (loudnorm/acompressor latency) is fine — the caller
    //     uses the returned count. `chain == NULL` or empty applies a pass-through (anull).
    fn fpx_au_apply(
        sr: c_int,
        ch: c_int,
        chain: *const c_char,
        input: *const f32,
        nb: c_int,
        out: *mut f32,
        out_cap: c_int,
    ) -> c_int;

    // Audio / asset shims (fpx_aread.c).
    //   fpx_audio_envelope: whole-track peak envelope into out[nbuckets] (0..1).
    //   fpx_decode_audio_range: decode [start,start+dur) -> interleaved f32 (out_ch).
    //   fpx_load_cube: parse a .cube 3D LUT into out (N*N*N*3 interleaved RGB floats); returns the
    //     grid size N on success (so out holds N^3*3 floats), or a negative error (-1 null arg,
    //     -2 open fail, -3 1D LUT unsupported, -4 out too small, -5 no LUT_3D_SIZE, -6 incomplete).
    fn fpx_audio_envelope(path: *const c_char, nbuckets: c_int, out: *mut f32) -> c_int;
    fn fpx_load_cube(path: *const c_char, out: *mut f32, max_floats: c_int) -> c_int;
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

/// Scope-image dimensions (fpx_gpu.c renders waveform/vectorscope as a fixed SVW×SVH RGBA8 image,
/// and the histogram is rendered into the same size here). Matches the C shim's hard-coded 256×256
/// scope grid / image buffers, and the pinned worker.rs SW/SH the UI reads back.
pub const SVW: usize = 256;
pub const SVH: usize = 256;
/// Number of histogram bins fpx_gpu_histogram fills: 256 each for R, G, B = 768 ints.
pub const HIST_BINS: usize = 768;

/// Max LUT grid size the OpenCL shim accepts (matches `MAXLUTF = 33*33*33*3` in fpx_gpu.c). A
/// `.cube` whose `LUT_3D_SIZE` exceeds 33 will overflow `fpx_load_cube`'s `max_floats` guard and
/// return -4 (caller degrades to no look). `MAX_LUT_N` is the grid edge; `MAX_LUT_FLOATS` the
/// interleaved-RGB float count we stage.
pub const MAX_LUT_N: usize = 33;
pub const MAX_LUT_FLOATS: usize = MAX_LUT_N * MAX_LUT_N * MAX_LUT_N * 3;

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

    /// Initialize OpenCL with a few in-process retries for a SOFT init failure (`fpx_gpu_init`
    /// returning rc != 0). The hard flake is a process-death segfault inside the driver, which no
    /// in-process retry can fix (only the client respawn can) — but a clean non-zero rc (transient
    /// driver/device-busy) is worth retrying a couple of times before giving up. Returns None only
    /// if every attempt fails; the caller (serve) then exits non-zero so the client respawns.
    ///
    /// LIMITATION (finding #2): this retry only meaningfully recovers a genuinely TRANSIENT failure
    /// (device-busy on `clGetDeviceIDs`, a momentarily unavailable device). The C `fpx_gpu_init`
    /// returns early-0 only once `g_ready` is set, and its error paths (rc -1..-13) leave the
    /// partially-created OpenCL objects (`g_ctx`/`g_q`/`g_prog`/`g_buf`…) allocated WITHOUT freeing
    /// them — so each retry re-creates and leaks the prior partial handles. For a DETERMINISTIC
    /// failure (e.g. a kernel build error, rc -6) every attempt fails identically and only leaks
    /// `attempts-1` extra contexts before we give up and exit (the OS then reclaims on process
    /// exit). Keeping `attempts` small bounds that leak. Real handle-reuse recovery would require
    /// the C side to `goto cleanup`/release partial handles on each failure path; that lives in
    /// `csrc/fpx_gpu.c` (not this crate's wrapper) and is out of scope for this slice.
    pub fn init_retry(attempts: usize) -> Option<Gpu> {
        let attempts = attempts.max(1);
        for a in 0..attempts {
            if let Some(g) = Gpu::init() {
                return Some(g);
            }
            eprintln!("[gpu] init attempt {} of {attempts} failed (rc != 0)", a + 1);
        }
        None
    }

    /// Tiny end-to-end self-check after init: upload a KNOWN non-black frame, run an identity
    /// compose (no overlay, no grade change), download the result, and confirm the GPU actually
    /// ROUND-TRIPPED the pixels — not merely that the buffer is the right length. This catches a
    /// compositor that "init'd" (g_ready==1) but whose kernels can't launch / whose
    /// `clEnqueueReadBuffer` silently fails (the C download swallows every CL error and returns
    /// void), so `--serve` fails fast + clean BEFORE printing "serve ready" rather than serving a
    /// broken worker whose first real PREVIEW/ENC would produce garbage.
    ///
    /// Finding #1: the old check only asserted `out.len() == GVW*GVH*4`. But `compose()` allocates
    /// `vec![0u8; GVW*GVH*4]` and the C void downloads never resize it, so the length is ALWAYS
    /// exact and the check was a tautology that could never fail. The real signal is the pixel
    /// VALUES: we pre-fill the output with a sentinel that the upload value can't produce, upload a
    /// mid-gray (0x7F) frame, run an identity grade, and require the download to have (a) overwritten
    /// the sentinel and (b) landed near mid-gray. The `k_unpack`→`k_pack` round-trip of 0x7F is
    /// `round(127/255*255) == 127`, so an identity pipeline must yield ≈0x7F; an all-zero (dead
    /// read) or all-sentinel (read never ran) buffer fails.
    ///
    /// The check is cheap (one frame, op=0 so the overlay path is skipped) and side-effect-free
    /// w.r.t. the encoder/decoder caches: it only touches the GPU slots, which every real compose
    /// overwrites anyway.
    pub fn self_check(&self) -> bool {
        const FILL: u8 = 0x7F; // mid-gray upload; identity grade should preserve it (≈127 out)
        const SENTINEL: u8 = 0xAB; // pre-fill the download buffer with a value upload can't produce

        // Upload a uniform mid-gray frame into slot 0; op=0 disables PiP so slot 1 isn't required.
        let gray = vec![FILL; GVW * GVH * 4];
        self.upload(0, &gray);

        // Identity grade (bright=0, contrast=1, sat=1) and look=none: out should ≈ the uploaded gray.
        // We pre-seed `out` with SENTINEL so a download that never actually ran (CL read failed and
        // was swallowed) leaves the sentinel and is detectable.
        let mut out = vec![SENTINEL; GVW * GVH * 4];
        unsafe {
            fpx_gpu_track1(-1, 0.0, 4.0); // no transition: copy base (slot 0)
            fpx_gpu_pip(0.0, 0.0, 0.0, 1.0, 1.0); // op=0: no overlay
            fpx_gpu_grade(0.0, 1.0, 1.0); // identity grade
            let fin = fpx_gpu_look(0, 0.0, 0); // look kind 0 = none
            fpx_gpu_download_u8(fin, out.as_mut_ptr());
            fpx_gpu_finish();
        }

        if out.len() != GVW * GVH * 4 {
            eprintln!(
                "[gpu] self-check FAILED: compose returned {} bytes (expected {})",
                out.len(),
                GVW * GVH * 4
            );
            return false;
        }

        // The download must have OVERWRITTEN our sentinel (proves the read ran) and produced a
        // non-degenerate, near-mid-gray result (proves the kernels ran). Sample on a stride so a
        // huge frame doesn't make the check expensive; the frame is uniform so a stride is faithful.
        const STRIDE: usize = 997; // coprime-ish stride to spread the samples across the buffer
        let mut samples = 0usize;
        let mut in_band = 0usize; // count of sampled bytes within an identity-of-0x7F band
        let mut sentinel_seen = 0usize;
        let mut nonzero_seen = false;
        let mut i = 0usize;
        while i < out.len() {
            let b = out[i];
            samples += 1;
            if b == SENTINEL {
                sentinel_seen += 1;
            }
            if b != 0 {
                nonzero_seen = true;
            }
            // Identity round-trip of 0x7F (127) should land at 127; allow generous slack for any
            // rounding in unpack/pack. Alpha bytes also round-trip 0x7F here (uniform fill).
            if (0x60..=0x9F).contains(&b) {
                in_band += 1;
            }
            i += STRIDE;
        }

        if samples == 0 {
            eprintln!("[gpu] self-check FAILED: no samples (empty buffer)");
            return false;
        }
        // Sentinel survivors mean the download never wrote those bytes (read failed silently).
        if sentinel_seen > 0 {
            eprintln!(
                "[gpu] self-check FAILED: {sentinel_seen}/{samples} sampled bytes still hold the \
                 pre-download sentinel (0x{SENTINEL:02X}) — GPU download did not run"
            );
            return false;
        }
        if !nonzero_seen {
            eprintln!("[gpu] self-check FAILED: download is all-zero (dead read / kernels not run)");
            return false;
        }
        // The vast majority of an identity-graded uniform-gray frame must land in the gray band.
        // (A few stragglers tolerated for any driver-specific rounding, but a broken pipeline that
        // returns black/white/garbage will miss the band wholesale.)
        if in_band * 2 < samples {
            eprintln!(
                "[gpu] self-check FAILED: only {in_band}/{samples} sampled bytes near mid-gray after \
                 an identity compose of a 0x{FILL:02X} frame — kernels are not composing correctly"
            );
            return false;
        }
        true
    }

    /// Upload an RGBA8 GVW×GVH frame to a slot (0=base/V1, 1=over/V2, 2=transition partner).
    pub fn upload(&self, slot: i32, rgba: &[u8]) {
        debug_assert_eq!(rgba.len(), GVW * GVH * 4);
        unsafe { fpx_gpu_upload_u8(slot as c_int, rgba.as_ptr()) };
    }

    /// Run the FIRST stage of the on-device pipeline directly (the transition / track-1 blend).
    ///
    /// `tt` = transition kind: -1 = none (copy slot-0 base into the track-1 buffer, today's
    /// no-transition behavior), 0..7 = the fpx_gpu transition kernels (0=crossfade, 1=wipe_lr,
    /// 2=wipe_rl, 3=wipe_up, 4=wipe_down, 5=slide_lr, 6=zoom, 7=dissolve). `t` is the transition
    /// progress in [0,1]; `param` is the per-transition parameter (4.0 default, dissolve Power).
    /// When `tt` in 0..7 the caller MUST have `upload(2, rgba)`'d the INCOMING (slot-2 / partner)
    /// frame first — the kernel blends slot-0 base toward slot-2 trans by `t`. Mirrors MojoMedia's
    /// `fpx_gpu_track1(tt_id, rtt, tt_p)` (main_editor.mojo ~699 preview / ~1300 render).
    ///
    /// This is exposed so the serve loop can drive a non-(-1) transition; the bundled `compose`/
    /// `compose_f32` keep their hardcoded no-transition `track1(-1,..)` for callers that never
    /// transition. `compose_trans`/`compose_trans_f32` below thread a real `tt` through instead.
    pub fn track1(&self, tt: i32, t: f32, param: f32) {
        unsafe { fpx_gpu_track1(tt as c_int, t, param) };
    }

    /// Upload a parsed 3D LUT to the device for the LUT3D look. `lut` must be `N*N*N*3` interleaved
    /// RGB floats (exactly what `load_cube` returns). Returns true on success; a bad length / CL
    /// failure returns false (the caller then degrades the look to none). Must be called before a
    /// `compose(.., look_kind=2, .., lut_n=N)` so the LUT3D kernel reads the intended grid.
    pub fn upload_lut(&self, lut: &Lut) -> bool {
        let n = lut.n;
        let want = n * n * n * 3;
        if n == 0 || lut.data.len() < want || want > MAX_LUT_FLOATS {
            return false;
        }
        // Pass exactly N^3*3 floats (load_cube may have a longer staging buffer; only the first
        // want floats are the LUT). The C side validates 0 < nfloats <= MAXLUTF.
        let rc = unsafe { fpx_gpu_upload_lut(lut.data.as_ptr(), want as c_int) };
        rc == 0
    }

    /// Run the on-device pipeline (no transition → PiP composite of slot1 over slot0 → grade →
    /// look) and download the result as an RGBA8 GVW×GVH buffer.
    ///
    /// `look_kind` selects the look (0=none, 1=VHS, 2=LUT3D), `look_amt` is the mix, and `lut_n` is
    /// the uploaded LUT's grid size N (only read by the kernel when `look_kind==2`; pass 0 otherwise).
    /// For a LUT3D look the caller MUST `upload_lut` the matching grid first. Returns the composed
    /// RGBA8 buffer AND `final_is_look` (true when the frame ended up in the LOOK buffer, i.e. kind
    /// 1/2) so the serve loop can point a subsequent SCOPE at the post-look buffer.
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
        look_kind: i32,
        look_amt: f32,
        lut_n: i32,
    ) -> (Vec<u8>, bool) {
        let mut out = vec![0u8; GVW * GVH * 4];
        let fin = unsafe {
            fpx_gpu_track1(-1, 0.0, 4.0); // no transition: copy base (slot 0)
            fpx_gpu_pip(op, px, py, pw, ph); // composite slot 1 over, into the PiP rect
            fpx_gpu_grade(bright, contrast, sat);
            let fin = fpx_gpu_look(look_kind as c_int, look_amt, lut_n as c_int);
            fpx_gpu_download_u8(fin, out.as_mut_ptr());
            fpx_gpu_finish();
            fin
        };
        (out, fin != 0)
    }

    /// Like `compose`, but runs a TRANSITION at the start of the pipeline (Wave 8). `tt` is the
    /// transition kind (-1 = none, copy base — identical to `compose`; 0..7 = a transition kernel),
    /// `trans_prog` is the progress in [0,1], `trans_param` the per-transition parameter (default
    /// 4.0). For `tt` in 0..7 the caller MUST have `upload(2, rgba)`'d the INCOMING frame first.
    /// Pipeline order matches MojoMedia: track1(tt, prog, param) → pip → grade → look. Returns the
    /// composed RGBA8 buffer + `final_is_look` (see `compose`).
    #[allow(clippy::too_many_arguments)]
    pub fn compose_trans(
        &self,
        tt: i32,
        trans_prog: f32,
        trans_param: f32,
        op: f32,
        px: f32,
        py: f32,
        pw: f32,
        ph: f32,
        cbright: f32,
        ccontrast: f32,
        csat: f32,
        bright: f32,
        contrast: f32,
        sat: f32,
        look_kind: i32,
        look_amt: f32,
        lut_n: i32,
        // P2 per-clip effects (pinned wire order: lift3, gamma3, gain3, rot, scale, blur).
        lift_r: f32,
        lift_g: f32,
        lift_b: f32,
        gamma_r: f32,
        gamma_g: f32,
        gamma_b: f32,
        gain_r: f32,
        gain_g: f32,
        gain_b: f32,
        rot: f32,
        scale: f32,
        blur: f32,
    ) -> (Vec<u8>, bool) {
        let mut out = vec![0u8; GVW * GVH * 4];
        let fin = unsafe {
            fpx_gpu_track1(tt as c_int, trans_prog, trans_param); // transition (or -1 copy base)
            fpx_gpu_transform(rot, scale); // P2: rotate+scale the BASE frame (TRACK1), before pip
            fpx_gpu_pip(op, px, py, pw, ph); // composite slot 1 over, into the PiP rect
            fpx_gpu_grade_clip(cbright, ccontrast, csat); // PER-CLIP grade (in place on INB), P1
            fpx_gpu_grade(bright, contrast, sat); // PROGRAM grade, stacked on top
            // P2: 3-way color wheels (LGG) then gaussian blur, in place on OUTB, before look.
            fpx_gpu_lgg(lift_r, lift_g, lift_b, gamma_r, gamma_g, gamma_b, gain_r, gain_g, gain_b);
            fpx_gpu_blur(blur);
            let fin = fpx_gpu_look(look_kind as c_int, look_amt, lut_n as c_int);
            fpx_gpu_download_u8(fin, out.as_mut_ptr());
            fpx_gpu_finish();
            fin
        };
        (out, fin != 0)
    }

    /// f32 sibling of `compose_trans` (Wave 8): same transition-first pipeline, but downloads RGBA
    /// **f32** in [0,1] for `Encoder::video_frame`. Mirrors MojoMedia's render loop, which runs
    /// `track1(r_tt_id, rtt, r_tt_p)` → pip → grade → look → `download_f32` (main_editor.mojo
    /// ~1300-1308). Same args + `final_is_look` return as `compose_trans`.
    #[allow(clippy::too_many_arguments)]
    pub fn compose_trans_f32(
        &self,
        tt: i32,
        trans_prog: f32,
        trans_param: f32,
        op: f32,
        px: f32,
        py: f32,
        pw: f32,
        ph: f32,
        cbright: f32,
        ccontrast: f32,
        csat: f32,
        bright: f32,
        contrast: f32,
        sat: f32,
        look_kind: i32,
        look_amt: f32,
        lut_n: i32,
        // P2 per-clip effects (pinned wire order: lift3, gamma3, gain3, rot, scale, blur).
        lift_r: f32,
        lift_g: f32,
        lift_b: f32,
        gamma_r: f32,
        gamma_g: f32,
        gamma_b: f32,
        gain_r: f32,
        gain_g: f32,
        gain_b: f32,
        rot: f32,
        scale: f32,
        blur: f32,
    ) -> (Vec<f32>, bool) {
        let mut out = vec![0f32; GVW * GVH * 4];
        let fin = unsafe {
            fpx_gpu_track1(tt as c_int, trans_prog, trans_param); // transition (or -1 copy base)
            fpx_gpu_transform(rot, scale); // P2: rotate+scale the BASE frame (TRACK1), before pip
            fpx_gpu_pip(op, px, py, pw, ph); // composite slot 1 over, into the PiP rect
            fpx_gpu_grade_clip(cbright, ccontrast, csat); // PER-CLIP grade (in place on INB), P1
            fpx_gpu_grade(bright, contrast, sat); // PROGRAM grade, stacked on top
            // P2: 3-way color wheels (LGG) then gaussian blur, in place on OUTB, before look.
            fpx_gpu_lgg(lift_r, lift_g, lift_b, gamma_r, gamma_g, gamma_b, gain_r, gain_g, gain_b);
            fpx_gpu_blur(blur);
            let fin = fpx_gpu_look(look_kind as c_int, look_amt, lut_n as c_int);
            fpx_gpu_download_f32(fin, out.as_mut_ptr());
            fpx_gpu_finish();
            fin
        };
        (out, fin != 0)
    }

    /// Same pipeline as `compose`, but downloads the result as RGBA **f32** in [0,1] — the
    /// exact buffer `Encoder::video_frame` (fpx_enc_video_frame_f32) expects. Mirrors
    /// MojoMedia's render loop, which feeds the encoder via `fpx_gpu_download_f32`. Same look
    /// arguments + `final_is_look` return as `compose`.
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
        look_kind: i32,
        look_amt: f32,
        lut_n: i32,
    ) -> (Vec<f32>, bool) {
        let mut out = vec![0f32; GVW * GVH * 4];
        let fin = unsafe {
            fpx_gpu_track1(-1, 0.0, 4.0); // no transition: copy base (slot 0)
            fpx_gpu_pip(op, px, py, pw, ph); // composite slot 1 over, into the PiP rect
            fpx_gpu_grade(bright, contrast, sat);
            let fin = fpx_gpu_look(look_kind as c_int, look_amt, lut_n as c_int);
            fpx_gpu_download_f32(fin, out.as_mut_ptr());
            fpx_gpu_finish();
            fin
        };
        (out, fin != 0)
    }

    /// RGB histogram of the LAST composed buffer (`final_is_look`=0 reads g_buf[OUTB], =1 reads
    /// g_buf[LOOKB]). Returns `HIST_BINS` (768) int bins: indices 0..256 = R, 256..512 = G,
    /// 512..768 = B, each bin = pixel count at that 8-bit value. The C shim does NOT render these
    /// into an image (unlike waveform/vectorscope) — `main.rs` rasterizes the bins into a 256×256
    /// RGBA graph for the SCOPE command. `fpx_gpu_finish` is called so the blocking read is complete
    /// before we return the buffer.
    pub fn histogram(&self, final_is_look: bool) -> Vec<i32> {
        let mut bins = vec![0i32; HIST_BINS];
        unsafe {
            fpx_gpu_histogram(final_is_look as c_int, bins.as_mut_ptr());
            fpx_gpu_finish();
        }
        bins
    }

    /// GPU-rendered luma-waveform image of the LAST composed buffer -> RGBA8 SVW×SVH (256×256).
    /// Reads g_buf[OUTB] (final_is_look=false) or g_buf[LOOKB] (true). The C shim clears its grid,
    /// accumulates over the frame, and renders directly to a 256×256×4 byte image; we just receive
    /// it. The Genesis preview path always composes with look=none, so callers pass false.
    pub fn waveform(&self, final_is_look: bool) -> Vec<u8> {
        let mut out = vec![0u8; SVW * SVH * 4];
        unsafe {
            fpx_gpu_waveform(final_is_look as c_int, out.as_mut_ptr());
            fpx_gpu_finish();
        }
        out
    }

    /// GPU-rendered vectorscope (U/V scatter) image of the LAST composed buffer -> RGBA8 SVW×SVH
    /// (256×256). Reads g_buf[OUTB] (final_is_look=false) or g_buf[LOOKB] (true). Same direct-image
    /// path as `waveform`.
    pub fn vectorscope(&self, final_is_look: bool) -> Vec<u8> {
        let mut out = vec![0u8; SVW * SVH * 4];
        unsafe {
            fpx_gpu_vectorscope(final_is_look as c_int, out.as_mut_ptr());
            fpx_gpu_finish();
        }
        out
    }

    /// GPU-rendered RGB PARADE (Triad-B P1, scope kind 3) of the LAST composed buffer -> RGBA8
    /// SVW×SVH (256×256): three side-by-side per-channel column waveforms (R|G|B), value on the
    /// y-axis. Reads g_buf[OUTB] (final_is_look=false) or g_buf[LOOKB] (true). Same direct-image path
    /// as `waveform`/`vectorscope`, but its own 3-panel kernel + dedicated 3×256×256 grid.
    pub fn parade(&self, final_is_look: bool) -> Vec<u8> {
        let mut out = vec![0u8; SVW * SVH * 4];
        unsafe {
            fpx_gpu_parade(final_is_look as c_int, out.as_mut_ptr());
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

    /// Set constant-quality (CRF) rate control. Call AFTER `config_video` and BEFORE `start` for a
    /// rate_mode=1 export. `crf` is the quality value (lower = better). Returns true on success;
    /// best-effort — a false return leaves the average-bitrate config in place (still a valid encode).
    pub fn set_quality(&mut self, crf: i32) -> bool {
        unsafe { fpx_enc_set_quality(self.h, crf as c_int) >= 0 }
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

/// A parsed 3D LUT: the grid edge `n` (so the data is `n*n*n*3` interleaved RGB floats in [0,1])
/// plus the float payload. Produced by `load_cube`, consumed by `Gpu::upload_lut`. Cached per
/// `.cube` path by the serve loop so repeated frames with the same look don't reparse the file.
#[derive(Clone)]
pub struct Lut {
    pub n: usize,
    pub data: Vec<f32>,
}

/// Parse a `.cube` 3D LUT file via the C `fpx_load_cube` shim. Returns the loaded `Lut` (grid N +
/// `N^3*3` floats) on success, or None on any failure — a missing/malformed/too-large/1D LUT.
/// The caller treats None as "no look" (degrade gracefully, never fail the frame). The staging
/// buffer is sized to the shim's `MAX_LUT_FLOATS` capacity; on success it is truncated to the
/// exact `N^3*3` floats the LUT used.
pub fn load_cube(path: &str) -> Option<Lut> {
    let c = CString::new(path).ok()?;
    let mut data = vec![0f32; MAX_LUT_FLOATS];
    let n = unsafe { fpx_load_cube(c.as_ptr(), data.as_mut_ptr(), MAX_LUT_FLOATS as c_int) };
    if n <= 0 {
        // negative = parse/open error; 0 should never happen (C returns N>0 or negative).
        return None;
    }
    let n = n as usize;
    let want = n.checked_mul(n)?.checked_mul(n)?.checked_mul(3)?;
    // Defensive: the C side guarantees count == N^3*3 <= max_floats before returning N, but clamp
    // anyway so a future C change can't hand us a length we can't slice.
    if want == 0 || want > data.len() {
        return None;
    }
    data.truncate(want);
    Some(Lut { n, data })
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

/// Apply a libavfilter `chain` (NO spaces; commas between filters, `=`/`:` inside) to `input`
/// (interleaved-float, `ch` channels, `nb` samples-per-channel) via the C `fpx_au_apply` shim.
/// Returns the FILTERED interleaved-float samples (length = out_frames * ch), or None on a hard
/// filter-graph error (bad chain / alloc fail) so the caller can fall back to the UNFILTERED input.
///
/// `chain` should be a real filter expression; an empty string is a pass-through (the C side maps
/// it to `anull`), but the P3 caller only ever calls this when the chain is non-trivial. The output
/// can have a DIFFERENT sample count than the input (loudnorm/acompressor add latency or trim), so
/// the returned Vec is truncated to exactly the floats the filter produced.
///
/// CAPACITY: a generous headroom over the input length is allocated (`nb*ch` + 1 s + slack) so a
/// filter that lengthens the stream isn't truncated for the common per-clip range. A filter that
/// produces MORE than that headroom has its tail clamped by the C side (`(total+n)*ch <= out_cap`),
/// which for these effects is inaudible (sub-frame mixing tail); the returned count still matches
/// the bytes actually written.
pub fn au_apply(chain: &str, samples: &[f32], sr: i32, ch: i32) -> Option<Vec<f32>> {
    if ch <= 0 || samples.is_empty() {
        return Some(samples.to_vec()); // nothing to filter; pass through.
    }
    let ch_us = ch as usize;
    let nb = samples.len() / ch_us; // samples-per-channel
    if nb == 0 {
        return Some(samples.to_vec());
    }
    let c = CString::new(chain).ok()?;
    // Output headroom: input frames + 1 s of slack (covers filter latency) per channel, clamped to
    // a c_int. Bounds the temp buffer and keeps the `as c_int` narrowing lossless and positive.
    let extra_frames = sr.max(0) as usize; // ~1 s of slack at this sample rate
    let cap_frames = nb.saturating_add(extra_frames).saturating_add(4096);
    let cap = cap_frames.saturating_mul(ch_us).min(c_int::MAX as usize);
    let mut out = vec![0f32; cap];
    let rc = unsafe {
        fpx_au_apply(
            sr as c_int,
            ch as c_int,
            c.as_ptr(),
            samples.as_ptr(),
            nb as c_int,
            out.as_mut_ptr(),
            cap as c_int,
        )
    };
    if rc < 0 {
        return None; // hard graph error: caller falls back to the unfiltered range.
    }
    // rc = OUTPUT samples-per-channel; truncate to the floats actually produced (clamp to cap so a
    // filter reporting more than it wrote — shouldn't happen — never over-reads the buffer).
    let out_floats = (rc as usize).saturating_mul(ch_us).min(out.len());
    out.truncate(out_floats);
    Some(out)
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
