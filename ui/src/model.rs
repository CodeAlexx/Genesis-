//! Project data model — the Rust win over MojoMedia's flat parallel lists: real structs.
//!
//! SHARED CONTRACT. Owned by the timeline/model team; consumed by worker + app + pool.
//! Frame units are timeline frames (30 fps assumed for now).

#[derive(Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Clip {
    pub media: usize,  // index into Project.media
    pub src_in: i64,   // source in-point (frames)
    pub len: i64,      // length on the timeline (frames)
    pub t0: i64,       // timeline start (frames)
    pub track: u8,     // 0 = V1, 1 = V2, 2 = A1
    pub look: i32,     // per-clip LOOK index (0 = none)
    pub look_amt: f32, // look mix 0..1
    pub fade_in: i64,
    pub fade_out: i64,
    pub px: f32, // PiP rect (fractions of frame)
    pub py: f32,
    pub pw: f32,
    pub ph: f32,
    // Per-clip LUT path for look == 2 (LUT3D); empty = none. PINNED this wave: produced here
    // (Team B), consumed by Team A (engine look — loaded via fpx_load_cube + uploaded with
    // fpx_gpu_upload_lut, cached per path) and Team C (Look picker UI). `#[serde(default)]` so
    // pre-LUT .json projects still deserialize (the field defaults to "" = no LUT). Clip.look
    // semantics: 0 = None, 1 = VHS, 2 = LUT3D (uses this `lut`).
    #[serde(default)]
    pub lut: String,

    // ----- Triad-B P1 per-clip AUDIO + COLOR (all #[serde(default ..)] so pre-P1 .json loads) -----
    // Per-clip audio gain as a LINEAR multiplier (1.0 = unity). Surfaced in the properties panel as
    // a dB slider (Shotcut "Gain / Volume" range −70..+24 dB → linear), but stored linear so the
    // worker can pass it straight onto the AUDIO wire line. The clip's fade_in/fade_out (already
    // used for VIDEO opacity) ALSO ramp this gain 0→1 / 1→0 at the clip edges (applied in gcompose
    // at mix time). Defaults to 1.0 via `default_gain` so a clip with no stored gain is unchanged.
    #[serde(default = "default_gain")]
    pub gain: f32,
    // Per-clip color grade, ADDITIVE on top of the program grade (grade_at). Same kernels/semantics
    // as the program grade: `bright` is added (−1..1), `contrast`/`sat` are multipliers (0..2, 1.0 =
    // identity). resolve_frame combines per-clip with program (see worker::resolve_frame for the
    // documented order). Defaults reproduce a neutral grade (0/1/1) so an un-graded clip is a no-op.
    #[serde(default)]
    pub bright: f32,
    #[serde(default = "default_one")]
    pub contrast: f32,
    #[serde(default = "default_one")]
    pub sat: f32,

    // ----- Triad-B P2 per-clip COLOR-WHEELS (LIFT/GAMMA/GAIN) + TRANSFORM + BLUR -----
    // PINNED P2 wire extension: these 9 + 3 scalar values are FOLDED+APPENDED to the ENC/PREVIEW
    // lines (after csat) so the engine's new fpx_gpu_lgg / fpx_gpu_transform / fpx_gpu_blur kernels
    // apply them per-clip. All `#[serde(default ..)]` with IDENTITY defaults so pre-P2 .json projects
    // load unchanged AND reproduce the current render (the engine no-ops at identity). Mirrors
    // Shotcut's movit.lift_gamma_gain (Color Grading), white balance, rotate, and blur_gaussian.
    //
    // 3-WAY COLOR WHEELS. Engine semantics (PINNED Team A): per channel
    //   out = pow(clamp(in*gain + lift, 0, 1), 1/gamma).
    // `lift` is an additive shadow offset (Shotcut lift_r = wheel.redF*2-1, range −1..1, def 0).
    // `gamma` is a midtone power (Shotcut gamma factor V0 = 2 → range 0..2, def 1).
    // `gain_rgb` is a highlight multiplier (Shotcut gain factor V0 = 4 → range 0..4, def 1).
    // NOTE: named `gain_rgb` to stay distinct from the P1 AUDIO `gain` (linear audio multiplier).
    #[serde(default = "default_lift")]
    pub lift: [f32; 3], // R,G,B additive lift, identity [0,0,0]
    #[serde(default = "default_gamma")]
    pub gamma: [f32; 3], // R,G,B gamma power, identity [1,1,1]
    #[serde(default = "default_gain_rgb")]
    pub gain_rgb: [f32; 3], // R,G,B highlight gain, identity [1,1,1]

    // WHITE BALANCE (NOT a wire field — folded into gain_rgb by the worker). `wb_temp` is a warm/
    // cool bias in [−1,1] (0 = neutral; >0 warmer → boosts gain_r, cuts gain_b; mirrors Shotcut's
    // color_temperature mapped about 6500 K). `wb_tint` is a green/magenta bias in [−1,1] (0 =
    // neutral; >0 greener → boosts gain_g, cuts gain_r/gain_b). The worker's resolve_frame folds
    // both into the 9 lift/gamma/gain values it sends, so the ENGINE only ever sees lift/gamma/gain.
    #[serde(default)]
    pub wb_temp: f32, // −1..1, def 0
    #[serde(default)]
    pub wb_tint: f32, // −1..1, def 0

    // TRANSFORM of the base frame (Shotcut rotate). `rot` is rotation in DEGREES about the frame
    // center (range −180..180 in the UI; def 0). `scale` is a uniform zoom about the center (UI
    // 0.1..4; def 1). Engine `fpx_gpu_transform(rot_deg, scale)` bilinear-samples; identity at 0/1.
    #[serde(default)]
    pub rot: f32, // degrees, def 0
    #[serde(default = "default_one")]
    pub scale: f32, // zoom, def 1

    // GAUSSIAN BLUR sigma (Shotcut blur_gaussian av.sigma). UI 0..~20; def 0 = no blur. Engine
    // `fpx_gpu_blur(sigma)` runs a separable gaussian; sigma <= 0 is a no-op.
    #[serde(default)]
    pub blur: f32, // sigma, def 0
}

/// serde default for `Clip.gain` (and any unity linear multiplier): 1.0.
fn default_gain() -> f32 {
    1.0
}

/// serde default for `Clip.contrast` / `Clip.sat` / `Clip.scale`: 1.0 (identity multiplier).
fn default_one() -> f32 {
    1.0
}

/// serde default for `Clip.lift`: [0,0,0] (no additive lift — identity).
fn default_lift() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}

/// serde default for `Clip.gamma`: [1,1,1] (unity gamma power — identity).
fn default_gamma() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

/// serde default for `Clip.gain_rgb`: [1,1,1] (unity highlight gain — identity).
fn default_gain_rgb() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

impl Clip {
    pub fn video(media: usize, t0: i64, len: i64, track: u8, name_hint: &str) -> Clip {
        let _ = name_hint;
        Clip {
            media, src_in: 0, len, t0, track, look: 0, look_amt: 1.0,
            fade_in: 0, fade_out: 0, px: 0.0, py: 0.0, pw: 1.0, ph: 1.0,
            lut: String::new(), gain: 1.0, bright: 0.0, contrast: 1.0, sat: 1.0,
            // P2 color-wheels / transform / blur — IDENTITY (no-op) so demo/render are unchanged.
            lift: [0.0, 0.0, 0.0],
            gamma: [1.0, 1.0, 1.0],
            gain_rgb: [1.0, 1.0, 1.0],
            wb_temp: 0.0,
            wb_tint: 0.0,
            rot: 0.0,
            scale: 1.0,
            blur: 0.0,
        }
    }
    pub fn end(&self) -> i64 {
        self.t0 + self.len
    }
}

/// A per-boundary TRANSITION between two same-track clips (Wave 8). Unlike MojoMedia's
/// per-boundary `trans_type[boundary]` parallel list keyed by clip index, this is a real struct
/// keyed by (track, center) so it survives split/delete/reorder without re-indexing: the
/// transition is anchored at a timeline `center` frame on a given `track`, animated over the
/// half-open window `[center - dur/2, center + dur/2)`. `kind` maps to the fpx_gpu track1
/// transition ids (0=crossfade, 1=wipe_lr, 2=wipe_rl, 3=wipe_up, 4=wipe_down, 5=slide_lr,
/// 6=zoom, 7=dissolve). PINNED this wave: produced + edited here (Team B), consumed by Team A
/// (worker::resolve_frame → ENC/PREVIEW trans fields) and Team C (timeline transition UI).
#[derive(Clone, Copy)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Transition {
    pub track: u8,    // 0 = V1, 1 = V2, 2 = A1 (Clip.track index space)
    pub center: i64,  // timeline frame the transition is centered on (typically a clip boundary)
    pub dur: i64,     // window length in frames (clamped >= 2); window = [center - dur/2, center + dur/2)
    pub kind: i32,    // fpx_gpu track1 transition id 0..7 (0=crossfade .. 7=dissolve)
}

impl Transition {
    /// Start frame of the animated window (inclusive).
    pub fn start(&self) -> i64 {
        self.center - self.dur / 2
    }
    /// End frame of the animated window (exclusive).
    pub fn end(&self) -> i64 {
        self.center - self.dur / 2 + self.dur
    }
    /// True when timeline frame `t` is inside this transition's half-open window.
    pub fn contains(&self, t: i64) -> bool {
        t >= self.start() && t < self.end()
    }
    /// Animation progress 0..1 for frame `t`, mirroring MojoMedia's `rtt` ramp (clamped to the
    /// window so the worker never feeds track1 a `t` outside [0,1]). At window start prog=0
    /// (full outgoing clip), at window end prog→1 (full incoming clip).
    pub fn progress(&self, t: i64) -> f32 {
        let span = self.dur.max(1);
        let p = (t - self.start()) as f64 / span as f64;
        p.clamp(0.0, 1.0) as f32
    }
}

/// One keyframe on a scalar track: `v` is the value at timeline (or clip-local) frame `t`.
/// Mirrors MojoMedia's parallel `KfTrack { frames, values }` but as a real struct (the Rust
/// win): a `Vec<Kf>` kept sorted ascending by `t` replaces the two parallel lists.
#[derive(Clone, Copy)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Kf {
    pub t: i64,
    pub v: f32,
}

/// One per-clip PiP keyframe, stored flat (mirrors MojoMedia `PipKf`): which clip, which
/// param (0=px,1=py,2=pw,3=ph), the CLIP-LOCAL frame, and the value. Flat storage (one Vec
/// for the whole project) is chosen over a Vec-per-clip so the set survives split/delete
/// without re-indexing nested vectors; see `remap_clip_keys` for the index-stability policy.
#[derive(Clone, Copy)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PipKey {
    pub clip: usize, // clip index these keys animate
    pub par: u8,     // 0 = px, 1 = py, 2 = pw, 3 = ph
    pub t_local: i64, // clip-local frame (t - clip.t0)
    pub v: f32,
}

/// Linear keyframe eval shared by grade + PiP: value of a sorted-ascending `Vec<Kf>` at `t`,
/// or `fallback` when the track is empty. Clamps to the first/last value outside the range.
/// Formula matches MojoMedia kf_eval / pip_eval: `blend*(vb-va)+va`.
fn eval_track(track: &[Kf], t: i64, fallback: f32) -> f32 {
    let n = track.len();
    if n == 0 {
        return fallback;
    }
    if t <= track[0].t {
        return track[0].v;
    }
    if t >= track[n - 1].t {
        return track[n - 1].v;
    }
    // find i such that track[i].t <= t < track[i+1].t
    let mut i = 0;
    while i < n - 1 && track[i + 1].t <= t {
        i += 1;
    }
    let fa = track[i].t;
    let fb = track[i + 1].t;
    let blend = (t - fa) as f64 / (fb - fa) as f64;
    (blend * (track[i + 1].v - track[i].v) as f64) as f32 + track[i].v
}

/// Sorted insert-or-replace into a `Vec<Kf>` keyed on `t` (mirrors MojoMedia kf_set): if a
/// key already exists at `t` its value is overwritten, otherwise the key is inserted so the
/// track stays ascending in `t`.
fn set_track(track: &mut Vec<Kf>, t: i64, v: f32) {
    match track.binary_search_by(|k| k.t.cmp(&t)) {
        Ok(idx) => track[idx].v = v,        // replace at existing frame
        Err(idx) => track.insert(idx, Kf { t, v }), // sorted insert
    }
}

/// Export / render settings (Triad-B P1). Carried on the `Project` so `worker::render_program`
/// (which keeps its `(&Project, &str)` signature — no app.rs change) can read them and pass them
/// through the `OPEN` wire line to the encoder. The OpenCL working canvas stays GVW×GVH (1280×856);
/// these only drive the ENCODER: `out_w`/`out_h` are the swscaled output dims, `fps_num`/`fps_den`
/// the output framerate, `rate_mode` selects bitrate (0) vs CRF (1), `rate_value` is the bitrate in
/// bits/s (rate_mode 0) OR the CRF quality value (rate_mode 1), and `vcodec` is the encoder name.
///
/// DEFAULTS REPRODUCE TODAY'S BEHAVIOR (1280×856 @ 30/1, mpeg4, 4 Mbit/s bitrate) so existing render
/// gates pass unchanged. All fields are `#[serde(default = ..)]` so pre-P1 .json projects load with
/// the defaults. `mlt`-style values are intentionally avoided — these map straight to fpx_encode.c.
#[derive(Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ExportSettings {
    #[serde(default = "default_out_w")]
    pub out_w: u32,
    #[serde(default = "default_out_h")]
    pub out_h: u32,
    #[serde(default = "default_fps_num")]
    pub fps_num: u32,
    #[serde(default = "default_fps_den")]
    pub fps_den: u32,
    /// 0 = average bitrate, 1 = constant quality (CRF/qscale).
    #[serde(default)]
    pub rate_mode: u8,
    /// bits/s when `rate_mode == 0`; the CRF/quality value when `rate_mode == 1`.
    #[serde(default = "default_bitrate")]
    pub rate_value: i64,
    /// CRF/quality value (kept separate from `rate_value` so toggling rate_mode in the UI doesn't
    /// clobber the other mode's last value). Used as `rate_value` source when rate_mode switches to 1.
    #[serde(default = "default_crf")]
    pub crf: i64,
    #[serde(default = "default_vcodec")]
    pub vcodec: String,
}

fn default_out_w() -> u32 {
    1280
}
fn default_out_h() -> u32 {
    856
}
fn default_fps_num() -> u32 {
    30
}
fn default_fps_den() -> u32 {
    1
}
fn default_bitrate() -> i64 {
    4_000_000
}
fn default_crf() -> i64 {
    23
}
fn default_vcodec() -> String {
    "mpeg4".to_string()
}

impl Default for ExportSettings {
    fn default() -> Self {
        ExportSettings {
            out_w: default_out_w(),
            out_h: default_out_h(),
            fps_num: default_fps_num(),
            fps_den: default_fps_den(),
            rate_mode: 0,
            rate_value: default_bitrate(),
            crf: default_crf(),
            vcodec: default_vcodec(),
        }
    }
}

#[derive(Clone, Default)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Project {
    pub media: Vec<String>, // media file paths; clips index into this
    pub names: Vec<String>, // display names per media
    pub clips: Vec<Clip>,
    #[serde(default)]
    pub trans: Vec<i32>, // LEGACY transition id per boundary (-1 = none); unused — superseded by `transitions`
    // ----- per-boundary transitions (Wave 8; PINNED) -----------------------------------------
    // The real transition store: a list of (track, center, dur, kind) structs anchored at
    // timeline frames, replacing the legacy index-keyed `trans` Vec. `#[serde(default)]` so
    // pre-Wave-8 .json projects still deserialize (the field defaults to empty = no transitions).
    // Produced + edited here via add/remove_transition; queried by Team A (resolve) via
    // transition_at() and by Team C (timeline UI) via boundaries() + transition_at().
    #[serde(default)]
    pub transitions: Vec<Transition>,
    #[serde(default)]
    pub bright: f32,
    #[serde(default)]
    pub contrast: f32,
    #[serde(default)]
    pub sat: f32,
    #[serde(default)]
    pub markers: Vec<i64>, // timeline markers (frames); the scrub/playhead can snap to them

    // ----- keyframe storage (Slice C; all #[serde(default)] so pre-keyframe .json loads) -----
    // Program-wide grade tracks, each a Vec<Kf> sorted ascending by t (timeline frames). An
    // EMPTY track means "use the static bright/contrast/sat field" — grade_at() falls back to
    // the static value so the existing non-animated grade keeps working unchanged. opacity_kf
    // is reserved for a future V2-opacity animation (worker does not yet read it; harmless to
    // store). Consumed by Team A worker::resolve_frame via grade_at(t).
    #[serde(default)]
    pub bright_kf: Vec<Kf>,
    #[serde(default)]
    pub contrast_kf: Vec<Kf>,
    #[serde(default)]
    pub sat_kf: Vec<Kf>,
    #[serde(default)]
    pub opacity_kf: Vec<Kf>,

    // Per-clip PiP keyframes, flat (mirrors MojoMedia PipKf). Each entry binds (clip, param,
    // clip-local frame) -> value. An (clip, param) with NO entries falls back to that clip's
    // static px/py/pw/ph in pip_at(). Consumed by Team A worker::resolve_frame via pip_at().
    #[serde(default)]
    pub pip_kf: Vec<PipKey>,

    // ----- per-track state (PINNED this wave; index 0 = V1, 1 = V2, 2 = A1) -----
    // serde(default) so older .json projects (without these keys) still deserialize to
    // [false; 3]. These fields are exposed for the worker via is_hidden()/is_muted()/
    // is_locked() below. INTEGRATOR WIRING REQUIRED (Team A): worker.rs does NOT yet
    // consult them — resolve_frame() composites by Clip.track alone and build_audio_lines()
    // gates on a static track_is_audible(track). Until Team A calls project.is_hidden(track)
    // in resolve_frame (skip a hidden VIDEO track) and project.is_muted(track) in
    // build_audio_lines (drop a muted track's audio), these toggles change neither the
    // preview nor the export. track_lock is advisory this wave (edits to a locked track
    // should be blocked; timeline.rs already calls is_locked()).
    #[serde(default)]
    pub track_hide: [bool; 3], // true => that VIDEO track is not shown/composited
    #[serde(default)]
    pub track_mute: [bool; 3], // true => that track contributes NO audio to the render
    #[serde(default)]
    pub track_lock: [bool; 3], // true => edits to that track are blocked (advisory)

    // ----- Export / render settings (Triad-B P1) -----
    // serde(default) so pre-P1 .json projects deserialize with today's-behavior defaults
    // (1280×856 @ 30, mpeg4, 4 Mbit/s). Read by worker::render_program → OPEN wire line; edited via
    // the Export Settings block in panels::properties_ui. Decouples the OUTPUT resolution from the
    // fixed GVW×GVH OpenCL working canvas (the encoder swscales the composed frame to out_w×out_h).
    #[serde(default)]
    pub export: ExportSettings,
}

impl Project {
    /// A demo project (3 clips) used until the media pool + import land.
    pub fn demo(media: String) -> Project {
        Project {
            media: vec![media],
            names: vec!["clip".into()],
            clips: vec![
                Clip::video(0, 0, 120, 0, "intro"),
                Clip::video(0, 70, 90, 1, "overlay"),
                Clip::video(0, 0, 160, 2, "audio"),
            ],
            trans: vec![],
            transitions: vec![],
            bright: 0.0,
            contrast: 1.0,
            sat: 1.0,
            markers: vec![],
            track_hide: [false; 3],
            track_mute: [false; 3],
            track_lock: [false; 3],
            bright_kf: vec![],
            contrast_kf: vec![],
            sat_kf: vec![],
            opacity_kf: vec![],
            pip_kf: vec![],
            export: ExportSettings::default(),
        }
    }

    pub fn total_frames(&self) -> i64 {
        self.clips.iter().map(|c| c.end()).max().unwrap_or(1).max(1)
    }

    // ----- keyframe eval (PINNED; consumed by Team A worker::resolve_frame) -------------
    // Both methods return the STATIC field when the relevant track has no keys, so a project
    // with no keyframes behaves exactly as before (worker can call these unconditionally).

    /// (bright, contrast, sat) at timeline frame `t`. Each component linearly interpolates its
    /// keyframe track; an empty track falls back to the static `bright`/`contrast`/`sat` field.
    pub fn grade_at(&self, t: i64) -> (f32, f32, f32) {
        (
            eval_track(&self.bright_kf, t, self.bright),
            eval_track(&self.contrast_kf, t, self.contrast),
            eval_track(&self.sat_kf, t, self.sat),
        )
    }

    /// V2-overlay opacity multiplier at timeline frame `t` from the `opacity_kf` track (Triad-B P1
    /// wiring of the previously-stored-but-unread track, model.rs:177). Returns 1.0 (fully opaque)
    /// when the track is EMPTY, so a project with no opacity keyframes composites the overlay exactly
    /// as before; otherwise it linearly interpolates the keyframes (clamped to the first/last value
    /// outside the range), mirroring `grade_at`. Consumed by worker::resolve_frame, which multiplies
    /// the overlay's composite `op` by this so a keyframed fade of the V2 overlay actually changes
    /// its opacity in BOTH the preview and the render.
    pub fn opacity_at(&self, t: i64) -> f32 {
        eval_track(&self.opacity_kf, t, 1.0)
    }

    /// (px, py, pw, ph) for clip `clip_idx` at CLIP-LOCAL frame `t_local`. Each param linearly
    /// interpolates its per-clip PiP keyframes; a param with no keys for this clip falls back to
    /// the clip's static `px`/`py`/`pw`/`ph`. An out-of-range `clip_idx` returns the full-frame
    /// default (0,0,1,1) so the worker never panics on a stale index.
    pub fn pip_at(&self, clip_idx: usize, t_local: i64) -> (f32, f32, f32, f32) {
        let (sx, sy, sw, sh) = match self.clips.get(clip_idx) {
            Some(c) => (c.px, c.py, c.pw, c.ph),
            None => (0.0, 0.0, 1.0, 1.0),
        };
        (
            self.eval_pip(clip_idx, 0, t_local, sx),
            self.eval_pip(clip_idx, 1, t_local, sy),
            self.eval_pip(clip_idx, 2, t_local, sw),
            self.eval_pip(clip_idx, 3, t_local, sh),
        )
    }

    /// Interpolate one (clip, param) PiP track from the flat `pip_kf` store at clip-local frame
    /// `t`. Mirrors MojoMedia pip_eval: scan the flat list for matching (clip,par) entries,
    /// track the nearest key at/below `t` (lo) and above `t` (hi), then linearly blend; empty
    /// -> `fallback`, clamp to lo/hi at the ends. The flat list is unsorted, so this is an O(n)
    /// scan (n = total PiP keys, small) rather than a binary search.
    fn eval_pip(&self, clip: usize, par: u8, t: i64, fallback: f32) -> f32 {
        let mut lo: Option<(i64, f32)> = None;
        let mut hi: Option<(i64, f32)> = None;
        for k in self.pip_kf.iter().filter(|k| k.clip == clip && k.par == par) {
            if k.t_local <= t {
                if lo.is_none_or(|(lf, _)| k.t_local > lf) {
                    lo = Some((k.t_local, k.v));
                }
            } else if hi.is_none_or(|(hf, _)| k.t_local < hf) {
                hi = Some((k.t_local, k.v));
            }
        }
        match (lo, hi) {
            (None, None) => fallback,
            (Some((_, lv)), None) => lv,          // clamp after the last key
            (None, Some((_, hv))) => hv,          // clamp before the first key
            (Some((lf, lv)), Some((hf, hv))) => {
                let blend = (t - lf) as f64 / (hf - lf) as f64;
                (blend * (hv - lv) as f64) as f32 + lv
            }
        }
    }

    // ----- keyframe edit ops (Slice C; called by panels::properties_ui Key buttons) -----

    /// Snapshot the CURRENT static grade (bright/contrast/sat) into a keyframe at timeline
    /// frame `t` on all three grade tracks (sorted insert-or-replace). Mirrors MojoMedia's
    /// "K" buttons keying brightness/contrast/saturation at the playhead with the live values.
    pub fn add_grade_key(&mut self, t: i64) {
        let (b, c, s) = (self.bright, self.contrast, self.sat);
        set_track(&mut self.bright_kf, t, b);
        set_track(&mut self.contrast_kf, t, c);
        set_track(&mut self.sat_kf, t, s);
    }

    /// Snapshot clip `clip_idx`'s CURRENT static PiP rect (px/py/pw/ph) into PiP keyframes at
    /// CLIP-LOCAL frame `t_local` (one per param 0..3, insert-or-replace). Mirrors MojoMedia's
    /// "Key PiP" button keying all four params at the clip-local frame. No-op for a bad index.
    pub fn add_pip_key(&mut self, clip_idx: usize, t_local: i64) {
        let (px, py, pw, ph) = match self.clips.get(clip_idx) {
            Some(c) => (c.px, c.py, c.pw, c.ph),
            None => return,
        };
        self.set_pip(clip_idx, 0, t_local, px);
        self.set_pip(clip_idx, 1, t_local, py);
        self.set_pip(clip_idx, 2, t_local, pw);
        self.set_pip(clip_idx, 3, t_local, ph);
    }

    /// Insert-or-replace a single PiP keyframe in the flat store (mirrors MojoMedia pip_set):
    /// overwrite the value if an entry already exists for (clip, par, t_local), else append.
    fn set_pip(&mut self, clip: usize, par: u8, t_local: i64, v: f32) {
        if let Some(k) = self
            .pip_kf
            .iter_mut()
            .find(|k| k.clip == clip && k.par == par && k.t_local == t_local)
        {
            k.v = v;
        } else {
            self.pip_kf.push(PipKey { clip, par, t_local, v });
        }
    }

    /// Count of PiP keyframes bound to clip `clip_idx` (any param) — for the panel's key-count
    /// readout. O(n) over the small flat store.
    pub fn pip_key_count(&self, clip_idx: usize) -> usize {
        self.pip_kf.iter().filter(|k| k.clip == clip_idx).count()
    }

    // ----- keyframe DRAG/DELETE edit ops (Slice B; called by timeline diamond/tick drags) -----
    // `track` selects the grade keyframe track: 0 = bright_kf, 1 = contrast_kf, 2 = sat_kf,
    // 3 = opacity_kf. All ops bounds-check (bad track / idx -> no-op) so a stale index from a
    // mid-drag mutation can never panic. PiP ops address the flat `pip_kf` store by index.

    /// Mutable borrow of one grade track by its PINNED index, or `None` for an out-of-range
    /// track. Internal helper for the move/delete ops below.
    fn grade_track_mut(&mut self, track: u8) -> Option<&mut Vec<Kf>> {
        match track {
            0 => Some(&mut self.bright_kf),
            1 => Some(&mut self.contrast_kf),
            2 => Some(&mut self.sat_kf),
            3 => Some(&mut self.opacity_kf),
            _ => None,
        }
    }

    /// Move grade keyframe `idx` of `track` to timeline frame `new_t` (clamped to `>= 0`), then
    /// re-sort that track ascending by `t` so eval/draw stay correct (mirrors MojoMedia kf_set's
    /// sorted ordering — here applied to a moved key rather than a fresh one). The key keeps its
    /// VALUE; only its frame changes. No-op for a bad track or `idx`. If the move lands the key
    /// exactly onto another key's frame, BOTH are kept (a stable sort preserves their relative
    /// order) — eval_track still returns a well-defined value, and a later add_grade_key at that
    /// frame would collapse them via set_track's replace.
    pub fn move_grade_key(&mut self, track: u8, idx: usize, new_t: i64) {
        let nt = new_t.max(0);
        if let Some(t) = self.grade_track_mut(track) {
            if idx < t.len() {
                t[idx].t = nt;
                // stable sort by frame so the moved key slots into ascending order
                t.sort_by_key(|k| k.t);
            }
        }
    }

    /// Delete grade keyframe `idx` of `track` (no-op for a bad track or `idx`). Removing a key
    /// keeps the track sorted (Vec::remove preserves order).
    pub fn delete_grade_key(&mut self, track: u8, idx: usize) {
        if let Some(t) = self.grade_track_mut(track) {
            if idx < t.len() {
                t.remove(idx);
            }
        }
    }

    /// Move PiP keyframe `idx` (index into the flat `pip_kf` store) to clip-local frame
    /// `new_t_local` (clamped to `>= 0`). The flat store is unsorted (eval_pip scans it), so no
    /// re-sort is needed; only the entry's `t_local` changes (its clip/par/value are preserved).
    /// No-op for an out-of-range `idx`.
    pub fn move_pip_key(&mut self, idx: usize, new_t_local: i64) {
        if let Some(k) = self.pip_kf.get_mut(idx) {
            k.t_local = new_t_local.max(0);
        }
    }

    /// Delete PiP keyframe `idx` from the flat store (no-op for an out-of-range `idx`).
    pub fn delete_pip_key(&mut self, idx: usize) {
        if idx < self.pip_kf.len() {
            self.pip_kf.remove(idx);
        }
    }

    // ----- per-track state helpers -------------------------------------
    // `track` is the Clip.track index space: 0 = V1, 1 = V2, 2 = A1. Each helper
    // bounds-checks the index (out-of-range tracks are treated as visible / audible /
    // unlocked) so callers never index a 3-element array out of bounds.

    /// True if the given track's VIDEO is hidden (skipped in base/over resolution).
    pub fn is_hidden(&self, track: u8) -> bool {
        (track as usize) < 3 && self.track_hide[track as usize]
    }

    /// True if the given track's AUDIO is muted (contributes nothing to the render).
    pub fn is_muted(&self, track: u8) -> bool {
        (track as usize) < 3 && self.track_mute[track as usize]
    }

    /// True if edits to the given track are blocked (advisory this wave).
    pub fn is_locked(&self, track: u8) -> bool {
        (track as usize) < 3 && self.track_lock[track as usize]
    }

    // ----- per-boundary transitions (Wave 8; PINNED) -----------------------------------------
    // Transitions are keyed by (track, center frame), NOT by clip index — so unlike the PiP
    // keyframe store they need NO remap on split/delete/reorder; a transition simply animates
    // whatever two clips happen to straddle its window. Callers pass CURRENT clip indices (from
    // boundaries()) into resolve and never cache them across a mutation.

    /// The transition on `track` whose window `[center - dur/2, center + dur/2)` contains `t`,
    /// or `None`. If several overlap `t` (windows can overlap after editing), the one whose
    /// `center` is NEAREST to `t` wins (and on a tie, the earliest in the list) so the worker
    /// gets a single deterministic transition per frame. Consumed by Team A (resolve_frame) to
    /// fill the ENC/PREVIEW `<trans_kind> <trans_prog> <trans_param> <trans_path> <trans_frame>`
    /// fields and by Team C (timeline) to highlight the active boundary.
    pub fn transition_at(&self, track: u8, t: i64) -> Option<&Transition> {
        self.transitions
            .iter()
            .filter(|tr| tr.track == track && tr.contains(t))
            .min_by_key(|tr| (tr.center - t).abs())
    }

    /// Add a transition on `track` centered at `center` over `dur` frames (clamped to `>= 2`)
    /// with the given `kind`. If a transition already exists on the SAME `track` at the SAME
    /// `center`, it is replaced in place (its `dur`/`kind` updated) rather than duplicated —
    /// mirroring MojoMedia's per-boundary "Cycle transition type" which mutates the existing
    /// boundary entry. Otherwise the new transition is pushed.
    pub fn add_transition(&mut self, track: u8, center: i64, dur: i64, kind: i32) {
        let d = dur.max(2);
        if let Some(tr) = self
            .transitions
            .iter_mut()
            .find(|tr| tr.track == track && tr.center == center)
        {
            tr.dur = d;
            tr.kind = kind;
        } else {
            self.transitions.push(Transition { track, center, dur: d, kind });
        }
    }

    /// Remove transition `idx` from the store (no-op for an out-of-range `idx`). `Vec::remove`
    /// preserves the relative order of the remaining transitions.
    pub fn remove_transition(&mut self, idx: usize) {
        if idx < self.transitions.len() {
            self.transitions.remove(idx);
        }
    }

    /// Adjacent/overlapping same-track clip boundaries as `(outgoing_clip_idx, incoming_clip_idx,
    /// boundary_frame)`. Scans the clips on `track` ordered by `t0` (without reordering the
    /// underlying `clips` Vec — indices in the result are into `self.clips`) and, for each
    /// consecutive pair (A then B) that touch or overlap within `BOUNDARY_GAP` frames, emits a
    /// boundary. The boundary frame is `A.end()` when the clips merely abut/gap, or the MIDPOINT
    /// of the overlap when they overlap (so a centered transition window straddles the seam).
    /// Consumed by Team C to place/seed transitions and by Team A to find the incoming partner
    /// clip to stage into slot 2.
    pub fn boundaries(&self, track: u8) -> Vec<(usize, usize, i64)> {
        // Collect (clip_index, t0, end) for this track, then sort by t0 (then end) for a stable
        // left-to-right scan. We sort a list of indices so the emitted indices stay valid into
        // self.clips.
        let mut order: Vec<usize> = (0..self.clips.len())
            .filter(|&i| self.clips[i].track == track)
            .collect();
        order.sort_by_key(|&i| (self.clips[i].t0, self.clips[i].end()));

        let mut out = Vec::new();
        for w in order.windows(2) {
            let a = w[0];
            let b = w[1];
            let a_end = self.clips[a].end();
            let b_t0 = self.clips[b].t0;
            let overlap = b_t0 < a_end; // B starts before A ends
            let gap = (b_t0 - a_end).abs();
            if overlap || gap <= BOUNDARY_GAP {
                // overlap -> seam at the midpoint of [b_t0, a_end); abut/gap -> at A.end().
                let boundary = if overlap {
                    (b_t0 + a_end) / 2
                } else {
                    a_end
                };
                out.push((a, b, boundary));
            }
        }
        out
    }

    // ----- edit ops -----------------------------------------------------
    // Mirror MojoMedia editor/main_editor.mojo: positioned clips (explicit t0),
    // split keeps src_in/len/t0 math identical, trims clamp len>=1 and t0>=0.

    /// Append a clip to the timeline.
    pub fn add_clip(&mut self, clip: Clip) {
        self.clips.push(clip);
    }

    /// Remove clip `i` (no-op if out of range). Also keeps the flat PiP keyframe store
    /// clip-stable: drop every PiP key bound to the removed clip, then shift the `clip` index
    /// of keys for higher clips down by one so they keep pointing at the same clip after the
    /// `Vec::remove`. (Grade keys are program-wide and need no remap.)
    pub fn delete_clip(&mut self, i: usize) {
        if i < self.clips.len() {
            self.clips.remove(i);
            self.pip_kf.retain(|k| k.clip != i);
            for k in self.pip_kf.iter_mut() {
                if k.clip > i {
                    k.clip -= 1;
                }
            }
        }
    }

    /// Split clip `i` at timeline frame `t` into two clips. The second clip starts at
    /// `t` with `src_in += off`, `len -= off`, `t0 = t`, same track/look/etc. Mirrors
    /// the razor in main_editor.mojo (`off = cur_T - seg_t0`; split only when 0 < off < len).
    /// Returns the index of the new (right-hand) clip if a split occurred.
    pub fn split_clip(&mut self, i: usize, t: i64) -> Option<usize> {
        if i >= self.clips.len() {
            return None;
        }
        let off = t - self.clips[i].t0;
        if off <= 0 || off >= self.clips[i].len {
            return None; // playhead not strictly inside the clip body
        }
        let mut right = self.clips[i].clone();
        // left half: shorten to the cut point
        self.clips[i].len = off;
        // right half: advance source + start, shorten remaining length
        right.src_in += off;
        right.len -= off;
        right.t0 = self.clips[i].t0 + off;
        // a fresh fade-in on the right half is not inherited from the left's fade-in
        right.fade_in = 0;
        // the left half no longer carries a fade-out (that belongs to the right edge now)
        self.clips[i].fade_out = 0;
        let idx = i + 1;
        self.clips.insert(idx, right);
        // PiP keyframe clip-stability: inserting the right half at `idx` shifts every clip
        // above `i` up by one, so bump the `clip` index of keys for those clips to match. Keys
        // on the split clip itself (clip == i) stay with the LEFT half — their clip-local
        // frames still measure from the same t0, so they animate the (now shorter) left clip.
        // The right half starts with no PiP keys (it inherits the static rect, like a fresh
        // fade-in), mirroring MojoMedia's razor which does not duplicate per-clip keyframes.
        for k in self.pip_kf.iter_mut() {
            if k.clip > i {
                k.clip += 1;
            }
        }
        Some(idx)
    }

    /// Trim the start of clip `i` to a new timeline start `new_t0`, holding the right
    /// edge fixed. Advances src_in by the delta and shortens len, with len>=1 / t0>=0
    /// clamping. Mirrors a left-edge trim (the inverse of the right-edge trim drag).
    pub fn trim_start(&mut self, i: usize, new_t0: i64) {
        if i >= self.clips.len() {
            return;
        }
        let end = self.clips[i].end(); // right edge stays put
        let src_in = self.clips[i].src_in;
        let t0 = self.clips[i].t0;
        let mut nt0 = new_t0.max(0);
        if nt0 >= end {
            nt0 = end - 1; // keep len >= 1
        }
        // Frames trimmed off the head (negative = extend left). On an extend, the source
        // can only supply `src_in` extra frames before hitting frame 0; clamp the timeline
        // extension to that so len/src_in/t0 stay consistent (no out-of-range source request
        // in worker.rs). This mirrors MojoMedia holding the trim within available source.
        let mut delta = nt0 - t0;
        if delta < 0 && src_in + delta < 0 {
            delta = -src_in; // only extend by what the source has
            nt0 = t0 + delta;
        }
        self.clips[i].src_in = (src_in + delta).max(0);
        self.clips[i].t0 = nt0;
        self.clips[i].len = (end - nt0).max(1);
    }

    /// Trim the end of clip `i` to a new length `new_len` (right-edge drag). Clamps to
    /// MIN_CLIP frames (matching MojoMedia's `nl < 15 → 15` trim floor) so dragging the
    /// right edge left past the start — or snapping it to an earlier edge — cannot collapse
    /// the clip below a usable length.
    pub fn trim_end(&mut self, i: usize, new_len: i64) {
        if i >= self.clips.len() {
            return;
        }
        self.clips[i].len = new_len.max(MIN_CLIP);
    }
}

/// Minimum clip length in frames. Mirrors MojoMedia's trim floor (`nl < 15 → 15`).
pub const MIN_CLIP: i64 = 15;

/// Max abutment/gap (frames) between two same-track clips for `boundaries()` to treat them as a
/// transition-eligible pair. Clips that touch (gap 0), slightly gap, or overlap within this many
/// frames produce a boundary; anything farther apart is two unrelated clips, not a seam.
pub const BOUNDARY_GAP: i64 = 30;

/// Snapshot-based undo/redo for the whole project. `Project` derives `Clone`, so each
/// snapshot is a full copy — simple + correct (mirrors MojoMedia's `Snap` stacks).
///
/// NOTE: not yet wired into `app.rs` (no caller pushes pre-edit state). `#[allow(dead_code)]`
/// keeps the build green under `-D warnings` until the integrator wires undo/redo keybindings
/// + a `push` before each edit. Remove the allow once consumed.
#[allow(dead_code)]
#[derive(Default)]
pub struct History {
    undo: Vec<Project>,
    redo: Vec<Project>,
}

impl History {
    pub fn new() -> History {
        History { undo: Vec::new(), redo: Vec::new() }
    }

    /// Push the *current* project state onto the undo stack before a mutation, clearing
    /// the redo stack (a new edit invalidates the redo future). Call this with the state
    /// as it is *before* applying an edit.
    pub fn push(&mut self, project: &Project) {
        self.undo.push(project.clone());
        self.redo.clear();
    }

    /// Undo: restore the most recent snapshot into `project`, stashing the current state
    /// onto the redo stack. Returns true if anything was undone.
    pub fn undo(&mut self, project: &mut Project) -> bool {
        if let Some(prev) = self.undo.pop() {
            self.redo.push(project.clone());
            *project = prev;
            true
        } else {
            false
        }
    }

    /// Redo: re-apply the most recently undone snapshot, stashing the current state back
    /// onto the undo stack. Returns true if anything was redone.
    pub fn redo(&mut self, project: &mut Project) -> bool {
        if let Some(next) = self.redo.pop() {
            self.undo.push(project.clone());
            *project = next;
            true
        } else {
            false
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grade_keyframe_interp() {
        let mut p = Project::demo("x".into());
        p.bright = 0.0;
        p.add_grade_key(0);
        p.bright = 1.0;
        p.add_grade_key(100);
        assert!((p.grade_at(0).0 - 0.0).abs() < 1e-4);
        assert!((p.grade_at(50).0 - 0.5).abs() < 1e-3, "b50={}", p.grade_at(50).0);
        assert!((p.grade_at(100).0 - 1.0).abs() < 1e-4);
        assert!((p.grade_at(150).0 - 1.0).abs() < 1e-4); // clamp past last key
    }

    #[test]
    fn pip_keyframe_interp() {
        let mut p = Project::demo("x".into());
        p.clips[0].px = 0.0;
        p.add_pip_key(0, 0);
        p.clips[0].px = 0.6;
        p.add_pip_key(0, 100);
        assert!((p.pip_at(0, 0).0 - 0.0).abs() < 1e-4);
        assert!((p.pip_at(0, 50).0 - 0.3).abs() < 1e-3, "px50={}", p.pip_at(0, 50).0);
        assert!((p.pip_at(0, 100).0 - 0.6).abs() < 1e-4);
    }

    #[test]
    fn transition_window_and_progress() {
        let mut p = Project::demo("x".into());
        // centered at 100, dur 20 -> window [90, 110), progress 0 at 90, ~1 at 109.
        p.add_transition(0, 100, 20, 0);
        assert_eq!(p.transitions.len(), 1);
        assert!(p.transition_at(0, 89).is_none(), "before window");
        assert!(p.transition_at(0, 90).is_some(), "window start inclusive");
        assert!(p.transition_at(0, 109).is_some(), "inside window");
        assert!(p.transition_at(0, 110).is_none(), "window end exclusive");
        assert!(p.transition_at(1, 100).is_none(), "other track has no transition");
        let tr = p.transition_at(0, 90).unwrap();
        assert!((tr.progress(90) - 0.0).abs() < 1e-4, "prog start=0");
        assert!((tr.progress(100) - 0.5).abs() < 1e-3, "prog mid=0.5 got {}", tr.progress(100));
        assert!((tr.progress(120) - 1.0).abs() < 1e-4, "prog clamped past end");
    }

    #[test]
    fn transition_add_replaces_same_center_and_clamps_dur() {
        let mut p = Project::demo("x".into());
        p.add_transition(0, 100, 20, 0);
        // same track+center -> replace in place (dur/kind updated), not a duplicate.
        p.add_transition(0, 100, 1, 7);
        assert_eq!(p.transitions.len(), 1, "same track+center replaced");
        assert_eq!(p.transitions[0].kind, 7);
        assert_eq!(p.transitions[0].dur, 2, "dur clamped to >= 2");
        // different center -> a new entry.
        p.add_transition(0, 200, 10, 1);
        assert_eq!(p.transitions.len(), 2);
        p.remove_transition(0);
        assert_eq!(p.transitions.len(), 1);
        assert_eq!(p.transitions[0].center, 200);
        p.remove_transition(99); // out of range -> no-op
        assert_eq!(p.transitions.len(), 1);
    }

    #[test]
    fn transition_at_picks_nearest_center_on_overlap() {
        let mut p = Project::demo("x".into());
        // two overlapping windows: [90,110) center 100 and [95,115) center 105; at t=104 both
        // contain it, nearest center (105) wins.
        p.add_transition(0, 100, 20, 0);
        p.add_transition(0, 105, 20, 1);
        let tr = p.transition_at(0, 104).expect("some transition at 104");
        assert_eq!(tr.center, 105, "nearest center wins");
    }

    #[test]
    fn boundaries_abut_and_overlap() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        // V1 (track 0): A [0,100), B [100,200) abut exactly -> boundary at 100.
        p.clips.push(Clip::video(0, 0, 100, 0, "A"));   // idx 0
        p.clips.push(Clip::video(0, 100, 100, 0, "B")); // idx 1
        // V2 (track 1): C [0,100), D [80,180) overlap -> boundary at midpoint (80+100)/2 = 90.
        p.clips.push(Clip::video(0, 0, 100, 1, "C"));   // idx 2
        p.clips.push(Clip::video(0, 80, 100, 1, "D"));  // idx 3

        let bv1 = p.boundaries(0);
        assert_eq!(bv1.len(), 1);
        assert_eq!(bv1[0], (0, 1, 100));

        let bv2 = p.boundaries(1);
        assert_eq!(bv2.len(), 1);
        assert_eq!(bv2[0], (2, 3, 90));

        // a far-apart pair (gap > BOUNDARY_GAP) produces no boundary.
        let mut q = Project::demo("x".into());
        q.clips.clear();
        q.clips.push(Clip::video(0, 0, 100, 0, "A"));
        q.clips.push(Clip::video(0, 200, 100, 0, "B")); // gap 100 > 30
        assert!(q.boundaries(0).is_empty());
    }
}
