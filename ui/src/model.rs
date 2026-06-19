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

    // ----- P3 per-clip AUDIO FX (consumed by the audio triad: worker builds a libavfilter chain
    // from these and passes it to gcompose's fpx_au_apply). All-neutral default = no audio change,
    // so pre-P3 projects load + render identically. Structured (not a raw filter string) so the UI
    // can present sliders/toggles; the worker maps them to volume/pan/equalizer/acompressor/agate/
    // loudnorm. Mirrors Shotcut's audio_gain/audio_pan/audio_eq3band/compressor/noisegate/normalize.
    #[serde(default)]
    pub audio_fx: AudioFx,

    // ----- P4 per-clip CHROMA KEY (consumed by the chroma triad: worker sends the key params on the
    // wire; gcompose's k_chroma zeroes the OVER clip's alpha where the pixel matches the key color so
    // pip composites only the non-keyed pixels over V1). Disabled by default = no change, so pre-P4
    // projects render identically. Mirrors Shotcut's bluescreen0r (Chroma Key: Simple).
    #[serde(default)]
    pub chroma: ChromaKey,

    // ----- P5 TEXT / TITLE overlay (consumed by the text triad: the worker rasterizes `title.text`
    // with ab_glyph into a full-frame transparent RGBA and composites it over the clip's frame).
    // Empty text = no title (default) so pre-P5 projects are unchanged. Mirrors Shotcut's
    // dynamictext (Text: Simple) filter.
    #[serde(default)]
    pub title: Title,

    // ----- P5 CURVE: a 5-point master tone curve (Shotcut Curves). The 5 outputs are at fixed
    // inputs 0, 0.25, 0.5, 0.75, 1.0; the engine piecewise-linear interpolates and applies it to all
    // 3 channels after blur, before look. Default = identity (y=x) so an un-curved clip is a no-op.
    #[serde(default = "default_curve")]
    pub curve: [f32; 5],

    // ----- P6 STYLIZE / UTILITY filters (consumed by the filter triad; all per-pixel/spatial on the
    // composited OUTB after the curve, before look). All defaults are no-ops so pre-P6 projects are
    // unchanged. Mirror Shotcut's vignette / sharpen / flip / invert+sepia+grayscale+posterize.
    #[serde(default)]
    pub vignette: f32, // 0 = off; 0..1 darkens the frame edges radially
    #[serde(default)]
    pub sharpen: f32, // 0 = off; unsharp amount (~0..2)
    #[serde(default)]
    pub flip: u8, // 0 none, 1 horizontal, 2 vertical, 3 both
    #[serde(default)]
    pub fx: i32, // simple per-pixel FX: 0 none, 1 invert, 2 sepia, 3 grayscale, 4 posterize

    // ----- P7 COLOR filters (consumed by the color triad; per-pixel on OUTB after the P6 filters,
    // before look). Identity defaults = no-ops. Mirror Shotcut's hue/lightness/saturation + levels.
    #[serde(default = "default_hsl")]
    pub hsl: [f32; 3], // [hue_shift_degrees (0), saturation_mult (1), lightness_add (0)]
    #[serde(default = "default_levels")]
    pub levels: [f32; 3], // [in_black (0), in_white (1), gamma (1)]

    // ----- P8 STYLIZE filters (consumed by the P8 triad; on OUTB after the P7 color filters, before
    // look). Identity defaults = no-ops. Mirror Shotcut's mosaic (pixelate) + gradient-map.
    #[serde(default)]
    pub mosaic: u32, // block size in px; 0 or 1 = off (no pixelation)
    #[serde(default)]
    pub gmap_amt: f32, // gradient-map mix 0..1; 0 = off
    #[serde(default = "default_zero3")]
    pub gmap_lo: [f32; 3], // shadow colour (luma 0), default black
    #[serde(default = "default_one3")]
    pub gmap_hi: [f32; 3], // highlight colour (luma 1), default white

    // ----- P9 STYLIZE-3 / FX filters (consumed by the P9 wave; on OUTB after the P8 stylize filters,
    // before look). Identity defaults = no-ops (engine skips each at its off value). Mirror Shotcut's
    // Reduce-Noise (denoise), Glow, and RGB-Shift (chromatic aberration).
    #[serde(default)]
    pub denoise: f32, // edge-preserving denoise strength 0..1; 0 = off
    #[serde(default)]
    pub glow_amt: f32, // glow/bloom mix 0..1; 0 = off
    #[serde(default = "default_glow_thr")]
    pub glow_thr: f32, // glow luma threshold (only pixels brighter than this bloom); default 0.7
    #[serde(default)]
    pub rgbshift: f32, // RGB-shift / chromatic-aberration offset in px (R +shift, B -shift); 0 = off

    // ----- P10 STYLIZE-4 filters (consumed by the P10 wave; on OUTB after the P9 FX filters, before
    // look). Identity defaults = no-ops (engine skips each at its off value). Mirror Shotcut's
    // Halftone, Emboss, and Sketch/Edge-detect.
    #[serde(default)]
    pub halftone: u32, // halftone cell size in px; 0 or 1 = off
    #[serde(default)]
    pub emboss: f32, // emboss relief strength 0..1; 0 = off
    #[serde(default)]
    pub edge: f32, // edge-detect (sketch) mix 0..1; 0 = off

    // ----- P13 OLD-FILM / DISTORT filters (consumed by the P13 wave; on OUTB after the P10 stylize-4
    // filters, before look). Identity defaults = no-ops (engine skips each at its off value). Mirror
    // Shotcut's Old Film: Grain, Old Film: Scratches, and a Diffusion (frosted-glass) distort.
    #[serde(default)]
    pub grain: f32, // film-grain noise strength 0..1; 0 = off
    #[serde(default)]
    pub scratches: f32, // old-film vertical scratch density/amount 0..1; 0 = off
    #[serde(default)]
    pub diffusion: f32, // diffusion / frosted-glass radius in px (0..16); 0 = off

    // ----- P16 DISTORT filters (consumed by the P16 wave; on OUTB after the P13 old-film filters,
    // before look). Identity defaults = no-ops (engine skips each at its off value). Mirror Shotcut's
    // Wave, Swirl, and Threshold distort/stylize filters.
    #[serde(default)]
    pub wave: f32, // sinusoidal wave displacement amplitude in px; 0 = off
    #[serde(default)]
    pub swirl: f32, // swirl rotation strength in radians (at centre); 0 = off
    #[serde(default)]
    pub threshold: f32, // luma threshold/binarize level 0..1; 0 = off

    // ----- P17 GEOMETRIC/DISTORT filters (consumed by the P17 wave; on OUTB after the P16 distort
    // filters, before look). Identity defaults = no-ops. Mirror Shotcut's Lens Correction, Crop, and
    // a Glitch (per-band channel shift).
    #[serde(default)]
    pub lens: f32, // lens distortion: + barrel / - pincushion (radial); 0 = off
    #[serde(default)]
    pub crop: f32, // crop margin fraction 0..0.49 (outside -> black); 0 = off
    #[serde(default)]
    pub glitch: f32, // glitch per-band horizontal channel shift, max px; 0 = off

    // ----- P23 360 REFRAME (consumed by the P23 wave; on OUTB after the P17 geometry filters, before
    // look). When `eq360` is true the clip is treated as a 360 equirectangular source and reprojected
    // to a flat rectilinear view at (eq_yaw, eq_pitch) with field-of-view eq_fov — the standard
    // equirect->rectilinear "360 viewer" (mirrors Shotcut/bigsh0t's 360 reframe). eq360=false = no-op.
    #[serde(default)]
    pub eq360: bool, // enable 360 equirectangular -> rectilinear reprojection
    #[serde(default)]
    pub eq_yaw: f32, // view yaw (degrees), 0 = forward
    #[serde(default)]
    pub eq_pitch: f32, // view pitch (degrees), 0 = level
    #[serde(default = "default_eq_fov")]
    pub eq_fov: f32, // view field-of-view (degrees), default 90

    // ----- P24 CLIP SPEED / TIME-REMAP + REVERSE (consumed by the P24 wave). Model A: the clip keeps
    // its timeline footprint (t0,len); `speed` scales how fast the SOURCE is consumed (2.0 = 2x faster
    // / reads every other source frame; 0.5 = slow-mo), and `reverse` plays the consumed source range
    // backward. Identity speed=1.0 + reverse=false reads src_in+(t-t0) exactly (byte-identical).
    #[serde(default = "default_speed")]
    pub speed: f32, // source consumption rate; 1.0 = normal, 2.0 = 2x faster, 0.5 = slow-mo
    #[serde(default)]
    pub reverse: bool, // play the consumed source range backward
}

/// serde default for `Clip.eq_fov`: a 90° rectilinear field of view.
fn default_eq_fov() -> f32 {
    90.0
}

/// serde default for `Clip.speed`: normal (1.0) playback rate. Pre-P24 projects load at 1.0 → identity.
fn default_speed() -> f32 {
    1.0
}

/// serde default [0,0,0] (gradient-map shadow colour = black).
fn default_zero3() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}

/// serde default for `Clip.glow_thr`: only pixels brighter than 0.7 luma contribute to the glow.
fn default_glow_thr() -> f32 {
    0.7
}

/// serde default [1,1,1] (gradient-map highlight colour = white).
fn default_one3() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

/// serde default for `Clip.hsl`: identity (no hue shift, unit saturation, no lightness change).
fn default_hsl() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

/// serde default for `Clip.levels`: identity (in 0..1, gamma 1).
fn default_levels() -> [f32; 3] {
    [0.0, 1.0, 1.0]
}

/// serde default for `Clip.curve`: the identity tone curve (outputs == inputs at the 5 control points).
fn default_curve() -> [f32; 5] {
    [0.0, 0.25, 0.5, 0.75, 1.0]
}

/// Per-clip text/title overlay (P5). `text` empty (default) is a no-op. `size_frac` is the font
/// height as a fraction of the frame height; `x`/`y` are the normalized top-left anchor in [0,1];
/// `rgb` is the text colour in [0,1]. Mirrors Shotcut's Text: Simple (dynamictext).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Title {
    pub text: String,
    pub size_frac: f32, // font height / frame height, e.g. 0.1
    pub x: f32,         // normalized left anchor [0,1]
    pub y: f32,         // normalized top anchor [0,1]
    pub rgb: [f32; 3],  // text colour [0,1]
}

impl Default for Title {
    fn default() -> Self {
        Title { text: String::new(), size_frac: 0.1, x: 0.05, y: 0.05, rgb: [1.0, 1.0, 1.0] }
    }
}

impl Title {
    /// True when there is no text to render (the worker then composites the clip normally).
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// A "lower third" preset (Shotcut's common title placement): the given text anchored toward
    /// the lower-left of the frame at a modest size, white. A convenience for the title-editor UI's
    /// preset button — it only sets the layout/colour fields (worker reads them unchanged), so a
    /// project that never builds one is byte-identical. `size_frac`/`x`/`y` are normalized as on the
    /// struct (font height / frame height; top-left anchor in [0,1]).
    pub fn lower_third(text: &str) -> Title {
        Title { text: text.to_string(), size_frac: 0.07, x: 0.06, y: 0.78, rgb: [1.0, 1.0, 1.0] }
    }
}

/// Per-clip chroma-key (green-screen) settings (P4). `enabled=false` (default) is a no-op: the worker
/// sends a disabled sentinel and the engine skips keying, so the composite is identical to P3. Applies
/// to a clip when it is the V2 OVERLAY (keyed pixels become transparent so V1 shows through).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ChromaKey {
    pub enabled: bool,
    pub key: [f32; 3],   // key colour RGB in [0,1], default green [0,1,0]
    pub similarity: f32, // 0..1 colour-distance threshold to key out (larger = more keyed), def 0.4
    pub smoothness: f32, // 0..1 edge softness band beyond `similarity`, def 0.1
}

impl Default for ChromaKey {
    fn default() -> Self {
        ChromaKey { enabled: false, key: [0.0, 1.0, 0.0], similarity: 0.4, smoothness: 0.1 }
    }
}

/// Per-clip audio-filter settings (P3). Neutral default (all 0 / false) is a no-op: the worker emits
/// no audio filter chain, so the mix is byte-identical to P2. Ranges mirror Shotcut's audio filters.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioFx {
    pub eq_low_db: f32,   // low-shelf gain, dB (Shotcut EQ: 3-band), 0 = flat
    pub eq_mid_db: f32,   // mid peak gain, dB, 0 = flat
    pub eq_high_db: f32,  // high-shelf gain, dB, 0 = flat
    pub pan: f32,         // -1 = full left, 0 = center, +1 = full right
    pub compress: bool,   // acompressor (sensible defaults)
    pub gate: bool,       // agate
    pub normalize: bool,  // loudnorm (single-pass)
    // ----- P11 per-clip audio effects (Shotcut Reverb / Delay / Pitch). All `#[serde(default ..)]`
    // so pre-P11 .json (an audio_fx object lacking these keys) loads to the neutral, off state. Each
    // is a no-op at its default, so is_neutral() stays true and the chain stays "-" (P10 identity).
    #[serde(default)]
    pub reverb: f32,      // reverb amount 0..1 (0 = off) → multi-tap aecho
    #[serde(default)]
    pub delay_ms: f32,    // echo delay in ms (0 = off) → aecho
    #[serde(default = "default_delay_decay")]
    pub delay_decay: f32, // echo feedback 0..0.95 (only meaningful when delay_ms>0)
    #[serde(default)]
    pub pitch: f32,       // pitch shift in SEMITONES (0 = off) → rubberband (tempo-preserving)
    // ----- P12 per-clip audio filters (Shotcut Low Pass / High Pass / Tremolo). All `#[serde(default)]`
    // (each defaults to 0.0), so pre-P12 .json (an audio_fx object lacking these keys) loads to the
    // neutral, off state. Each is a no-op at its default, so is_neutral() stays true and the chain
    // stays "-" (identity preserved).
    #[serde(default)]
    pub lowpass_hz: f32,  // low-pass cutoff in Hz (0 = off) → lowpass=f=<hz>
    #[serde(default)]
    pub highpass_hz: f32, // high-pass cutoff in Hz (0 = off) → highpass=f=<hz>
    #[serde(default)]
    pub tremolo: f32,     // tremolo depth 0..0.95 (0 = off) → tremolo=f=5:d=<depth>
    // ----- P15 per-clip audio filters (Shotcut Bass & Treble / Notch / Chorus). All `#[serde(default)]`
    // (each defaults to 0.0), so pre-P15 .json (an audio_fx object lacking these keys) loads to the
    // neutral, off state. Each is a no-op at its default, so is_neutral() stays true and the chain
    // stays "-" (identity preserved).
    #[serde(default)]
    pub bass_db: f32,    // low-shelf gain in dB (0 = flat / off) → bass=g=<db>
    #[serde(default)]
    pub treble_db: f32,  // high-shelf gain in dB (0 = flat / off) → treble=g=<db>
    #[serde(default)]
    pub notch_hz: f32,   // band-reject centre frequency in Hz (0 = off) → bandreject=f=<hz>
    #[serde(default)]
    pub chorus: f32,     // chorus depth 0..1 (0 = off) → chorus=0.5:0.9:50:0.4:0.25:<2*depth ms>
    // ----- P22 per-clip audio filters (Shotcut Flanger / Phaser / Limiter). All `#[serde(default)]`
    // (each defaults to 0.0), so pre-P22 .json (an audio_fx object lacking these keys) loads to the
    // neutral, off state. Each is a no-op at its default, so is_neutral() stays true and the chain
    // stays "-" (identity preserved).
    #[serde(default)]
    pub flanger: f32,    // flanger depth 0..1 (0 = off) → flanger=depth=<0..8 ms>:speed=0.5
    #[serde(default)]
    pub phaser: f32,     // phaser intensity 0..1 (0 = off) → aphaser=speed=<0.1..2.1 Hz>
    #[serde(default)]
    pub limiter: f32,    // limiter peak ceiling 0..1 (0 = off) → alimiter=limit=<0.05..1.0 linear>
}

impl Default for AudioFx {
    fn default() -> Self {
        AudioFx {
            eq_low_db: 0.0,
            eq_mid_db: 0.0,
            eq_high_db: 0.0,
            pan: 0.0,
            compress: false,
            gate: false,
            normalize: false,
            reverb: 0.0,
            delay_ms: 0.0,
            delay_decay: default_delay_decay(),
            pitch: 0.0,
            lowpass_hz: 0.0,
            highpass_hz: 0.0,
            tremolo: 0.0,
            bass_db: 0.0,
            treble_db: 0.0,
            notch_hz: 0.0,
            chorus: 0.0,
            flanger: 0.0,
            phaser: 0.0,
            limiter: 0.0,
        }
    }
}

/// serde default for `AudioFx.delay_decay`: 0.5 (echo feedback midpoint). Kept as a fn so a pre-P11
/// project that has an `audio_fx` object without `delay_decay` deserializes to 0.5 rather than 0.0 —
/// 0.5 is the neutral resting value the UI shows, and decay alone never makes the FX non-neutral.
fn default_delay_decay() -> f32 {
    0.5
}

impl AudioFx {
    /// True when every control is at its neutral value — the worker can then skip the audio filter
    /// chain entirely (no fpx_au_apply call), keeping the no-FX mix path byte-identical to P2.
    pub fn is_neutral(&self) -> bool {
        self.eq_low_db == 0.0
            && self.eq_mid_db == 0.0
            && self.eq_high_db == 0.0
            && self.pan == 0.0
            && !self.compress
            && !self.gate
            && !self.normalize
            // P11: only the "active when > 0" effects gate neutrality. `delay_decay` is a parameter
            // of the delay (not an effect by itself), so a clip with the default decay 0.5 but no
            // delay_ms stays neutral and still emits "-".
            && self.reverb == 0.0
            && self.delay_ms == 0.0
            && self.pitch == 0.0
            // P12: low-pass / high-pass / tremolo each gate neutrality only when active (> 0). All
            // default 0.0, so a pre-P12 clip (and any clip with these untouched) stays neutral → "-".
            && self.lowpass_hz == 0.0
            && self.highpass_hz == 0.0
            && self.tremolo == 0.0
            // P15: bass / treble shelves, notch (band-reject) and chorus each gate neutrality only
            // when active. bass_db / treble_db are "off" at 0 (flat shelf); notch_hz / chorus are
            // "off" at 0. All default 0.0, so a pre-P15 clip (and any clip untouched) stays neutral → "-".
            && self.bass_db == 0.0
            && self.treble_db == 0.0
            && self.notch_hz == 0.0
            && self.chorus == 0.0
            // P22: flanger / phaser / limiter each gate neutrality only when active (> 0). All
            // default 0.0, so a pre-P22 clip (and any clip untouched) stays neutral → "-".
            && self.flanger == 0.0
            && self.phaser == 0.0
            && self.limiter == 0.0
    }
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
            audio_fx: AudioFx::default(),
            chroma: ChromaKey::default(),
            title: Title::default(),
            curve: default_curve(),
            vignette: 0.0,
            sharpen: 0.0,
            flip: 0,
            fx: 0,
            hsl: default_hsl(),
            levels: default_levels(),
            mosaic: 0,
            gmap_amt: 0.0,
            gmap_lo: default_zero3(),
            gmap_hi: default_one3(),
            denoise: 0.0,
            glow_amt: 0.0,
            glow_thr: default_glow_thr(),
            rgbshift: 0.0,
            halftone: 0,
            emboss: 0.0,
            edge: 0.0,
            grain: 0.0,
            scratches: 0.0,
            diffusion: 0.0,
            wave: 0.0,
            swirl: 0.0,
            threshold: 0.0,
            lens: 0.0,
            crop: 0.0,
            glitch: 0.0,
            eq360: false,
            eq_yaw: 0.0,
            eq_pitch: 0.0,
            eq_fov: default_eq_fov(),
            speed: default_speed(),
            reverse: false,
        }
    }
    pub fn end(&self) -> i64 {
        self.t0 + self.len
    }

    /// True when this clip carries a non-empty TITLE overlay (P5) — the worker then rasterizes
    /// `title.text` into a full-frame transparent RGBA and composites it over the clip's frame. An
    /// empty title (the default) returns false, so a pre-P5 / untitled clip is unchanged. Mirrors
    /// the `Title::is_empty` no-op contract: `is_title()` ≡ `!self.title.is_empty()`.
    // Retained predicate (unit-tested via `title_is_empty_and_clip_is_title`): the worker inlines
    // `!title.is_empty()` in resolve_frame, so this convenience accessor is currently unwired.
    #[allow(dead_code)]
    pub fn is_title(&self) -> bool {
        !self.title.is_empty()
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

/// Keyframe interpolation TYPE (P14), mirroring Shotcut/MLT's practical keyframe modes. The interp
/// is PER-KEYFRAME and controls the curve of the segment whose LOWER keyframe carries it:
///   - `Discrete`: HOLD the lower keyframe's value until the next key (step). MLT "discrete".
///   - `Linear`:   straight-line blend between the two keys (the pre-P14 behavior). MLT "linear".
///   - `Smooth`:   smoothstep ease-in/out (`s = b*b*(3-2b)`). This is an HONEST approximation of
///                 Shotcut's "Smooth" mode — it is a smoothstep ease, NOT a bit-exact MLT
///                 Catmull-Rom spline. Endpoints match Linear (eased toward the mid).
/// `Default = Linear`, and the `#[serde(default)]` on the `interp` fields means a pre-P14 .json
/// keyframe (a bare `{t,v}` with no `interp` key) deserializes as `Linear` — so an old project's
/// render is byte-identical (Linear is the previous, only mode).
#[derive(Clone, Copy, PartialEq, Debug)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum KfInterp {
    Discrete,
    Linear,
    Smooth,
    // P19: MLT-exact Catmull-Rom smooth variants (mlt_animation.c interpolate_value /
    // catmull_rom_interpolate). Each maps to a (alpha, tension) pair fed to the same spline:
    //   SmoothNatural = centripetal, peak-flattening (alpha 0.5, tension -1.0)  [MLT smooth_natural]
    //   SmoothLoose   = uniform Catmull-Rom, overshoots    (alpha 0.0, tension 1.0)  [MLT smooth_loose / "~"]
    //   SmoothTight   = zero tangents (= smoothstep ease)  (alpha 0.5, tension 0.0)  [MLT smooth_tight]
    SmoothNatural,
    SmoothLoose,
    SmoothTight,
    // P20: MLT easing keyframe types (mlt_animation.c interpolate_value, Robert-Penner easings).
    // Each is a closed-form factor on the linear blend (no neighbours needed) — see `ease_factor`.
    SineIn, SineOut, SineInOut,
    QuadIn, QuadOut, QuadInOut,
    CubicIn, CubicOut, CubicInOut,
    QuartIn, QuartOut, QuartInOut,
    QuintIn, QuintOut, QuintInOut,
    ExpoIn, ExpoOut, ExpoInOut,
    CircIn, CircOut, CircInOut,
    BackIn, BackOut, BackInOut,
    ElasticIn, ElasticOut, ElasticInOut,
    BounceIn, BounceOut, BounceInOut,
}

impl KfInterp {
    /// The MLT (alpha, tension) pair for the Catmull-Rom variants; `None` for the non-spline kinds
    /// (Discrete/Linear/Smooth + the easings, which use the 2-point `interp_segment`).
    fn catmull_params(self) -> Option<(f64, f64)> {
        match self {
            KfInterp::SmoothNatural => Some((0.5, -1.0)),
            KfInterp::SmoothLoose => Some((0.0, 1.0)),
            KfInterp::SmoothTight => Some((0.5, 0.0)),
            _ => None,
        }
    }

    /// Human label for the keyframe-interp picker (single source of truth for the UI combo).
    pub fn label(self) -> &'static str {
        use KfInterp::*;
        match self {
            Discrete => "Discrete (hold)", Linear => "Linear", Smooth => "Smooth (eased)",
            SmoothNatural => "Smooth Natural", SmoothLoose => "Smooth Loose", SmoothTight => "Smooth Tight",
            SineIn => "Sine In", SineOut => "Sine Out", SineInOut => "Sine In-Out",
            QuadIn => "Quad In", QuadOut => "Quad Out", QuadInOut => "Quad In-Out",
            CubicIn => "Cubic In", CubicOut => "Cubic Out", CubicInOut => "Cubic In-Out",
            QuartIn => "Quart In", QuartOut => "Quart Out", QuartInOut => "Quart In-Out",
            QuintIn => "Quint In", QuintOut => "Quint Out", QuintInOut => "Quint In-Out",
            ExpoIn => "Expo In", ExpoOut => "Expo Out", ExpoInOut => "Expo In-Out",
            CircIn => "Circ In", CircOut => "Circ Out", CircInOut => "Circ In-Out",
            BackIn => "Back In", BackOut => "Back Out", BackInOut => "Back In-Out",
            ElasticIn => "Elastic In", ElasticOut => "Elastic Out", ElasticInOut => "Elastic In-Out",
            BounceIn => "Bounce In", BounceOut => "Bounce Out", BounceInOut => "Bounce In-Out",
        }
    }

    /// All keyframe-interp kinds, in picker order (the UI iterates this).
    pub const ALL: [KfInterp; 36] = {
        use KfInterp::*;
        [
            Discrete, Linear, Smooth, SmoothNatural, SmoothLoose, SmoothTight,
            SineIn, SineOut, SineInOut, QuadIn, QuadOut, QuadInOut,
            CubicIn, CubicOut, CubicInOut, QuartIn, QuartOut, QuartInOut,
            QuintIn, QuintOut, QuintInOut, ExpoIn, ExpoOut, ExpoInOut,
            CircIn, CircOut, CircInOut, BackIn, BackOut, BackInOut,
            ElasticIn, ElasticOut, ElasticInOut, BounceIn, BounceOut, BounceInOut,
        ]
    };
}

/// Easing direction (Robert-Penner): the three phases each easing family comes in.
#[derive(Clone, Copy)]
enum EaseDir {
    In,
    Out,
    InOut,
}

/// MLT easing FACTOR for an easing `KfInterp` at fractional progress `t` ∈ [0,1] — `None` for the
/// non-easing kinds. Each family is a verbatim port of the matching function in MLT's
/// mlt_animation.c (sinusoidal/power/exponential/circular/back/elastic/bounce). The caller applies
/// it as `y1 + (y2-y1)*factor`, exactly like MLT.
fn ease_factor(kind: KfInterp, t: f64) -> Option<f64> {
    use EaseDir::*;
    use KfInterp::*;
    let f = match kind {
        SineIn => ease_sine(t, In), SineOut => ease_sine(t, Out), SineInOut => ease_sine(t, InOut),
        QuadIn => ease_pow(t, 2.0, In), QuadOut => ease_pow(t, 2.0, Out), QuadInOut => ease_pow(t, 2.0, InOut),
        CubicIn => ease_pow(t, 3.0, In), CubicOut => ease_pow(t, 3.0, Out), CubicInOut => ease_pow(t, 3.0, InOut),
        QuartIn => ease_pow(t, 4.0, In), QuartOut => ease_pow(t, 4.0, Out), QuartInOut => ease_pow(t, 4.0, InOut),
        QuintIn => ease_pow(t, 5.0, In), QuintOut => ease_pow(t, 5.0, Out), QuintInOut => ease_pow(t, 5.0, InOut),
        ExpoIn => ease_expo(t, In), ExpoOut => ease_expo(t, Out), ExpoInOut => ease_expo(t, InOut),
        CircIn => ease_circ(t, In), CircOut => ease_circ(t, Out), CircInOut => ease_circ(t, InOut),
        BackIn => ease_back(t, In), BackOut => ease_back(t, Out), BackInOut => ease_back(t, InOut),
        ElasticIn => ease_elastic(t, In), ElasticOut => ease_elastic(t, Out), ElasticInOut => ease_elastic(t, InOut),
        BounceIn => ease_bounce(t, In), BounceOut => ease_bounce(t, Out), BounceInOut => ease_bounce(t, InOut),
        _ => return None,
    };
    Some(f)
}

// --- Robert-Penner easing factors, verbatim from MLT mlt_animation.c ---
fn ease_sine(t: f64, e: EaseDir) -> f64 {
    use std::f64::consts::{PI, FRAC_PI_2};
    match e {
        EaseDir::In => (t - 1.0).mul_add(FRAC_PI_2, 0.0).sin() + 1.0,
        EaseDir::Out => (t * FRAC_PI_2).sin(),
        EaseDir::InOut => 0.5 * (1.0 - (t * PI).cos()),
    }
}
fn ease_pow(t: f64, order: f64, e: EaseDir) -> f64 {
    match e {
        EaseDir::In => t.powf(order),
        EaseDir::Out => 1.0 - (1.0 - t).powf(order),
        EaseDir::InOut => {
            if t < 0.5 {
                2f64.powf(order) * t.powf(order) / 2.0
            } else {
                1.0 - (-2.0 * t + 2.0).powf(order) / 2.0
            }
        }
    }
}
fn ease_expo(t: f64, e: EaseDir) -> f64 {
    if t == 0.0 {
        return 0.0;
    }
    if t == 1.0 {
        return 1.0;
    }
    match e {
        EaseDir::In => 2f64.powf(10.0 * t - 10.0),
        EaseDir::Out => 1.0 - 2f64.powf(-10.0 * t),
        EaseDir::InOut => {
            if t < 0.5 {
                2f64.powf(20.0 * t - 10.0) / 2.0
            } else {
                (2.0 - 2f64.powf(-20.0 * t + 10.0)) / 2.0
            }
        }
    }
}
fn ease_circ(t: f64, e: EaseDir) -> f64 {
    match e {
        EaseDir::In => 1.0 - (1.0 - t.powi(2)).sqrt(),
        EaseDir::Out => (1.0 - (t - 1.0).powi(2)).sqrt(),
        EaseDir::InOut => {
            if t < 0.5 {
                0.5 * (1.0 - (1.0 - 4.0 * (t * t)).sqrt())
            } else {
                0.5 * ((-((2.0 * t) - 3.0) * ((2.0 * t) - 1.0)).sqrt() + 1.0)
            }
        }
    }
}
fn ease_back(t: f64, e: EaseDir) -> f64 {
    use std::f64::consts::PI;
    match e {
        EaseDir::In => t * t * t - t * (t * PI).sin(),
        EaseDir::Out => {
            let f = 1.0 - t;
            1.0 - (f * f * f - f * (f * PI).sin())
        }
        EaseDir::InOut => {
            if t < 0.5 {
                let f = 2.0 * t;
                0.5 * (f * f * f - f * (f * PI).sin())
            } else {
                let f = 1.0 - (2.0 * t - 1.0);
                0.5 * (1.0 - (f * f * f - f * (f * PI).sin())) + 0.5
            }
        }
    }
}
fn ease_elastic(t: f64, e: EaseDir) -> f64 {
    use std::f64::consts::FRAC_PI_2;
    let c = 13.0 * FRAC_PI_2;
    match e {
        EaseDir::In => (c * t).sin() * 2f64.powf(10.0 * (t - 1.0)),
        EaseDir::Out => (-c * (t + 1.0)).sin() * 2f64.powf(-10.0 * t) + 1.0,
        EaseDir::InOut => {
            if t < 0.5 {
                0.5 * (c * (2.0 * t)).sin() * 2f64.powf(10.0 * ((2.0 * t) - 1.0))
            } else {
                0.5 * ((-c * ((2.0 * t - 1.0) + 1.0)).sin() * 2f64.powf(-10.0 * (2.0 * t - 1.0)) + 2.0)
            }
        }
    }
}
fn ease_bounce(t: f64, e: EaseDir) -> f64 {
    match e {
        EaseDir::In => 1.0 - ease_bounce(1.0 - t, EaseDir::Out),
        EaseDir::Out => {
            if t < 4.0 / 11.0 {
                (121.0 * t * t) / 16.0
            } else if t < 8.0 / 11.0 {
                (363.0 / 40.0 * t * t) - (99.0 / 10.0 * t) + 17.0 / 5.0
            } else if t < 9.0 / 10.0 {
                (4356.0 / 361.0 * t * t) - (35442.0 / 1805.0 * t) + 16061.0 / 1805.0
            } else {
                (54.0 / 5.0 * t * t) - (513.0 / 25.0 * t) + 268.0 / 25.0
            }
        }
        EaseDir::InOut => {
            if t < 0.5 {
                0.5 * ease_bounce(t * 2.0, EaseDir::In)
            } else {
                0.5 * ease_bounce(2.0 * t - 1.0, EaseDir::Out) + 0.5
            }
        }
    }
}

impl Default for KfInterp {
    fn default() -> Self {
        KfInterp::Linear
    }
}

/// One keyframe on a scalar track: `v` is the value at timeline (or clip-local) frame `t`.
/// Mirrors MojoMedia's parallel `KfTrack { frames, values }` but as a real struct (the Rust
/// win): a `Vec<Kf>` kept sorted ascending by `t` replaces the two parallel lists.
/// `interp` (P14) is this keyframe's interpolation type; it controls the SEGMENT that STARTS at
/// this key (i.e. the curve from this key up to the next). `#[serde(default)]` → pre-P14 `{t,v}`
/// keyframes load as `Linear`.
#[derive(Clone, Copy)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Kf {
    pub t: i64,
    pub v: f32,
    #[serde(default)]
    pub interp: KfInterp,
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
    // P14 interpolation type; controls the SEGMENT starting at this key (the lower key of an
    // (clip,par) pair). `#[serde(default)]` → pre-P14 flat PiP keyframes load as `Linear`.
    #[serde(default)]
    pub interp: KfInterp,
}

/// Keyframe eval shared by grade + PiP: value of a sorted-ascending `Vec<Kf>` at `t`, or
/// `fallback` when the track is empty. Clamps to the first/last value outside the range. The
/// interpolation of the segment `[i, i+1]` is selected by the LOWER keyframe's `interp` (P14):
/// `Discrete` HOLDS `track[i].v`, `Linear` is the pre-P14 straight blend, `Smooth` applies a
/// smoothstep ease. See `interp_segment` for the shared blend math (also used by `eval_pip`).
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
    let kind = track[i].interp;
    // P19: the MLT Catmull-Rom variants need the two NEIGHBOURING keys (the one before `i` and the
    // one after `i+1`) for their tangents. At the ends, duplicate the boundary key — MLT's
    // catmull_rom_interpolate then shoves the duplicate ±10000 frames away to make a horizontal end
    // tangent. `progress` is the fractional position in [i, i+1] (MLT's (frame-p1)/(p2-p1)).
    if let Some((alpha, tension)) = kind.catmull_params() {
        let p1 = track[i];
        let p2 = track[i + 1];
        let p0 = if i > 0 { track[i - 1] } else { p1 };
        let p3 = if i + 2 < n { track[i + 2] } else { p2 };
        let prog = (t - p1.t) as f64 / (p2.t - p1.t) as f64;
        return catmull_rom(
            p0.t as f64, p0.v as f64, p1.t as f64, p1.v as f64,
            p2.t as f64, p2.v as f64, p3.t as f64, p3.v as f64,
            prog, alpha, tension,
        ) as f32;
    }
    interp_segment(kind, track[i].t, track[i].v, track[i + 1].t, track[i + 1].v, t)
}

/// Euclidean distance between two control points (MLT `distance`, mlt_animation.c).
fn kf_distance(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
}

/// MLT-exact Catmull-Rom spline (mlt_animation.c `catmull_rom_interpolate`), translated line-for-line.
/// 4 control points by FRAME (x) + value (y): `(x0,y0)` before, `(x1,y1)` segment start, `(x2,y2)`
/// segment end, `(x3,y3)` after; `t` ∈ [0,1] is the fractional progress between p1 and p2; `alpha`
/// selects the parameterisation (0 uniform / 0.5 centripetal / 1 chordal); `tension` scales the
/// tangents (|tension|; sign + the monotonic-between-neighbours test gate whether a tangent is
/// computed at all, so a peak gets a flat tangent = no overshoot). Returns the interpolated value.
#[allow(clippy::too_many_arguments)]
fn catmull_rom(
    mut x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64, mut x3: f64, y3: f64,
    t: f64, alpha: f64, tension: f64,
) -> f64 {
    // Duplicated boundary point → push it far away so the end segment gets a horizontal tangent.
    if x0 == x1 {
        x0 -= 10000.0;
    }
    if x3 == x2 {
        x3 += 10000.0;
    }
    let mut m1 = 0.0;
    let mut m2 = 0.0;
    let t12 = kf_distance(x1, y1, x2, y2).powf(alpha);
    if tension > 0.0 || (y1 < y0 && y1 > y2) || (y1 > y0 && y1 < y2) {
        let t01 = kf_distance(x0, y0, x1, y1).powf(alpha);
        m1 = tension.abs() * (y2 - y1 + t12 * ((y1 - y0) / t01 - (y2 - y0) / (t01 + t12)));
    }
    if tension > 0.0 || (y2 < y1 && y2 > y3) || (y2 > y1 && y2 < y3) {
        let t23 = kf_distance(x2, y2, x3, y3).powf(alpha);
        m2 = tension.abs() * (y2 - y1 + t12 * ((y3 - y2) / t23 - (y3 - y1) / (t12 + t23)));
    }
    let a = 2.0 * (y1 - y2) + m1 + m2;
    let b = -3.0 * (y1 - y2) - m1 - m1 - m2;
    let c = m1;
    let d = y1;
    a * t * t * t + b * t * t + c * t + d
}

/// Evaluate a single keyframe SEGMENT at frame `t` using the lower key's interpolation `kind`.
/// `(fa, va)` is the lower keyframe, `(fb, vb)` the upper; `t` is assumed in `[fa, fb)` (the
/// endpoint-clamp cases are handled by the callers). With `blend = (t-fa)/(fb-fa) ∈ [0,1)`:
///   - `Discrete`: return `va` (HOLD until the next key — step interpolation).
///   - `Linear`:   `va + blend*(vb-va)` (the pre-P14 behavior, unchanged).
///   - `Smooth`:   `s = blend*blend*(3 - 2*blend)` (smoothstep ease-in/out), return `va + s*(vb-va)`.
/// Shared by `eval_track` (grade tracks) and `eval_pip` (flat PiP store) so both honor the same
/// per-segment curve. A degenerate `fb == fa` would only arise from coincident keys; the callers
/// never feed that case (eval_track advances past equal-frame keys; eval_pip's lo<t<hi guarantees
/// fb>fa), but Discrete still returns `va` safely regardless.
fn interp_segment(kind: KfInterp, fa: i64, va: f32, fb: i64, vb: f32, t: i64) -> f32 {
    match kind {
        KfInterp::Discrete => va,
        KfInterp::Linear => {
            let blend = (t - fa) as f64 / (fb - fa) as f64;
            (blend * (vb - va) as f64) as f32 + va
        }
        KfInterp::Smooth => {
            let blend = (t - fa) as f64 / (fb - fa) as f64;
            let s = blend * blend * (3.0 - 2.0 * blend); // smoothstep ease-in/out
            (s * (vb - va) as f64) as f32 + va
        }
        // Everything else: the easing kinds (P20) apply a closed-form factor on the linear blend;
        // the Catmull-Rom variants are NEIGHBOUR-aware and handled in `eval_track` (they never reach
        // this 2-point helper, but a linear fallback keeps this safe if mis-routed).
        kind => {
            let blend = (t - fa) as f64 / (fb - fa) as f64;
            let factor = ease_factor(kind, blend).unwrap_or(blend); // easing factor, else linear
            (factor * (vb - va) as f64) as f32 + va
        }
    }
}

/// Sorted insert-or-replace into a `Vec<Kf>` keyed on `t` (mirrors MojoMedia kf_set): if a
/// key already exists at `t` its value AND its `interp` are overwritten, otherwise the key is
/// inserted so the track stays ascending in `t`. P14: `interp` is the CURRENT create mode
/// (`Project.kf_interp`) threaded through by `add_grade_key`, so re-keying a frame while the mode
/// is Smooth makes that key Smooth.
fn set_track(track: &mut Vec<Kf>, t: i64, v: f32, interp: KfInterp) {
    match track.binary_search_by(|k| k.t.cmp(&t)) {
        Ok(idx) => {
            // replace at existing frame — value AND interp follow the current create mode.
            track[idx].v = v;
            track[idx].interp = interp;
        }
        Err(idx) => track.insert(idx, Kf { t, v, interp }), // sorted insert
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
    // P5 ARBITRARY TRACKS: an ordered list (bottom -> top) replacing the fixed V1/V2/A1 + [bool;3]
    // hide/mute/lock. Video tracks composite bottom-as-base + top-as-overlay; audio tracks all mix.
    // `Clip.track` indexes into this. serde default rebuilds the legacy 3 (V1 video, V2 video, A1
    // audio) so pre-P5 .json projects (no "tracks" field) load with today's layout.
    #[serde(default = "default_tracks")]
    pub tracks: Vec<Track>,

    // ----- Export / render settings (Triad-B P1) -----
    // serde(default) so pre-P1 .json projects deserialize with today's-behavior defaults
    // (1280×856 @ 30, mpeg4, 4 Mbit/s). Read by worker::render_program → OPEN wire line; edited via
    // the Export Settings block in panels::properties_ui. Decouples the OUTPUT resolution from the
    // fixed GVW×GVH OpenCL working canvas (the encoder swscales the composed frame to out_w×out_h).
    #[serde(default)]
    pub export: ExportSettings,

    // ----- P14 keyframe interpolation CREATE mode -----
    // The interpolation TYPE applied to NEW keyframes created via add_grade_key / add_pip_key (and
    // to a re-keyed frame). Per-keyframe interp lives on Kf/PipKey; this is the single "current
    // mode" the create path reads, so those add_* signatures stay unchanged (no-ripple design).
    // `#[serde(default)]` → pre-P14 .json (no "kf_interp" key) loads as `Linear`, and the derived
    // `Default for Project` also yields `Linear` (KfInterp::default), matching the pre-P14-only mode.
    #[serde(default)]
    pub kf_interp: KfInterp,
}

/// Video vs audio track (P5 arbitrary tracks). Video tracks composite; audio tracks mix.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TrackKind {
    Video,
    Audio,
}

/// One timeline track (P5). `Project.tracks` is ordered bottom -> top; `Clip.track` indexes it.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Track {
    pub kind: TrackKind,
    pub name: String,
    #[serde(default)]
    pub hidden: bool,
    #[serde(default)]
    pub muted: bool,
    #[serde(default)]
    pub locked: bool,
}

impl Track {
    pub fn new(kind: TrackKind, name: &str) -> Track {
        Track { kind, name: name.to_string(), hidden: false, muted: false, locked: false }
    }
}

/// The legacy default track set — V1 video, V2 video, A1 audio — matching the old fixed `Clip.track`
/// 0/1/2 so the demo and pre-P5 projects keep their layout. The serde default for `Project.tracks`.
pub fn default_tracks() -> Vec<Track> {
    vec![
        Track::new(TrackKind::Video, "V1"),
        Track::new(TrackKind::Video, "V2"),
        Track::new(TrackKind::Audio, "A1"),
    ]
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
            tracks: default_tracks(),
            bright_kf: vec![],
            contrast_kf: vec![],
            sat_kf: vec![],
            opacity_kf: vec![],
            pip_kf: vec![],
            export: ExportSettings::default(),
            kf_interp: KfInterp::Linear,
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
        // P19: build the (clip,par) track as a sorted `Vec<Kf>` (t_local as the frame) and delegate
        // to the unified `eval_track`, so the flat PiP store gets the SAME interpolation as grade
        // tracks — including the neighbour-aware Catmull-Rom variants (eval_track needs the keys in
        // ascending order to find each segment's neighbours). The flat list is small, so the
        // collect+sort per query is cheap. Empty -> fallback; endpoint clamp handled by eval_track.
        let mut keys: Vec<Kf> = self
            .pip_kf
            .iter()
            .filter(|k| k.clip == clip && k.par == par)
            .map(|k| Kf {
                t: k.t_local,
                v: k.v,
                interp: k.interp,
            })
            .collect();
        keys.sort_by_key(|k| k.t);
        eval_track(&keys, t, fallback)
    }

    // ----- keyframe edit ops (Slice C; called by panels::properties_ui Key buttons) -----

    /// Snapshot the CURRENT static grade (bright/contrast/sat) into a keyframe at timeline
    /// frame `t` on all three grade tracks (sorted insert-or-replace). Mirrors MojoMedia's
    /// "K" buttons keying brightness/contrast/saturation at the playhead with the live values.
    pub fn add_grade_key(&mut self, t: i64) {
        let (b, c, s) = (self.bright, self.contrast, self.sat);
        // P14: the new keys take the project's CURRENT create mode. Same signature as before —
        // no caller (panels.rs) changes — the mode is read off the Project, not passed in.
        let interp = self.kf_interp;
        set_track(&mut self.bright_kf, t, b, interp);
        set_track(&mut self.contrast_kf, t, c, interp);
        set_track(&mut self.sat_kf, t, s, interp);
    }

    /// Snapshot clip `clip_idx`'s CURRENT static PiP rect (px/py/pw/ph) into PiP keyframes at
    /// CLIP-LOCAL frame `t_local` (one per param 0..3, insert-or-replace). Mirrors MojoMedia's
    /// "Key PiP" button keying all four params at the clip-local frame. No-op for a bad index.
    pub fn add_pip_key(&mut self, clip_idx: usize, t_local: i64) {
        let (px, py, pw, ph) = match self.clips.get(clip_idx) {
            Some(c) => (c.px, c.py, c.pw, c.ph),
            None => return,
        };
        // P14: all four new param keys take the project's CURRENT create mode. Signature
        // unchanged — the mode is read off the Project, so panels.rs's call site is untouched.
        let interp = self.kf_interp;
        self.set_pip(clip_idx, 0, t_local, px, interp);
        self.set_pip(clip_idx, 1, t_local, py, interp);
        self.set_pip(clip_idx, 2, t_local, pw, interp);
        self.set_pip(clip_idx, 3, t_local, ph, interp);
    }

    /// Insert-or-replace a single PiP keyframe in the flat store (mirrors MojoMedia pip_set):
    /// overwrite the value AND `interp` if an entry already exists for (clip, par, t_local), else
    /// append. P14: `interp` is the current create mode (`Project.kf_interp`) threaded through by
    /// `add_pip_key`, so re-keying a frame while the mode is Smooth makes that key Smooth.
    fn set_pip(&mut self, clip: usize, par: u8, t_local: i64, v: f32, interp: KfInterp) {
        if let Some(k) = self
            .pip_kf
            .iter_mut()
            .find(|k| k.clip == clip && k.par == par && k.t_local == t_local)
        {
            // replace — value AND interp follow the current create mode.
            k.v = v;
            k.interp = interp;
        } else {
            self.pip_kf.push(PipKey { clip, par, t_local, v, interp });
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
                // stable sort by frame so the moved key slots into ascending order. `sort_by_key`
                // is GUARANTEED stable by std (do NOT swap to `sort_unstable_by_key`): the doc
                // invariant "two keys landing on the same frame keep their relative order" relies
                // on it, so a moved key dropped exactly onto another's frame stays well-defined.
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

    // ----- per-track state helpers (P5: read Project.tracks) -----------------------------------
    // `track` is the Clip.track index into Project.tracks. Each helper bounds-checks the index
    // (an out-of-range track is treated as visible / audible / unlocked / video) so callers never
    // index out of bounds.

    /// True if the given track's VIDEO is hidden (skipped in base/over resolution).
    pub fn is_hidden(&self, track: u8) -> bool {
        self.tracks.get(track as usize).is_some_and(|t| t.hidden)
    }

    /// True if the given track's AUDIO is muted (contributes nothing to the render).
    pub fn is_muted(&self, track: u8) -> bool {
        self.tracks.get(track as usize).is_some_and(|t| t.muted)
    }

    /// True if edits to the given track are blocked.
    pub fn is_locked(&self, track: u8) -> bool {
        self.tracks.get(track as usize).is_some_and(|t| t.locked)
    }

    /// True if the given track is an AUDIO track (its clips contribute audio, not video).
    pub fn is_audio(&self, track: u8) -> bool {
        self.tracks.get(track as usize).is_some_and(|t| t.kind == TrackKind::Audio)
    }

    /// Number of tracks (timeline lane count).
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Append a new track of `kind` (named V/A + the next ordinal). Returns its index.
    pub fn add_track(&mut self, kind: TrackKind) -> usize {
        let n = self.tracks.iter().filter(|t| t.kind == kind).count() + 1;
        let name = match kind {
            TrackKind::Video => format!("V{n}"),
            TrackKind::Audio => format!("A{n}"),
        };
        self.tracks.push(Track::new(kind, &name));
        self.tracks.len() - 1
    }

    /// Remove track `idx`: drop all clips on it and decrement the `track` index of every clip on a
    /// higher track (and rebase transitions/keyframes that key by track). No-op for the last track
    /// or an out-of-range index. Returns true if a track was removed.
    pub fn remove_track(&mut self, idx: usize) -> bool {
        if idx >= self.tracks.len() || self.tracks.len() <= 1 {
            return false;
        }
        let ti = idx as u8;
        // Remove clips on the track (descending so delete_clip's index/PiP remap stays valid).
        let doomed: Vec<usize> =
            self.clips.iter().enumerate().filter(|(_, c)| c.track == ti).map(|(i, _)| i).collect();
        for &i in doomed.iter().rev() {
            self.delete_clip(i);
        }
        // Shift higher clips + transitions down one track.
        for c in self.clips.iter_mut() {
            if c.track > ti {
                c.track -= 1;
            }
        }
        self.transitions.retain(|tr| tr.track != ti);
        for tr in self.transitions.iter_mut() {
            if tr.track > ti {
                tr.track -= 1;
            }
        }
        self.tracks.remove(idx);
        true
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

    // ----- RIPPLE edit ops (P3 editing slice; Shotcut "Ripple Delete" / ripple trim) ----------
    // Unlike the LIFT (`delete_clip`, leaves a gap) and the plain trims above (hold the far edge,
    // leave/open a gap), the RIPPLE ops CLOSE the gap on the SAME track: deleting/shortening a clip
    // shifts every later same-track clip left so the timeline has no hole; extending shifts them
    // right. Other tracks are never touched (Shotcut's default is ripple-current-track-only; the
    // "ripple all tracks" setting is out of scope this wave). PiP keyframes are clip-stable through
    // the underlying `delete_clip` (which already remaps the flat `pip_kf` store); the t0 shifts
    // here move clips on the timeline but keep their CLIP-LOCAL keyframes (t_local) intact, so PiP
    // animation rides along with the rippled clip exactly as it would with a body-drag move.

    /// Shift every clip on `track` whose `t0 >= from` by `delta` frames (t0 clamped to `>= 0`).
    /// Internal helper for the ripple ops: closes/opens the gap left by a ripple delete/trim. A
    /// `delta < 0` ripples earlier (gap-close); `delta > 0` ripples later (gap-open). `skip` is a
    /// clip index NOT to move (the clip whose edit triggered the ripple, when it must stay put);
    /// pass `usize::MAX` to move nothing-excluded.
    fn shift_after(&mut self, track: u8, from: i64, delta: i64, skip: usize) {
        if delta == 0 {
            return;
        }
        for (k, c) in self.clips.iter_mut().enumerate() {
            if k != skip && c.track == track && c.t0 >= from {
                c.t0 = (c.t0 + delta).max(0);
            }
        }
    }

    /// RIPPLE DELETE clip `i`: remove it AND shift every later same-track clip left by the deleted
    /// clip's length, closing the gap (Shotcut "Ripple Delete", X / Shift+Delete). Contrast with
    /// `delete_clip` (the LIFT), which removes the clip but leaves a hole. The downstream shift uses
    /// the deleted clip's `end()` as the cutoff and `-len` as the delta, computed BEFORE the
    /// `delete_clip` call (which renumbers higher clip indices via its PiP remap). No-op out of range.
    // Retained single-clip helper: the UI ripple-delete (X / Shift+Del) routes through
    // `ripple_delete_many` (handles multi-select + index/t0-order correctly), so this is unwired but
    // kept as the documented single-clip contract `ripple_delete_many` contrasts against.
    #[allow(dead_code)]
    pub fn ripple_delete(&mut self, i: usize) {
        let (track, end, len) = match self.clips.get(i) {
            Some(c) => (c.track, c.end(), c.len),
            None => return,
        };
        // Close the gap first (the surviving downstream clips keep their indices through this), then
        // remove the clip itself (which renumbers the higher indices + remaps PiP keys).
        self.shift_after(track, end, -len, i);
        self.delete_clip(i);
    }

    /// RIPPLE DELETE a SET of clips at once, closing the gap on each affected track correctly
    /// REGARDLESS of how the clips' Vec-index order relates to their t0 order (skeptic #1). The
    /// single-clip `ripple_delete` cannot be applied in a loop for a multi-clip selection: its
    /// per-call gap-close is t0-based (`shift_after`), so deleting one clip moves the *t0/end* of
    /// the other still-selected clips before they are removed, and the per-call `end()` cutoffs
    /// compound — over-shifting survivors when index order ≠ t0 order (e.g. two same-track clips
    /// where the lower index has the higher t0). Shotcut closes a multi-ripple by accumulating the
    /// removed length per track and shifting each survivor by the total removed length that sits
    /// at/before it. This does exactly that, atomically:
    ///   1. snapshot every selected clip's `(track, t0, len)` BEFORE any mutation;
    ///   2. remove all selected clips in ONE pass (descending index so Vec indices stay valid and
    ///      `delete_clip`'s PiP-key remap stays correct);
    ///   3. for each SURVIVING clip, subtract from its `t0` the summed `len` of every removed clip
    ///      on the SAME track whose original `t0 <= survivor.t0` (clamped to `>= 0`).
    /// Other tracks are untouched (ripple-current-track-only, like `ripple_delete`). PiP keyframes
    /// ride along with their clips via `delete_clip`'s remap; only survivor `t0`s move here, so
    /// clip-local keyframes are preserved exactly as in the single-clip path. Out-of-range / dup
    /// indices are ignored. No-op (no mutation) when `indices` selects nothing valid.
    pub fn ripple_delete_many(&mut self, indices: &[usize]) {
        // Gather valid, unique indices and snapshot the removed clips' (track, t0, len) up front.
        let mut idxs: Vec<usize> = Vec::new();
        for &i in indices {
            if i < self.clips.len() && !idxs.contains(&i) {
                idxs.push(i);
            }
        }
        if idxs.is_empty() {
            return;
        }
        // Removed-clip snapshots: their original positions, captured before any index renumbering.
        let removed: Vec<(u8, i64, i64)> =
            idxs.iter().map(|&i| (self.clips[i].track, self.clips[i].t0, self.clips[i].len)).collect();

        // Remove all selected clips in descending index order: higher indices first keeps the lower
        // indices (and `delete_clip`'s `clip > i` PiP remap) valid through the whole batch.
        idxs.sort_unstable();
        for &i in idxs.iter().rev() {
            self.delete_clip(i);
        }

        // Close the gaps: each survivor slides left by the total removed length on its track that
        // sat at/before its ORIGINAL t0. Using the pre-delete snapshot makes the result independent
        // of deletion order and of any index↔t0 mismatch.
        for c in self.clips.iter_mut() {
            let shift: i64 = removed
                .iter()
                .filter(|&&(rt, rt0, _)| rt == c.track && rt0 <= c.t0)
                .map(|&(_, _, rl)| rl)
                .sum();
            if shift != 0 {
                c.t0 = (c.t0 - shift).max(0);
            }
        }
    }

    /// RIPPLE TRIM the START of clip `i` to a new timeline start `new_t0`, then shift the downstream
    /// same-track clips by the resulting length delta so the gap stays closed (Shotcut ripple
    /// trim-in). Holds the clip's RIGHT edge via `trim_start` (which advances src_in + reshapes len),
    /// then moves every later same-track clip by `old_len - new_len` (a head trim that SHORTENS the
    /// clip ripples downstream LEFT; extending the head ripples them RIGHT). No-op out of range.
    pub fn ripple_trim_start(&mut self, i: usize, new_t0: i64) {
        let (track, old_t0, old_end) = match self.clips.get(i) {
            Some(c) => (c.track, c.t0, c.end()),
            None => return,
        };
        self.trim_start(i, new_t0);
        // Head-trim amount actually applied (trim_start clamps new_t0 into [0, end-1] / source limits).
        let d = self.clips[i].t0 - old_t0;
        // A head trim HOLDS the right edge, so it opens a gap at the FRONT (between old_t0 and the
        // new t0), NOT downstream. Ripple = close that gap: re-anchor the (now shorter) clip at its
        // original start and slide every later same-track clip left by the same amount, keeping the
        // sequence tight. Mirrors ripple_trim_end (clip stays anchored; followers ride the delta).
        self.clips[i].t0 = old_t0;
        self.shift_after(track, old_end, -d, i);
    }

    /// RIPPLE TRIM the END of clip `i` to a new length `new_len`, then shift the downstream same-track
    /// clips by the length delta so the gap stays closed (Shotcut ripple trim-out). `trim_end`
    /// applies the MIN_CLIP floor; we read the clip's ACTUAL new length back (so the ripple matches
    /// what was really applied) and shift every later same-track clip by `actual_new - old_len`
    /// (shortening ripples them LEFT, lengthening ripples them RIGHT). No-op out of range.
    pub fn ripple_trim_end(&mut self, i: usize, new_len: i64) {
        let (track, old_end, old_len) = match self.clips.get(i) {
            Some(c) => (c.track, c.end(), c.len),
            None => return,
        };
        self.trim_end(i, new_len);
        let actual_new = self.clips[i].len;
        let delta = actual_new - old_len;
        // Downstream = clips that started at/after this clip's ORIGINAL end (so the trimmed clip's
        // own t0 is untouched and only the followers slide). Use old_end as the cutoff.
        self.shift_after(track, old_end, delta, i);
    }

    /// SLIP clip `i` by `delta` source frames: re-time the SOURCE under a fixed timeline window
    /// (Shotcut slip / 3-point slip). `t0` and `len` are UNCHANGED — only `src_in` moves, so the
    /// clip occupies the exact same span on the timeline but shows an earlier/later part of its
    /// media. `src_in` is clamped to `>= 0` (a slip cannot pull source before frame 0). A positive
    /// `delta` slips the source forward (later media under the same window); negative, backward.
    /// No-op out of range. Mirrors `slipTrim` holding the timeline rect while sliding the cut.
    pub fn slip(&mut self, i: usize, delta: i64) {
        if let Some(c) = self.clips.get_mut(i) {
            c.src_in = (c.src_in + delta).max(0);
        }
    }

    /// ROLL the shared cut between two adjacent same-track clips by `delta` frames (Shotcut roll /
    /// 3-point roll edit): the LEFT clip's OUT point and the RIGHT clip's IN point move together so
    /// the boundary slides while the pair's combined timeline span is unchanged. `delta > 0` moves
    /// the cut RIGHT (left clip grows, right clip shrinks + starts later); `delta < 0` moves it LEFT.
    ///
    /// Both edges are clamped to keep each clip `>= MIN_CLIP` and the right clip's `src_in >= 0`, and
    /// the EFFECTIVE delta is the most either side can take, so the seam stays a single shared cut
    /// (no gap, no overlap). `left_i`/`right_i` must be distinct, same-track, and abut (right starts
    /// where left ends); otherwise it is a no-op. Returns the effective delta actually applied.
    pub fn roll_edit(&mut self, left_i: usize, right_i: usize, delta: i64) -> i64 {
        if left_i == right_i {
            return 0;
        }
        let (l_track, l_t0, l_len, l_end) = match self.clips.get(left_i) {
            Some(c) => (c.track, c.t0, c.len, c.end()),
            None => return 0,
        };
        let (r_track, r_t0, r_len, r_src) = match self.clips.get(right_i) {
            Some(c) => (c.track, c.t0, c.len, c.src_in),
            None => return 0,
        };
        // The two must share the cut (right starts exactly where left ends) and be on one track.
        if l_track != r_track || r_t0 != l_end {
            return 0;
        }
        // Clamp so neither clip drops below MIN_CLIP and the right source can't go negative.
        //   moving the cut right by d: left.len += d (max = anything), right.len -= d, right.src_in += d
        //   moving the cut left  by d (<0): left.len += d (>= MIN_CLIP), right grows.
        let mut d = delta;
        // left length floor: l_len + d >= MIN_CLIP  ->  d >= MIN_CLIP - l_len
        d = d.max(MIN_CLIP - l_len);
        // right length floor: r_len - d >= MIN_CLIP  ->  d <= r_len - MIN_CLIP
        d = d.min(r_len - MIN_CLIP);
        // right source floor: r_src + d >= 0  ->  d >= -r_src
        d = d.max(-r_src);
        let _ = l_t0; // left t0 is untouched by a roll (only its OUT point moves); bound for clarity.
        if d == 0 {
            return 0;
        }
        // Apply: left out-point moves by +d (grow/shrink its tail); right in-point moves by +d
        // (advance its source + start, shrink/grow its length), keeping the seam a single cut.
        self.clips[left_i].len = l_len + d;
        self.clips[right_i].t0 = r_t0 + d;
        self.clips[right_i].src_in = r_src + d;
        self.clips[right_i].len = r_len - d;
        d
    }

    /// Convenience ROLL by `boundary` frame: find the same-track clip pair whose shared cut sits at
    /// `boundary` on `track` (left.end() == boundary == right.t0) and roll it by `delta`. Returns the
    /// effective delta, or 0 if no abutting pair sits exactly at that boundary. Lets a caller roll by
    /// a timeline frame (e.g. a dragged boundary x) without resolving the two clip indices itself.
    // Retained roll-by-boundary convenience API (unit-tested): the timeline's roll hot-zone already
    // has the two clip indices, so it calls `roll_edit` directly — this frame-addressed variant is
    // unwired but kept (covered by `roll_moves_shared_cut_preserving_total`).
    #[allow(dead_code)]
    pub fn roll(&mut self, track: u8, boundary: i64, delta: i64) -> i64 {
        let mut left_i: Option<usize> = None;
        let mut right_i: Option<usize> = None;
        for (k, c) in self.clips.iter().enumerate() {
            if c.track != track {
                continue;
            }
            if c.end() == boundary {
                left_i = Some(k);
            }
            if c.t0 == boundary {
                right_i = Some(k);
            }
        }
        match (left_i, right_i) {
            (Some(l), Some(r)) if l != r => self.roll_edit(l, r, delta),
            _ => 0,
        }
    }

    // ----- COPY / PASTE clipboard helpers (P3 editing slice; Shotcut Ctrl+C / Ctrl+V) -----------
    // The clipboard itself lives on the app (`Genesis.clipboard: Vec<Clip>`) so it survives across
    // edits and project loads independent of the model. These helpers do the OFFSET-PRESERVING math:
    // copy snapshots a selection (rebased so the earliest clip sits at t0 = 0); paste re-anchors that
    // snapshot at the playhead. Cloning keeps every per-clip field (look/grade/transform/audio_fx/
    // fades/PiP rect) — audio_fx is preserved verbatim (Team A never reads/writes it, just carries it).

    /// Snapshot the clips at `indices` into a fresh `Vec<Clip>`, REBASED so the earliest selected
    /// clip starts at `t0 = 0` (offsets between the copied clips, and their tracks, are preserved).
    /// Paste then re-anchors the whole group at the playhead. Out-of-range indices are skipped;
    /// duplicate indices are de-duped so a clip is never copied twice. Order follows ascending t0
    /// so the rebase origin is deterministic. Returns an empty Vec if nothing valid was selected.
    pub fn copy_clips(&self, indices: &[usize]) -> Vec<Clip> {
        // Gather valid, unique clip indices.
        let mut picked: Vec<usize> = Vec::new();
        for &i in indices {
            if i < self.clips.len() && !picked.contains(&i) {
                picked.push(i);
            }
        }
        if picked.is_empty() {
            return Vec::new();
        }
        // Rebase origin = the earliest t0 among the picked clips.
        let base = picked.iter().map(|&i| self.clips[i].t0).min().unwrap_or(0);
        // Sort the snapshot by t0 so paste lays them down left-to-right (cosmetic; offsets carry).
        picked.sort_by_key(|&i| self.clips[i].t0);
        picked
            .into_iter()
            .map(|i| {
                let mut c = self.clips[i].clone();
                c.t0 -= base; // rebase: earliest clip lands at 0, the rest keep their relative offset
                c
            })
            .collect()
    }

    /// PASTE a clipboard snapshot (from `copy_clips`) at timeline frame `at`, OFFSET-PRESERVING:
    /// each clipboard clip is cloned with `t0 += at` so the group lands with the same internal
    /// spacing/tracks, its earliest clip at `at` (Shotcut paste-at-playhead). Appends the new clips
    /// and returns the index of the FIRST pasted clip (for the caller to select), or `None` if the
    /// clipboard is empty. Drops onto LOCKED tracks are skipped (advisory lock enforcement, matching
    /// the drop path); if every clip is on a locked track nothing is added and `None` is returned.
    pub fn paste_clips(&mut self, clips: &[Clip], at: i64) -> Option<usize> {
        let first = self.clips.len();
        let mut added = 0usize;
        for c in clips {
            if self.is_locked(c.track) {
                continue; // refuse a paste onto a locked track (advisory; mirrors the drop path)
            }
            let mut nc = c.clone();
            nc.t0 = (c.t0 + at).max(0);
            self.clips.push(nc);
            added += 1;
        }
        if added == 0 {
            None
        } else {
            Some(first)
        }
    }

    // ----- 3-POINT EDITING ops (P4 editing slice; Shotcut Append / Overwrite / Insert) ----------
    // These three are the timeline-target half of a 3-point edit: a SOURCE clip (already cut to its
    // length, with its src_in/len/look/grade/audio_fx/chroma carried verbatim) is dropped onto a
    // TRACK at a TIME with one of three placement policies, mirroring Shotcut's MultitrackModel:
    //   * INSERT   (TimelineDock::insert / InsertCommand)    — RIPPLE: if a clip straddles `t0`,
    //     SPLIT it at `t0` (Shotcut's insertClip splits the clip under the insert point), then open
    //     a hole of `clip.len` at `t0` by shifting every same-track clip whose t0 >= t0 RIGHT by
    //     clip.len (incl. that right remnant), then drop the clip at t0. Downstream content is
    //     preserved, just pushed later. (Shotcut default-ripple V.)
    //   * OVERWRITE(TimelineDock::overwrite / OverwriteCommand) — REPLACE the range [t0, t0+len) on
    //     that track: trim/split/remove whatever the new clip covers, then drop the clip. NO ripple
    //     (the timeline length is unchanged; the range is simply replaced). (Shotcut B.)
    //   * APPEND   (TimelineDock::append / AppendCommand)    — drop the clip at the track's END
    //     (max end() of clips on that track, or 0 for an empty track). NO ripple. (Shotcut A.)
    // The caller (app.rs) clones the source clip, sets its `track`, pushes undo, then calls these;
    // each method sets the placed clip's `track`/`t0` itself from its arguments so the caller need
    // not pre-set t0. All three return the index of the newly-placed clip (so the caller can select
    // it). LOCKED tracks are refused (advisory, matching paste/drop): a locked target is a no-op and
    // returns `None`. PiP keyframes ride along correctly: ripple/overwrite only move/trim/remove
    // EXISTING clips via the t0 shift + the PiP-stable `delete_clip`/`split_clip`/`trim_*` already
    // in this file, and the freshly-placed clip starts with no PiP keys (it is appended LAST, so its
    // new index never collides with a remapped key). IDENTITY: none of these run unless the app
    // fires the gesture, so a project with no 3-point edit is byte-identical to before.

    /// INSERT (ripple) `clip` at timeline frame `t0` on `track` (the clip's own `track`/`t0` are set
    /// from these args). Opens a hole of `clip.len` frames at `t0`: if an existing same-track clip
    /// STRADDLES `t0` (its body spans the insert point), it is first SPLIT at `t0` so its right half
    /// starts exactly at `t0`; then every same-track clip whose `t0 >= t0` (which now includes that
    /// right half) is shifted RIGHT by `clip.len`, and the new clip is placed in the opened hole at
    /// `t0`. Returns the new clip's index, or `None` if `track` is locked. `t0` is clamped to `>= 0`.
    /// Mirrors Shotcut's InsertCommand → MultitrackModel::insertClip (multitrackmodel.cpp:1294 splits
    /// the clip under the insert point via `splitClip` when `position > clip_start(target)` before
    /// inserting; ripple-current-track-only — the "ripple all tracks" setting is out of scope).
    pub fn insert_clip(&mut self, track: u8, t0: i64, mut clip: Clip) -> Option<usize> {
        if self.is_locked(track) {
            return None;
        }
        let at = t0.max(0);
        let len = clip.len.max(1);
        // SHOTCUT PARITY: split the clip the insert point falls strictly inside, so its right
        // remnant starts at `at` and rides the ripple (rather than being overlapped by the new
        // clip). `split_clip` only acts when `at` is strictly inside a clip body (off>0 && off<len),
        // returns None otherwise, and remaps PiP keys for the inserted right half. There can be at
        // most one such straddling clip on a track (clips on a track don't overlap), so a single
        // scan + split suffices. We must do this BEFORE `shift_after`, which moves clips by `t0`:
        // the fresh right half has `t0 == at`, so the cutoff `at` catches it (`at >= at`).
        if let Some(i) = self
            .clips
            .iter()
            .position(|c| c.track == track && c.t0 < at && c.end() > at)
        {
            self.split_clip(i, at);
        }
        // Open the gap: shift every same-track clip at/after `at` right by `len`. `usize::MAX`
        // excludes nothing (the new clip is not in `clips` yet, so there is no index to skip).
        self.shift_after(track, at, len, usize::MAX);
        clip.track = track;
        clip.t0 = at;
        let idx = self.clips.len();
        self.clips.push(clip);
        Some(idx)
    }

    /// OVERWRITE `clip` onto `track` at timeline frame `t0`, REPLACING the range `[t0, t0+len)` on
    /// that track (no ripple). Every same-track clip the range touches is trimmed to its surviving
    /// portion(s) and/or removed via `lift_range`, then the new clip is placed. Returns the new
    /// clip's index, or `None` if `track` is locked. `t0` is clamped to `>= 0`. Mirrors Shotcut's
    /// OverwriteCommand (the timeline length is unchanged — only the covered range is replaced).
    pub fn overwrite_clip(&mut self, track: u8, t0: i64, mut clip: Clip) -> Option<usize> {
        if self.is_locked(track) {
            return None;
        }
        let at = t0.max(0);
        let len = clip.len.max(1);
        // Clear the covered range on this track (trim/split/remove existing clips under it).
        self.lift_range(track, at, at + len);
        clip.track = track;
        clip.t0 = at;
        let idx = self.clips.len();
        self.clips.push(clip);
        Some(idx)
    }

    /// APPEND `clip` to the END of `track` (no ripple): the clip is placed at the maximum `end()` of
    /// the clips already on that track, or at 0 for an empty track. Returns the new clip's index, or
    /// `None` if `track` is locked. Mirrors Shotcut's AppendCommand (Append, `A`).
    pub fn append_clip(&mut self, track: u8, mut clip: Clip) -> Option<usize> {
        if self.is_locked(track) {
            return None;
        }
        let end = self.track_end(track);
        clip.track = track;
        clip.t0 = end;
        let idx = self.clips.len();
        self.clips.push(clip);
        Some(idx)
    }

    /// The end frame of `track` = the maximum `end()` of every clip on it, or 0 if the track is
    /// empty. The landing point for `append_clip` (Shotcut appends after the last clip on the track).
    pub fn track_end(&self, track: u8) -> i64 {
        self.clips
            .iter()
            .filter(|c| c.track == track)
            .map(|c| c.end())
            .max()
            .unwrap_or(0)
    }

    /// Clear the timeline range `[from, to)` on `track`: every same-track clip is reshaped so that no
    /// part of it remains inside the range, by trimming its overlapping head/tail and/or splitting it
    /// (a clip that STRADDLES the whole range is cut into a left remnant before `from` and a right
    /// remnant after `to`). Used by `overwrite_clip` to make room for the overwriting clip (NO
    /// ripple — surviving remnants keep their timeline positions). A `to <= from` range is a no-op.
    ///
    /// Implementation reuses the PiP-stable primitives already in this file so keyframes ride along:
    ///   * fully-covered clip (from <= c.t0 && c.end() <= to)  -> `delete_clip` (remaps PiP keys).
    ///   * straddling clip (c.t0 < from && to < c.end())       -> `split_clip` at `from`, then the
    ///     right remnant is `trim_start`-ed to `to` (advancing its src_in), leaving a left remnant
    ///     ending at `from` and a right remnant starting at `to`.
    ///   * head-overlap (c.t0 < to && c.t0 >= from ... i.e. starts inside the range, ends after) ->
    ///     `trim_start` to `to`.
    ///   * tail-overlap (ends inside the range, starts before) -> `trim_end` so the clip ends at
    ///     `from`.
    /// The clip Vec is mutated (split inserts, delete removes), so we rescan from scratch until no
    /// same-track clip still intersects `[from, to)`. The loop terminates because each pass strictly
    /// reduces the total covered frames (every action removes or shrinks an intersecting clip), and
    /// a clip whose remaining length would fall below `MIN_CLIP` via a trim is removed instead so a
    /// trim can never get "stuck" at the floor while still intersecting.
    pub fn lift_range(&mut self, track: u8, from: i64, to: i64) {
        if to <= from {
            return;
        }
        // Bounded rescans (defensive cap: at most one action per existing clip, plus splits). Each
        // pass performs exactly ONE structural action then rescans, so the index it found stays valid.
        let max_passes = self.clips.len() * 4 + 8;
        for _ in 0..max_passes {
            // Find the first same-track clip that still intersects [from, to).
            let hit = self.clips.iter().position(|c| {
                c.track == track && c.t0 < to && c.end() > from
            });
            let i = match hit {
                Some(i) => i,
                None => return, // range is clear
            };
            let (c_t0, c_end) = (self.clips[i].t0, self.clips[i].end());

            if from <= c_t0 && c_end <= to {
                // fully covered -> remove it (delete_clip remaps PiP keys).
                self.delete_clip(i);
            } else if c_t0 < from && to < c_end {
                // STRADDLES the whole range. Each surviving remnant (left = [c_t0, from), right =
                // [to, c_end)) is kept only if it is at least MIN_CLIP long; otherwise that side is
                // dropped (a remnant clamped UP to MIN_CLIP by trim_* would spill back into the range
                // and the rescan would never clear it — an infinite loop). We split at `from` first,
                // which yields a left half [c_t0, from) and a right half [from, c_end); then advance
                // the right half's start to `to`. If a side is too short to survive, we remove it.
                let left_len = from - c_t0;
                let right_len = c_end - to;
                if left_len < MIN_CLIP {
                    // left remnant too small: drop the whole clip and let the (large enough) right
                    // side, if any, be re-created by re-processing — simplest correct path is to
                    // head-trim the original clip to `to` (keeps the right remnant, drops the left).
                    if right_len < MIN_CLIP {
                        self.delete_clip(i); // neither side survives -> remove entirely
                    } else {
                        self.trim_start(i, to); // keep only the right remnant
                    }
                } else if right_len < MIN_CLIP {
                    // right remnant too small: keep only the left remnant by tail-trimming to `from`.
                    self.trim_end(i, left_len);
                } else if let Some(right) = self.split_clip(i, from) {
                    // both remnants survive: split made the right half [from, c_end); advance it to `to`.
                    self.trim_start(right, to);
                } else {
                    // split refused (shouldn't happen here: from is strictly inside) — keep the left.
                    self.trim_end(i, left_len);
                }
            } else if c_t0 >= from {
                // HEAD-OVERLAP: starts inside the range, extends past `to`. The survivor is
                // [to, c_end); if that is shorter than MIN_CLIP, drop the clip (a trim_start clamped
                // up would leave it intersecting the range). Otherwise head-trim to `to`.
                if c_end - to < MIN_CLIP {
                    self.delete_clip(i);
                } else {
                    self.trim_start(i, to);
                }
            } else {
                // TAIL-OVERLAP: starts before `from`, ends inside the range. The survivor is
                // [c_t0, from); if that is shorter than MIN_CLIP, drop the clip (a trim_end clamped
                // up to MIN_CLIP would spill back into the range). Otherwise tail-trim to `from`.
                let left_len = from - c_t0;
                if left_len < MIN_CLIP {
                    self.delete_clip(i);
                } else {
                    self.trim_end(i, left_len);
                }
            }
        }
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

    // ----- P14 keyframe INTERPOLATION TYPES (eval_track + eval_pip) -------------------------------
    // All on a 2-key track 0@frame0 -> 10@frame10. The interp of the LOWER key (frame 0) selects
    // the segment curve. `kf_interp` is the create mode read by add_grade_key/add_pip_key.

    // Build a grade (bright) track 0@0 -> 10@10 with the given create interp; eval via grade_at.
    fn grade_0_10(interp: KfInterp) -> Project {
        let mut p = Project::demo("x".into());
        p.kf_interp = interp;
        p.bright = 0.0;
        p.add_grade_key(0); // lower key carries `interp`
        p.bright = 10.0;
        p.add_grade_key(10);
        p
    }

    // Build a PiP px track 0@0 -> 10@10 with the given create interp; eval via pip_at(.).0.
    fn pip_0_10(interp: KfInterp) -> Project {
        let mut p = Project::demo("x".into());
        p.kf_interp = interp;
        p.clips[0].px = 0.0;
        p.add_pip_key(0, 0); // lower key carries `interp`
        p.clips[0].px = 10.0;
        p.add_pip_key(0, 10);
        p
    }

    #[test]
    fn interp_linear_eval_track_and_pip() {
        // (a) LINEAR: eval@5 == 5.0 (midpoint), both tracks. This is the pre-P14 behavior and the
        //     KfInterp::default, so it is also what an old .json loads as.
        let g = grade_0_10(KfInterp::Linear);
        assert!((g.grade_at(5).0 - 5.0).abs() < 1e-3, "grade linear@5={}", g.grade_at(5).0);
        let pp = pip_0_10(KfInterp::Linear);
        assert!((pp.pip_at(0, 5).0 - 5.0).abs() < 1e-3, "pip linear@5={}", pp.pip_at(0, 5).0);
        // sanity: default create mode is Linear, so an un-set kf_interp behaves identically.
        assert_eq!(KfInterp::default(), KfInterp::Linear);
    }

    #[test]
    fn interp_discrete_eval_track_and_pip() {
        // (b) DISCRETE: HOLD the lower key's value across the segment. eval@5 == 0.0, eval@9 == 0.0,
        //     eval@10 == 10.0 (the upper key's frame is its own value — endpoint clamp / next key).
        let g = grade_0_10(KfInterp::Discrete);
        assert!((g.grade_at(5).0 - 0.0).abs() < 1e-3, "grade discrete@5={}", g.grade_at(5).0);
        assert!((g.grade_at(9).0 - 0.0).abs() < 1e-3, "grade discrete@9={}", g.grade_at(9).0);
        assert!((g.grade_at(10).0 - 10.0).abs() < 1e-3, "grade discrete@10={}", g.grade_at(10).0);
        let pp = pip_0_10(KfInterp::Discrete);
        assert!((pp.pip_at(0, 5).0 - 0.0).abs() < 1e-3, "pip discrete@5={}", pp.pip_at(0, 5).0);
        assert!((pp.pip_at(0, 9).0 - 0.0).abs() < 1e-3, "pip discrete@9={}", pp.pip_at(0, 9).0);
        assert!((pp.pip_at(0, 10).0 - 10.0).abs() < 1e-3, "pip discrete@10={}", pp.pip_at(0, 10).0);
    }

    #[test]
    fn interp_smooth_eval_track_and_pip() {
        // (c) SMOOTH: smoothstep s = b*b*(3-2b). Symmetric → eval@5 == 5.0; ease-IN below midpoint
        //     (eval@2 = smoothstep(0.2)*10 = 1.04 < 2.0); ease-OUT above midpoint
        //     (eval@8 = smoothstep(0.8)*10 = 8.96 > 8.0).
        let g = grade_0_10(KfInterp::Smooth);
        assert!((g.grade_at(5).0 - 5.0).abs() < 1e-3, "grade smooth@5={}", g.grade_at(5).0);
        assert!((g.grade_at(2).0 - 1.04).abs() < 1e-3, "grade smooth@2={}", g.grade_at(2).0);
        assert!(g.grade_at(2).0 < 2.0, "grade smooth ease-in@2={}", g.grade_at(2).0);
        assert!((g.grade_at(8).0 - 8.96).abs() < 1e-3, "grade smooth@8={}", g.grade_at(8).0);
        assert!(g.grade_at(8).0 > 8.0, "grade smooth ease-out@8={}", g.grade_at(8).0);
        let pp = pip_0_10(KfInterp::Smooth);
        assert!((pp.pip_at(0, 5).0 - 5.0).abs() < 1e-3, "pip smooth@5={}", pp.pip_at(0, 5).0);
        assert!((pp.pip_at(0, 2).0 - 1.04).abs() < 1e-3, "pip smooth@2={}", pp.pip_at(0, 2).0);
        assert!(pp.pip_at(0, 2).0 < 2.0, "pip smooth ease-in@2={}", pp.pip_at(0, 2).0);
        assert!((pp.pip_at(0, 8).0 - 8.96).abs() < 1e-3, "pip smooth@8={}", pp.pip_at(0, 8).0);
        assert!(pp.pip_at(0, 8).0 > 8.0, "pip smooth ease-out@8={}", pp.pip_at(0, 8).0);
    }

    #[test]
    fn interp_rekey_updates_interp() {
        // Re-keying the SAME frame while the create mode changed updates that key's interp (replace
        // path threads the new mode). Start Linear@0..10, then re-key frame 0 as Discrete -> hold.
        let mut g = grade_0_10(KfInterp::Linear);
        assert!((g.grade_at(5).0 - 5.0).abs() < 1e-3);
        g.kf_interp = KfInterp::Discrete;
        g.bright = 0.0;
        g.add_grade_key(0); // replace frame-0 key; its interp becomes Discrete
        assert!((g.grade_at(5).0 - 0.0).abs() < 1e-3, "rekey->discrete holds, @5={}", g.grade_at(5).0);

        let mut pp = pip_0_10(KfInterp::Linear);
        assert!((pp.pip_at(0, 5).0 - 5.0).abs() < 1e-3);
        pp.kf_interp = KfInterp::Discrete;
        pp.clips[0].px = 0.0;
        pp.add_pip_key(0, 0); // replace frame-0 param keys; their interp becomes Discrete
        assert!((pp.pip_at(0, 5).0 - 0.0).abs() < 1e-3, "pip rekey->discrete holds, @5={}", pp.pip_at(0, 5).0);
    }

    // P19: a 3-key grade track 0@f0 -> 10@f10 -> 30@f20 with the SEGMENT [10,20] (lower key = f10)
    // carrying the given Catmull variant. Frame 0 / frame 20 keys are Linear (their interp is
    // irrelevant to an eval inside [10,20]).
    fn grade_3key(seg_interp: KfInterp) -> Project {
        let mut p = Project::demo("x".into());
        p.kf_interp = KfInterp::Linear;
        p.bright = 0.0;
        p.add_grade_key(0);
        p.kf_interp = seg_interp;
        p.bright = 10.0;
        p.add_grade_key(10);
        p.kf_interp = KfInterp::Linear;
        p.bright = 30.0;
        p.add_grade_key(20);
        p
    }
    fn pip_3key(seg_interp: KfInterp) -> Project {
        let mut p = Project::demo("x".into());
        p.kf_interp = KfInterp::Linear;
        p.clips[0].px = 0.0;
        p.add_pip_key(0, 0);
        p.kf_interp = seg_interp;
        p.clips[0].px = 10.0;
        p.add_pip_key(0, 10);
        p.kf_interp = KfInterp::Linear;
        p.clips[0].px = 30.0;
        p.add_pip_key(0, 20);
        p
    }

    #[test]
    fn interp_catmull_variants_match_mlt() {
        // P19: values are the EXACT output of MLT's catmull_rom_interpolate (mlt_animation.c) computed
        // from the verbatim formula on the same track, segment [10,20] (p0=(0,0),p1=(10,10),
        // p2=(20,30),p3=dup(20,30)) at frame 15 (progress 0.5):
        //   smooth_loose   (alpha 0.0, tension  1.0) = 20.625000
        //   smooth_natural (alpha 0.5, tension -1.0) = 21.982970
        //   smooth_tight   (alpha 0.5, tension  0.0) = 20.000000  (zero tangents == smoothstep)
        for (interp, want) in [
            (KfInterp::SmoothLoose, 20.625_f32),
            (KfInterp::SmoothNatural, 21.982_97_f32),
            (KfInterp::SmoothTight, 20.0_f32),
        ] {
            let g = grade_3key(interp);
            assert!((g.grade_at(15).0 - want).abs() < 1e-2, "grade {:?}@15 = {} (want {})", interp, g.grade_at(15).0, want);
            let pp = pip_3key(interp);
            assert!((pp.pip_at(0, 15).0 - want).abs() < 1e-2, "pip {:?}@15 = {} (want {})", interp, pp.pip_at(0, 15).0, want);
        }
        // smooth_loose overshoot signature off the midpoint (MLT reference): @12=13.68, @18=27.12.
        let g = grade_3key(KfInterp::SmoothLoose);
        assert!((g.grade_at(12).0 - 13.68).abs() < 1e-2, "loose@12={}", g.grade_at(12).0);
        assert!((g.grade_at(18).0 - 27.12).abs() < 1e-2, "loose@18={}", g.grade_at(18).0);
    }

    // P20: a grade track 0@f0 -> 10@f100 with the given easing on the f0 (lower) key; grade_at(frame)
    // returns 10*factor(frame/100), so the expected values are the MLT easing factors *10.
    fn ease_track(interp: KfInterp) -> Project {
        let mut p = Project::demo("x".into());
        p.kf_interp = interp;
        p.bright = 0.0;
        p.add_grade_key(0);
        p.kf_interp = KfInterp::Linear;
        p.bright = 10.0;
        p.add_grade_key(100);
        p
    }

    #[test]
    fn interp_easings_match_mlt() {
        // Values are 10 * the easing factor at t = frame/100, computed from the VERBATIM MLT
        // mlt_animation.c easing functions (sinusoidal/power/exponential/circular/back/elastic/bounce).
        let cases = [
            (KfInterp::SineInOut, 50, 5.0_f32),
            (KfInterp::QuadIn, 25, 0.625),
            (KfInterp::CubicIn, 25, 0.15625),
            (KfInterp::CubicOut, 25, 5.78125),
            (KfInterp::QuartIn, 50, 0.625),
            (KfInterp::QuintIn, 50, 0.3125),
            (KfInterp::ExpoOut, 50, 9.6875),
            (KfInterp::CircIn, 50, 1.33975),
            (KfInterp::BackIn, 50, -3.75), // back anticipates BELOW the start value (overshoot)
            (KfInterp::ElasticOut, 50, 10.22097), // elastic overshoots ABOVE the target
            (KfInterp::BounceOut, 50, 7.1875),
            (KfInterp::BounceOut, 25, 4.72656),
        ];
        for (interp, frame, want) in cases {
            let g = ease_track(interp);
            let got = g.grade_at(frame).0;
            assert!((got - want).abs() < 1e-2, "{:?}@f{} = {} (want {})", interp, frame, got, want);
        }
        // every easing kind is reachable from ALL + has a label (UI invariant).
        assert_eq!(KfInterp::ALL.len(), 36);
        for k in KfInterp::ALL {
            assert!(!k.label().is_empty());
        }
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

    // ----- P3 EDITING ops -------------------------------------------------------------------

    #[test]
    fn ripple_delete_closes_gap_same_track_only() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        // V1: A [0,100), B [100,150), C [150,210). V2: D [120,220) must NOT move.
        p.clips.push(Clip::video(0, 0, 100, 0, "A"));   // idx 0
        p.clips.push(Clip::video(0, 100, 50, 0, "B"));  // idx 1
        p.clips.push(Clip::video(0, 150, 60, 0, "C"));  // idx 2
        p.clips.push(Clip::video(0, 120, 100, 1, "D")); // idx 3 (other track)
        p.ripple_delete(1); // delete B (len 50), C shifts left by 50
        // B gone -> 3 clips. C now at 100, ends 160. D unchanged at 120.
        assert_eq!(p.clips.len(), 3);
        // find C (media 0 track 0 len 60) and D (track 1)
        let c = p.clips.iter().find(|c| c.track == 0 && c.len == 60).unwrap();
        assert_eq!(c.t0, 100, "C rippled left by B.len");
        let d = p.clips.iter().find(|c| c.track == 1).unwrap();
        assert_eq!(d.t0, 120, "other-track clip unmoved");
    }

    #[test]
    fn ripple_delete_many_non_contiguous_survivor() {
        // Track 0: A [0,100), B [100,150), C [150,210). Ripple-delete A and C (non-contiguous):
        // B is after A (removed, 100 frames) -> shifts left 100; C is removed. B -> t0=0, len=50.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A")); // idx 0
        p.clips.push(Clip::video(0, 100, 50, 0, "B")); // idx 1
        p.clips.push(Clip::video(0, 150, 60, 0, "C")); // idx 2
        p.ripple_delete_many(&[0, 2]);
        assert_eq!(p.clips.len(), 1, "A and C removed, B survives");
        assert_eq!(p.clips[0].len, 50, "survivor is B");
        assert_eq!(p.clips[0].t0, 0, "B closed the 100-frame gap left by A");
    }

    #[test]
    fn ripple_delete_many_index_order_ne_t0_order() {
        // Skeptic #1: two same-track clips where the LOWER index has the HIGHER t0. The old
        // descending-index loop (per-clip t0-based shift) over-shifted the survivor; the batch
        // must be order-independent and identical to deleting them as one contiguous block.
        // Track 0: idx0 @ t0=100 len=50 (end 150), idx1 @ t0=0 len=100 (end 100), then a
        // downstream survivor S @ t0=150 len=40 (end 190). Selected block [0,150) removes 150
        // frames before S -> S slides left 150 to t0=0.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 100, 50, 0, "later-but-idx0")); // idx 0, t0=100
        p.clips.push(Clip::video(0, 0, 100, 0, "earlier-idx1"));    // idx 1, t0=0
        p.clips.push(Clip::video(0, 150, 40, 0, "S"));              // idx 2, downstream survivor
        p.ripple_delete_many(&[0, 1]);
        assert_eq!(p.clips.len(), 1, "both block clips removed, S survives");
        assert_eq!(p.clips[0].len, 40, "survivor is S");
        assert_eq!(p.clips[0].t0, 0, "S slid left by the full 150-frame removed block (no double-shift)");
    }

    #[test]
    fn ripple_delete_many_other_track_unmoved() {
        // A same-track ripple must never move clips on another track (ripple-current-track-only).
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A"));   // idx 0, track 0
        p.clips.push(Clip::video(0, 100, 50, 0, "B"));  // idx 1, track 0 downstream
        p.clips.push(Clip::video(0, 50, 100, 1, "D"));  // idx 2, track 1 (other track)
        p.ripple_delete_many(&[0]);
        assert_eq!(p.clips.len(), 2);
        let b = p.clips.iter().find(|c| c.track == 0).unwrap();
        assert_eq!(b.t0, 0, "B rippled left by A.len");
        let d = p.clips.iter().find(|c| c.track == 1).unwrap();
        assert_eq!(d.t0, 50, "other-track clip unmoved");
    }

    #[test]
    fn ripple_trim_end_shifts_downstream() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A")); // idx 0
        p.clips.push(Clip::video(0, 100, 50, 0, "B")); // idx 1 (starts at A.end)
        p.ripple_trim_end(0, 80); // A 100 -> 80, B ripples left by 20 to t0 80
        assert_eq!(p.clips[0].len, 80);
        assert_eq!(p.clips[1].t0, 80, "downstream clip closed the 20-frame gap");
    }

    #[test]
    fn ripple_trim_start_shifts_downstream() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A")); // idx 0
        p.clips.push(Clip::video(0, 100, 50, 0, "B")); // idx 1
        // Ripple head-trim: 20 frames come off A's head (src_in advances), the shortened clip is
        // re-anchored at its original start, and the sequence slides left to stay gapless.
        p.ripple_trim_start(0, 20);
        assert_eq!(p.clips[0].t0, 0, "re-anchored at original start — no front gap");
        assert_eq!(p.clips[0].len, 80, "head trimmed by 20 frames");
        assert_eq!(p.clips[0].src_in, 20, "head trim advances the source in-point");
        // A now ends at 80; downstream B ripples left by 20 to stay tight (100 -> 80).
        assert_eq!(p.clips[1].t0, 80, "downstream rippled left by the head-trim delta");
    }

    #[test]
    fn slip_moves_source_only() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        let mut c = Clip::video(0, 50, 100, 0, "A");
        c.src_in = 30;
        p.clips.push(c);
        p.slip(0, 10);
        assert_eq!(p.clips[0].src_in, 40);
        assert_eq!(p.clips[0].t0, 50, "t0 unchanged by slip");
        assert_eq!(p.clips[0].len, 100, "len unchanged by slip");
        p.slip(0, -1000); // clamp at 0
        assert_eq!(p.clips[0].src_in, 0);
    }

    #[test]
    fn roll_moves_shared_cut_preserving_total() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        let mut a = Clip::video(0, 0, 100, 0, "A");
        a.src_in = 0;
        let mut b = Clip::video(0, 100, 100, 0, "B");
        b.src_in = 200;
        p.clips.push(a); // idx 0
        p.clips.push(b); // idx 1
        let total_before = p.clips[0].len + p.clips[1].len;
        let d = p.roll_edit(0, 1, 15); // cut moves right 15
        assert_eq!(d, 15);
        assert_eq!(p.clips[0].len, 115, "left grew");
        assert_eq!(p.clips[1].t0, 115, "right starts later");
        assert_eq!(p.clips[1].src_in, 215, "right source advanced with the cut");
        assert_eq!(p.clips[1].len, 85, "right shrank");
        assert_eq!(p.clips[0].len + p.clips[1].len, total_before, "combined span unchanged");
        // boundary-keyed convenience: roll the cut at frame 115 back left by 15.
        let d2 = p.roll(0, 115, -15);
        assert_eq!(d2, -15);
        assert_eq!(p.clips[0].len, 100);
        assert_eq!(p.clips[1].t0, 100);
    }

    #[test]
    fn copy_paste_offset_preserving() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 40, 30, 0, "A"));  // idx 0
        p.clips.push(Clip::video(0, 90, 20, 1, "B"));  // idx 1 (later, other track)
        let clip = p.copy_clips(&[0, 1]);
        assert_eq!(clip.len(), 2);
        // rebased: earliest (A at 40) -> 0; B keeps its +50 offset.
        assert_eq!(clip[0].t0, 0);
        assert_eq!(clip[1].t0, 50);
        let first = p.paste_clips(&clip, 200).unwrap();
        assert_eq!(p.clips.len(), 4);
        assert_eq!(p.clips[first].t0, 200, "first pasted at the playhead");
        assert_eq!(p.clips[first + 1].t0, 250, "offset preserved");
        assert_eq!(p.clips[first + 1].track, 1, "track preserved");
    }

    #[test]
    fn paste_skips_locked_track() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.tracks[1].locked = true; // V2 locked
        let clips = vec![Clip::video(0, 0, 30, 0, "ok"), Clip::video(0, 10, 30, 1, "locked")];
        let first = p.paste_clips(&clips, 100).unwrap();
        assert_eq!(p.clips.len(), 1, "only the unlocked-track clip pasted");
        assert_eq!(p.clips[first].track, 0);
    }

    // ----- 3-POINT EDITING ops (P4) ---------------------------------------------------------

    #[test]
    fn insert_clip_ripples_downstream_by_len() {
        // Track 0: A [0,100) len 100, B [100,160) len 60. Insert a 40-frame clip at t0=50 (inside
        // A). SHOTCUT PARITY: A straddles the insert point, so it is SPLIT at 50 -> A-left [0,50)
        // len 50 src_in 0, A-right [50,100) len 50 src_in 50. Then every same-track clip with
        // t0 >= 50 (A-right at 50, B at 100) shifts RIGHT by 40; the new clip lands at 50 in the
        // opened hole. B's distinct len (60) keeps it uniquely identifiable past the split.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A")); // idx 0
        p.clips.push(Clip::video(0, 100, 60, 0, "B")); // idx 1
        let src = Clip::video(0, 0, 40, 0, "NEW");
        let new_i = p.insert_clip(0, 50, src).expect("insert lands on unlocked track");
        // A split into two halves (+1) and the new clip pushed (+1): 2 -> 4 clips.
        assert_eq!(p.clips.len(), 4);
        assert_eq!(p.clips[new_i].t0, 50, "new clip at the insert frame");
        assert_eq!(p.clips[new_i].len, 40);
        // B (len 60, the only len-60 clip) rippled right by the inserted length (40): 100 -> 140.
        let b = p.clips.iter().find(|c| c.len == 60).unwrap();
        assert_eq!(b.t0, 140, "downstream clip shifted right by the inserted length");
        // A-left is the [0,50) remnant: len 50, src_in 0, unmoved (t0 0 < 50, not shifted right).
        let a_left = p
            .clips
            .iter()
            .find(|c| c.len == 50 && c.src_in == 0)
            .expect("A-left [0,50) remnant");
        assert_eq!(a_left.t0, 0, "left remnant of the split clip stays put");
        // A-right is the [50,100) remnant: split at 50 (src_in advanced to 50) then rippled +40 to 90.
        let a_right = p
            .clips
            .iter()
            .find(|c| c.len == 50 && c.src_in == 50)
            .expect("A-right [50,100) remnant rides the ripple");
        assert_eq!(a_right.t0, 90, "right remnant of the split clip rippled right by the inserted length");
    }

    #[test]
    fn insert_clip_at_clip_boundary_does_not_split() {
        // Insert exactly at a clip's start (a boundary, not strictly inside a body): no split, the
        // clip just ripples right. Track 0: A [0,100). Insert 30 frames at t0=0 (== A's start).
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A"));
        let new_i = p.insert_clip(0, 0, Clip::video(0, 0, 30, 0, "NEW")).unwrap();
        assert_eq!(p.clips.len(), 2, "no split at a clip boundary — only the new clip is added");
        assert_eq!(p.clips[new_i].t0, 0, "new clip at the insert frame");
        let a = p.clips.iter().find(|c| c.len == 100).unwrap();
        assert_eq!(a.t0, 30, "A (t0 0 >= 0) rippled right by the inserted length, intact (unsplit)");
    }

    #[test]
    fn insert_clip_other_track_unmoved_and_clamps() {
        // Insert must not move clips on a different track; t0 clamps to >= 0.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A")); // track 0, t0 >= 0
        p.clips.push(Clip::video(0, 0, 100, 1, "D")); // track 1 (other)
        let new_i = p.insert_clip(0, -10, Clip::video(0, 0, 30, 0, "NEW")).unwrap();
        assert_eq!(p.clips[new_i].t0, 0, "negative insert frame clamped to 0");
        // A (track 0, t0 0 >= 0) shifted right by 30; D (track 1) unmoved.
        let a = p.clips.iter().find(|c| c.track == 0 && c.len == 100).unwrap();
        assert_eq!(a.t0, 30, "same-track clip rippled");
        let d = p.clips.iter().find(|c| c.track == 1).unwrap();
        assert_eq!(d.t0, 0, "other-track clip unmoved");
    }

    #[test]
    fn overwrite_clip_replaces_covered_range_no_ripple() {
        // Track 0: A [0,100), B [100,200). Overwrite a 60-frame clip at t0=80, covering [80,140):
        // A is tail-trimmed to end at 80; B is head-trimmed to start at 140; nothing ripples (B's
        // tail and the timeline length are unchanged). The new clip occupies [80,140).
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A")); // idx 0
        p.clips.push(Clip::video(0, 100, 100, 0, "B")); // idx 1
        let new_i = p.overwrite_clip(0, 80, Clip::video(0, 0, 60, 0, "OVR")).expect("unlocked");
        assert_eq!(p.clips[new_i].t0, 80);
        assert_eq!(p.clips[new_i].len, 60);
        // A tail-trimmed: now ends at 80 (len 80).
        let a = p.clips.iter().find(|c| c.t0 == 0).unwrap();
        assert_eq!(a.end(), 80, "A tail-trimmed to the overwrite start");
        // B head-trimmed: now starts at 140 (its tail at 200 is untouched -> no ripple).
        let b = p.clips.iter().find(|c| c.end() == 200).unwrap();
        assert_eq!(b.t0, 140, "B head-trimmed to the overwrite end; tail unmoved (no ripple)");
    }

    #[test]
    fn overwrite_clip_straddle_splits_into_two_remnants() {
        // One big clip A [0,300) on track 0. Overwrite [100,200) with a 100-frame clip: A is cut
        // into a left remnant [0,100) and a right remnant [200,300); the new clip fills [100,200).
        let mut p = Project::demo("x".into());
        p.clips.clear();
        let mut a = Clip::video(0, 0, 300, 0, "A");
        a.src_in = 0;
        p.clips.push(a);
        let new_i = p.overwrite_clip(0, 100, Clip::video(0, 0, 100, 0, "OVR")).unwrap();
        // 3 clips now: left remnant, new clip, right remnant.
        assert_eq!(p.clips.len(), 3);
        assert_eq!(p.clips[new_i].t0, 100);
        assert_eq!(p.clips[new_i].end(), 200);
        let left = p.clips.iter().find(|c| c.t0 == 0).unwrap();
        assert_eq!(left.end(), 100, "left remnant ends at overwrite start");
        let right = p.clips.iter().find(|c| c.t0 == 200).unwrap();
        assert_eq!(right.end(), 300, "right remnant starts at overwrite end");
        assert_eq!(right.src_in, 200, "right remnant source advanced past the covered span");
    }

    #[test]
    fn overwrite_clip_fully_covered_is_removed() {
        // A small clip A [50,90) entirely inside the overwrite range [0,200) is removed.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 50, 40, 0, "A")); // [50,90)
        p.clips.push(Clip::video(0, 50, 40, 1, "D")); // other track, must survive
        let new_i = p.overwrite_clip(0, 0, Clip::video(0, 0, 200, 0, "OVR")).unwrap();
        // A removed; new clip + D remain (the new clip is appended LAST).
        assert_eq!(p.clips.len(), 2);
        assert_eq!(p.clips[new_i].t0, 0);
        assert!(p.clips.iter().any(|c| c.track == 1), "other-track clip survived");
        assert!(
            !p.clips.iter().any(|c| c.track == 0 && c.len == 40),
            "fully-covered same-track clip was removed"
        );
    }

    #[test]
    fn append_clip_lands_at_track_end() {
        // Track 0: A [0,100), B [100,150) -> track end 150. Track 1: C [0,80) -> track end 80.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A"));
        p.clips.push(Clip::video(0, 100, 50, 0, "B"));
        p.clips.push(Clip::video(0, 0, 80, 1, "C"));
        let v0 = p.append_clip(0, Clip::video(0, 0, 30, 0, "AP0")).unwrap();
        assert_eq!(p.clips[v0].t0, 150, "appended at the end of track 0");
        let v1 = p.append_clip(1, Clip::video(0, 0, 30, 1, "AP1")).unwrap();
        assert_eq!(p.clips[v1].t0, 80, "appended at the end of track 1");
    }

    #[test]
    fn append_clip_empty_track_lands_at_zero() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        let i = p.append_clip(2, Clip::video(0, 0, 30, 2, "A1")).unwrap();
        assert_eq!(p.clips[i].t0, 0, "append onto an empty track lands at frame 0");
        assert_eq!(p.track_end(2), 30, "track_end now reflects the appended clip");
    }

    #[test]
    fn three_point_ops_refuse_locked_track() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.tracks[0].locked = true; // V1 locked
        let n0 = p.clips.len();
        assert!(p.insert_clip(0, 0, Clip::video(0, 0, 30, 0, "x")).is_none());
        assert!(p.overwrite_clip(0, 0, Clip::video(0, 0, 30, 0, "x")).is_none());
        assert!(p.append_clip(0, Clip::video(0, 0, 30, 0, "x")).is_none());
        assert_eq!(p.clips.len(), n0, "no clip placed on a locked track");
    }

    #[test]
    fn overwrite_clip_sets_track_and_carries_source_fields() {
        // The placed clip takes its `track`/`t0` from the args but carries its other fields verbatim
        // (here a non-default look/gain), and overwrite onto an EMPTY track is just a placement.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        let mut src = Clip::video(0, 5, 40, 0, "S");
        src.src_in = 5; // a non-default source in-point (Clip::video's 2nd arg is t0, not src_in)
        src.look = 1;
        src.gain = 0.5;
        let i = p.overwrite_clip(1, 20, src).unwrap();
        assert_eq!(p.clips[i].track, 1, "track set from the arg");
        assert_eq!(p.clips[i].t0, 20, "t0 set from the arg");
        assert_eq!(p.clips[i].src_in, 5, "source in-point carried");
        assert_eq!(p.clips[i].look, 1, "look carried");
        assert!((p.clips[i].gain - 0.5).abs() < 1e-6, "gain carried");
    }

    // ----- P5 arbitrary tracks -----
    #[test]
    fn add_track_appends_and_names() {
        let mut p = Project::demo("x".into());
        assert_eq!(p.tracks.len(), 3); // default V1 V2 A1
        let vi = p.add_track(TrackKind::Video);
        assert_eq!(vi, 3);
        assert_eq!(p.tracks[3].name, "V3"); // third video
        assert_eq!(p.tracks[3].kind, TrackKind::Video);
        let ai = p.add_track(TrackKind::Audio);
        assert_eq!(p.tracks[ai].name, "A2"); // second audio
    }

    #[test]
    fn remove_track_drops_clips_and_reindexes() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 50, 0, "a")); // track 0 (V1)
        p.clips.push(Clip::video(0, 0, 50, 1, "b")); // track 1 (V2) -> removed
        p.clips.push(Clip::video(0, 0, 50, 2, "c")); // track 2 (A1) -> reindexes to 1
        assert!(p.remove_track(1)); // remove V2
        assert_eq!(p.tracks.len(), 2);
        assert_eq!(p.clips.len(), 2, "the clip on the removed track is gone");
        assert!(p.clips.iter().any(|c| c.track == 0), "V1 clip stays on track 0");
        assert!(p.clips.iter().any(|c| c.track == 1), "A1 clip reindexed 2 -> 1");
        assert!(!p.clips.iter().any(|c| c.track == 2), "no clip left on the old track 2");
    }

    #[test]
    fn is_audio_and_hidden_read_tracks() {
        let mut p = Project::demo("x".into());
        assert!(!p.is_audio(0) && !p.is_audio(1) && p.is_audio(2)); // V1 V2 video, A1 audio
        p.tracks[0].hidden = true;
        p.tracks[2].muted = true;
        assert!(p.is_hidden(0) && !p.is_hidden(1));
        assert!(p.is_muted(2) && !p.is_muted(0));
        assert!(!p.is_hidden(99), "out-of-range track is not hidden");
    }

    #[test]
    fn title_is_empty_and_clip_is_title() {
        // Default title -> empty -> the clip is NOT a title (untitled clips render unchanged).
        let mut t = Title::default();
        assert!(t.is_empty(), "default title is empty");
        // Whitespace-only counts as empty (worker still composites normally).
        t.text = "   \t \n".into();
        assert!(t.is_empty(), "whitespace-only title is empty");
        // Real text -> not empty.
        t.text = "Hello".into();
        assert!(!t.is_empty(), "non-blank title is not empty");

        // Clip::is_title mirrors !title.is_empty(): the demo clips have a default (empty) title.
        let mut c = Clip::video(0, 0, 100, 0, "x");
        assert!(!c.is_title(), "a fresh clip has no title");
        c.title.text = "Lower third".into();
        assert!(c.is_title(), "a clip with text is a title");
        c.title.text = "  ".into();
        assert!(!c.is_title(), "whitespace title is not a title");

        // The lower_third preset is a real (non-empty) title with the expected layout.
        let lt = Title::lower_third("Name / Role");
        assert!(!lt.is_empty(), "lower_third has text");
        assert_eq!(lt.text, "Name / Role");
        assert!(lt.y > 0.5, "lower_third anchors toward the lower part of the frame");
        assert_eq!(lt.rgb, [1.0, 1.0, 1.0], "lower_third defaults to white");
    }
}
