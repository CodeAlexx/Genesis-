//! Side panels — properties / filters (right) + scopes.
//!
//! Owned by the layout/panels team. Mirrors MojoMedia's properties ribbon (Color + Comp
//! tabs): per-clip PiP rect (X/Y/W/H, fractions 0..1), fades, look index/mix, plus the
//! program-wide grade (brightness/contrast/saturation). The SCOPES section (Slice C) shows a
//! live histogram / luma waveform / vectorscope of the composited program frame at the playhead,
//! computed on the GPU by the `gcompose` worker (`worker::scope`) and blitted as a 256×256 image
//! (mirrors MojoMedia main_editor.mojo's Shotcut-style Hist/Wave/Vec scope selector).

use crate::icons;
use crate::model::{History, KfInterp, Project};
use crate::theme;
use crate::worker;
use eframe::egui;
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Spawn a native `zenity --file-selection` dialog filtered to `*.cube` and return the chosen
/// path, or `None` if the user cancelled / zenity is unavailable. Mirrors MojoMedia's "Load .cube"
/// flow (it lists a luts dir; we let the user pick any .cube anywhere). Blocking by design — the
/// click handler waits for the modal, exactly as a file-open dialog should. A non-zero zenity exit
/// (cancel) or a missing zenity binary both fold into `None` so the UI never panics on the picker.
fn pick_cube_file() -> Option<String> {
    let out = Command::new("zenity")
        .args([
            "--file-selection",
            "--title=Load .cube LUT",
            "--file-filter=Cube LUT (*.cube) | *.cube *.CUBE",
            "--file-filter=All files | *",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None; // user cancelled, or zenity errored
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if path.is_empty() {
        None
    } else {
        Some(path)
    }
}

/// The basename of a path for compact display (`/a/b/teal_orange.cube` -> `teal_orange.cube`).
/// Empty input -> "no LUT". Splits on '/' only (paths here are POSIX from zenity / the project).
fn lut_basename(path: &str) -> &str {
    if path.is_empty() {
        return "no LUT";
    }
    match path.rfind('/') {
        Some(i) => &path[i + 1..],
        None => path,
    }
}

/// Linear gain multiplier -> decibels (for the per-clip Gain slider; Shotcut "Gain / Volume" is in
/// dB, −70..+24). A non-positive linear value floors at −70 dB (effectively silence).
fn lin_to_db(lin: f32) -> f32 {
    if lin <= 0.0 {
        -70.0
    } else {
        20.0 * lin.log10()
    }
}

/// Decibels -> linear gain multiplier (inverse of `lin_to_db`). −70 dB (or below) maps to 0.0
/// (silence) so the slider's floor is a true mute.
fn db_to_lin(db: f32) -> f32 {
    if db <= -70.0 {
        0.0
    } else {
        10.0f32.powf(db / 20.0)
    }
}

/// A thin labeled section header inside a panel.
fn section(ui: &mut egui::Ui, label: &str) {
    ui.add_space(4.0);
    ui.label(egui::RichText::new(label).color(egui::Color32::from_rgb(150, 150, 160)).size(11.0));
    ui.separator();
}

/// Display label for a `Clip.track` index (0 = V1, 1 = V2, 2 = A1). Out-of-range
/// tracks fall through to a numeric "T{n}" so the header never lies (audio clips on
/// track 2 must read "A1", not "V2").
fn track_label(track: u8) -> String {
    match track {
        0 => "V1".into(),
        1 => "V2".into(),
        2 => "A1".into(),
        n => format!("T{n}"),
    }
}

/// Display label for media index `m` in the P35 "Replace media" combo: the per-media display name
/// (mirrors pool_ui's `project.names.get(i)`), falling back to `media {m}` for an unnamed / past-end
/// index. Read-only — just builds a string for the picker.
fn media_label(project: &Project, m: usize) -> String {
    project.names.get(m).cloned().unwrap_or_else(|| format!("media {m}"))
}

pub fn properties_ui(
    ui: &mut egui::Ui,
    project: &mut Project,
    selected: usize,
    selection: &[usize],
    history: &mut History,
    playhead: i64,
) {
    section(ui, "PROPERTIES");

    // ---- Comp tab: the selected clip's picture-in-picture rect + fades + look ----
    if let Some(c) = project.clips.get(selected) {
        ui.label(
            egui::RichText::new(format!("clip {selected}  \u{2022}  track {}  \u{2022}  t0 {}  \u{2022}  len {}", track_label(c.track), c.t0, c.len))
                .color(theme::TEXT)
                .size(11.0),
        );
    } else {
        ui.weak("no clip selected");
    }

    // The clip's timeline start (captured before the mutable borrow below) so the PiP Key
    // button can compute the CLIP-LOCAL frame (playhead - t0) once the borrow has ended.
    let mut clip_t0: Option<i64> = None;
    // P30: per-slider "◆" Key clicks queued while the `&mut c` slider borrow is live. Each entry is
    // (par, value): par per the PipKey registry (4=bright 5=contrast 6=sat 7=blur 8=rot 9=scale),
    // value = the slider's CURRENT field value. The buttons sit INLINE next to their slider but only
    // PUSH here (a plain local Vec, no project borrow); the actual `add_clip_param_key` calls run
    // AFTER the clip borrow ends (mirrors how the "Key PiP" button defers to `clip_t0`), so the
    // mutable slider borrow and the `&mut project` add never overlap.
    let mut pending_param_keys: Vec<(u8, f32)> = Vec::new();
    if let Some(c) = project.clips.get_mut(selected) {
        clip_t0 = Some(c.t0);
        section(ui, "PiP (picture-in-picture)");
        ui.add(egui::Slider::new(&mut c.px, 0.0..=1.0).text("X"));
        ui.add(egui::Slider::new(&mut c.py, 0.0..=1.0).text("Y"));
        ui.add(egui::Slider::new(&mut c.pw, 0.0..=1.0).text("W"));
        ui.add(egui::Slider::new(&mut c.ph, 0.0..=1.0).text("H"));
        if ui.button("Reset PiP (full frame)").clicked() {
            c.px = 0.0;
            c.py = 0.0;
            c.pw = 1.0;
            c.ph = 1.0;
        }

        // ---- P31 BLEND MODE (V2 overlay compositing). Mirrors Shotcut's per-clip blend modes
        // (qtblend/cairoblend): when THIS clip is the V2 OVERLAY composited over the V1 base, its RGB
        // is combined with the base via this mode BEFORE the alpha-over. Only meaningful for the
        // overlay clip (a base/single clip ignores it), but shown for any selected clip — like the
        // Chroma Key controls. Binds clip.blend_mode (u8); 0 = Normal = plain alpha-over =
        // byte-identical to pre-P31. Mutating the selected clip in place IS the dirty signal (same as
        // the adjacent PiP/grade controls).
        ui.label(egui::RichText::new("Blend (V2 overlay)").color(theme::TEXT).size(10.0));
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut c.blend_mode, 0u8, "Normal");
            ui.selectable_value(&mut c.blend_mode, 1u8, "Multiply");
            ui.selectable_value(&mut c.blend_mode, 2u8, "Screen");
            ui.selectable_value(&mut c.blend_mode, 3u8, "Overlay");
            ui.selectable_value(&mut c.blend_mode, 4u8, "Add");
            ui.selectable_value(&mut c.blend_mode, 5u8, "Darken");
            ui.selectable_value(&mut c.blend_mode, 6u8, "Lighten");
            ui.selectable_value(&mut c.blend_mode, 7u8, "Difference");
        });

        section(ui, "Fades (frames)");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut c.fade_in).speed(1.0).range(0..=600).prefix("in "));
            ui.add(egui::DragValue::new(&mut c.fade_out).speed(1.0).range(0..=600).prefix("out "));
        });

        // ---- Per-clip SPEED / REVERSE (P24, Model A). Mirrors Shotcut's clip Properties > Speed /
        // Reverse. The clip keeps its timeline footprint (t0/len UNCHANGED); `speed` scales how fast
        // the SOURCE is consumed (2.0 = 2x faster, 0.5 = slow-mo) and `reverse` plays the consumed
        // source range backward. Mutating the selected clip in place IS the dirty signal (same as the
        // adjacent controls). Identity (1.0x, no reverse) is byte-identical to pre-P24.
        section(ui, "Speed");
        ui.add(egui::Slider::new(&mut c.speed, 0.25..=4.0).text("Speed").suffix("x"));
        ui.checkbox(&mut c.reverse, "Reverse");
        if ui.button("Reset speed (1x)").clicked() {
            c.speed = 1.0;
            c.reverse = false;
        }

        // ---- Per-clip AUDIO GAIN (Triad-B P1). Stored linear (Clip.gain, 1.0 = unity); surfaced as
        // a dB slider matching Shotcut's "Gain / Volume" range (−70..+24 dB). The same fade_in/
        // fade_out above ALSO ramp the audio at mix time (worker passes the fades on the AUDIO line).
        section(ui, "Audio");
        let mut db = lin_to_db(c.gain);
        if ui.add(egui::Slider::new(&mut db, -70.0..=24.0).text("Gain (dB)")).changed() {
            c.gain = db_to_lin(db);
        }
        if ui.button("Reset gain (0 dB)").clicked() {
            c.gain = 1.0;
        }

        // ---- Per-clip AUDIO FX (Triad-B P3). Binds to clip.audio_fx (the worker maps these to a
        // libavfilter chain; a neutral AudioFx → no chain → byte-identical to P2). Ranges mirror
        // Shotcut: EQ 3-band ±20 dB (audio_eq3band), Pan −1..1 (audio_pan), and Compress / Gate /
        // Normalize toggles (audio_compressor / audio_noisegate / audio_normalize_1p). Team B
        // READS/WRITES audio_fx only — it never edits model.rs.
        section(ui, "Audio FX");
        let fx = &mut c.audio_fx;
        ui.label(egui::RichText::new("EQ (dB)").color(theme::TEXT).size(10.0));
        ui.add(egui::Slider::new(&mut fx.eq_low_db, -20.0..=20.0).text("Low"));
        ui.add(egui::Slider::new(&mut fx.eq_mid_db, -20.0..=20.0).text("Mid"));
        ui.add(egui::Slider::new(&mut fx.eq_high_db, -20.0..=20.0).text("High"));
        ui.add(egui::Slider::new(&mut fx.pan, -1.0..=1.0).text("Pan (L \u{2194} R)"));
        ui.horizontal(|ui| {
            ui.checkbox(&mut fx.compress, "Compress");
            ui.checkbox(&mut fx.gate, "Gate");
            ui.checkbox(&mut fx.normalize, "Normalize");
        });

        // ---- P11 audio effects (Shotcut Reverb / Delay / Pitch). 0 / off defaults keep AudioFx
        // neutral → worker emits "-" → byte-identical audio to P10. Worker maps these to aecho
        // (delay + multi-tap reverb) and rubberband (tempo-preserving pitch shift).
        ui.add_space(4.0);
        ui.label(egui::RichText::new("Effects").color(theme::TEXT).size(10.0));
        ui.add(egui::Slider::new(&mut fx.reverb, 0.0..=1.0).text("Reverb"));
        ui.add(egui::Slider::new(&mut fx.delay_ms, 0.0..=1000.0).text("Delay (ms)"));
        ui.add(egui::Slider::new(&mut fx.delay_decay, 0.0..=0.95).text("Delay decay"));
        ui.add(egui::Slider::new(&mut fx.pitch, -24.0..=24.0).text("Pitch (semitones)"));

        // ---- P12 audio filters (Shotcut Low Pass / High Pass / Tremolo). 0 / off defaults keep
        // AudioFx neutral → worker emits "-" → byte-identical audio. Worker maps these to the
        // libavfilter `lowpass` / `highpass` / `tremolo` filters (cutoff Hz; tremolo depth 0..0.95).
        ui.add(egui::Slider::new(&mut fx.lowpass_hz, 0.0..=20000.0).text("Low Pass (Hz, 0=off)"));
        ui.add(egui::Slider::new(&mut fx.highpass_hz, 0.0..=20000.0).text("High Pass (Hz, 0=off)"));
        ui.add(egui::Slider::new(&mut fx.tremolo, 0.0..=0.95).text("Tremolo"));

        // ---- P15 audio filters (Shotcut Bass & Treble / Notch / Chorus). 0 / off defaults keep
        // AudioFx neutral → worker emits "-" → byte-identical audio. Worker maps these to the
        // libavfilter `bass` / `treble` shelves (gain dB, 0 = flat), `bandreject` (notch centre Hz)
        // and `chorus` (single-voice, depth 0..1).
        ui.add(egui::Slider::new(&mut fx.bass_db, -30.0..=30.0).text("Bass (dB)"));
        ui.add(egui::Slider::new(&mut fx.treble_db, -30.0..=30.0).text("Treble (dB)"));
        ui.add(egui::Slider::new(&mut fx.notch_hz, 0.0..=20000.0).text("Notch (Hz, 0=off)"));
        ui.add(egui::Slider::new(&mut fx.chorus, 0.0..=1.0).text("Chorus"));

        // ---- P22 audio filters (Shotcut Flanger / Phaser / Limiter). 0 / off defaults keep
        // AudioFx neutral → worker emits "-" → byte-identical audio. Worker maps these to the
        // libavfilter `flanger` (depth 0..8 ms), `aphaser` (sweep speed Hz) and `alimiter`
        // (linear peak ceiling) filters.
        ui.add(egui::Slider::new(&mut fx.flanger, 0.0..=1.0).text("Flanger"));
        ui.add(egui::Slider::new(&mut fx.phaser, 0.0..=1.0).text("Phaser"));
        ui.add(egui::Slider::new(&mut fx.limiter, 0.0..=1.0).text("Limiter (0=off)"));

        // ---- P32 GRAPHIC 10-BAND EQ (Shotcut audio_eq15band-style). One slider per ISO octave band;
        // each is a peaking gain in dB at that band centre. All 0 dB → AudioFx stays neutral → worker
        // emits "-" → byte-identical audio. Worker maps each non-zero band to a one-octave `equalizer`
        // peaking part. "Flat" zeroes all 10 bands. egui 0.31 Sliders bound directly to fx.geq[i].
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Graphic EQ (dB)").color(theme::TEXT).size(10.0));
            if ui.button("Flat").clicked() {
                fx.geq = [0.0; 10];
            }
        });
        const GEQ_LABELS: [&str; 10] =
            ["31", "62", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"];
        for i in 0..10 {
            ui.add(egui::Slider::new(&mut fx.geq[i], -12.0..=12.0).text(GEQ_LABELS[i]));
        }

        if ui.button("Reset audio FX").clicked() {
            // Default() restores every control — EQ/pan/dynamics, the P11 effects (reverb 0,
            // delay_ms 0, delay_decay 0.5, pitch 0), the P12 filters (lowpass_hz 0, highpass_hz 0,
            // tremolo 0), the P15 filters (bass_db 0, treble_db 0, notch_hz 0, chorus 0) AND the
            // P22 filters (flanger 0, phaser 0, limiter 0) — so the clip returns to the neutral
            // "-" state.
            *fx = crate::model::AudioFx::default();
        }

        // ---- Per-clip COLOR grade (Triad-B P1; ADDITIVE on top of the program grade below). Same
        // ranges as the program grade: brightness −1..1 (added), contrast/saturation 0..2 (multiply,
        // 1.0 = identity). gcompose applies the per-clip grade FIRST, then the program grade.
        section(ui, "Clip Grade");
        // P30: each slider gets an inline "◆" Key button that snapshots the slider's CURRENT value
        // into a per-clip keyframe at the clip-local playhead frame (par 4=bright, 5=contrast,
        // 6=sat). The click only queues into `pending_param_keys`; the add runs after the borrow.
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut c.bright, -1.0..=1.0).text("Brightness"));
            if ui.small_button("\u{25C6}").on_hover_text("Key brightness @ playhead").clicked() {
                pending_param_keys.push((4, c.bright));
            }
        });
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut c.contrast, 0.0..=2.0).text("Contrast"));
            if ui.small_button("\u{25C6}").on_hover_text("Key contrast @ playhead").clicked() {
                pending_param_keys.push((5, c.contrast));
            }
        });
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut c.sat, 0.0..=2.0).text("Saturation"));
            if ui.small_button("\u{25C6}").on_hover_text("Key saturation @ playhead").clicked() {
                pending_param_keys.push((6, c.sat));
            }
        });
        if ui.button("Reset clip grade").clicked() {
            c.bright = 0.0;
            c.contrast = 1.0;
            c.sat = 1.0;
        }

        // ---- Color wheels (3-way LIFT / GAMMA / GAIN). Mirrors Shotcut's movit.lift_gamma_gain
        // (Color Grading). Engine semantics (PINNED Team A): per channel
        //   out = pow(clamp(in*gain + lift, 0, 1), 1/gamma).
        // Ranges follow Shotcut: lift −1..1 (additive shadow offset, identity 0); gamma 0..2
        // (midtone power, identity 1); gain 0..4 (highlight multiplier, identity 1). Three labeled
        // groups each with R/G/B drag-sliders. White balance (below) is folded into GAIN by the
        // worker — these are the RAW wheel values the user dials.
        section(ui, "Color Wheels (Lift / Gamma / Gain)");
        ui.label(egui::RichText::new("Lift (shadows)").color(theme::TEXT).size(10.0));
        ui.add(egui::Slider::new(&mut c.lift[0], -1.0..=1.0).text("R"));
        ui.add(egui::Slider::new(&mut c.lift[1], -1.0..=1.0).text("G"));
        ui.add(egui::Slider::new(&mut c.lift[2], -1.0..=1.0).text("B"));
        ui.label(egui::RichText::new("Gamma (midtones)").color(theme::TEXT).size(10.0));
        ui.add(egui::Slider::new(&mut c.gamma[0], 0.0..=2.0).text("R"));
        ui.add(egui::Slider::new(&mut c.gamma[1], 0.0..=2.0).text("G"));
        ui.add(egui::Slider::new(&mut c.gamma[2], 0.0..=2.0).text("B"));
        ui.label(egui::RichText::new("Gain (highlights)").color(theme::TEXT).size(10.0));
        ui.add(egui::Slider::new(&mut c.gain_rgb[0], 0.0..=4.0).text("R"));
        ui.add(egui::Slider::new(&mut c.gain_rgb[1], 0.0..=4.0).text("G"));
        ui.add(egui::Slider::new(&mut c.gain_rgb[2], 0.0..=4.0).text("B"));
        if ui.button("Reset wheels").clicked() {
            c.lift = [0.0, 0.0, 0.0];
            c.gamma = [1.0, 1.0, 1.0];
            c.gain_rgb = [1.0, 1.0, 1.0];
        }

        // ---- White balance (Temp / Tint). NOT a wire field — the worker FOLDS these into the GAIN
        // channels before sending (the engine only ever sees lift/gamma/gain). Both are biases in
        // −1..1 (0 = neutral): Temp >0 warms (red up / blue down), Tint >0 greens (green up /
        // red+blue down). Mirrors Shotcut's white-balance temperature, simplified to two sliders.
        section(ui, "White Balance");
        ui.add(egui::Slider::new(&mut c.wb_temp, -1.0..=1.0).text("Temp (cool \u{2194} warm)"));
        ui.add(egui::Slider::new(&mut c.wb_tint, -1.0..=1.0).text("Tint (magenta \u{2194} green)"));
        if ui.button("Reset white balance").clicked() {
            c.wb_temp = 0.0;
            c.wb_tint = 0.0;
        }

        // ---- Transform (rotate + scale of the base clip). Mirrors Shotcut's rotate filter:
        // rotation in degrees (−180..180, identity 0) and uniform scale (0.1..4, identity 1), both
        // about the frame center. Engine fpx_gpu_transform(rot_deg, scale) bilinear-samples the base.
        section(ui, "Transform");
        // P30: inline "◆" Key buttons (par 8=rot, 9=scale) — queue the current value, add after borrow.
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut c.rot, -180.0..=180.0).text("Rotation (deg)"));
            if ui.small_button("\u{25C6}").on_hover_text("Key rotation @ playhead").clicked() {
                pending_param_keys.push((8, c.rot));
            }
        });
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut c.scale, 0.1..=4.0).text("Scale"));
            if ui.small_button("\u{25C6}").on_hover_text("Key scale @ playhead").clicked() {
                pending_param_keys.push((9, c.scale));
            }
        });
        if ui.button("Reset transform").clicked() {
            c.rot = 0.0;
            c.scale = 1.0;
        }

        // ---- Gaussian blur (sigma). Mirrors Shotcut's blur_gaussian (av.sigma). sigma 0 = no blur;
        // the engine fpx_gpu_blur(sigma) runs a separable gaussian (no-op at sigma <= 0).
        section(ui, "Blur");
        // P30: inline "◆" Key button (par 7=blur) — queue the current value, add after borrow.
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut c.blur, 0.0..=20.0).text("Gaussian (sigma)"));
            if ui.small_button("\u{25C6}").on_hover_text("Key blur @ playhead").clicked() {
                pending_param_keys.push((7, c.blur));
            }
        });
        if ui.button("Reset blur").clicked() {
            c.blur = 0.0;
        }

        // ---- P5 master tone CURVE: 5 control-point outputs at fixed inputs 0/.25/.5/.75/1. The
        // engine piecewise-linear interpolates and applies it to all 3 channels (after blur, before
        // look). Identity = [0,.25,.5,.75,1] (no-op). Lifting the mid point brightens midtones, etc.
        section(ui, "Curve (tone)");
        const CURVE_LABELS: [&str; 5] = ["Black", "Shadow", "Mid", "Highlight", "White"];
        for (i, label) in CURVE_LABELS.iter().enumerate() {
            ui.add(egui::Slider::new(&mut c.curve[i], 0.0..=1.0).text(*label));
        }
        if ui.button("Reset curve (identity)").clicked() {
            c.curve = [0.0, 0.25, 0.5, 0.75, 1.0];
        }

        // ---- P6 STYLIZE / UTILITY filters. Four per-clip filters applied by the engine on the
        // composited OUTB AFTER the master curve (above) and BEFORE the look (below), in the engine
        // order simple-fx -> vignette -> sharpen -> flip. Each is no-op at its default (vignette 0,
        // sharpen 0, flip None, fx None), so an un-stylized clip renders byte-identically. Binds the
        // pre-added Clip fields vignette:f32 / sharpen:f32 / flip:u8 / fx:i32 (Team B reads/writes
        // them; never edits model.rs).
        //   Vignette  : radial edge-darken amount (0..1; Shotcut "Vignette").
        //   Sharpen   : unsharp-mask amount (0..2; Shotcut "Sharpen").
        //   Flip      : mirror None / Horizontal / Vertical / Both (Shotcut "Flip").
        //   Simple FX : None / Invert / Sepia / Grayscale / Posterize (utility colour ops).
        section(ui, "Stylize / Utility");
        ui.add(egui::Slider::new(&mut c.vignette, 0.0..=1.0).text("Vignette"));
        ui.add(egui::Slider::new(&mut c.sharpen, 0.0..=2.0).text("Sharpen"));

        // Flip selector (mirror). clip.flip: 0 None / 1 Horizontal / 2 Vertical / 3 Both.
        ui.label(egui::RichText::new("Flip (mirror)").color(theme::TEXT).size(10.0));
        ui.horizontal(|ui| {
            ui.selectable_value(&mut c.flip, 0u8, "None");
            ui.selectable_value(&mut c.flip, 1u8, "Horizontal");
            ui.selectable_value(&mut c.flip, 2u8, "Vertical");
            ui.selectable_value(&mut c.flip, 3u8, "Both");
        });

        // Simple FX selector. clip.fx: 0 None / 1 Invert / 2 Sepia / 3 Grayscale / 4 Posterize.
        ui.label(egui::RichText::new("Simple FX").color(theme::TEXT).size(10.0));
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut c.fx, 0i32, "None");
            ui.selectable_value(&mut c.fx, 1i32, "Invert");
            ui.selectable_value(&mut c.fx, 2i32, "Sepia");
            ui.selectable_value(&mut c.fx, 3i32, "Grayscale");
            ui.selectable_value(&mut c.fx, 4i32, "Posterize");
        });

        if ui.button("Reset stylize").clicked() {
            c.vignette = 0.0;
            c.sharpen = 0.0;
            c.flip = 0;
            c.fx = 0;
        }

        // ---- P7 HSL ADJUST. Per-pixel RGB->HSL color filter applied by the engine on the composited
        // OUTB AFTER the P6 stylize filters (flip) and BEFORE the look, in the order HSL -> LEVELS.
        // Mirrors Shotcut's "Hue/Lightness/Saturation". Binds the pre-added Clip.hsl:[f32;3]
        // (Team B reads/writes it; never edits model.rs):
        //   hsl[0] = hue shift in degrees (−180..180, wraps 360; identity 0),
        //   hsl[1] = saturation multiplier (0..2, identity 1),
        //   hsl[2] = lightness add (−1..1, identity 0).
        // Identity (0/1/0) is a no-op (the engine skips the HSL kernel), so an un-adjusted clip
        // renders byte-identically.
        section(ui, "HSL Adjust");
        ui.add(egui::Slider::new(&mut c.hsl[0], -180.0..=180.0).text("Hue shift (deg)"));
        ui.add(egui::Slider::new(&mut c.hsl[1], 0.0..=2.0).text("Saturation"));
        ui.add(egui::Slider::new(&mut c.hsl[2], -1.0..=1.0).text("Lightness"));
        if ui.button("Reset HSL").clicked() {
            c.hsl = [0.0, 1.0, 0.0];
        }

        // ---- P7 LEVELS. Per-channel input black/white point + gamma applied by the engine right
        // AFTER the HSL adjust (and before the look). Mirrors Shotcut's "Levels". Binds the pre-added
        // Clip.levels:[f32;3] (Team B reads/writes it; never edits model.rs):
        //   levels[0] = input black point (0..1, identity 0),
        //   levels[1] = input white point (0..1, identity 1),
        //   levels[2] = gamma (0.1..4, identity 1).
        // Per channel: out = clamp01(pow(clamp01((in - in_black)/(in_white - in_black)), 1/gamma)).
        // Identity (0/1/1) is a no-op (the engine skips the LEVELS kernel), byte-identical.
        section(ui, "Levels");
        ui.add(egui::Slider::new(&mut c.levels[0], 0.0..=1.0).text("Input Black"));
        ui.add(egui::Slider::new(&mut c.levels[1], 0.0..=1.0).text("Input White"));
        ui.add(egui::Slider::new(&mut c.levels[2], 0.1..=4.0).text("Gamma"));
        if ui.button("Reset levels").clicked() {
            c.levels = [0.0, 1.0, 1.0];
        }

        // ---- P8 STYLIZE 2. Two per-clip filters applied by the engine on the composited OUTB
        // AFTER the P7 color filters (levels) and BEFORE the look, in the order MOSAIC ->
        // GRADIENT MAP. Mirrors Shotcut's "Mosaic" (pixelate) + "Gradient Map". Binds the
        // pre-added Clip.mosaic:u32 / Clip.gmap_amt:f32 / Clip.gmap_lo:[f32;3] /
        // Clip.gmap_hi:[f32;3] (Team B reads/writes them; never edits model.rs):
        //   mosaic   = pixelate block size in px (0..128; 0 or 1 = off, identity),
        //   gmap_amt = gradient-map mix (0..1; 0 = off, identity),
        //   gmap_lo  = shadow ramp colour (luma 0; identity [0,0,0] black),
        //   gmap_hi  = highlight ramp colour (luma 1; identity [1,1,1] white).
        // Identity (mosaic 0/1, gmap_amt 0) is a no-op (the engine skips both kernels), so an
        // un-stylized clip renders byte-identically.
        section(ui, "Stylize 2");
        // Mosaic block size. The slider edits an f32 we round back into the u32 field so egui's
        // Slider (float-only) can drive the integer block size; 0/1 = off (no pixelation).
        let mut mosaic_px = c.mosaic as f32;
        if ui
            .add(egui::Slider::new(&mut mosaic_px, 0.0..=128.0).text("Mosaic (px)"))
            .changed()
        {
            c.mosaic = mosaic_px.round().clamp(0.0, 128.0) as u32;
        }
        // Gradient-map amount (luma -> lo..hi colour ramp mix).
        ui.add(egui::Slider::new(&mut c.gmap_amt, 0.0..=1.0).text("Gradient Map"));
        // Shadow + highlight ramp colours (each [r,g,b] in [0,1] via egui's RGB picker).
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Shadows").color(theme::TEXT).size(10.0));
            ui.color_edit_button_rgb(&mut c.gmap_lo);
            ui.label(egui::RichText::new("Highlights").color(theme::TEXT).size(10.0));
            ui.color_edit_button_rgb(&mut c.gmap_hi);
        });
        if ui.button("Reset Stylize 2").clicked() {
            c.mosaic = 0;
            c.gmap_amt = 0.0;
            c.gmap_lo = [0.0, 0.0, 0.0];
            c.gmap_hi = [1.0, 1.0, 1.0];
        }

        // ---- P9 FX FILTERS. Three per-clip filters applied by the engine on the composited OUTB
        // AFTER the P8 stylize-2 filters (gradient map) and BEFORE the look, in the engine order
        // DENOISE -> GLOW -> RGB-SHIFT. Each is no-op at its default (denoise 0, glow_amt 0,
        // rgbshift 0), so an un-FX'd clip renders byte-identically. Binds the pre-added Clip fields
        // denoise:f32 / glow_amt:f32 / glow_thr:f32 / rgbshift:f32 (Team B reads/writes them; never
        // edits model.rs).
        //   Denoise        : edge-preserving (bilateral) smooth strength (0..1; Shotcut "Reduce Noise").
        //   Glow amount    : bloom mix — bright-pass blur added back (0..1; Shotcut "Glow").
        //   Glow threshold : luma above which a pixel blooms (0..1; only matters when amount > 0).
        //   RGB Shift      : chromatic-aberration channel offset in px (0..32; R +shift, B −shift).
        section(ui, "FX");
        ui.add(egui::Slider::new(&mut c.denoise, 0.0..=1.0).text("Denoise"));
        ui.add(egui::Slider::new(&mut c.glow_amt, 0.0..=1.0).text("Glow amount"));
        ui.add(egui::Slider::new(&mut c.glow_thr, 0.0..=1.0).text("Glow threshold"));
        ui.add(egui::Slider::new(&mut c.rgbshift, 0.0..=32.0).text("RGB Shift (px)"));
        if ui.button("Reset FX").clicked() {
            c.denoise = 0.0;
            c.glow_amt = 0.0;
            c.glow_thr = 0.7;
            c.rgbshift = 0.0;
        }

        // ---- P10 STYLIZE-4 FILTERS. Three per-clip filters applied by the engine on the composited
        // OUTB AFTER the P9 FX filters (RGB-shift) and BEFORE the look, in the engine order
        // HALFTONE -> EMBOSS -> EDGE. Each is no-op at its default (halftone 0/1, emboss 0, edge 0),
        // so an un-stylized clip renders byte-identically. Binds the pre-added Clip fields
        // halftone:u32 / emboss:f32 / edge:f32 (Team B reads/writes them; never edits model.rs).
        //   Halftone     : luma-driven dot-screen cell size in px (0/1 = off; darker => bigger dot).
        //   Emboss       : directional (NW) relief strength (0..1; Shotcut "Emboss").
        //   Edge/Sketch  : Sobel edge-detect mix back over the image (0..1; Shotcut "Sketch").
        section(ui, "Stylize 3");
        // Halftone cell size. The slider edits an f32 we round back into the u32 field so egui's
        // Slider (float-only) can drive the integer cell size; 0/1 = off (no dots).
        let mut halftone_px = c.halftone as f32;
        if ui
            .add(egui::Slider::new(&mut halftone_px, 0.0..=64.0).text("Halftone (px)"))
            .changed()
        {
            c.halftone = halftone_px.round().clamp(0.0, 64.0) as u32;
        }
        ui.add(egui::Slider::new(&mut c.emboss, 0.0..=1.0).text("Emboss"));
        ui.add(egui::Slider::new(&mut c.edge, 0.0..=1.0).text("Edge / Sketch"));
        if ui.button("Reset Stylize 3").clicked() {
            c.halftone = 0;
            c.emboss = 0.0;
            c.edge = 0.0;
        }

        // ---- P13 OLD FILM / DISTORT. Three per-clip filters applied by the engine on the composited
        // OUTB AFTER the P10 stylize-4 filters (edge) and BEFORE the look, in the engine order
        // GRAIN -> SCRATCHES -> DIFFUSION. Each is no-op at its default (grain 0, scratches 0,
        // diffusion 0), so an un-aged clip renders byte-identically. The effects are DETERMINISTIC (a
        // coordinate integer hash, not time/RNG) so a held frame is stable. Binds the pre-added Clip
        // fields grain:f32 / scratches:f32 / diffusion:f32 (Team B reads/writes them; never edits
        // model.rs).
        //   Grain      : film-noise strength (0..1; luma noise added per pixel; Shotcut "Old Film: Grain").
        //   Scratches  : old-film vertical-scratch density/amount (0..1; bright/dark vertical lines;
        //                Shotcut "Old Film: Scratches").
        //   Diffusion  : frosted-glass jitter radius in px (0..16; per-pixel random spatial offset).
        section(ui, "Old Film");
        ui.add(egui::Slider::new(&mut c.grain, 0.0..=1.0).text("Grain"));
        ui.add(egui::Slider::new(&mut c.scratches, 0.0..=1.0).text("Scratches"));
        ui.add(egui::Slider::new(&mut c.diffusion, 0.0..=16.0).text("Diffusion (px)"));
        if ui.button("Reset Old Film").clicked() {
            c.grain = 0.0;
            c.scratches = 0.0;
            c.diffusion = 0.0;
        }

        // ---- P16 DISTORT. Three per-clip filters applied by the engine on the composited OUTB
        // AFTER the P13 old-film filters (diffusion) and BEFORE the look, in the engine order
        // WAVE -> SWIRL -> THRESHOLD. Each is no-op at its default (wave 0, swirl 0, threshold 0),
        // so an un-distorted clip renders byte-identically. Binds the pre-added Clip fields
        // wave:f32 / swirl:f32 / threshold:f32 (Team B reads/writes them; never edits model.rs).
        //   Wave      : sinusoidal horizontal row-displacement amplitude in px (0..40; Shotcut "Wave").
        //   Swirl     : rotational distortion strength in radians at the centre (0..π; Shotcut
        //               "Swirl"); falls off to 0 at the frame edge.
        //   Threshold : luma binarize level (0..1; pixels with luma >= level go white, else black;
        //               Shotcut "Threshold"). 0 = off.
        section(ui, "Distort");
        ui.add(egui::Slider::new(&mut c.wave, 0.0..=40.0).text("Wave (px)"));
        ui.add(egui::Slider::new(&mut c.swirl, 0.0..=std::f32::consts::PI).text("Swirl (rad)"));
        ui.add(egui::Slider::new(&mut c.threshold, 0.0..=1.0).text("Threshold"));
        if ui.button("Reset Distort").clicked() {
            c.wave = 0.0;
            c.swirl = 0.0;
            c.threshold = 0.0;
        }

        // ---- P17 GEOMETRY. Three per-clip filters applied by the engine on the composited OUTB
        // AFTER the P16 distort filters (threshold) and BEFORE the look, in the engine order
        // LENS -> CROP -> GLITCH. Each is no-op at its default (lens 0, crop 0, glitch 0), so an
        // un-distorted clip renders byte-identically. GLITCH is DETERMINISTIC (an integer band hash,
        // not time/RNG) so a held frame is stable. Binds the pre-added Clip fields lens:f32 /
        // crop:f32 / glitch:f32 (Team B reads/writes them; never edits model.rs).
        //   Lens   : radial barrel/pincushion distortion (+ barrel bulge / - pincushion pinch;
        //            0 = off). Mirrors Shotcut's "Lens Correction".
        //   Crop   : margin fraction cropped to black on all four sides (0..0.49; 0 = off).
        //            Mirrors Shotcut's "Crop: Rectangle" margin.
        //   Glitch : per-band horizontal channel-split shift, max px (0..60; 0 = off). Splits the R
        //            and B channels by a per-band offset for a datamosh/RGB-split look.
        section(ui, "Geometry");
        ui.add(egui::Slider::new(&mut c.lens, -1.0..=1.0).text("Lens (- pinch / + bulge)"));
        ui.add(egui::Slider::new(&mut c.crop, 0.0..=0.49).text("Crop (margin frac)"));
        ui.add(egui::Slider::new(&mut c.glitch, 0.0..=60.0).text("Glitch (px)"));
        if ui.button("Reset Geometry").clicked() {
            c.lens = 0.0;
            c.crop = 0.0;
            c.glitch = 0.0;
        }

        // ---- P23 360 REFRAME. When `eq360` is on the engine treats the composited OUTB as a full
        // 360x180 EQUIRECTANGULAR panorama and reprojects it to a flat RECTILINEAR view at
        // (eq_yaw, eq_pitch) degrees with horizontal field-of-view eq_fov degrees — the standard
        // pinhole "360 viewer" model (the same projection bigsh0t / Shotcut 360 use; NOT bit-exact
        // bigsh0t). When `eq360` is OFF the engine returns immediately (no kernel run), so the frame
        // is byte-identical to pre-P23 and a non-360 clip renders unchanged. Binds the pre-added Clip
        // fields eq360:bool / eq_yaw:f32 / eq_pitch:f32 / eq_fov:f32 (Team B reads/writes them; never
        // edits model.rs). Mutating `c` here is the dirty signal, exactly like the lens/crop/glitch
        // Geometry controls above.
        //   360 equirectangular : enable the equirect->rectilinear reprojection (off = no-op).
        //   Yaw   : view yaw in degrees (-180..180; 0 = forward).
        //   Pitch : view pitch in degrees (-90..90; 0 = level).
        //   FOV   : view horizontal field of view in degrees (30..170; 90 = default).
        section(ui, "360 Reframe");
        ui.checkbox(&mut c.eq360, "360 equirectangular");
        ui.add(egui::Slider::new(&mut c.eq_yaw, -180.0..=180.0).text("Yaw\u{00b0}"));
        ui.add(egui::Slider::new(&mut c.eq_pitch, -90.0..=90.0).text("Pitch\u{00b0}"));
        ui.add(egui::Slider::new(&mut c.eq_fov, 30.0..=170.0).text("FOV\u{00b0}"));
        if ui.button("Reset 360").clicked() {
            c.eq360 = false;
            c.eq_yaw = 0.0;
            c.eq_pitch = 0.0;
            c.eq_fov = 90.0;
        }

        // ---- P34 SHAPE MASK. A per-clip shape mask applied by the engine on the composited OUTB
        // AFTER the P17 geometry (lens/crop/glitch) and the P23 360 reframe, BEFORE the look — the
        // SAME slot the Geometry / 360 Reframe controls above use. The engine zeroes (to black) every
        // pixel OUTSIDE a centred Rectangle (mask_shape 1) or Ellipse (mask_shape 2) at (mask_cx,
        // mask_cy) with half-extents (mask_rw, mask_rh) in normalized [0,1] frame coords, softening
        // the edge over mask_feather and flipping inside/outside when mask_invert is set. When
        // mask_shape is None (0) the engine returns immediately (no kernel run), so the frame is
        // byte-identical to pre-P34 and an un-masked clip renders unchanged. Binds the pre-added Clip
        // fields mask_shape:u8 / mask_cx:f32 / mask_cy:f32 / mask_rw:f32 / mask_rh:f32 /
        // mask_feather:f32 / mask_invert:bool (Team B reads/writes them; never edits model.rs).
        // Mutating `c` here is the dirty signal, exactly like the lens/crop/glitch Geometry controls
        // above. Mirrors Shotcut's "Mask: Simple Shape" (mask_shape) filter.
        //   Shape   : None (0, no-op) / Rectangle (1) / Ellipse (2).
        //   X / Y   : mask centre in normalized [0,1] frame coords (0.5 = centred).
        //   W / H   : mask half-width / half-height in normalized [0,1] (0.5 = full extent).
        //   Feather : edge softness fraction (0..1; 0 = hard edge).
        //   Invert  : keep OUTSIDE / zero INSIDE instead of keep inside.
        section(ui, "Mask");
        ui.horizontal(|ui| {
            // Segmented selector. selectable_value sets c.mask_shape to the variant on click
            // (u8, matching the pre-added model field; mirrors the flip/fx selectors above).
            ui.selectable_value(&mut c.mask_shape, 0u8, "None");
            ui.selectable_value(&mut c.mask_shape, 1u8, "Rectangle");
            ui.selectable_value(&mut c.mask_shape, 2u8, "Ellipse");
        });
        ui.add(egui::Slider::new(&mut c.mask_cx, 0.0..=1.0).text("X"));
        ui.add(egui::Slider::new(&mut c.mask_cy, 0.0..=1.0).text("Y"));
        ui.add(egui::Slider::new(&mut c.mask_rw, 0.0..=1.0).text("W"));
        ui.add(egui::Slider::new(&mut c.mask_rh, 0.0..=1.0).text("H"));
        ui.add(egui::Slider::new(&mut c.mask_feather, 0.0..=1.0).text("Feather"));
        ui.checkbox(&mut c.mask_invert, "Invert");
        if ui.button("Reset Mask").clicked() {
            c.mask_shape = 0;
            c.mask_cx = 0.5;
            c.mask_cy = 0.5;
            c.mask_rw = 0.5;
            c.mask_rh = 0.5;
            c.mask_feather = 0.0;
            c.mask_invert = false;
        }

        // ---- P38 DISTORT 3 (MIRROR / KALEIDOSCOPE / DITHER). Three per-clip filters applied by the
        // engine on the composited OUTB AFTER the P34 shape mask and BEFORE the look — the SAME slot
        // the Geometry / 360 Reframe / Mask controls above use, in the engine order
        // MIRROR -> KALEIDOSCOPE -> DITHER. Each is no-op at its default (mirror off, kaleido < 2,
        // dither 0), so an un-distorted clip renders byte-identically and pre-P38 projects load the
        // defaults via serde unchanged. Binds the pre-added Clip fields mirror_x:u8 / kaleido:i32 /
        // dither:f32 (Team B reads/writes them; never edits model.rs). Mutating `c` here is the dirty
        // signal, exactly like the lens/crop/glitch Geometry controls above. Mirrors Shotcut's
        // mirror / kaleidoscope / dither distort family.
        //   Mirror       : reflect the LEFT half of the frame onto the right (off = no-op).
        //   Kaleidoscope : N-fold radial segment count (0/1 = off; 2..12 = segments). Shotcut
        //                  "Kaleidoscope".
        //   Dither       : ordered 4x4 Bayer dither strength (0 = off .. 1 = full; reduces banding).
        section(ui, "Distort 3");
        // Mirror is a u8 on the model (0=off, 1=on); bind a local bool to a checkbox then map back so
        // the control reads naturally (mirrors the flip/fx selector style above but as a toggle).
        let mut mirror_on = c.mirror_x != 0;
        if ui.checkbox(&mut mirror_on, "Mirror (X)").changed() {
            c.mirror_x = if mirror_on { 1 } else { 0 };
        }
        ui.add(egui::Slider::new(&mut c.kaleido, 0..=12).text("Kaleidoscope segments"));
        ui.add(egui::Slider::new(&mut c.dither, 0.0..=1.0).text("Dither"));
        if ui.button("Reset Distort 3").clicked() {
            c.mirror_x = 0;
            c.kaleido = 0;
            c.dither = 0.0;
        }

        // ---- P39 SELECTIVE COLOR (one HUE BAND). A per-clip filter applied by the engine on the
        // composited OUTB AFTER the P38 distort and BEFORE the look — the SAME OUTB slot the Distort 3
        // controls above use. Selects ONE hue band (Reds / Yellows / ... / Magentas) and rotates its
        // hue and/or scales its saturation; every other hue is untouched. No-op at its default
        // (sel_band 0 = None), so an un-graded clip renders byte-identically and pre-P39 projects load
        // the defaults via serde unchanged. Binds the pre-added Clip fields sel_band:u8 /
        // sel_hshift:f32 / sel_sat:f32 (Team B reads/writes them; never edits model.rs). Mutating `c`
        // here is the dirty signal, exactly like the Distort 3 controls above. Mirrors Shotcut's
        // "Selective Color" (hue-vs-hue / hue-vs-sat single-band grade).
        //   Band       : the hue band to adjust. None = off (no-op). 1=Reds 2=Yellows 3=Greens
        //                4=Cyans 5=Blues 6=Magentas.
        //   Hue shift  : rotate the selected band's hue, -1..1 = -180..180 deg (0 = unchanged).
        //   Saturation : saturation MULTIPLIER for the selected band, 1.0 = unchanged (the no-op
        //                default), 0 = desaturate the band to grey, 2 = double.
        section(ui, "Selective Color");
        // Band is a u8 on the model (0=None..6=Magentas); a segmented selector via selectable_value
        // (mirrors the Look selector below). Each (label, value) sets c.sel_band on click. None = 0 is
        // the byte-identical no-op.
        ui.horizontal(|ui| {
            for (label, band) in [
                ("None", 0u8),
                ("Reds", 1u8),
                ("Yellows", 2u8),
                ("Greens", 3u8),
                ("Cyans", 4u8),
                ("Blues", 5u8),
                ("Magentas", 6u8),
            ] {
                ui.selectable_value(&mut c.sel_band, band, label);
            }
        });
        // Hue shift / Saturation always shown so the controls don't jump as the band changes. Hue
        // shift centres on 0 (unchanged); Saturation centres on 1.0 (a MULTIPLIER — NOT 0, which would
        // desaturate the band). Both are ignored by the engine when sel_band is None (0).
        ui.add(egui::Slider::new(&mut c.sel_hshift, -1.0..=1.0).text("Hue shift"));
        ui.add(egui::Slider::new(&mut c.sel_sat, 0.0..=2.0).text("Saturation"));
        if ui.button("Reset Selective Color").clicked() {
            c.sel_band = 0;
            c.sel_hshift = 0.0;
            // Reset to the NEUTRAL saturation MULTIPLIER 1.0 (NOT 0.0 — 0.0 would desaturate the band).
            c.sel_sat = 1.0;
        }

        // ---- Look: per-clip color look. Clip.look semantics (PINNED): 0=None, 1=VHS,
        // 2=LUT3D (uses clip.lut, a .cube path). Mirrors MojoMedia's per-clip LOOK list
        // (None / VHS / <luts>), collapsed here to a 3-way segmented selector + a LUT picker
        // that only appears for LUT3D. `clip.lut: String` is added by Team B; we read/write it
        // (and clip.look + clip.look_amt) — no other clip field is touched here.
        section(ui, "Look");
        ui.horizontal(|ui| {
            // Segmented selector. selectable_value sets c.look to the variant on click.
            ui.selectable_value(&mut c.look, 0, "None");
            ui.selectable_value(&mut c.look, 1, "VHS");
            ui.selectable_value(&mut c.look, 2, "LUT3D");
        });

        // Mix amount applies to VHS + LUT3D (None ignores it). Always shown so the control
        // doesn't jump as the user switches looks.
        ui.add(egui::Slider::new(&mut c.look_amt, 0.0..=1.0).text("Mix"));

        // LUT picker row — only for LUT3D (look == 2). Switching away leaves clip.lut intact
        // (cheap to keep; re-selecting LUT3D restores the previous .cube without re-picking).
        if c.look == 2 {
            ui.horizontal(|ui| {
                if ui.button("Load .cube").on_hover_text("Pick a 3D LUT (.cube) for this clip").clicked() {
                    if let Some(path) = pick_cube_file() {
                        c.lut = path;
                    }
                }
                ui.weak(lut_basename(&c.lut)).on_hover_text(if c.lut.is_empty() { "no LUT loaded" } else { c.lut.as_str() });
            });
            if !c.lut.is_empty() && ui.button("Clear LUT").clicked() {
                c.lut.clear();
            }
        }

        // ---- Chroma Key (green-screen). Mirrors Shotcut's bluescreen0r ("Chroma Key: Simple"):
        // a key colour + a distance/threshold. Engine semantics (PINNED Team A): when ENABLED on a
        // clip used as the V2 OVERLAY, the engine zeroes/softens the OVER alpha where the pixel's
        // chroma matches the key colour, so V1 shows through the keyed (e.g. green) pixels. Disabled
        // (default) is a no-op (byte-identical to P3). Only meaningful for a V2 overlay clip — the key
        // applies to the overlay layer — but shown for any selected clip. Binds clip.chroma.* only
        // (pre-added by Team C; Team A reads/writes it but never edits model.rs).
        section(ui, "Chroma Key (green-screen)");
        ui.checkbox(&mut c.chroma.enabled, "Enable (keys this clip when used as V2 overlay)");
        // Key colour swatch: edit the [r,g,b] in [0,1] via egui's RGB colour picker.
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Key colour").color(theme::TEXT).size(10.0));
            ui.color_edit_button_rgb(&mut c.chroma.key);
        });
        // Similarity = how much colour-distance is keyed out (Shotcut "Distance"); Smoothness = the
        // soft edge band beyond it. Both 0..1; defaults 0.4 / 0.1.
        ui.add(egui::Slider::new(&mut c.chroma.similarity, 0.0..=1.0).text("Similarity"));
        ui.add(egui::Slider::new(&mut c.chroma.smoothness, 0.0..=1.0).text("Smoothness"));
        // P37 SPILL SUPPRESSION (Shotcut spillsuppress/keyspillm0pup): removes the green/key-colour
        // tint that bleeds onto the kept subject's edges. Runs in the SAME k_chroma kernel on the OVER
        // (V2) buffer, AFTER the alpha key. 0..1; default 0 = off (byte-identical to pre-P37).
        ui.add(egui::Slider::new(&mut c.chroma.spill, 0.0..=1.0).text("Spill"));
        ui.horizontal(|ui| {
            if ui.button("Reset to green").clicked() {
                c.chroma.key = [0.0, 1.0, 0.0];
                c.chroma.similarity = 0.4;
                c.chroma.smoothness = 0.1;
                c.chroma.spill = 0.0;
            }
            if ui.button("Disable chroma").clicked() {
                c.chroma.enabled = false;
            }
        });

        // ---- Title / Text overlay (P5). Mirrors Shotcut's dynamictext ("Text: Simple"): a per-clip
        // text string rasterized by the worker (ab_glyph) into a full-frame transparent RGBA that
        // composites over THIS clip's frame (so V1 footage + a V2 title shows the text over the
        // video). Binds clip.title.* only — Team B never edits model.rs (the `Title` struct + the
        // fields/types are PINNED there; the worker READS them to rasterize). Empty text (the
        // default) is a no-op: the worker sends no RAW: title layer, so an untitled clip renders
        // byte-identically. The clip need not be on V2 — a title on V1 simply overlays its own
        // base — but a title on the upper (overlay) track is the usual lower-third placement.
        section(ui, "Title / Text");
        ui.label(
            egui::RichText::new("Text (multi-line)").color(theme::TEXT).size(10.0),
        );
        ui.add(
            egui::TextEdit::multiline(&mut c.title.text)
                .desired_rows(2)
                .desired_width(f32::INFINITY)
                .hint_text("Title text (empty = no overlay)"),
        );

        // Font height as a fraction of the frame height (Shotcut sizes text by % of frame). 0.02..0.5
        // keeps it from vanishing or swamping the frame. x/y are the normalized TOP-LEFT anchor in
        // [0,1] (0,0 = top-left of the frame), matching the worker's rasterizer placement.
        ui.add(egui::Slider::new(&mut c.title.size_frac, 0.02..=0.5).text("Size (frac of height)"));
        ui.add(egui::Slider::new(&mut c.title.x, 0.0..=1.0).text("X (left)"));
        ui.add(egui::Slider::new(&mut c.title.y, 0.0..=1.0).text("Y (top)"));

        // Text colour — egui's RGB colour button edits the [r,g,b] in [0,1] in place (same control
        // used for the chroma key colour above). The worker rasterizes glyphs in this colour.
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Colour").color(theme::TEXT).size(10.0));
            ui.color_edit_button_rgb(&mut c.title.rgb);
        });

        ui.horizontal(|ui| {
            // Lower-third preset: re-anchors the CURRENT text toward the lower-left at a modest
            // size in white (keeps whatever text is typed; only the layout/colour change).
            if ui.button("Lower third").on_hover_text("Anchor as a lower-third title").clicked() {
                let preset = crate::model::Title::lower_third(&c.title.text);
                c.title.size_frac = preset.size_frac;
                c.title.x = preset.x;
                c.title.y = preset.y;
                c.title.rgb = preset.rgb;
            }
            if ui.button("Clear title").on_hover_text("Remove the text overlay").clicked() {
                c.title = crate::model::Title::default();
            }
        });

        // When there's text, hint that this clip now renders as a title overlay (so the user knows
        // the worker will composite the rasterized text over the clip's frame).
        if !c.title.is_empty() {
            ui.label(
                egui::RichText::new("\u{2713} renders as a title overlay over this clip")
                    .color(egui::Color32::from_rgb(120, 200, 140))
                    .size(10.0),
            );
        } else {
            ui.weak("no title (empty text)");
        }
    }

    // ---- Audio LEVEL METERS (Triad-B P3): stereo peak + RMS (dBFS) of the ASSEMBLED program audio
    // around the playhead. Self-contained worker fetch + throttle cache (the `&mut Project` is only
    // reborrowed immutably here — no clip borrow is held). Drawn whether or not a clip is selected,
    // since it reflects the whole program mix at the playhead, not the selected clip.
    meters_ui(ui, project, playhead);

    // ---- AUDIO SPECTRUM scope (Triad-B P26): frequency-spectrum (FFT) display of the ASSEMBLED
    // program audio around the playhead — mirrors Shotcut's Audio Spectrum scope. Same cadence/cache
    // pattern as the level meter; the worker round-trip is READ-ONLY (changes nothing in the
    // render/mix/LEVELS path). Drawn whether or not a clip is selected (reflects the whole program mix).
    spectrum_ui(ui, project, playhead);

    // ---- AUDIO WAVEFORM scope (Triad-B P40): time-domain oscilloscope of the ASSEMBLED program
    // audio around the playhead — mirrors Shotcut's Audio Waveform scope. Same cadence/cache pattern
    // as the spectrum; the worker round-trip is READ-ONLY (changes nothing in the
    // render/mix/LEVELS/SPECTRUM path). Drawn whether or not a clip is selected (reflects the whole
    // program mix).
    waveform_ui(ui, project, playhead);

    // ---- PiP keyframes (only meaningful when a clip is selected) ----
    // Snapshot the clip's current px/py/pw/ph at the CLIP-LOCAL playhead frame. The mutable
    // clip borrow has ended, so we can now take &mut project for add_pip_key / pip_key_count.
    if let Some(t0) = clip_t0 {
        let local = playhead - t0;
        let n_pip = project.pip_key_count(selected);
        ui.horizontal(|ui| {
            if ui.button("Key PiP @ playhead").clicked() {
                project.add_pip_key(selected, local);
            }
            ui.weak(format!("{n_pip} key{}", if n_pip == 1 { "" } else { "s" }));
        });
        ui.label(
            egui::RichText::new(format!("PiP local frame {local}"))
                .color(egui::Color32::from_rgb(150, 150, 160))
                .size(10.0),
        );

        // P30: flush the per-slider "◆" Key clicks queued above (the clip borrow has ended, so
        // `&mut project` is now free). The clip-LOCAL frame is `playhead - t0`, clamped to >= 0
        // exactly like the PiP path; `add_clip_param_key` snapshots the slider value queued at
        // click time into a (clip, par) keyframe using the project's current create-mode interp.
        if !pending_param_keys.is_empty() {
            let lf = local.max(0);
            for (par, v) in pending_param_keys.drain(..) {
                project.add_clip_param_key(selected, par, lf, v);
            }
        }
        // Per-param video keyframe readout (par 4..9): a compact count line so the user can see how
        // many keys each keyframeable filter param holds on the selected clip.
        let counts = [
            ("bri", project.clip_param_key_count(selected, 4)),
            ("con", project.clip_param_key_count(selected, 5)),
            ("sat", project.clip_param_key_count(selected, 6)),
            ("blur", project.clip_param_key_count(selected, 7)),
            ("rot", project.clip_param_key_count(selected, 8)),
            ("scl", project.clip_param_key_count(selected, 9)),
        ];
        if counts.iter().any(|(_, n)| *n > 0) {
            let txt = counts
                .iter()
                .filter(|(_, n)| *n > 0)
                .map(|(name, n)| format!("{name} {n}"))
                .collect::<Vec<_>>()
                .join("  ");
            ui.label(
                egui::RichText::new(format!("clip-param keys: {txt}"))
                    .color(egui::Color32::from_rgb(150, 150, 160))
                    .size(10.0),
            );
        }
    }

    // ---- P35 CLIP EDITING: Replace media + Group / Ungroup (Shotcut clip ops) ----------------
    // All three are pure timeline/model edits (no render-path change): each snapshots history BEFORE
    // mutating, mirroring the file-wide edit-button undo discipline (history.push(project) → mutate).
    // Replace swaps the selected clip's source media keeping its position/length/effects; Group links
    // the current multi-selection (>=2 clips) so they move together; Ungroup unlinks the selected
    // clip's group. A no chosen-media / lone-selection / ungrouped path stays a no-op (no dead undo).
    if project.clips.get(selected).is_some() {
        section(ui, "Clip editing");

        // -- Replace media: a combo over project.media; picking a NEW index re-targets the selected
        //    clip's media in place (t0/len/track/effects preserved). Snapshots history on commit only.
        let cur_media = project.clips[selected].media;
        let cur_label = media_label(project, cur_media);
        let mut pick_media: Option<usize> = None;
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Replace media").color(theme::TEXT).size(11.0));
            egui::ComboBox::from_id_salt("replace_media")
                .selected_text(cur_label)
                .show_ui(ui, |ui| {
                    for m in 0..project.media.len() {
                        let label = media_label(project, m);
                        // selectable_label highlights the current media; a click on a DIFFERENT one
                        // records the pick (applied after the combo, where history.push is free).
                        if ui.selectable_label(m == cur_media, label).clicked() && m != cur_media {
                            pick_media = Some(m);
                        }
                    }
                });
        });
        if let Some(m) = pick_media {
            history.push(project); // pre-edit snapshot (mirrors every other edit button)
            let ok = project.replace_clip(selected, m);
            // replace_clip is in-range here (selected valid, m < media.len()), so this always applies;
            // the guard keeps the discipline explicit (an out-of-range pick would no-op + false).
            let _ = ok;
        }

        // -- Group / Ungroup. Group acts on the multi-selection (needs >= 2 clips); Ungroup clears the
        //    selected clip's group. Both snapshot history before mutating.
        let cur_group = project.clips[selected].group;
        // Valid, deduped multi-selection (filter stale indices defensively, like effective_selection).
        let mut sel: Vec<usize> =
            selection.iter().copied().filter(|&i| i < project.clips.len()).collect();
        sel.sort_unstable();
        sel.dedup();
        ui.horizontal(|ui| {
            let can_group = sel.len() >= 2;
            if ui
                .add_enabled(can_group, egui::Button::new("Group"))
                .on_hover_text("Link the selected clips so they move together")
                .clicked()
            {
                history.push(project);
                project.group_clips(&sel);
            }
            if ui
                .add_enabled(cur_group != 0, egui::Button::new("Ungroup"))
                .on_hover_text("Unlink this clip's group")
                .clicked()
            {
                history.push(project);
                project.ungroup(cur_group);
            }
        });
        // Status readout: the selected clip's group + how many clips share it (0 = ungrouped).
        let members = project.clips.iter().filter(|c| c.group != 0 && c.group == cur_group).count();
        if cur_group != 0 {
            ui.label(
                egui::RichText::new(format!("grouped (#{cur_group}, {members} clips)"))
                    .color(egui::Color32::from_rgb(150, 180, 150))
                    .size(10.0),
            );
        } else {
            ui.weak("ungrouped");
        }
    }

    // ---- Color tab: program-wide grade ----
    section(ui, "Grade");
    ui.add(egui::Slider::new(&mut project.bright, -1.0..=1.0).text("Brightness"));
    ui.add(egui::Slider::new(&mut project.contrast, 0.0..=2.0).text("Contrast"));
    ui.add(egui::Slider::new(&mut project.sat, 0.0..=2.0).text("Saturation"));
    if ui.button("Reset grade").clicked() {
        project.bright = 0.0;
        project.contrast = 1.0;
        project.sat = 1.0;
    }

    // ---- P14 keyframe interpolation TYPE (Discrete / Linear / Smooth) ----
    // The CURRENT create mode: a NEW keyframe (grade or PiP) takes this interp, so the user picks
    // the mode BEFORE keying. Per-keyframe interp lives on Kf/PipKey; this combo just sets the
    // single `project.kf_interp` the add_* path reads (no add_* signature change). Smooth is an
    // honest smoothstep ease, NOT a bit-exact MLT Catmull-Rom spline.
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Keyframe Interp").color(theme::TEXT).size(11.0));
        egui::ComboBox::from_id_salt("kf_interp")
            .selected_text(project.kf_interp.label())
            .show_ui(ui, |ui| {
                // P19/P20: iterate the single-source-of-truth list (Discrete/Linear/Smooth + the 3
                // Catmull variants + the 30 MLT easings), labelled by KfInterp::label().
                for k in KfInterp::ALL {
                    ui.selectable_value(&mut project.kf_interp, k, k.label());
                }
            });
    });

    // Drop a grade keyframe (bright+contrast+sat snapshot) at the playhead, plus a per-track
    // key count so the user can see the animation building up. Empty tracks read "0 keys" and
    // the worker falls back to the static slider values above.
    ui.horizontal(|ui| {
        if ui.button("Key grade @ playhead").clicked() {
            project.add_grade_key(playhead);
        }
        // P27: drop a LINEAR master audio-gain key at the playhead (lane 4 of the grade strip). Uses
        // add_gain_key (NOT add_grade_key) so it keys the gain_kf automation track at its current
        // envelope value. Empty gain_kf → no GAINENV emitted → byte-identical audio (identity).
        if ui.button("Key master gain @ playhead").clicked() {
            project.add_gain_key(playhead);
        }
        ui.weak(format!(
            "B {}  C {}  S {}  G {}",
            project.bright_kf.len(),
            project.contrast_kf.len(),
            project.sat_kf.len(),
            project.gain_kf.len(),
        ));
    });

    // ---- Export / render settings (Triad-B P1; folded in so app.rs need not change) ----
    export_ui(ui, project);

    // ---- per-track Hide / Mute / Lock state (folded in so app.rs need not change) ----
    tracks_ui(ui, project);
}

/// EXPORT SETTINGS block (Triad-B P1): resolution preset/custom, fps, quality (CRF or bitrate), and
/// codec — written into `project.export`, which `worker::render_program` reads onto the OPEN wire
/// line. The OpenCL working canvas stays GVW×GVH; these only drive the ENCODER (the composed frame is
/// swscaled to out_w×out_h). DEFAULTS reproduce today's behavior (1280×856 @ 30/1, mpeg4, 4 Mbit/s).
/// Mirrors Shotcut's encode dock (resolution + fps spinners, average-bitrate vs constant-quality
/// rate control, codec). Folded into the right panel because app.rs is not editable this slice.
fn export_ui(ui: &mut egui::Ui, project: &mut Project) {
    section(ui, "EXPORT");
    let ex = &mut project.export;

    // P5 NAMED PRESETS (Shotcut encode-dock style): one click sets resolution + fps + codec + rate
    // control together. (name, w, h, fps_num, fps_den, rate_mode 0=bitrate/1=crf, rate_value, vcodec).
    // rate_value is bits/s when rate_mode==0, else the CRF. Mirrors Shotcut's YouTube/H.264/H.265
    // defaults; the individual controls below still allow tweaking after a preset is applied.
    const NAMED: [(&str, u32, u32, u32, u32, u8, i64, &str); 4] = [
        ("YouTube 1080p", 1920, 1080, 30, 1, 1, 21, "libx264"),
        ("YouTube 720p", 1280, 720, 30, 1, 1, 23, "libx264"),
        ("H.265 1080p", 1920, 1080, 30, 1, 1, 24, "libx265"),
        ("Default 1280x856", 1280, 856, 30, 1, 0, 4_000_000, "mpeg4"),
    ];
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new("Preset:").size(11.0).color(theme::TEXT));
        for (name, w, h, fn_, fd, rm, rv, codec) in NAMED {
            let active = ex.out_w == w && ex.out_h == h && ex.vcodec == codec && ex.rate_mode == rm;
            if ui.selectable_label(active, name).clicked() {
                ex.out_w = w;
                ex.out_h = h;
                ex.fps_num = fn_;
                ex.fps_den = fd;
                ex.rate_mode = rm;
                ex.rate_value = rv;
                if rm == 1 {
                    ex.crf = rv;
                }
                ex.vcodec = codec.to_string();
            }
        }
    });

    // Resolution presets + custom. Each preset just sets out_w/out_h; "Custom" keeps the current
    // values editable via the spinners below. Common 16:9 + the engine-native 1280×856 default.
    const PRESETS: [(&str, u32, u32); 5] = [
        ("1280x856", 1280, 856),
        ("1920x1080", 1920, 1080),
        ("1280x720", 1280, 720),
        ("854x480", 854, 480),
        ("640x480", 640, 480),
    ];
    ui.horizontal_wrapped(|ui| {
        for (label, w, h) in PRESETS {
            let active = ex.out_w == w && ex.out_h == h;
            if ui.selectable_label(active, label).clicked() {
                ex.out_w = w;
                ex.out_h = h;
            }
        }
    });
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut ex.out_w).speed(2.0).range(16..=7680).prefix("W "));
        ui.add(egui::DragValue::new(&mut ex.out_h).speed(2.0).range(16..=4320).prefix("H "));
    });

    // FPS (numerator over a fixed denominator of 1 for the common integer rates; the den spinner
    // covers fractional NTSC-style rates like 30000/1001).
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut ex.fps_num).speed(1.0).range(1..=240000).prefix("fps "));
        ui.add(egui::DragValue::new(&mut ex.fps_den).speed(1.0).range(1..=1001).prefix("/ "));
    });

    // Rate control: average bitrate (rate_mode 0) vs constant quality / CRF (rate_mode 1). The two
    // values are kept independent (rate_value vs crf) so toggling the mode doesn't clobber the other.
    ui.horizontal(|ui| {
        if ui.selectable_label(ex.rate_mode == 0, "Bitrate").clicked() {
            ex.rate_mode = 0;
            // rate_value holds the bitrate in this mode; seed a sane default if it looks like a CRF.
            if ex.rate_value < 1000 {
                ex.rate_value = 4_000_000;
            }
        }
        if ui.selectable_label(ex.rate_mode == 1, "Quality (CRF)").clicked() {
            ex.rate_mode = 1;
            ex.rate_value = ex.crf;
        }
    });
    if ex.rate_mode == 0 {
        // Bitrate in kbit/s for a friendlier control; stored as bits/s in rate_value.
        let mut kbps = (ex.rate_value / 1000).max(1);
        if ui.add(egui::Slider::new(&mut kbps, 250..=50_000).text("Bitrate (kbit/s)")).changed() {
            ex.rate_value = kbps * 1000;
        }
    } else {
        // CRF: lower = better quality. Keep `crf` and `rate_value` in lockstep so render reads it.
        if ui.add(egui::Slider::new(&mut ex.crf, 0..=51).text("CRF (lower = better)")).changed() {
            ex.rate_value = ex.crf;
        }
        // Ensure rate_value tracks crf even without a drag (e.g. just switched into this mode).
        ex.rate_value = ex.crf;
    }

    // Codec selector. mpeg4 is the engine default (always available); x264/x265 give CRF support.
    ui.horizontal(|ui| {
        for codec in ["mpeg4", "libx264", "libx265"] {
            let active = ex.vcodec == codec;
            if ui.selectable_label(active, codec).clicked() {
                ex.vcodec = codec.to_string();
            }
        }
    });

    // EXPORT DEPTH (Triad-B P25): GOP / encoder preset / audio bitrate — three more knobs that ride the
    // OPEN wire (after total_s) and are applied engine-side. DEFAULTS (gop 0 / preset "" / abitrate 0)
    // reproduce today's render exactly: gop 0 = encoder default keyframe interval, preset "" sets none
    // (mpeg4 ignores presets anyway), abitrate 0 = the existing 128 kbit/s audio. Mirrors Shotcut's
    // encode-dock GOP, x264/x265 preset, and audio-bitrate fields.

    // GOP / keyframe interval in frames. 0 = auto (encoder default gop_size left untouched).
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut ex.gop).speed(1.0).range(0..=600).prefix("GOP "));
        ui.label(egui::RichText::new("0 = auto").size(10.0).color(egui::Color32::from_rgb(150, 150, 160)));
    });

    // Encoder preset (libx264 / libx265 speed-vs-quality). Only meaningful for x264/x265; mpeg4 ignores
    // it engine-side. "none" -> empty string -> emitted as "-" on the wire (no preset set).
    const PRESETS_X: [&str; 10] = [
        "none", "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower",
        "veryslow",
    ];
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new("Preset:").size(11.0).color(theme::TEXT));
        for name in PRESETS_X {
            // "none" maps to an empty preset; any other word selects that preset verbatim.
            let active = if name == "none" { ex.preset.is_empty() } else { ex.preset == name };
            if ui.selectable_label(active, name).clicked() {
                ex.preset = if name == "none" { String::new() } else { name.to_string() };
            }
        }
    });

    // Audio bitrate in bits/s. 0 = Default (engine keeps its hardcoded 128 kbit/s aac).
    const ABITRATES: [(&str, i64); 6] = [
        ("Default", 0),
        ("96k", 96_000),
        ("128k", 128_000),
        ("192k", 192_000),
        ("256k", 256_000),
        ("320k", 320_000),
    ];
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new("Audio:").size(11.0).color(theme::TEXT));
        for (label, bps) in ABITRATES {
            let active = ex.abitrate == bps;
            if ui.selectable_label(active, label).clicked() {
                ex.abitrate = bps;
            }
        }
    });

    // Audio codec (P29 export depth). "Default" -> empty -> emitted as "-" -> the engine's aac
    // (byte-identical to pre-P29). Each codec must be muxable into the chosen container (the output
    // extension picks the muxer): aac/ac3 -> mp4/mkv, libmp3lame -> mp3/mkv, pcm_s16le -> mov/mkv/wav.
    const ACODECS: [(&str, &str); 5] = [
        ("Default", ""),
        ("AAC", "aac"),
        ("MP3", "libmp3lame"),
        ("AC-3", "ac3"),
        ("PCM", "pcm_s16le"),
    ];
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new("A.codec:").size(11.0).color(theme::TEXT));
        for (label, name) in ACODECS {
            let active = ex.acodec == name;
            if ui.selectable_label(active, label).clicked() {
                ex.acodec = name.to_string();
            }
        }
    });
}

/// An icon-or-text toggle button. Tries `icons::icon(ctx, icon_name)` for the glyph and
/// falls back to `text` when the blob/icon is unavailable (mirrors the toolbar's discipline).
/// `on` selects the active visual: an active toggle gets the theme accent tint, inactive a
/// muted neutral. Returns true on click (caller flips the backing bool).
fn toggle_button(ui: &mut egui::Ui, icon_name: &str, text: &str, on: bool, tooltip: &str) -> bool {
    let size = egui::vec2(26.0, 22.0);
    let tint = if on { theme::TEXT } else { egui::Color32::from_rgb(120, 120, 130) };

    let resp = if let Some(tex) = icons::icon(ui.ctx(), icon_name) {
        // (TextureId, Vec2) -> SizedTexture (egui 0.31 load::SizedTexture::from), fed to
        // Image::from_texture; ImageButton::new takes `impl Into<Image>`.
        let image = egui::Image::from_texture((tex, egui::vec2(16.0, 16.0))).tint(tint);
        ui.add_sized(size, egui::ImageButton::new(image).frame(true))
    } else {
        // Text fallback: dim the label when the toggle is "off" so state still reads.
        let label = egui::RichText::new(text).size(11.0).color(tint);
        ui.add_sized(size, egui::Button::new(label))
    };

    resp.on_hover_text(tooltip).clicked()
}

/// TRACKS section: one row per track (V2 / V1 / A1, timeline top-to-bottom) with Hide /
/// Mute / Lock toggles. Rows are displayed in V2,V1,A1 order but each maps to its Clip.track
/// index (0 = V1, 1 = V2, 2 = A1) so toggles write the correct slot of the `[bool; 3]` arrays.
///
/// Hide uses the eye glyphs ("visible" / "hidden") and applies to video tracks (V1/V2). Mute
/// uses the speaker glyphs ("volume" / "muted"). Lock uses the padlock glyphs ("unlocked" /
/// "locked"). Worker.rs honors track_hide (video) + track_mute (audio); lock is advisory.
pub fn tracks_ui(ui: &mut egui::Ui, project: &mut Project) {
    use crate::model::TrackKind;
    section(ui, "TRACKS");

    // Add controls. Tracks are stored bottom -> top; new tracks append (go on top).
    let mut add_kind: Option<TrackKind> = None;
    let mut remove_idx: Option<usize> = None;
    ui.horizontal(|ui| {
        if ui.small_button("+ Video").on_hover_text("Add a video track").clicked() {
            add_kind = Some(TrackKind::Video);
        }
        if ui.small_button("+ Audio").on_hover_text("Add an audio track").clicked() {
            add_kind = Some(TrackKind::Audio);
        }
    });

    // List TOP (highest index) -> BOTTOM so the panel order matches the timeline stacking.
    let n = project.tracks.len();
    for t in (0..n).rev() {
        let is_video = project.tracks[t].kind == TrackKind::Video;
        ui.horizontal(|ui| {
            let label = project.tracks[t].name.clone();
            ui.add_sized(
                egui::vec2(30.0, 22.0),
                egui::Label::new(egui::RichText::new(label).color(theme::TEXT).size(11.0)),
            );

            // Hide (eye) — video tracks only; the worker skips a hidden video track in base/over.
            if is_video {
                let hidden = project.tracks[t].hidden;
                let (name, txt) = if hidden { ("hidden", "H\u{0335}") } else { ("visible", "H") };
                if toggle_button(ui, name, txt, !hidden, "Hide / show this video track") {
                    project.tracks[t].hidden = !hidden;
                }
            } else {
                ui.add_sized(egui::vec2(26.0, 22.0), egui::Label::new(egui::RichText::new("\u{2014}").weak()));
            }

            // Mute (speaker) — every track; the worker drops a muted track's audio.
            let muted = project.tracks[t].muted;
            let (mname, mtxt) = if muted { ("muted", "M\u{0335}") } else { ("volume", "M") };
            if toggle_button(ui, mname, mtxt, !muted, "Mute / unmute this track's audio") {
                project.tracks[t].muted = !muted;
            }

            // Lock (padlock) — the editor blocks edits to a locked track.
            let locked = project.tracks[t].locked;
            let (lname, ltxt) = if locked { ("locked", "L\u{0335}") } else { ("unlocked", "L") };
            if toggle_button(ui, lname, ltxt, locked, "Lock / unlock edits on this track") {
                project.tracks[t].locked = !locked;
            }

            // Remove — only when more than one track remains. Drops the track's clips + reindexes.
            if n > 1 && ui.small_button("\u{2715}").on_hover_text("Remove this track (and its clips)").clicked() {
                remove_idx = Some(t);
            }
        });
    }

    if let Some(k) = add_kind {
        project.add_track(k);
    }
    if let Some(idx) = remove_idx {
        project.remove_track(idx);
    }
}

// ===========================================================================================
//  SCOPES (Slice C) — live histogram / luma waveform / vectorscope of the program frame.
// ===========================================================================================
//
// The scope image is computed on the GPU by the persistent `gcompose` worker
// (`worker::scope(project, playhead, kind) -> Option<Vec<u8>>`, RGBA8 SW×SH) and uploaded to an
// egui texture for display. Because the right-panel UI is a stateless free function, the selected
// scope kind AND the cached texture (so we don't re-run the worker every repaint) live in a
// process-global `OnceLock<Mutex<ScopeCache>>` — exactly the pattern `thumbs.rs` uses for the
// timeline visual cache.
//
// Fetch discipline (so the single serial worker is not hammered every repaint): the worker is
// only re-queried when the inputs that change the image change — the playhead frame or the
// selected kind — or when the user clicks Refresh. A successful fetch stores (kind, frame) +
// the uploaded `TextureHandle`; a failed fetch caches `None` for that (kind, frame) so an
// undecodable/worker-down frame is not retried every repaint (it shows "scope unavailable" until
// the playhead/kind moves or the user clicks Refresh).
//
// PLAYBACK NOTE (skeptic #4 / #7): during wall-clock playback the playhead advances on (almost)
// every repaint. A naive "refetch whenever (kind, frame) changed" therefore hits the worker EVERY
// playing frame — and each `worker::scope` is NOT a cheap round-trip: the worker re-composites the
// program frame (PREVIEW) and then runs the scope kernel (SCOPE) under one mutex hold. Stacked on
// the preview `compose()` (which also re-composites) this puts ~2 full GPU composites per frame on
// the single UI thread, all serialized on the worker mutex, dragging playback FPS well below 30 and
// starving the wall-clock pacer (stutter) whenever the SCOPES panel is mounted.
//
// The clean fix (gate auto-refetch on a `playing` flag) is not available: `playing` is app.rs state
// and this is a PINNED-signature free fn (`scopes_ui(ui, project, playhead)`) — adding a param is a
// contract change. Instead we THROTTLE the auto-refetch by WALL-CLOCK time, entirely inside this
// module: when the playhead is moving fast (playback), we recompute the scope at most ~5 Hz rather
// than every repaint, which keeps the scopes panel live (it visibly tracks playback) while leaving
// the bulk of the per-frame worker budget to the preview composite. A genuine seek/scrub or a kind
// change still updates promptly because the throttle window is short, and the Refresh button always
// forces an immediate recompute (it bypasses the throttle). Stationary playhead → no refetch at all
// (the (kind, frame) key is unchanged), so a paused frame still costs exactly one fetch.
const SCOPE_REFETCH_MIN_INTERVAL: f64 = 0.20; // seconds → ~5 Hz auto-refresh ceiling during playback

/// The scope kinds, in the worker's `kind` order (0=histogram, 1=luma-waveform, 2=vectorscope,
/// 3=RGB parade — Triad-B P1). `as u8` yields the worker wire value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScopeKind {
    Histogram = 0,
    Waveform = 1,
    Vectorscope = 2,
    Parade = 3,
}

impl ScopeKind {
    fn label(self) -> &'static str {
        match self {
            ScopeKind::Histogram => "Hist",
            ScopeKind::Waveform => "Wave",
            ScopeKind::Vectorscope => "Vec",
            ScopeKind::Parade => "Parade",
        }
    }
}

/// Process-global scope state: the selected kind, the (kind, frame) the cached texture was built
/// for, and the cached texture itself. `tex == None` with a matching key means "we tried this
/// (kind, frame) and the worker returned None" — the failure sentinel, so we don't refetch every
/// repaint.
struct ScopeCache {
    kind: ScopeKind,
    /// (kind-as-u8, playhead frame) the cached `tex` was computed for. `None` = nothing fetched.
    key: Option<(u8, i64)>,
    /// The uploaded scope texture, or `None` when the last fetch for `key` failed.
    tex: Option<egui::TextureHandle>,
    /// Wall-clock time of the last *worker* fetch (success or failure). Used to throttle the
    /// auto-refetch during playback (skeptic #4): a stale key only triggers a worker call once the
    /// `SCOPE_REFETCH_MIN_INTERVAL` window has elapsed, capping the playback-time scope recompute
    /// rate. A kind change still bypasses the throttle (see `scopes_ui`). `None` = never fetched.
    last_fetch: Option<Instant>,
}

impl ScopeCache {
    fn new() -> ScopeCache {
        ScopeCache { kind: ScopeKind::Histogram, key: None, tex: None, last_fetch: None }
    }
}

static SCOPE: OnceLock<Mutex<ScopeCache>> = OnceLock::new();

fn scope_slot() -> &'static Mutex<ScopeCache> {
    SCOPE.get_or_init(|| Mutex::new(ScopeCache::new()))
}

/// Upload an SW×SH RGBA8 scope buffer as an egui texture (NEAREST — scopes are crisp synthetic
/// images; we don't want them blurred when scaled to fill the panel). Mirrors
/// `worker::rgba_to_texture` but at scope dims and nearest filtering.
fn scope_to_texture(ctx: &egui::Context, buf: &[u8]) -> egui::TextureHandle {
    let img = egui::ColorImage::from_rgba_unmultiplied([worker::SW, worker::SH], buf);
    ctx.load_texture("scope", img, egui::TextureOptions::NEAREST)
}

/// SCOPES section: a Hist / Wave / Vec selector, a Refresh button, and the 256×256 scope image of
/// the composited program frame at `playhead`. Auto-refreshes when the playhead or selected kind
/// changes; "scope unavailable" when the worker can't produce one (e.g. an empty timeline / a
/// worker flake). State (selected kind + cached texture) lives in the process-global `SCOPE`
/// cache because this is a stateless free fn (Slice C; signature changed from `scopes_ui(ui)`).
pub fn scopes_ui(ui: &mut egui::Ui, project: &Project, playhead: i64) {
    section(ui, "SCOPES");

    // On a poisoned lock just skip scopes this frame (never panic the whole UI), matching the
    // defensive posture of thumbs.rs's `with_cache`.
    let mut guard = match scope_slot().lock() {
        Ok(g) => g,
        Err(_) => {
            ui.weak("scope unavailable");
            return;
        }
    };

    // ---- kind selector (Hist / Wave / Vec) — a small segmented row of selectable labels ----
    ui.horizontal(|ui| {
        for kind in [
            ScopeKind::Histogram,
            ScopeKind::Waveform,
            ScopeKind::Vectorscope,
            ScopeKind::Parade,
        ] {
            let selected = guard.kind == kind;
            if ui.selectable_label(selected, kind.label()).clicked() {
                guard.kind = kind;
            }
        }
    });

    // ---- Refresh button (force a re-fetch even if the key is unchanged) ----
    let force = ui.button("Refresh").on_hover_text("Recompute the scope at the playhead").clicked();

    // Decide whether to (re)fetch: the playhead moved OR the kind changed (both fold into the key,
    // which embeds `kind as u8`), we have no cached entry for the current (kind, frame), or the
    // user clicked Refresh.
    let kind = guard.kind;
    let want_key = (kind as u8, playhead);
    // Split staleness into "the selected kind changed" vs "only the frame changed": a kind change
    // must update promptly (the user clicked a different scope), whereas a frame-only change during
    // playback is throttled (skeptic #4) so we don't issue a worker recomposite every repaint.
    let kind_changed = guard.key.map(|(k, _)| k) != Some(kind as u8);
    let frame_stale = guard.key != Some(want_key);
    // Throttle gate: a frame-only stale key only earns a worker call once SCOPE_REFETCH_MIN_INTERVAL
    // has elapsed since the last fetch. `force` (Refresh) and `kind_changed` bypass the throttle;
    // a first-ever fetch (last_fetch == None) also bypasses it so the scope appears immediately.
    let throttle_ok = match guard.last_fetch {
        None => true,
        Some(t) => t.elapsed().as_secs_f64() >= SCOPE_REFETCH_MIN_INTERVAL,
    };
    if force || kind_changed || (frame_stale && throttle_ok) {
        // Ask the worker for the scope of the program frame at the playhead. None => store the
        // failure sentinel (tex = None) under this key so we don't refetch every repaint.
        let tex = worker::scope(project, playhead, kind as u8)
            .filter(|buf| buf.len() == worker::SW * worker::SH * 4)
            .map(|buf| scope_to_texture(ui.ctx(), &buf));
        guard.tex = tex;
        guard.key = Some(want_key);
        guard.last_fetch = Some(Instant::now());
    }

    // ---- display ----
    match &guard.tex {
        Some(tex) => {
            // Draw the scope square, capped to the panel width so it never overflows the side
            // panel. The source is square (SW == SH), so an exact square fit keeps its aspect.
            // Floor at SCOPE_MIN_SIDE (skeptic #9): on a collapsed/just-laid-out panel
            // `available_width()` can be ~0, which would render a 0×0 (invisible) scope that reads
            // as "no scope" rather than a shrunk one. Clamp up so it always stays visible.
            const SCOPE_MIN_SIDE: f32 = 64.0;
            let side = ui.available_width().clamp(SCOPE_MIN_SIDE, worker::SW as f32);
            let src = egui::load::SizedTexture::new(tex.id(), egui::vec2(side, side));
            ui.add(egui::Image::new(src).fit_to_exact_size(egui::vec2(side, side)));
        }
        None => {
            ui.weak("scope unavailable");
        }
    }
}

// ===========================================================================================
//  AUDIO LEVEL METERS (Triad-B P3) — stereo peak + RMS (dBFS) of the assembled program mix.
// ===========================================================================================
//
// The levels are computed by the persistent `gcompose` worker (`worker::program_levels(project,
// playhead) -> Option<AudioLevels>`): the worker assembles a short window of the program audio from
// the playhead (mixing each clip's filtered + gained range, exactly as render/playback does) and
// reports per-channel peak + RMS in dBFS. Like the SCOPES panel, this is a stateless free fn, so the
// last good reading + a wall-clock throttle live in a process-global `OnceLock<Mutex<MeterCache>>`
// so we don't hammer the single serial worker every repaint.
//
// THROTTLE (same rationale as the scopes): a meter that refetched every repaint would re-run a
// decode+mix round-trip on the worker mutex every frame, starving the preview composite. We refetch
// at most ~10 Hz (a touch faster than the scopes — a meter wants to feel live), keep the last reading
// between fetches, and on a contended worker (program_levels returns None via try_lock) we simply
// keep showing the last reading.
const METER_REFETCH_MIN_INTERVAL: f64 = 0.10; // seconds → ~10 Hz meter refresh ceiling

/// Process-global meter state: the last good reading + when it was fetched (for the throttle).
struct MeterCache {
    last: Option<worker::AudioLevels>,
    last_fetch: Option<Instant>,
}

impl MeterCache {
    fn new() -> MeterCache {
        MeterCache { last: None, last_fetch: None }
    }
}

static METER: OnceLock<Mutex<MeterCache>> = OnceLock::new();

fn meter_slot() -> &'static Mutex<MeterCache> {
    METER.get_or_init(|| Mutex::new(MeterCache::new()))
}

/// AUDIO METERS section: a small stereo peak + RMS meter (dBFS) of the assembled program audio at the
/// playhead. Self-contained (process-global throttle cache); the `project` borrow is immutable. Shows
/// "no audio" until the first successful worker reading.
fn meters_ui(ui: &mut egui::Ui, project: &Project, playhead: i64) {
    section(ui, "AUDIO METERS");

    // Poisoned lock → just skip the meter this frame (never panic the whole UI), matching scopes_ui.
    let mut guard = match meter_slot().lock() {
        Ok(g) => g,
        Err(_) => {
            ui.weak("meter unavailable");
            return;
        }
    };

    // Throttle the worker fetch (skeptic: a meter that refetched every repaint would run a decode+mix
    // on the worker mutex every frame). A first-ever fetch (last_fetch == None) bypasses the throttle
    // so the meter appears promptly; otherwise we only refetch once the interval has elapsed.
    let throttle_ok = match guard.last_fetch {
        None => true,
        Some(t) => t.elapsed().as_secs_f64() >= METER_REFETCH_MIN_INTERVAL,
    };
    if throttle_ok {
        // None (nothing to measure / worker busy via try_lock) KEEPS the last reading rather than
        // blanking the meter — only a successful measurement replaces it. We still stamp last_fetch
        // so a busy worker doesn't get re-polled faster than the throttle.
        if let Some(levels) = worker::program_levels(project, playhead) {
            guard.last = Some(levels);
        }
        guard.last_fetch = Some(Instant::now());
    }

    match guard.last {
        Some(levels) => {
            draw_meter_row(ui, "L", levels.peak_l, levels.rms_l);
            draw_meter_row(ui, "R", levels.peak_r, levels.rms_r);
            ui.label(
                egui::RichText::new(format!(
                    "peak {:>5.1} / {:>5.1} dB   rms {:>5.1} / {:>5.1} dB",
                    levels.peak_l, levels.peak_r, levels.rms_l, levels.rms_r
                ))
                .color(egui::Color32::from_rgb(150, 150, 160))
                .size(9.0),
            );
        }
        None => {
            ui.weak("no audio");
        }
    }
}

/// Draw one channel's meter row: a label, then a horizontal bar whose FILL maps the RMS level (the
/// solid body of the bar, green→yellow→red by loudness) with a thin bright tick at the PEAK position.
/// Both inputs are dBFS; the bar spans `worker::LEVELS_FLOOR_DB`..0 dBFS left→right.
fn draw_meter_row(ui: &mut egui::Ui, label: &str, peak_db: f32, rms_db: f32) {
    // Map a dBFS value to a 0..1 fraction along the FLOOR..0 scale (clamped).
    let frac = |db: f32| -> f32 {
        let floor = worker::LEVELS_FLOOR_DB;
        ((db - floor) / (0.0 - floor)).clamp(0.0, 1.0)
    };
    let rms_f = frac(rms_db);
    let peak_f = frac(peak_db);

    // Loudness color for the RMS body: green up to ~−18 dB, yellow toward −6, red near 0.
    let body_color = if rms_db >= -6.0 {
        egui::Color32::from_rgb(232, 72, 72) // hot
    } else if rms_db >= -18.0 {
        egui::Color32::from_rgb(230, 200, 64) // warm
    } else {
        egui::Color32::from_rgb(72, 200, 110) // safe
    };

    ui.horizontal(|ui| {
        ui.add_sized(
            egui::vec2(14.0, 14.0),
            egui::Label::new(egui::RichText::new(label).color(theme::TEXT).size(10.0)),
        );

        // Allocate the bar rect (fill the remaining width, fixed height).
        let bar_w = (ui.available_width() - 4.0).max(40.0);
        let bar_h = 12.0;
        let (rect, _resp) =
            ui.allocate_exact_size(egui::vec2(bar_w, bar_h), egui::Sense::hover());
        let painter = ui.painter_at(rect);

        // Track (dark background).
        painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(28, 30, 38));

        // RMS body fill from the left.
        if rms_f > 0.0 {
            let mut fill = rect;
            fill.set_width(rect.width() * rms_f);
            painter.rect_filled(fill, 2.0, body_color);
        }

        // Peak tick: a thin bright vertical line at the peak fraction.
        if peak_f > 0.0 {
            let x = rect.left() + rect.width() * peak_f;
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(2.0, egui::Color32::from_rgb(245, 245, 250)),
            );
        }

        // 0 dBFS edge marker (right border tick) so the user can read where full-scale is.
        painter.line_segment(
            [egui::pos2(rect.right() - 0.5, rect.top()), egui::pos2(rect.right() - 0.5, rect.bottom())],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(90, 90, 100)),
        );
    });
}

// ===========================================================================================
//  AUDIO SPECTRUM SCOPE (Triad-B P26) — frequency magnitudes (FFT) of the assembled program mix.
// ===========================================================================================
//
// Mirrors the AUDIO METERS path EXACTLY but with the SPECTRUM query: the persistent `gcompose` worker
// (`worker::program_spectrum(project, playhead) -> Option<Vec<f32>>`) assembles the SAME short window
// of program audio from the playhead (the audible, filtered+gained mix) and returns
// `worker::SPECTRUM_BINS` LINEAR frequency-bin magnitudes over [0, sr/2] (sr = 48000 → 93.75 Hz/bar
// at 256 bins). READ-ONLY analysis — it changes nothing in the render/mix/LEVELS pipeline.
//
// THROTTLE / CACHE: same rationale and cadence as the level meter — a stateless free fn with a
// process-global last-reading + wall-clock throttle so we don't re-run a decode+mix on the single
// serial worker every repaint. A contended worker (None via try_lock) keeps the last reading.
const SPECTRUM_REFETCH_MIN_INTERVAL: f64 = 0.10; // seconds → ~10 Hz spectrum refresh ceiling

/// Process-global spectrum state: the last good bins + when they were fetched (for the throttle).
struct SpectrumCache {
    last: Option<Vec<f32>>,
    last_fetch: Option<Instant>,
}

impl SpectrumCache {
    fn new() -> SpectrumCache {
        SpectrumCache { last: None, last_fetch: None }
    }
}

static SPECTRUM: OnceLock<Mutex<SpectrumCache>> = OnceLock::new();

fn spectrum_slot() -> &'static Mutex<SpectrumCache> {
    SPECTRUM.get_or_init(|| Mutex::new(SpectrumCache::new()))
}

/// AUDIO SPECTRUM section: a frequency-spectrum display (FFT) of the assembled program audio at the
/// playhead — mirrors Shotcut's Audio Spectrum scope. Self-contained (process-global throttle cache);
/// the `project` borrow is immutable. Draws the per-bin magnitudes as a row of vertical bars, low→high
/// frequency left→right, normalized by the max bin. Shows "no audio" until the first worker reading.
fn spectrum_ui(ui: &mut egui::Ui, project: &Project, playhead: i64) {
    section(ui, "AUDIO SPECTRUM");

    // Poisoned lock → just skip the spectrum this frame (never panic the whole UI), matching scopes/meter.
    let mut guard = match spectrum_slot().lock() {
        Ok(g) => g,
        Err(_) => {
            ui.weak("spectrum unavailable");
            return;
        }
    };

    // Throttle the worker fetch (same rationale as the meter). First-ever fetch bypasses the throttle.
    let throttle_ok = match guard.last_fetch {
        None => true,
        Some(t) => t.elapsed().as_secs_f64() >= SPECTRUM_REFETCH_MIN_INTERVAL,
    };
    if throttle_ok {
        // None (nothing to measure / worker busy via try_lock) KEEPS the last reading rather than
        // blanking the scope — only a successful measurement replaces it. We still stamp last_fetch
        // so a busy worker doesn't get re-polled faster than the throttle.
        if let Some(bins) = worker::program_spectrum(project, playhead) {
            guard.last = Some(bins);
        }
        guard.last_fetch = Some(Instant::now());
    }

    match &guard.last {
        Some(bins) if !bins.is_empty() => {
            // Normalize by the max bin (guard max == 0 → flat, no div-by-zero). Bars are drawn
            // low→high frequency left→right; bar height ∝ magnitude / max.
            let max = bins.iter().cloned().fold(0.0_f32, f32::max);
            let inv_max = if max > 0.0 { 1.0 / max } else { 0.0 };

            let w = (ui.available_width() - 4.0).max(64.0);
            let h = 64.0_f32;
            let (rect, _resp) = ui.allocate_exact_size(egui::vec2(w, h), egui::Sense::hover());
            let painter = ui.painter_at(rect);

            // Background track.
            painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(20, 22, 28));

            let n = bins.len().max(1);
            let bar_w = rect.width() / n as f32;
            for (i, &m) in bins.iter().enumerate() {
                let frac = (m * inv_max).clamp(0.0, 1.0); // flat when max == 0
                let bar_h = rect.height() * frac;
                let x0 = rect.left() + i as f32 * bar_w;
                // Leave a hairline gap between bars; floor the width so adjacent bins don't merge to 0.
                let bar = egui::Rect::from_min_max(
                    egui::pos2(x0, rect.bottom() - bar_h),
                    egui::pos2(x0 + (bar_w - 0.5).max(0.5), rect.bottom()),
                );
                // Low freq green → high freq cyan/blue across the row (purely cosmetic).
                let t = i as f32 / n as f32;
                let color = egui::Color32::from_rgb(
                    (72.0 + t * 40.0) as u8,
                    (200.0 - t * 70.0) as u8,
                    (110.0 + t * 130.0) as u8,
                );
                painter.rect_filled(bar, 0.0, color);
            }
            // Label the linear axis: bins span [0, sr/2] with sr = 48000 (PROG_SR), i.e. up to the
            // 24 kHz Nyquist, drawn low→high left→right.
            ui.label(
                egui::RichText::new(format!("{} bins · 0–24 kHz", bins.len()))
                    .color(egui::Color32::from_rgb(150, 150, 160))
                    .size(9.0),
            );
        }
        _ => {
            ui.weak("no audio");
        }
    }
}

// ---- AUDIO WAVEFORM scope (Triad-B P40) ----
//
// A time-domain oscilloscope of the assembled program audio at the playhead — mirrors Shotcut's Audio
// Waveform scope. The worker round-trip (`worker::program_samples(project, playhead) -> Option<Vec<f32>>`)
// assembles the SAME short window of program audio from the playhead (the audible, filtered+gained mix)
// and returns `worker::SAMPLES_N` RAW time-domain amplitudes (~[-1,1], LEFT channel, decimated over the
// window). READ-ONLY analysis — it changes nothing in the render/mix/LEVELS/SPECTRUM pipeline.
//
// THROTTLE / CACHE: same rationale and cadence as the spectrum scope — a stateless free fn with a
// process-global last-reading + wall-clock throttle so we don't re-run a decode+mix on the single
// serial worker every repaint. A contended worker (None via try_lock) keeps the last reading.
const WAVEFORM_REFETCH_MIN_INTERVAL: f64 = 0.10; // seconds → ~10 Hz waveform refresh ceiling

/// Process-global waveform state: the last good samples + when they were fetched (for the throttle).
struct WaveformCache {
    last: Option<Vec<f32>>,
    last_fetch: Option<Instant>,
}

impl WaveformCache {
    fn new() -> WaveformCache {
        WaveformCache { last: None, last_fetch: None }
    }
}

static WAVEFORM: OnceLock<Mutex<WaveformCache>> = OnceLock::new();

fn waveform_slot() -> &'static Mutex<WaveformCache> {
    WAVEFORM.get_or_init(|| Mutex::new(WaveformCache::new()))
}

/// AUDIO WAVEFORM section: a time-domain oscilloscope of the assembled program audio at the playhead —
/// mirrors Shotcut's Audio Waveform scope. Self-contained (process-global throttle cache); the
/// `project` borrow is immutable. Draws the raw samples as a CENTERED waveform line (a polyline across
/// the widget width: y = center − sample·half_height; x = i/(n−1)·width). Shows "no audio" until the
/// first worker reading.
fn waveform_ui(ui: &mut egui::Ui, project: &Project, playhead: i64) {
    section(ui, "AUDIO WAVEFORM");

    // Poisoned lock → just skip the waveform this frame (never panic the whole UI), matching spectrum/meter.
    let mut guard = match waveform_slot().lock() {
        Ok(g) => g,
        Err(_) => {
            ui.weak("waveform unavailable");
            return;
        }
    };

    // Throttle the worker fetch (same rationale as the spectrum). First-ever fetch bypasses the throttle.
    let throttle_ok = match guard.last_fetch {
        None => true,
        Some(t) => t.elapsed().as_secs_f64() >= WAVEFORM_REFETCH_MIN_INTERVAL,
    };
    if throttle_ok {
        // None (nothing to measure / worker busy via try_lock) KEEPS the last reading rather than
        // blanking the scope — only a successful measurement replaces it. We still stamp last_fetch
        // so a busy worker doesn't get re-polled faster than the throttle.
        if let Some(samples) = worker::program_samples(project, playhead) {
            guard.last = Some(samples);
        }
        guard.last_fetch = Some(Instant::now());
    }

    match &guard.last {
        Some(samples) if samples.len() >= 2 => {
            let w = (ui.available_width() - 4.0).max(64.0);
            let h = 64.0_f32;
            let (rect, _resp) = ui.allocate_exact_size(egui::vec2(w, h), egui::Sense::hover());
            let painter = ui.painter_at(rect);

            // Background track.
            painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(20, 22, 28));

            // Zero-amplitude center line (cosmetic reference).
            let center_y = rect.center().y;
            painter.line_segment(
                [egui::pos2(rect.left(), center_y), egui::pos2(rect.right(), center_y)],
                egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 48, 56)),
            );

            // Centered waveform polyline: x = i/(n−1)·width, y = center − sample·half_height. Samples
            // are raw amplitude ~[-1,1]; clamp so an over-unity sample stays inside the widget.
            let n = samples.len();
            let half_h = rect.height() * 0.5;
            let points: Vec<egui::Pos2> = samples
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let x = rect.left() + (i as f32 / (n - 1) as f32) * rect.width();
                    let y = center_y - s.clamp(-1.0, 1.0) * half_h;
                    egui::pos2(x, y)
                })
                .collect();
            painter.add(egui::Shape::line(
                points,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(110, 200, 140)),
            ));

            ui.label(
                egui::RichText::new(format!("{} samples · time domain", samples.len()))
                    .color(egui::Color32::from_rgb(150, 150, 160))
                    .size(9.0),
            );
        }
        _ => {
            ui.weak("no audio");
        }
    }
}
