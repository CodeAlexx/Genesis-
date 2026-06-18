//! Shared Shotcut-dark palette. Stable contract — referenced by app + timeline + panels.

use eframe::egui::Color32;

pub const WINDOW: Color32 = Color32::from_rgb(50, 50, 50); // #323232
pub const BASE: Color32 = Color32::from_rgb(36, 36, 36); // #242424
pub const ALT_BASE: Color32 = Color32::from_rgb(43, 43, 43); // #2b2b2b
pub const ACCENT: Color32 = Color32::from_rgb(48, 140, 198); // #308cc6
pub const CLIP_VIDEO: Color32 = Color32::from_rgb(23, 92, 118); // #175c76
pub const CLIP_AUDIO: Color32 = Color32::from_rgb(143, 188, 143); // darkseagreen
pub const TEXT: Color32 = Color32::from_rgb(240, 240, 240);

/// Apply the dark theme to an egui context.
pub fn apply(ctx: &eframe::egui::Context) {
    let mut v = eframe::egui::Visuals::dark();
    v.panel_fill = WINDOW;
    v.window_fill = WINDOW;
    v.extreme_bg_color = BASE;
    v.selection.bg_fill = ACCENT;
    ctx.set_visuals(v);
}
