//! Genesis — Phase 0 vertical slice.
//!
//! Proves the whole Rust/egui ↔ C-engine boundary before porting the rest of MojoMedia:
//!   - eframe window with a Shotcut dark theme,
//!   - a labeled toolbar,
//!   - a real video frame decoded by the vendored C shim (fpx_decode.c) shown as a texture,
//!   - a custom-painted timeline with draggable clips (egui interaction, not hand-rolled).
//!
//! Engine stays C (FFmpeg here; OpenCL compute comes in a later phase). egui owns the chrome.

mod ffi;

use eframe::egui;
use egui::{Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

const PREVIEW_W: usize = 640;
const PREVIEW_H: usize = 360;

// Shotcut dark palette (from the audit / UI_SHOTCUT_MAP).
const WINDOW: Color32 = Color32::from_rgb(50, 50, 50); // #323232
const BASE: Color32 = Color32::from_rgb(36, 36, 36); // #242424
const ALT_BASE: Color32 = Color32::from_rgb(43, 43, 43); // #2b2b2b
const ACCENT: Color32 = Color32::from_rgb(48, 140, 198); // #308cc6
const CLIP_VIDEO: Color32 = Color32::from_rgb(23, 92, 118); // #175c76
const CLIP_AUDIO: Color32 = Color32::from_rgb(143, 188, 143); // darkseagreen
const TEXT: Color32 = Color32::from_rgb(240, 240, 240);

struct Clip {
    name: String,
    start: f32, // timeline start, in frames
    len: f32,   // length in frames
    track: u8,  // 0 = V1, 1 = V2, 2 = A1
}

struct Genesis {
    preview: Option<egui::TextureHandle>,
    media_path: String,
    clips: Vec<Clip>,
    selected: usize,
    px_per_frame: f32,
    shot_path: Option<String>, // GENESIS_SHOT: capture one frame to this PPM then exit (gate)
    frames: u64,
}

/// Write an egui ColorImage as a binary PPM (P6) — no image-crate dependency; convert externally.
fn save_ppm(img: &egui::ColorImage, path: &str) {
    let [w, h] = img.size;
    let mut data = Vec::with_capacity(w * h * 3 + 32);
    data.extend_from_slice(format!("P6\n{} {}\n255\n", w, h).as_bytes());
    for px in &img.pixels {
        data.push(px.r());
        data.push(px.g());
        data.push(px.b());
    }
    let _ = std::fs::write(path, data);
}

impl Genesis {
    fn new(cc: &eframe::CreationContext<'_>, media_path: String) -> Self {
        // Shotcut dark theme.
        let mut v = egui::Visuals::dark();
        v.panel_fill = WINDOW;
        v.window_fill = WINDOW;
        v.extreme_bg_color = BASE;
        v.selection.bg_fill = ACCENT;
        cc.egui_ctx.set_visuals(v);

        let preview = decode_preview(&cc.egui_ctx, &media_path, 60);

        let clips = vec![
            Clip { name: "intro".into(), start: 0.0, len: 120.0, track: 0 },
            Clip { name: "overlay".into(), start: 70.0, len: 90.0, track: 1 },
            Clip { name: "audio".into(), start: 0.0, len: 160.0, track: 2 },
        ];

        let shot_path = std::env::var("GENESIS_SHOT").ok();
        Genesis { preview, media_path, clips, selected: 0, px_per_frame: 6.0, shot_path, frames: 0 }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add_space(6.0);
            ui.label(egui::RichText::new("Genesis").color(Color32::from_rgb(230, 214, 128)).size(15.0));
            ui.separator();
            for label in ["Add", "Open", "Save", "Render"] {
                let _ = ui.button(label);
            }
            ui.separator();
            for label in ["Undo", "Redo"] {
                let _ = ui.button(label);
            }
            ui.separator();
            if ui.button("Reload frame").clicked() {
                self.preview = decode_preview(ui.ctx(), &self.media_path, 60);
            }
        });
    }

    fn preview_pane(&mut self, ui: &mut egui::Ui) {
        let painter = ui.painter();
        painter.rect_filled(ui.max_rect(), CornerRadius::ZERO, Color32::from_rgb(10, 10, 12));
        if let Some(tex) = &self.preview {
            let src = egui::load::SizedTexture::new(tex.id(), tex.size_vec2());
            ui.centered_and_justified(|ui| {
                ui.add(egui::Image::new(src).maintain_aspect_ratio(true).max_size(ui.available_size()));
            });
        } else {
            ui.centered_and_justified(|ui| {
                ui.label(format!("no preview (couldn't decode {})", self.media_path));
            });
        }
    }

    fn timeline(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        let painter = ui.painter().clone();
        let track_h = 40.0;
        let gap = 4.0;
        let top = full.top() + 8.0;
        let left = full.left() + 8.0;
        let lane_w = full.width() - 16.0;
        let ppf = self.px_per_frame;

        // three lanes (V2, V1, A1) with alternating Shotcut stripes
        let lane_colors = [ALT_BASE, BASE, ALT_BASE];
        for (i, c) in lane_colors.iter().enumerate() {
            let y = top + i as f32 * (track_h + gap);
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(left, y), Vec2::new(lane_w, track_h)),
                CornerRadius::ZERO,
                *c,
            );
        }
        // lane labels
        for (i, name) in ["V2", "V1", "A1"].iter().enumerate() {
            let y = top + i as f32 * (track_h + gap);
            painter.text(Pos2::new(full.left() - 0.0 + 12.0, y + 4.0), egui::Align2::LEFT_TOP, *name, FontId::proportional(11.0), TEXT);
        }

        // map our track index (0=V1,1=V2,2=A1) to a lane row (V2 row 0, V1 row 1, A1 row 2)
        let row_of = |track: u8| -> usize {
            match track {
                1 => 0, // V2 top
                0 => 1, // V1 middle
                _ => 2, // A1 bottom
            }
        };

        for i in 0..self.clips.len() {
            let (start, len, track, name) = {
                let c = &self.clips[i];
                (c.start, c.len, c.track, c.name.clone())
            };
            let row = row_of(track);
            let x = left + 34.0 + start * ppf;
            let w = (len * ppf).max(6.0);
            let y = top + row as f32 * (track_h + gap);
            let rect = Rect::from_min_size(Pos2::new(x, y + 1.0), Vec2::new(w, track_h - 2.0));

            let resp = ui.interact(rect, ui.id().with(("clip", i)), Sense::click_and_drag());
            if resp.dragged() {
                let dx = resp.drag_delta().x;
                let ns = (self.clips[i].start + dx / ppf).max(0.0);
                self.clips[i].start = ns;
            }
            if resp.clicked() {
                self.selected = i;
            }

            let fill = if track == 2 { CLIP_AUDIO } else { CLIP_VIDEO };
            painter.rect_filled(rect, CornerRadius::same(3), fill);
            // lighter top band (faux gradient)
            let band = Rect::from_min_size(rect.min, Vec2::new(rect.width(), (rect.height() * 0.4).min(12.0)));
            painter.rect_filled(band, CornerRadius::same(3), fill.gamma_multiply(1.35));
            // border: white if selected, else black
            let border = if i == self.selected { Color32::WHITE } else { Color32::BLACK };
            painter.rect_stroke(rect, CornerRadius::same(3), Stroke::new(1.0, border), StrokeKind::Inside);
            // name chip
            painter.text(rect.min + Vec2::new(4.0, 2.0), egui::Align2::LEFT_TOP, &name, FontId::proportional(10.0), Color32::BLACK);
        }
    }
}

impl eframe::App for Genesis {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Deterministic screenshot gate: render a few frames, request a screenshot, save PPM, exit.
        self.frames += 1;
        if let Some(path) = self.shot_path.clone() {
            ctx.request_repaint();
            if self.frames == 4 {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            }
            let shot = ctx.input(|i| {
                i.events.iter().rev().find_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(img) = shot {
                save_ppm(&img, &path);
                std::process::exit(0);
            }
        }

        egui::TopBottomPanel::top("toolbar").exact_height(40.0).show(ctx, |ui| self.toolbar(ui));
        egui::TopBottomPanel::bottom("timeline")
            .resizable(true)
            .min_height(170.0)
            .default_height(200.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.label(egui::RichText::new("TIMELINE").color(Color32::from_rgb(150, 150, 160)).size(11.0));
                self.timeline(ui);
            });
        egui::CentralPanel::default().show(ctx, |ui| self.preview_pane(ui));
    }
}

/// Decode `frame` of `path` (letterboxed to PREVIEW_W×PREVIEW_H RGBA8) into an egui texture.
fn decode_preview(ctx: &egui::Context, path: &str, frame: i32) -> Option<egui::TextureHandle> {
    let mut dec = ffi::Decoder::open(path)?;
    let buf = dec.decode_rgba(frame, PREVIEW_W, PREVIEW_H)?;
    let img = egui::ColorImage::from_rgba_unmultiplied([PREVIEW_W, PREVIEW_H], &buf);
    Some(ctx.load_texture("preview", img, egui::TextureOptions::LINEAR))
}

fn main() -> eframe::Result<()> {
    let media_path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/editor_clip.mp4".to_string());
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("Genesis"),
        ..Default::default()
    };
    eframe::run_native(
        "Genesis",
        opts,
        Box::new(move |cc| Ok(Box::new(Genesis::new(cc, media_path)))),
    )
}
