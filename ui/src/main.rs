//! Genesis UI — egui front-end. Links NO engine/OpenCL; it spawns the `gcompose` worker
//! process to decode/composite frames and reads back the RGBA result as a texture. This keeps
//! the NVIDIA OpenCL driver out of the GL process (see workspace Cargo.toml).
//!
//! Phase 0 slice: Shotcut dark theme, labeled toolbar, the composited frame from gcompose in
//! the preview, and a custom-painted timeline with draggable clips.

use eframe::egui;
use egui::{Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

// Engine working resolution (matches GVW/GVH in the gcompose worker).
const PVW: usize = 1280;
const PVH: usize = 856;
const OVER_PATH: &str = "/tmp/pip_v2.mp4"; // demo V2 overlay for the composite preview
const PREVIEW_RGBA: &str = "/tmp/genesis_preview.rgba";

// Shotcut dark palette.
const WINDOW: Color32 = Color32::from_rgb(50, 50, 50);
const BASE: Color32 = Color32::from_rgb(36, 36, 36);
const ALT_BASE: Color32 = Color32::from_rgb(43, 43, 43);
const ACCENT: Color32 = Color32::from_rgb(48, 140, 198);
const CLIP_VIDEO: Color32 = Color32::from_rgb(23, 92, 118);
const CLIP_AUDIO: Color32 = Color32::from_rgb(143, 188, 143);
const TEXT: Color32 = Color32::from_rgb(240, 240, 240);

struct Clip {
    name: String,
    start: f32,
    len: f32,
    track: u8, // 0 = V1, 1 = V2, 2 = A1
}

struct Genesis {
    preview: Option<egui::TextureHandle>,
    media_path: String,
    clips: Vec<Clip>,
    selected: usize,
    px_per_frame: f32,
    preview_inited: bool,
    status: String,
    shot_path: Option<String>,
    frames: u64,
}

/// Locate the sibling `gcompose` worker binary (next to this executable).
fn worker_path() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.join("gcompose"))
}

/// Run `gcompose <base> <over> <out>` and read back the RGBA8 frame. Retries a few times:
/// the worker has a small residual OpenCL-init flake, but it's isolated here (a failed spawn
/// never touches this GL process), so a couple of retries makes the composite reliable.
fn run_worker(base: &str) -> Option<Vec<u8>> {
    let w = worker_path()?;
    for attempt in 0..4 {
        let ok = std::process::Command::new(&w)
            .arg(base)
            .arg(OVER_PATH)
            .arg(PREVIEW_RGBA)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            if let Ok(bytes) = std::fs::read(PREVIEW_RGBA) {
                if bytes.len() == PVW * PVH * 4 {
                    return Some(bytes);
                }
            }
        }
        eprintln!("gcompose attempt {} failed; retrying", attempt + 1);
    }
    None
}

fn rgba_to_texture(ctx: &egui::Context, buf: &[u8]) -> egui::TextureHandle {
    let img = egui::ColorImage::from_rgba_unmultiplied([PVW, PVH], buf);
    ctx.load_texture("preview", img, egui::TextureOptions::LINEAR)
}

impl Genesis {
    fn new(cc: &eframe::CreationContext<'_>, media_path: String) -> Self {
        let mut v = egui::Visuals::dark();
        v.panel_fill = WINDOW;
        v.window_fill = WINDOW;
        v.extreme_bg_color = BASE;
        v.selection.bg_fill = ACCENT;
        cc.egui_ctx.set_visuals(v);

        let clips = vec![
            Clip { name: "intro".into(), start: 0.0, len: 120.0, track: 0 },
            Clip { name: "overlay".into(), start: 70.0, len: 90.0, track: 1 },
            Clip { name: "audio".into(), start: 0.0, len: 160.0, track: 2 },
        ];

        let shot_path = std::env::var("GENESIS_SHOT").ok();
        Genesis {
            preview: None,
            media_path,
            clips,
            selected: 0,
            px_per_frame: 6.0,
            preview_inited: false,
            status: "compositing…".into(),
            shot_path,
            frames: 0,
        }
    }

    /// On frame 2, spawn the engine worker (separate process) to produce the composite.
    fn ensure_preview(&mut self, ctx: &egui::Context) {
        if self.preview_inited {
            return;
        }
        self.preview_inited = true;
        match run_worker(&self.media_path) {
            Some(bytes) => {
                self.preview = Some(rgba_to_texture(ctx, &bytes));
                self.status = "composite (gcompose worker)".into();
            }
            None => self.status = "worker failed — no preview".into(),
        }
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
            if ui.button("Reload").clicked() {
                self.preview_inited = false;
            }
            ui.label(egui::RichText::new(&self.status).color(ACCENT).size(11.0));
        });
    }

    fn preview_pane(&mut self, ui: &mut egui::Ui) {
        ui.painter().rect_filled(ui.max_rect(), CornerRadius::ZERO, Color32::from_rgb(10, 10, 12));
        if let Some(tex) = &self.preview {
            let src = egui::load::SizedTexture::new(tex.id(), tex.size_vec2());
            ui.centered_and_justified(|ui| {
                ui.add(egui::Image::new(src).maintain_aspect_ratio(true).max_size(ui.available_size()));
            });
        } else {
            let s = self.status.clone();
            ui.centered_and_justified(|ui| {
                ui.label(s);
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

        let lane_colors = [ALT_BASE, BASE, ALT_BASE];
        for (i, c) in lane_colors.iter().enumerate() {
            let y = top + i as f32 * (track_h + gap);
            painter.rect_filled(Rect::from_min_size(Pos2::new(left, y), Vec2::new(lane_w, track_h)), CornerRadius::ZERO, *c);
        }
        for (i, name) in ["V2", "V1", "A1"].iter().enumerate() {
            let y = top + i as f32 * (track_h + gap);
            painter.text(Pos2::new(full.left() + 12.0, y + 4.0), egui::Align2::LEFT_TOP, *name, FontId::proportional(11.0), TEXT);
        }

        let row_of = |track: u8| -> usize {
            match track {
                1 => 0,
                0 => 1,
                _ => 2,
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
                let ns = (self.clips[i].start + resp.drag_delta().x / ppf).max(0.0);
                self.clips[i].start = ns;
            }
            if resp.clicked() {
                self.selected = i;
            }

            let fill = if track == 2 { CLIP_AUDIO } else { CLIP_VIDEO };
            painter.rect_filled(rect, CornerRadius::same(3), fill);
            let band = Rect::from_min_size(rect.min, Vec2::new(rect.width(), (rect.height() * 0.4).min(12.0)));
            painter.rect_filled(band, CornerRadius::same(3), fill.gamma_multiply(1.35));
            let border = if i == self.selected { Color32::WHITE } else { Color32::BLACK };
            painter.rect_stroke(rect, CornerRadius::same(3), Stroke::new(1.0, border), StrokeKind::Inside);
            painter.text(rect.min + Vec2::new(4.0, 2.0), egui::Align2::LEFT_TOP, &name, FontId::proportional(10.0), Color32::BLACK);
        }
    }
}

impl eframe::App for Genesis {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frames += 1;
        if self.frames == 2 {
            self.ensure_preview(ctx);
        }
        if !self.preview_inited {
            ctx.request_repaint();
        }

        if let Some(path) = self.shot_path.clone() {
            ctx.request_repaint();
            if self.frames == 6 {
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

/// Write an egui ColorImage as a binary PPM (P6) — for the screenshot gate.
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

fn main() -> eframe::Result<()> {
    let media_path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/editor_clip.mp4".to_string());
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("Genesis"),
        ..Default::default()
    };
    eframe::run_native("Genesis", opts, Box::new(move |cc| Ok(Box::new(Genesis::new(cc, media_path)))))
}
