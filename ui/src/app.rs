//! App shell — eframe::App, toolbar, preview pane, panel/timeline layout, screenshot gate.
//!
//! Owned by the layout team. The 3-column layout here is plain SidePanels for now; the
//! egui_dock docking pass replaces it. Wires together model + worker + timeline + pool + panels.

use crate::model::Project;
use crate::{panels, pool, theme, timeline, worker};
use eframe::egui::{self, Color32};

pub struct Genesis {
    preview: Option<egui::TextureHandle>,
    project: Project,
    selected: usize,
    ppf: f32,
    playhead: i64,
    preview_inited: bool,
    status: String,
    shot_path: Option<String>,
    frames: u64,
}

impl Genesis {
    pub fn new(cc: &eframe::CreationContext<'_>, project: Project) -> Self {
        theme::apply(&cc.egui_ctx);
        let shot_path = std::env::var("GENESIS_SHOT").ok();
        Genesis {
            preview: None,
            project,
            selected: 0,
            ppf: 6.0,
            playhead: 0,
            preview_inited: false,
            status: "compositing…".into(),
            shot_path,
            frames: 0,
        }
    }

    fn ensure_preview(&mut self, ctx: &egui::Context) {
        if self.preview_inited {
            return;
        }
        self.preview_inited = true;
        match worker::request_frame(&self.project, self.playhead) {
            Some(b) => {
                self.preview = Some(worker::rgba_to_texture(ctx, &b));
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
            ui.label(egui::RichText::new(&self.status).color(theme::ACCENT).size(11.0));
        });
    }

    fn preview_pane(&mut self, ui: &mut egui::Ui) {
        ui.painter().rect_filled(ui.max_rect(), egui::CornerRadius::ZERO, Color32::from_rgb(10, 10, 12));
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
        egui::SidePanel::left("pool").default_width(220.0).show(ctx, |ui| pool::pool_ui(ui, &self.project));
        egui::SidePanel::right("props").default_width(260.0).show(ctx, |ui| {
            panels::properties_ui(ui, &mut self.project, self.selected);
            ui.add_space(10.0);
            panels::scopes_ui(ui);
        });
        egui::TopBottomPanel::bottom("timeline")
            .resizable(true)
            .min_height(170.0)
            .default_height(200.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.label(egui::RichText::new("TIMELINE").color(Color32::from_rgb(150, 150, 160)).size(11.0));
                timeline::timeline_ui(ui, &mut self.project, &mut self.selected, self.ppf);
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
