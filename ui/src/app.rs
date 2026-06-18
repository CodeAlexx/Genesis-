//! App shell — eframe::App, toolbar, preview pane, panel/timeline layout, screenshot gate.
//!
//! Owned by the layout team. A Shotcut-style 3-column layout built from plain SidePanels
//! (left = media pool, right = properties + scopes, bottom = timeline, center = preview),
//! each topped by a thin labeled dock-header bar. Wires together model + worker + timeline
//! + pool + panels. The preview re-composites whenever the playhead moves off the last
//! frame we composited (`last_composed`), in addition to the initial frame-2 gate.

use crate::model::Project;
use crate::{panels, pool, theme, timeline, worker};
use eframe::egui::{self, Color32};

pub struct Genesis {
    preview: Option<egui::TextureHandle>,
    project: Project,
    selected: usize,
    ppf: f32,
    playhead: i64,
    /// The playhead value of the frame currently in `preview`. `-1` = nothing composited yet.
    last_composed: i64,
    /// The playhead value at the end of the previous `update()`, used to detect actual
    /// movement so we don't re-enter the (blocking) worker round-trip every frame on a
    /// stationary playhead — including after a failed compose, where `last_composed` is
    /// intentionally left unchanged.
    prev_playhead: i64,
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
            last_composed: -1,
            prev_playhead: 0,
            preview_inited: false,
            status: "compositing\u{2026}".into(),
            shot_path,
            frames: 0,
        }
    }

    /// Composite the current playhead into the preview texture. Marks `last_composed`.
    fn compose(&mut self, ctx: &egui::Context) {
        match worker::request_frame(&self.project, self.playhead) {
            Some(b) => {
                self.preview = Some(worker::rgba_to_texture(ctx, &b));
                self.status = format!("composite (gcompose worker) \u{2022} f{}", self.playhead);
                self.last_composed = self.playhead;
            }
            None => self.status = "worker failed — no preview".into(),
        }
    }

    /// First-frame preview gate (kept on the frame-2 boundary for the screenshot path).
    fn ensure_preview(&mut self, ctx: &egui::Context) {
        if self.preview_inited {
            return;
        }
        self.preview_inited = true;
        self.compose(ctx);
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
                // Force a re-composite of the current frame. Do NOT clear `preview_inited`:
                // the frame-2 init gate only fires once, so clearing it would permanently
                // disable the line-135 re-composite path (which is gated on preview_inited).
                // Setting last_composed = -1 (which differs from any valid playhead >= 0)
                // makes the next update() re-composite via that path.
                self.last_composed = -1;
            }
            ui.separator();
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

/// A thin labeled dock-header bar drawn atop a panel.
fn dock_header(ui: &mut egui::Ui, label: &str) {
    let full = ui.available_rect_before_wrap();
    let h = 20.0;
    let bar = egui::Rect::from_min_size(full.min, egui::Vec2::new(full.width(), h));
    let painter = ui.painter();
    painter.rect_filled(bar, egui::CornerRadius::ZERO, theme::ALT_BASE);
    painter.text(
        bar.min + egui::Vec2::new(8.0, h * 0.5),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(11.0),
        Color32::from_rgb(160, 160, 170),
    );
    ui.allocate_rect(bar, egui::Sense::hover());
    ui.add_space(2.0);
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

        // Re-composite when the playhead has moved off the frame we last composited.
        // (Skip until the initial gate has fired so the screenshot path stays deterministic.)
        //
        // `compose()` is a synchronous, blocking subprocess round-trip (spawn/IO/file-read).
        // It is run on the UI thread, which is acceptable only because the persistent worker
        // is fast. To avoid hammering it every frame, we only attempt a re-composite when:
        //   - the playhead actually moved since the previous frame (scrub/seek), or
        //   - a forced re-composite is pending: `last_composed == -1` (set by Reload), which
        //     re-runs exactly once because a successful compose sets last_composed = playhead.
        // Note: on a failed compose, `last_composed` is left unchanged; if the playhead is
        // also stationary we will NOT retry every frame (no busy subprocess loop).
        let playhead_moved = self.playhead != self.prev_playhead;
        let force_recomposite = self.last_composed == -1;
        if self.preview_inited && self.playhead != self.last_composed && (playhead_moved || force_recomposite) {
            self.compose(ctx);
        }
        self.prev_playhead = self.playhead;

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
                // `process::exit` does NOT run destructors, so WorkerProc::Drop (which kills +
                // reaps the gcompose child) would never run — leaking the worker subprocess on
                // every screenshot/CI run. Tear it down explicitly before exiting.
                worker::shutdown();
                std::process::exit(0);
            }
        }

        egui::TopBottomPanel::top("toolbar").exact_height(40.0).show(ctx, |ui| self.toolbar(ui));

        egui::SidePanel::left("pool").default_width(220.0).show(ctx, |ui| {
            dock_header(ui, "MEDIA");
            pool::pool_ui(ui, &mut self.project);
        });

        egui::SidePanel::right("props").default_width(260.0).show(ctx, |ui| {
            dock_header(ui, "PROPERTIES \u{2022} SCOPES");
            panels::properties_ui(ui, &mut self.project, self.selected);
            ui.add_space(10.0);
            panels::scopes_ui(ui);
        });

        egui::TopBottomPanel::bottom("timeline")
            .resizable(true)
            .min_height(210.0)
            .default_height(250.0)
            .show(ctx, |ui| {
                dock_header(ui, "TIMELINE");
                timeline::timeline_ui(ui, &mut self.project, &mut self.selected, &mut self.playhead, self.ppf);
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
