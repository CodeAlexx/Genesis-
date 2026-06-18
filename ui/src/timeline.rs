//! Timeline widget — custom egui painting + interaction over the project model.
//!
//! Owned by the timeline/model team. Returns the playhead frame if it changed (for the preview).
//! Today: draggable clips + selection. To grow: trim edges, split, snap, markers, ruler/timecode,
//! per-clip thumbnails + in-clip waveforms, fade/trim handles, multi-track heads.

use crate::model::Project;
use crate::theme;
use eframe::egui::{self, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

/// Draw the timeline. `selected` is the selected clip index. `ppf` = pixels per frame.
pub fn timeline_ui(ui: &mut egui::Ui, project: &mut Project, selected: &mut usize, ppf: f32) {
    let full = ui.available_rect_before_wrap();
    let painter = ui.painter().clone();
    let track_h = 40.0;
    let gap = 4.0;
    let top = full.top() + 8.0;
    let left = full.left() + 8.0;
    let lane_w = full.width() - 16.0;

    let lane_colors = [theme::ALT_BASE, theme::BASE, theme::ALT_BASE];
    for (i, c) in lane_colors.iter().enumerate() {
        let y = top + i as f32 * (track_h + gap);
        painter.rect_filled(Rect::from_min_size(Pos2::new(left, y), Vec2::new(lane_w, track_h)), CornerRadius::ZERO, *c);
    }
    for (i, name) in ["V2", "V1", "A1"].iter().enumerate() {
        let y = top + i as f32 * (track_h + gap);
        painter.text(Pos2::new(full.left() + 12.0, y + 4.0), egui::Align2::LEFT_TOP, *name, FontId::proportional(11.0), theme::TEXT);
    }

    let row_of = |track: u8| -> usize {
        match track {
            1 => 0,
            0 => 1,
            _ => 2,
        }
    };

    for i in 0..project.clips.len() {
        let (start, len, track) = {
            let c = &project.clips[i];
            (c.t0 as f32, c.len as f32, c.track)
        };
        let row = row_of(track);
        let x = left + 34.0 + start * ppf;
        let w = (len * ppf).max(6.0);
        let y = top + row as f32 * (track_h + gap);
        let rect = Rect::from_min_size(Pos2::new(x, y + 1.0), Vec2::new(w, track_h - 2.0));

        let resp = ui.interact(rect, ui.id().with(("clip", i)), Sense::click_and_drag());
        if resp.dragged() {
            let ns = (project.clips[i].t0 as f32 + resp.drag_delta().x / ppf).max(0.0);
            project.clips[i].t0 = ns.round() as i64;
        }
        if resp.clicked() {
            *selected = i;
        }

        let fill = if track == 2 { theme::CLIP_AUDIO } else { theme::CLIP_VIDEO };
        painter.rect_filled(rect, CornerRadius::same(3), fill);
        let band = Rect::from_min_size(rect.min, Vec2::new(rect.width(), (rect.height() * 0.4).min(12.0)));
        painter.rect_filled(band, CornerRadius::same(3), fill.gamma_multiply(1.35));
        let border = if i == *selected { Color32::WHITE } else { Color32::BLACK };
        painter.rect_stroke(rect, CornerRadius::same(3), Stroke::new(1.0, border), StrokeKind::Inside);
        let name = project.names.get(project.clips[i].media).cloned().unwrap_or_default();
        painter.text(rect.min + Vec2::new(4.0, 2.0), egui::Align2::LEFT_TOP, &name, FontId::proportional(10.0), Color32::BLACK);
    }
}
