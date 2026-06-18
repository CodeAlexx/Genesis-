//! Timeline widget — custom egui painting + interaction over the project model.
//!
//! Owned by the timeline/model team. Drives the playhead frame (for the preview).
//! Today: ruler + timecode, playhead line, draggable clips, edge-trim handles, snapping,
//! markers, selection. Shotcut styling (blue video / green audio, white selection border,
//! name chip). To grow: per-clip thumbnails + in-clip waveforms, multi-track heads.

use crate::model;
use crate::theme;
use eframe::egui::{self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

const FPS: i64 = 30; // timeline framerate (matches MojoMedia editor + render config)
const LANE_X_OFF: f32 = 34.0; // clip area starts here, past the track-header column
const SNAP_PX: f32 = 6.0; // snap clip moves/trims to clip edges within this pixel distance
const EDGE_PX: f32 = 5.0; // hot zone (px) at each clip edge for trim handles

/// frame count -> "M:SS:ff" at FPS (mirrors MojoMedia fmt_timecode shape, frame-based).
fn fmt_tc(frame: i64) -> String {
    let f = frame.max(0);
    let total_secs = f / FPS;
    let ff = f % FPS;
    let m = total_secs / 60;
    let ss = total_secs % 60;
    format!("{}:{:02}:{:02}", m, ss, ff)
}

/// Map a timeline frame to its x pixel (clip-area origin = left + LANE_X_OFF).
#[inline]
fn frame_to_x(left: f32, frame: f32, ppf: f32) -> f32 {
    left + LANE_X_OFF + frame * ppf
}

/// Map an x pixel back to a timeline frame (inverse of frame_to_x).
#[inline]
fn x_to_frame(left: f32, x: f32, ppf: f32) -> i64 {
    (((x - left - LANE_X_OFF) / ppf).round() as i64).max(0)
}

/// Visual row for a track (V2 on top, then V1, then A1) — matches the original layout.
#[inline]
fn row_of(track: u8) -> usize {
    match track {
        1 => 0, // V2
        0 => 1, // V1
        _ => 2, // A1
    }
}

/// Collect snap candidates (every clip's left+right edge, plus markers), excluding the
/// clip currently being edited so it does not snap to itself.
fn snap_edges(project: &model::Project, skip: usize) -> Vec<i64> {
    let mut edges = Vec::with_capacity(project.clips.len() * 2 + project.markers.len() + 1);
    edges.push(0);
    for (k, c) in project.clips.iter().enumerate() {
        if k == skip {
            continue;
        }
        edges.push(c.t0);
        edges.push(c.end());
    }
    for &m in &project.markers {
        edges.push(m);
    }
    edges
}

/// Snap `frame` to the nearest edge if within SNAP_PX *pixels*; otherwise return it unchanged.
/// Tolerance is computed and compared in pixel space (frame distance * ppf) — matching
/// MojoMedia's `fpx_snap_frame` pixel threshold — so snapping stays alive when zoomed in.
/// (The old `(SNAP_PX / ppf).round()` frame tolerance collapsed to 0 for ppf >= ~9,
/// silently disabling snapping at high zoom.)
fn snap_frame(frame: i64, edges: &[i64], ppf: f32) -> i64 {
    let mut best = frame;
    let mut best_px = SNAP_PX; // only accept candidates within the pixel threshold
    for &e in edges {
        let d_px = (e - frame).abs() as f32 * ppf;
        if d_px <= SNAP_PX && d_px < best_px {
            best_px = d_px;
            best = e;
        }
    }
    best
}

/// Draw the timeline. `selected` = selected clip index. `playhead` = current frame (mutated
/// by ruler/lane clicks). `ppf` = pixels per frame.
pub fn timeline_ui(
    ui: &mut egui::Ui,
    project: &mut model::Project,
    selected: &mut usize,
    playhead: &mut i64,
    ppf: f32,
) {
    let full = ui.available_rect_before_wrap();
    let painter = ui.painter().clone();

    let ruler_h = 18.0;
    let track_h = 40.0;
    let gap = 4.0;
    let left = full.left() + 8.0;
    let ruler_top = full.top() + 8.0;
    let top = ruler_top + ruler_h + gap; // lanes start below the ruler
    let lane_w = full.width() - 16.0;
    let lanes_bottom = top + 3.0 * (track_h + gap);
    let total = project.total_frames().max(1);

    // ---- ruler: background, tick lines + M:SS:ff labels ----
    let ruler_rect = Rect::from_min_size(Pos2::new(left, ruler_top), Vec2::new(lane_w, ruler_h));
    painter.rect_filled(ruler_rect, CornerRadius::ZERO, theme::BASE);

    // Pick a tick spacing (in frames) that yields a comfortable on-screen gap (~64px),
    // snapped to whole seconds when zoomed out enough.
    let min_px = 64.0_f32;
    let mut step = (min_px / ppf).ceil().max(1.0) as i64;
    if step >= FPS {
        step = ((step + FPS - 1) / FPS) * FPS; // round up to whole seconds
    } else {
        // small steps: prefer 1, 5, 10, 15 frames
        step = if step <= 1 { 1 } else if step <= 5 { 5 } else if step <= 10 { 10 } else { 15 };
    }
    let ruler_x0 = frame_to_x(left, 0.0, ppf);
    let ruler_x_max = left + lane_w;
    let mut f = 0i64;
    loop {
        let x = ruler_x0 + f as f32 * ppf;
        if x > ruler_x_max || f > total + step {
            break;
        }
        painter.line_segment(
            [Pos2::new(x, ruler_top + ruler_h - 5.0), Pos2::new(x, ruler_top + ruler_h)],
            Stroke::new(1.0, theme::TEXT.gamma_multiply(0.5)),
        );
        painter.text(
            Pos2::new(x + 2.0, ruler_top + 1.0),
            Align2::LEFT_TOP,
            fmt_tc(f),
            FontId::proportional(9.0),
            theme::TEXT.gamma_multiply(0.8),
        );
        f += step;
    }

    // ---- lane backgrounds + track-header labels ----
    let lane_colors = [theme::ALT_BASE, theme::BASE, theme::ALT_BASE];
    for (i, c) in lane_colors.iter().enumerate() {
        let y = top + i as f32 * (track_h + gap);
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(left, y), Vec2::new(lane_w, track_h)),
            CornerRadius::ZERO,
            *c,
        );
    }
    for (i, name) in ["V2", "V1", "A1"].iter().enumerate() {
        let y = top + i as f32 * (track_h + gap);
        painter.text(
            Pos2::new(full.left() + 12.0, y + 4.0),
            Align2::LEFT_TOP,
            *name,
            FontId::proportional(11.0),
            theme::TEXT,
        );
    }

    // ---- click on ruler OR empty lane area sets the playhead ----
    let scrub_rect = Rect::from_min_max(
        Pos2::new(left + LANE_X_OFF, ruler_top),
        Pos2::new(left + lane_w, lanes_bottom),
    );
    let scrub_resp = ui.interact(scrub_rect, ui.id().with("tl_scrub"), Sense::click_and_drag());
    if scrub_resp.clicked() || scrub_resp.dragged() {
        if let Some(pos) = scrub_resp.interact_pointer_pos() {
            let frame = x_to_frame(left, pos.x, ppf).clamp(0, (total - 1).max(0));
            *playhead = frame;
        }
    }

    // ---- clips: draw body, name chip, selection border + handle interaction ----
    // Clip interactions are registered AFTER the scrub rect so they take pointer priority
    // (egui resolves overlapping interactions by the last-registered widget under the cursor).
    for i in 0..project.clips.len() {
        let (start, len, track) = {
            let c = &project.clips[i];
            (c.t0 as f32, c.len as f32, c.track)
        };
        let row = row_of(track);
        let x = frame_to_x(left, start, ppf);
        let w = (len * ppf).max(6.0);
        let y = top + row as f32 * (track_h + gap);
        let rect = Rect::from_min_size(Pos2::new(x, y + 1.0), Vec2::new(w, track_h - 2.0));

        // Edge hot zones (left/right) for trim handles.
        let left_edge = Rect::from_min_size(rect.min, Vec2::new(EDGE_PX, rect.height()));
        let right_edge = Rect::from_min_size(
            Pos2::new(rect.max.x - EDGE_PX, rect.min.y),
            Vec2::new(EDGE_PX, rect.height()),
        );

        // Register the body FIRST, then the edges, so the edge hot zones are the
        // last-registered widgets under the cursor. egui resolves overlapping interactions
        // in favour of the last-registered widget, so this makes the trim handles actually
        // win the pointer on overlap (e.g. narrow clips where the edges cover most of the
        // body). The drag cascade below then prefers trims via `if l … else if r … else body`.
        let body = ui.interact(rect, ui.id().with(("clip", i)), Sense::click_and_drag());
        let lresp = ui.interact(left_edge, ui.id().with(("clip_l", i)), Sense::click_and_drag());
        let rresp = ui.interact(right_edge, ui.id().with(("clip_r", i)), Sense::click_and_drag());

        if lresp.dragged() {
            // left-edge trim: move t0 to the pointer (snapped), holding the right edge fixed
            *selected = i;
            if let Some(pos) = lresp.interact_pointer_pos() {
                let raw = x_to_frame(left, pos.x, ppf);
                let edges = snap_edges(project, i);
                let nt0 = snap_frame(raw, &edges, ppf);
                project.trim_start(i, nt0);
            }
        } else if rresp.dragged() {
            // right-edge trim: new length from pointer x (snapped to nearby edges)
            *selected = i;
            if let Some(pos) = rresp.interact_pointer_pos() {
                let edges = snap_edges(project, i);
                let raw_end = x_to_frame(left, pos.x, ppf);
                let snapped_end = snap_frame(raw_end, &edges, ppf);
                // Guard against the snapped end landing at/left of the clip start (an earlier
                // edge or frame 0 in the snap set): never pass a non-positive length. The
                // real MIN_CLIP floor is enforced in `trim_end`.
                let new_len = (snapped_end - project.clips[i].t0).max(1);
                project.trim_end(i, new_len);
            }
        } else if body.dragged() {
            // move: reposition the clip from an ABSOLUTE cursor mapping anchored at the drag
            // start, snapping the new start to nearby edges. This mirrors MojoMedia
            // (`nt0 = drag_orig + (mx - drag_anchor)`) and avoids the per-frame `.round()`
            // loss the old `t0 + drag_delta()` approach suffered: slow drags (sub-frame
            // motion per frame) rounded to 0 and the clip never moved; fast drags lost
            // fractional frames cumulatively. We stash (origin_x, origin_t0) in egui temp
            // memory at drag start and map the live pointer x to frames each frame.
            *selected = i;
            let anchor_id = body.id.with("move_anchor");
            if body.drag_started() {
                if let Some(pos) = body.interact_pointer_pos() {
                    let origin_t0 = project.clips[i].t0;
                    ui.data_mut(|d| d.insert_temp(anchor_id, (pos.x, origin_t0)));
                }
            }
            if let Some(pos) = body.interact_pointer_pos() {
                let anchor: Option<(f32, i64)> = ui.data(|d| d.get_temp(anchor_id));
                if let Some((origin_x, origin_t0)) = anchor {
                    // Absolute mapping: frames moved = (live_x - origin_x) / ppf.
                    let moved = ((pos.x - origin_x) / ppf).round() as i64;
                    let raw = (origin_t0 + moved).max(0);
                    let edges = snap_edges(project, i);
                    let ns = snap_frame(raw, &edges, ppf).max(0);
                    project.clips[i].t0 = ns;
                }
            }
        }
        if body.clicked() || lresp.clicked() || rresp.clicked() {
            *selected = i;
        }

        // ---- Shotcut clip styling ----
        let fill = if track == 2 { theme::CLIP_AUDIO } else { theme::CLIP_VIDEO };
        painter.rect_filled(rect, CornerRadius::same(3), fill);
        let band = Rect::from_min_size(rect.min, Vec2::new(rect.width(), (rect.height() * 0.4).min(12.0)));
        painter.rect_filled(band, CornerRadius::same(3), fill.gamma_multiply(1.35));
        let border = if i == *selected { Color32::WHITE } else { Color32::BLACK };
        painter.rect_stroke(rect, CornerRadius::same(3), Stroke::new(1.0, border), StrokeKind::Inside);

        // edge handle highlight on hover/drag (subtle accent bars)
        for (er, hot) in [(left_edge, lresp.hovered() || lresp.dragged()), (right_edge, rresp.hovered() || rresp.dragged())] {
            if hot {
                painter.rect_filled(er, CornerRadius::ZERO, theme::ACCENT.gamma_multiply(0.8));
            }
        }

        let name = project.names.get(project.clips[i].media).cloned().unwrap_or_default();
        painter.text(
            rect.min + Vec2::new(4.0, 2.0),
            Align2::LEFT_TOP,
            &name,
            FontId::proportional(10.0),
            Color32::BLACK,
        );
    }

    // ---- markers: small ticks spanning the ruler + lanes ----
    for &m in &project.markers {
        let mx = frame_to_x(left, m as f32, ppf);
        if mx < left + LANE_X_OFF || mx > left + lane_w {
            continue;
        }
        painter.line_segment(
            [Pos2::new(mx, ruler_top), Pos2::new(mx, ruler_top + ruler_h)],
            Stroke::new(1.0, Color32::from_rgb(220, 200, 90)),
        );
        // a little downward triangle tab at the ruler base
        painter.text(
            Pos2::new(mx, ruler_top + ruler_h - 8.0),
            Align2::CENTER_TOP,
            "v",
            FontId::proportional(8.0),
            Color32::from_rgb(220, 200, 90),
        );
    }

    // ---- playhead: vertical line at left + LANE_X_OFF + playhead*ppf ----
    let ph = (*playhead).clamp(0, (total - 1).max(0));
    *playhead = ph;
    let px = frame_to_x(left, ph as f32, ppf);
    painter.line_segment(
        [Pos2::new(px, ruler_top), Pos2::new(px, lanes_bottom)],
        Stroke::new(1.0, theme::ACCENT),
    );
    // playhead head marker (small triangle-ish tab at the top of the ruler)
    let head = Rect::from_min_size(Pos2::new(px - 3.0, ruler_top), Vec2::new(6.0, 5.0));
    painter.rect_filled(head, CornerRadius::same(1), theme::ACCENT);
}
