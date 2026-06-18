//! Timeline widget — custom egui painting + interaction over the project model.
//!
//! Owned by the timeline/model team. Drives the playhead frame (for the preview).
//! Today: ruler + timecode, playhead line, draggable clips, edge-trim handles, snapping,
//! markers, selection. Shotcut styling (blue video / green audio, white selection border,
//! name chip). To grow: per-clip thumbnails + in-clip waveforms, multi-track heads.

use crate::icons;
use crate::model;
use crate::pool::DragMedia;
use crate::theme;
use crate::thumbs;
use eframe::egui::{self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, Vec2};

const FPS: i64 = 30; // timeline framerate (matches MojoMedia editor + render config)
const LANE_X_OFF: f32 = 34.0; // clip area starts here, past the track-header column
const SNAP_PX: f32 = 6.0; // snap clip moves/trims to clip edges within this pixel distance
const EDGE_PX: f32 = 5.0; // hot zone (px) at each clip edge for trim handles

// ---- fade triangles (slice C) ----
// Translucent near-black wedge drawn over a clip's head/tail to telegraph a fade. Mirrors
// MojoMedia's add_triangle_filled fill Col4(0.04, 0.04, 0.06, 0.75).
const FADE_FILL: Color32 = Color32::from_rgba_premultiplied(8, 8, 12, 191);

// ---- track-head column (slice C) ----
// Small Shotcut-style status icons drawn per lane in the left strip (DISPLAY-ONLY this wave —
// no mute/lock/visible state in the model yet; see summary follow-up note).
const HEAD_ICON: f32 = 16.0; // icon draw size (px)

// ---- per-clip visuals (slice C) ----
// A video clip narrower than this (px) is too small to host a thumbnail strip; it just shows
// its solid body + chip (mirrors MojoMedia hiding the filmstrip on sub-thumbnail-width clips).
const MIN_THUMB_CLIP_W: f32 = 24.0;
// Width (px) a single in/out thumbnail occupies inside a clip. Both thumbs only fit side by
// side once the clip is wide enough for two of them plus a small gutter.
const THUMB_W: f32 = 32.0;
// Waveform color (Shotcut-ish blue-on-green), drawn as a centered mirrored bar field.
const WAVE_COLOR: Color32 = Color32::from_rgb(40, 70, 95);

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

/// Inverse of `row_of`: the track id a visual lane row hosts. Used to map a drop Y (which lane
/// the pointer is over) back to a `Clip.track`. Rows clamp to the 3 lanes (0=V2, 1=V1, 2=A1).
#[inline]
fn track_of_row(row: usize) -> u8 {
    match row {
        0 => 1, // V2 (top video lane)
        1 => 0, // V1
        _ => 2, // A1 (audio)
    }
}

/// Default length (frames) for a clip dropped from the media pool onto a lane. Mirrors
/// MojoMedia's drop-to-lane placement (`seg_len.append(150)` in main_editor.mojo) — a fixed,
/// trimmable default until the source duration is known.
const DROP_CLIP_LEN: i64 = 150;

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

/// Draw in/out-point thumbnails for a VIDEO clip inside `rect`, UNDER the border + name chip.
///
/// The in thumbnail (source frame `src_in`) is blitted flush-left; the out thumbnail (source
/// frame `src_in + len - 1`) flush-right. For a narrow clip that only fits one, just the in
/// thumbnail is drawn. Each thumbnail is letterbox-fit into a `THUMB_W`-wide slot using the
/// 80×45 (16:9) thumbnail aspect, vertically centered, so it never stretches. Fetches go
/// through the lazy global `thumbs` cache (worker-backed); a decode miss simply draws nothing.
///
/// `painter.image(tex, dst, uv, tint)` blits the whole texture (uv = full 0..1) tinted WHITE
/// (no color change) into `dst`. We keep the worker calls to exactly the in + out frame.
fn draw_clip_thumbs(
    ctx: &egui::Context,
    painter: &egui::Painter,
    project: &model::Project,
    clip_i: usize,
    rect: Rect,
) {
    let clip = &project.clips[clip_i];
    let media_idx = clip.media;
    let media_path = match project.media.get(media_idx) {
        Some(p) => p.clone(),
        None => return,
    };

    // Thumbnail aspect (TW:TH). We fit each thumb into a THUMB_W-wide, full-clip-height slot
    // WITHOUT distorting it. The worker already letterboxed the decoded frame to TW×TH (16:9)
    // and we blit with full uv, so the dst rect must itself be 16:9 or the image stretches.
    let slot_h = rect.height();
    let aspect = thumbs::TW as f32 / thumbs::TH as f32; // 80/45 = 16:9
    // Fit by height first (clips are short). If the resulting width would overflow the THUMB_W
    // slot, clamp the WIDTH and derive the height back from it so the 16:9 aspect is preserved
    // (the old `img_h = slot_h; img_w = (img_h*aspect).min(THUMB_W)` produced a too-tall, too-
    // narrow box that squashed the frame horizontally). Vertically center the result in the slot.
    let mut img_h = slot_h;
    let mut img_w = img_h * aspect;
    if img_w > THUMB_W {
        img_w = THUMB_W;
        img_h = (img_w / aspect).min(slot_h);
    }
    let y_off = (slot_h - img_h) * 0.5; // center the (possibly shorter) thumb in the clip body
    let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));

    // In-point thumbnail (source frame at the clip's src_in), flush-left.
    let in_frame = clip.src_in.max(0);
    if let Some(Some(tex)) = thumbs::with_cache(|c| c.thumb(ctx, media_idx, &media_path, in_frame)) {
        let dst = Rect::from_min_size(
            Pos2::new(rect.min.x, rect.min.y + y_off),
            Vec2::new(img_w, img_h),
        );
        painter.image(tex, dst, uv, Color32::WHITE);
    }

    // Out-point thumbnail only when the clip is wide enough for two non-overlapping slots.
    if rect.width() >= img_w * 2.0 + 4.0 {
        let out_frame = (clip.src_in + clip.len - 1).max(in_frame);
        if let Some(Some(tex)) =
            thumbs::with_cache(|c| c.thumb(ctx, media_idx, &media_path, out_frame))
        {
            let dst = Rect::from_min_size(
                Pos2::new(rect.max.x - img_w, rect.min.y + y_off),
                Vec2::new(img_w, img_h),
            );
            painter.image(tex, dst, uv, Color32::WHITE);
        }
    }
}

/// Draw the audio envelope for an AUDIO clip as a centered, mirrored vertical-bar field inside
/// `rect`, UNDER the border + name chip. Mirrors MojoMedia's `waveform_strip` (center_y ±
/// amp*half_h). The full-media envelope is fetched once per media via the global cache and
/// stretched across the clip width (MojoMedia parity — the model has no media-length hint to
/// slice a trimmed window precisely). Peaks are down-sampled to `bars` on-screen bars.
fn draw_clip_waveform(
    painter: &egui::Painter,
    project: &model::Project,
    clip_i: usize,
    rect: Rect,
) {
    let clip = &project.clips[clip_i];
    let media_idx = clip.media;
    let media_path = match project.media.get(media_idx) {
        Some(p) => p.clone(),
        None => return,
    };

    // We compute the on-screen peaks into an owned Vec while the cache lock is held, then draw
    // after releasing it (the painter loop must not run under the cache mutex).
    let center_y = rect.center().y;
    let half_h = (rect.height() * 0.5 - 2.0).max(1.0);

    // How many on-screen bars to draw: roughly one per ~2px of clip width, bounded so we never
    // emit thousands of line segments for a wide clip.
    let bars = ((rect.width() / 2.0).round() as usize).clamp(1, 256);

    // Down-sample the per-media envelope into `bars` on-screen peaks, spanning the whole
    // envelope across the clip width (MojoMedia parity — see fn doc). Each on-screen bar takes
    // the max over its slice of envelope buckets so transients are not lost when zoomed out.
    let peaks: Vec<f32> = thumbs::with_cache(|c| {
        let env = c.envelope(media_idx, &media_path, thumbs::ENV_BUCKETS);
        let n = env.len();
        if n == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(bars);
        for b in 0..bars {
            // bucket range [lo, hi) of the envelope mapped to this on-screen bar
            let lo = b * n / bars;
            let hi = (((b + 1) * n) / bars).max(lo + 1).min(n);
            let mut peak = 0.0f32;
            for &v in &env[lo..hi] {
                if v > peak {
                    peak = v;
                }
            }
            out.push(peak.clamp(0.0, 1.0));
        }
        out
    })
    .unwrap_or_default();

    if peaks.is_empty() {
        return;
    }

    let bar_w = rect.width() / peaks.len() as f32;
    for (b, &amp) in peaks.iter().enumerate() {
        let x = rect.min.x + b as f32 * bar_w + bar_w * 0.5;
        let hh = amp * half_h;
        if hh <= 0.0 {
            continue;
        }
        painter.line_segment(
            [Pos2::new(x, center_y - hh), Pos2::new(x, center_y + hh)],
            Stroke::new((bar_w * 0.8).clamp(1.0, 2.0), WAVE_COLOR),
        );
    }
}

/// Draw a clip's fade-in / fade-out wedges over `rect`, UNDER the border + chip + thumbnails.
///
/// Mirrors MojoMedia (`add_triangle_filled`): the fade-in is a dark right-triangle anchored at
/// the clip's top-left, its top edge widening to the right over `fade_in * ppf` pixels and its
/// vertical leg running the full clip height at the left edge. The fade-out is its mirror at the
/// clip's top-right. Both use a translucent near-black fill (`FADE_FILL`). Widths are clamped to
/// the clip so a fade longer than the (trimmed) clip never spills past it. egui fills a triangle
/// via `Shape::convex_polygon(points, fill, stroke)` — three points, no border stroke.
fn draw_clip_fades(painter: &egui::Painter, clip: &model::Clip, rect: Rect, ppf: f32) {
    let no_stroke = Stroke::NONE;

    if clip.fade_in > 0 {
        // top edge widens to the right; clamp the wedge to the clip width.
        let fw = (clip.fade_in as f32 * ppf).min(rect.width());
        if fw > 0.5 {
            let pts = vec![
                Pos2::new(rect.min.x, rect.min.y),        // top-left
                Pos2::new(rect.min.x + fw, rect.min.y),   // top, fw to the right
                Pos2::new(rect.min.x, rect.max.y),        // bottom-left (full-height leg)
            ];
            painter.add(Shape::convex_polygon(pts, FADE_FILL, no_stroke));
        }
    }

    if clip.fade_out > 0 {
        let fw = (clip.fade_out as f32 * ppf).min(rect.width());
        if fw > 0.5 {
            let pts = vec![
                Pos2::new(rect.max.x - fw, rect.min.y),   // top, fw to the left of the right edge
                Pos2::new(rect.max.x, rect.min.y),        // top-right
                Pos2::new(rect.max.x, rect.max.y),        // bottom-right (full-height leg)
            ];
            painter.add(Shape::convex_polygon(pts, FADE_FILL, no_stroke));
        }
    }
}

/// Blit one track-head status icon (by PINNED name) at top-left `pos`, sized `HEAD_ICON`.
///
/// Resolves the icon texture via the PINNED `icons::icon(ctx, name)` (Team B produces it). If the
/// blob is missing or the name is unknown the icon is simply skipped (graceful — DISPLAY-ONLY).
/// Tinted with a slight dim so the heads sit quietly under the clips. Returns the x just past the
/// drawn icon (caller advances the cursor) regardless of whether the icon resolved, so the V/A
/// label spacing stays stable even when the blob is absent.
fn draw_head_icon(ctx: &egui::Context, painter: &egui::Painter, name: &str, pos: Pos2) -> f32 {
    if let Some(id) = icons::icon(ctx, name) {
        let dst = Rect::from_min_size(pos, Vec2::splat(HEAD_ICON));
        let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
        painter.image(id, dst, uv, theme::TEXT.gamma_multiply(0.85));
    }
    pos.x + HEAD_ICON
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
    // ---- track-head column (slice C): label + Shotcut status icons per lane -------------
    // Reserve the left strip (full.left() .. left + LANE_X_OFF). Per lane draw the V2/V1/A1
    // label, then a row of small Shotcut icons beneath it: video lanes show visible + locked,
    // the audio lane shows volume + locked. DISPLAY-ONLY this wave — there are no per-track
    // mute/lock/visible fields in the model yet, so these reflect no state and toggle nothing
    // (see summary: per-track state needs model fields). Icons are resolved via the PINNED
    // icons::icon(ctx, name) and skipped gracefully when the blob is unavailable.
    let head_left = full.left() + 4.0; // small inset inside the reserved strip
    // Names per lane row (row 0 = V2, 1 = V1, 2 = A1). Video lanes: visible + locked.
    // Audio lane: volume + locked.
    let head_icons: [(&str, [&str; 2]); 3] = [
        ("V2", ["visible", "locked"]),
        ("V1", ["visible", "locked"]),
        ("A1", ["volume", "locked"]),
    ];
    for (i, (label, names)) in head_icons.iter().enumerate() {
        let y = top + i as f32 * (track_h + gap);
        // label on the first text row
        painter.text(
            Pos2::new(head_left, y + 3.0),
            Align2::LEFT_TOP,
            *label,
            FontId::proportional(11.0),
            if i == 2 { theme::CLIP_AUDIO } else { theme::TEXT },
        );
        // icon row beneath the label, two icons side by side (2px gutter)
        let icon_y = y + 18.0;
        let mut ix = head_left;
        ix = draw_head_icon(ui.ctx(), &painter, names[0], Pos2::new(ix, icon_y));
        ix += 2.0;
        let _ = draw_head_icon(ui.ctx(), &painter, names[1], Pos2::new(ix, icon_y));
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

        // ---- Shotcut clip styling: rounded body + a stronger faux top->bottom gradient ----
        // egui has no gradient fill, so we fake one with three stacked bands (lightest at the
        // top, mid in the upper-middle, base body) — a stronger lift than the old single band.
        // Mirrors MojoMedia's lighter top band but with an extra mid step for more depth.
        let corner = CornerRadius::same(4);
        let fill = if track == 2 { theme::CLIP_AUDIO } else { theme::CLIP_VIDEO };
        painter.rect_filled(rect, corner, fill);
        // upper half: a clear lift (rounded top corners only — bottom of the band is square so
        // it blends into the body below it).
        let half_h = rect.height() * 0.5;
        let top_half = Rect::from_min_size(rect.min, Vec2::new(rect.width(), half_h));
        painter.rect_filled(
            top_half,
            CornerRadius { nw: 4, ne: 4, sw: 0, se: 0 },
            fill.gamma_multiply(1.22),
        );
        // top band: the brightest sliver along the very top edge.
        let band_h = (rect.height() * 0.32).min(11.0);
        let band = Rect::from_min_size(rect.min, Vec2::new(rect.width(), band_h));
        painter.rect_filled(
            band,
            CornerRadius { nw: 4, ne: 4, sw: 0, se: 0 },
            fill.gamma_multiply(1.45),
        );

        // ---- per-clip FADES (slice C): dark wedges over head/tail, UNDER border + chip ----
        // Drawn after the gradient bands but before thumbnails/waveform/border/chip so the fade
        // shading sits on the body without hiding the content that follows.
        draw_clip_fades(&painter, &project.clips[i], rect, ppf);

        // ---- per-clip visuals (slice C): drawn ON the body, UNDER the border + name chip ----
        // VIDEO clips (track 0=V1, 1=V2) get in/out thumbnails; AUDIO clips (track 2) get the
        // envelope waveform. Fetches are memoised + bounded (in/out thumb only, one envelope
        // per media) so the single serial worker is not hammered during a repaint.
        if track == 2 {
            draw_clip_waveform(&painter, project, i, rect);
        } else if w >= MIN_THUMB_CLIP_W {
            draw_clip_thumbs(ui.ctx(), &painter, project, i, rect);
        }

        let border = if i == *selected { Color32::WHITE } else { Color32::BLACK };
        painter.rect_stroke(rect, corner, Stroke::new(1.0, border), StrokeKind::Inside);

        // Trim-handle hover affordance: a subtle, thin highlight bar pinned to the very edge of
        // the clip (2px wide, full clip height) when the corresponding edge hot zone is hovered
        // or being dragged. Drawn ON TOP of the border so it reads as an active trim handle.
        // Left handle hugs the left edge; right handle hugs the right edge.
        let handle_w = 2.0_f32;
        if lresp.hovered() || lresp.dragged() {
            let bar = Rect::from_min_size(rect.min, Vec2::new(handle_w, rect.height()));
            painter.rect_filled(bar, CornerRadius::ZERO, theme::ACCENT);
        }
        if rresp.hovered() || rresp.dragged() {
            let bar = Rect::from_min_size(
                Pos2::new(rect.max.x - handle_w, rect.min.y),
                Vec2::new(handle_w, rect.height()),
            );
            painter.rect_filled(bar, CornerRadius::ZERO, theme::ACCENT);
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

    // ---- drag-to-timeline (slice B): drop a pool media item onto a lane ----
    // A DRAG SOURCE in pool.rs carries `DragMedia(media_idx)`; here the lane area is the DROP
    // TARGET. We register ONE interact response over the clip portion of all three lanes. egui
    // does NOT resolve overlaps by registration order — `hit_test` marks CONTAINS_POINTER on
    // EVERY widget whose interact_rect contains the pointer (hit_test.rs collects all close
    // widgets into hits.contains_pointer regardless of order). The DnD payload methods
    // (`dnd_hover_payload`/`dnd_release_payload`) key off `contains_pointer()`, NOT occlusion
    // order, so this lane-spanning `Sense::hover()` zone always receives the payload even when it
    // overlaps clip widgets. It only senses hover, so it never steals an active clip-move/trim
    // drag (those own the pointer once pressed). On the dragged payload we (a) highlight the
    // hovered lane for feedback while held, and (b) on release, place a new clip at the drop frame
    // (X, snapped) on the drop track (Y).
    //
    // Mirrors MojoMedia's drop-to-lane (main_editor.mojo ~L1384): df = x→frame (snapped to edges),
    // track from Y (top video lane = V2), default len 150, src_in 0, full-frame PiP / no fades.
    let drop_area = Rect::from_min_max(
        Pos2::new(left + LANE_X_OFF, top),
        Pos2::new(left + lane_w, lanes_bottom),
    );
    let drop_resp = ui.interact(drop_area, ui.id().with("tl_drop"), Sense::hover());

    // Which lane row the pointer is over, if any (used for both the hover highlight and the
    // release placement). Rows are stacked `top + row*(track_h+gap)`, each `track_h` tall.
    let row_at = |pos: Pos2| -> Option<usize> {
        if pos.x < left + LANE_X_OFF || pos.x > left + lane_w {
            return None;
        }
        for row in 0..3usize {
            let y = top + row as f32 * (track_h + gap);
            if pos.y >= y && pos.y <= y + track_h {
                return Some(row);
            }
        }
        None
    };

    // Pointer position usable DURING a drag. `Response::hover_pos()` is `None` while *another*
    // widget owns the active drag (egui zeroes `hovered()` then), and the pool drag is exactly
    // that case — so we read the live pointer straight from the context instead. This is the
    // position egui itself uses for `contains_pointer()`/`dnd_release_payload`, so it stays in
    // sync with the gates below.
    let ptr = ui.ctx().pointer_interact_pos();

    // Highlight the hovered lane while a `DragMedia` payload is being dragged over the timeline.
    // `dnd_hover_payload` only returns Some while the pointer is over `drop_resp` and a payload
    // of this type is in flight, so this draws nothing during normal interaction.
    if drop_resp.dnd_hover_payload::<DragMedia>().is_some() {
        if let Some(row) = ptr.and_then(row_at) {
            let y = top + row as f32 * (track_h + gap);
            let lane_rect = Rect::from_min_size(
                Pos2::new(left + LANE_X_OFF, y),
                Vec2::new((left + lane_w) - (left + LANE_X_OFF), track_h),
            );
            // translucent accent wash + a crisp accent border so the target lane reads clearly
            painter.rect_filled(lane_rect, CornerRadius::same(2), theme::ACCENT.gamma_multiply(0.22));
            painter.rect_stroke(
                lane_rect,
                CornerRadius::same(2),
                Stroke::new(1.5, theme::ACCENT),
                StrokeKind::Inside,
            );
            // a thin insertion marker at the drop frame (snapped), so the user sees where it lands
            if let Some(pos) = ptr {
                let raw = x_to_frame(left, pos.x, ppf);
                let edges = snap_edges(project, usize::MAX); // no clip to exclude during a drop
                let snapped = snap_frame(raw, &edges, ppf);
                let ix = frame_to_x(left, snapped as f32, ppf);
                painter.line_segment(
                    [Pos2::new(ix, y), Pos2::new(ix, y + track_h)],
                    Stroke::new(2.0, theme::ACCENT),
                );
            }
        }
    }

    // On release over a lane, add the clip. `dnd_release_payload` returns the `Arc<DragMedia>`
    // only on the frame the pointer is released over `drop_resp` (it uses `contains_pointer`, so
    // it fires even though `hovered()` is false during the foreign drag).
    if let Some(payload) = drop_resp.dnd_release_payload::<DragMedia>() {
        let DragMedia(media_idx) = *payload;
        // Guard against a stale index (media removed between drag start and release).
        if media_idx < project.media.len() {
            if let (Some(pos), Some(row)) = (ptr, ptr.and_then(row_at)) {
                let track = track_of_row(row);
                // Block drops onto a locked track (advisory this wave — Team C added track_lock).
                if !project.is_locked(track) {
                    let raw = x_to_frame(left, pos.x, ppf);
                    let edges = snap_edges(project, usize::MAX);
                    let t0 = snap_frame(raw, &edges, ppf).max(0);
                    // `Clip::video` builds full-frame PiP, no look/fades, src_in 0 — matching the
                    // MojoMedia drop default. `name_hint` is discarded by the model, so pass "".
                    let clip = model::Clip::video(media_idx, t0, DROP_CLIP_LEN, track, "");
                    let new_i = project.clips.len();
                    project.add_clip(clip);
                    *selected = new_i; // select the freshly dropped clip (MojoMedia parity)
                }
            }
        }
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
