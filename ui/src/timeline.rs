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
// Zoom (pixels-per-frame) bounds for zoom-to-fit + ctrl-wheel zoom (wave P1). MUST match the same
// MIN_PPF/MAX_PPF in app.rs (the `=`/`-`/`0` zoom keys) so keyboard + wheel zoom share one range.
const MIN_PPF: f32 = 0.25;
const MAX_PPF: f32 = 40.0;
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

// ---- keyframe strip (slice B) ----
// A thin band under the ruler (above the lanes) that hosts the GRADE keyframe diamonds. Its
// height reserves vertical room between the ruler and the first lane. Each grade keyframe draws
// a small DIAMOND (convex_polygon, 4 pts) at frame_to_x; PiP keyframes draw orange ticks on the
// clip bodies (NOT in this strip). Mirrors MojoMedia's ruler diamonds (main_editor.mojo ~L1081).
const KF_STRIP_H: f32 = 12.0;
// Half-width / half-height of a keyframe diamond (px). 4 -> an 8px-wide / 8px-tall diamond,
// matching MojoMedia's ±4 brightness diamond.
const KF_DIAMOND_R: f32 = 4.0;
// Click hit radius (px) for snapping the playhead to a diamond on the strip (per the slice spec).
const KF_HIT_PX: f32 = 4.0;
// Per-param diamond colors (mirror MojoMedia tcol values, converted from Col4 0..1 to 0..255):
// bright=cyan (0.35,0.85,0.95), contrast=yellow (0.95,0.85,0.35), sat=magenta (0.9,0.45,0.9),
// opacity=green (0.45,0.9,0.5).
const KF_COL_BRIGHT: Color32 = Color32::from_rgb(89, 217, 242);
const KF_COL_CONTRAST: Color32 = Color32::from_rgb(242, 217, 89);
const KF_COL_SAT: Color32 = Color32::from_rgb(230, 115, 230);
const KF_COL_OPACITY: Color32 = Color32::from_rgb(115, 230, 128);
// Per-clip PiP keyframe tick color (mirror MojoMedia orange (0.97,0.6,0.2)).
const KF_COL_PIP: Color32 = Color32::from_rgb(247, 153, 51);

// ---- per-boundary transitions (wave 8, slice C-trans-ui) ----
// A TRANSITION AFFORDANCE is drawn on a video clip lane at each same-track clip boundary
// (project.boundaries(track)). If a transition already exists at that boundary frame
// (project.transition_at(track, boundary).is_some()) we draw a filled MEDIUM-PURPLE bowtie/"X"
// marker; otherwise a faint "+" hint. Clicking an empty boundary adds a 30-frame crossfade;
// clicking an existing marker CYCLES its kind 0..7 (remove + re-add at kind+1 % 8); right-click
// removes it. The hit-test only fires within TRANS_HIT_PX of the boundary x so it never steals
// the lane scrub or a clip-body drag for clicks elsewhere on the lane. Mirrors MojoMedia's
// per-boundary "Bndry>/Trans>" cycling (main_editor.mojo ~L1006), surfaced directly on the
// timeline instead of behind toolbar buttons.
const TRANS_NEW_DUR: i64 = 30; // default crossfade window length (frames) for a freshly-added transition
const TRANS_KIND_N: i32 = 8;   // engine has 8 track1 kernels (0=crossfade .. 7=dissolve)
const TRANS_HIT_PX: f32 = 6.0; // pointer must land within this many px of the boundary x to hit it
const TRANS_R: f32 = 6.0;      // half-extent (px) of the bowtie/"+" marker
// CSS "mediumpurple" (147,112,219) — the active-transition marker fill. Distinct from the
// keyframe diamond palette (cyan/yellow/magenta/green/orange) and the clip bodies (blue/green),
// so an active transition reads clearly against everything else on the lane.
const TRANS_PURPLE: Color32 = Color32::from_rgb(147, 112, 219);
// Faint hint color for an UNused boundary (a barely-there "+" the user can click to add).
// mediumpurple (147,112,219) at ~47% alpha (120/255). `Color32::from_rgba_unmultiplied` is NOT a
// const fn in ecolor 0.31 (it does a runtime sRGB LUT lookup), so it cannot initialize a `const`.
// We instead store the PREMULTIPLIED bytes (which IS const-callable): premultiplied channel =
// round(linear_byte * a / 255) -> r=69, g=53, b=103 for (147,112,219)@120. This renders identically
// to the unmultiplied form since egui blends in premultiplied space.
const TRANS_HINT: Color32 = Color32::from_rgba_premultiplied(69, 53, 103, 120);

/// The 8 transition kind names, indexed by `Transition.kind` (0..7 = fpx_gpu track1 ids:
/// 0=crossfade .. 7=dissolve). Mirrors MojoMedia's `trans_names` (main_editor.mojo L242),
/// minus the index-8 "Cut" (the engine has no kernel 8; the pinned model is 0..7).
const TRANS_NAMES: [&str; 8] = [
    "Crossfade", // 0
    "Wipe L>R",  // 1
    "Wipe R<L",  // 2
    "Wipe Up",   // 3
    "Wipe Down", // 4
    "Slide",     // 5
    "Zoom",      // 6
    "Dissolve",  // 7
];

/// Human name for a transition kind (defensive: an out-of-range kind shows "Transition").
#[inline]
fn trans_name(kind: i32) -> &'static str {
    if (0..TRANS_KIND_N).contains(&kind) {
        TRANS_NAMES[kind as usize]
    } else {
        "Transition"
    }
}

/// Index of the transition on `track` whose WINDOW contains `t`, choosing the nearest-center on
/// overlap (earliest on a tie). This is the index-returning twin of `model::Project::transition_at`
/// and MUST use the same predicate so the edit ops (cycle/remove) act on EXACTLY the record the
/// marker was drawn for. Resolving by an exact `center == t` match instead would diverge after a
/// clip move shifts the boundary frame: the marker would still draw (window still covers the new
/// boundary) but the edit would find no record — making the transition un-removable and letting
/// Cycle spawn a duplicate. `None` if no transition's window contains `t`.
fn trans_idx_containing(project: &model::Project, track: u8, t: i64) -> Option<usize> {
    project
        .transitions
        .iter()
        .enumerate()
        .filter(|(_, tr)| tr.track == track && tr.contains(t))
        .min_by_key(|(_, tr)| (tr.center - t).abs())
        .map(|(idx, _)| idx)
}

// ---- per-clip visuals (slice C) ----
// A video clip narrower than this (px) is too small to host a thumbnail strip; it just shows
// its solid body + chip (mirrors MojoMedia hiding the filmstrip on sub-thumbnail-width clips).
const MIN_THUMB_CLIP_W: f32 = 24.0;
// Width (px) a single in/out thumbnail occupies inside a clip. Both thumbs only fit side by
// side once the clip is wide enough for two of them plus a small gutter.
const THUMB_W: f32 = 32.0;
// Waveform color (Shotcut-ish blue-on-green), drawn as a centered mirrored bar field.
const WAVE_COLOR: Color32 = Color32::from_rgb(40, 70, 95);

// ---- multi-select highlight (P3 editing) ----
// A clip that is in the multi-select `selection` set but is NOT the primary `selected` clip draws
// a distinct ORANGE-ish border (vs the primary's white border, vs an unselected clip's black). So
// the user can tell the panel-target/primary clip apart from the rest of the copy/cut set at a
// glance. Shotcut tints multi-selected clips; we use a 2px accent-orange stroke. Chosen warm so it
// reads against both the blue video and green audio bodies and the white primary border.
const MULTISEL_BORDER: Color32 = Color32::from_rgb(255, 170, 60);

// Half-width (px) of a SHARED-BOUNDARY roll hot-zone, centered on the cut x. A press-drag starting
// within this many px of an abutting same-track cut performs a ROLL (slide the cut); a press
// elsewhere on the body still moves/trims the clip. Kept small so it only claims the exact seam.
const ROLL_HIT_PX: f32 = 5.0;

/// frame count -> "M:SS:ff" at FPS (mirrors MojoMedia fmt_timecode shape, frame-based).
fn fmt_tc(frame: i64) -> String {
    let f = frame.max(0);
    let total_secs = f / FPS;
    let ff = f % FPS;
    let m = total_secs / 60;
    let ss = total_secs % 60;
    format!("{}:{:02}:{:02}", m, ss, ff)
}

/// Map a timeline frame to its x pixel (clip-area origin = left + LANE_X_OFF), shifted left by the
/// horizontal scroll offset `scroll` (in FRAMES). At scroll=0 the project starts flush at
/// LANE_X_OFF; a positive scroll pans the content left so a zoomed-in project can reveal later
/// frames. The SAME `scroll` is applied to every draw AND every hit-test (x_to_frame) so the two
/// stay in lockstep and clip/marker/diamond/transition hit-testing is never thrown off by a pan.
#[inline]
fn frame_to_x(left: f32, frame: f32, ppf: f32, scroll: f32) -> f32 {
    left + LANE_X_OFF + (frame - scroll) * ppf
}

/// Map an x pixel back to a timeline frame (inverse of frame_to_x), accounting for the same
/// horizontal `scroll` (frames) frame_to_x applies.
#[inline]
fn x_to_frame(left: f32, x: f32, ppf: f32, scroll: f32) -> i64 {
    (((x - left - LANE_X_OFF) / ppf + scroll).round() as i64).max(0)
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

/// Apply a clip CLICK to the primary + multi-select state (P3 editing). With a modifier
/// (Shift OR Ctrl held) the click TOGGLES clip `i` in/out of the `selection` set (and makes it the
/// primary when added); a PLAIN click sets `selected = i` and resets `selection` to just `[i]`.
/// Mirrors Shotcut's Shift/Ctrl-click additive selection vs a plain click replacing the selection.
fn apply_clip_click(i: usize, multi: bool, selected: &mut usize, selection: &mut Vec<usize>) {
    if multi {
        if let Some(pos) = selection.iter().position(|&s| s == i) {
            selection.remove(pos); // already selected -> remove from the set
            // Keep the primary pointing at a still-selected clip when possible (cosmetic).
            if *selected == i {
                if let Some(&first) = selection.first() {
                    *selected = first;
                }
            }
        } else {
            selection.push(i); // add to the set + make it the primary
            *selected = i;
        }
    } else {
        // Plain click: primary = i, selection collapses to just this clip.
        *selected = i;
        selection.clear();
        selection.push(i);
    }
}

/// Which lane ROW (0=V2, 1=V1, 2=A1) a vertical pixel `py` falls in, or `None` if it is outside
/// all three lanes. Lanes are stacked `lanes_top + row*(track_h+gap)`, each `track_h` tall. Used by
/// the cross-track clip-move (wave P1) to pick a destination track from the live pointer Y; mirrors
/// the `row_at` closure the pool-drop path uses (kept as a free fn so the clip loop can call it
/// without the closure's `pos.x` lane-bounds check, which the move does not want).
#[inline]
fn row_at_y(py: f32, lanes_top: f32, track_h: f32, gap: f32) -> Option<usize> {
    for row in 0..3usize {
        let y = lanes_top + row as f32 * (track_h + gap);
        if py >= y && py <= y + track_h {
            return Some(row);
        }
    }
    None
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

/// Draw a small filled diamond centered at `(cx, cy)` with half-extent `r` (px). egui fills a
/// convex polygon via `Shape::convex_polygon(points, fill, stroke)`; a diamond is four points
/// (top, right, bottom, left) wound consistently, no border stroke. Mirrors MojoMedia's two
/// stacked `add_triangle_filled` calls that together form the brightness diamond.
fn draw_kf_diamond(painter: &egui::Painter, cx: f32, cy: f32, r: f32, fill: Color32) {
    let pts = vec![
        Pos2::new(cx, cy - r), // top
        Pos2::new(cx + r, cy), // right
        Pos2::new(cx, cy + r), // bottom
        Pos2::new(cx - r, cy), // left
    ];
    painter.add(Shape::convex_polygon(pts, fill, Stroke::NONE));
}

/// Draw the ACTIVE-transition marker: a filled MEDIUM-PURPLE bowtie ("X"/hourglass) centered at
/// `(cx, cy)` with half-extent `r` (px). A bowtie is two opposed triangles sharing the center —
/// it reads as the classic "transition" glyph and stays visually distinct from the keyframe
/// DIAMOND (four points, single convex poly) and the PiP TICK (a thin line). egui fills each
/// triangle via `Shape::convex_polygon`. A thin dark outline is added so the marker reads on top
/// of the (similarly-toned) clip body.
fn draw_trans_marker(painter: &egui::Painter, cx: f32, cy: f32, r: f32, fill: Color32) {
    // left triangle: top-left, bottom-left, center  /  right triangle: top-right, bottom-right, center
    let left_tri = vec![
        Pos2::new(cx - r, cy - r),
        Pos2::new(cx - r, cy + r),
        Pos2::new(cx, cy),
    ];
    let right_tri = vec![
        Pos2::new(cx + r, cy - r),
        Pos2::new(cx + r, cy + r),
        Pos2::new(cx, cy),
    ];
    painter.add(Shape::convex_polygon(left_tri, fill, Stroke::NONE));
    painter.add(Shape::convex_polygon(right_tri, fill, Stroke::NONE));
    // crisp outline of the bowtie bounding box so the glyph separates from the clip body
    let bbox = Rect::from_center_size(Pos2::new(cx, cy), Vec2::splat(r * 2.0));
    painter.rect_stroke(
        bbox,
        CornerRadius::ZERO,
        Stroke::new(1.0, Color32::from_black_alpha(150)),
        StrokeKind::Inside,
    );
}

/// Draw the EMPTY-boundary hint: a faint "+" centered at `(cx, cy)` with half-extent `r` (px),
/// telegraphing that the boundary is clickable to add a transition. Two thin crossed line
/// segments in the faint purple hint color.
fn draw_trans_hint(painter: &egui::Painter, cx: f32, cy: f32, r: f32) {
    let s = Stroke::new(1.5, TRANS_HINT);
    painter.line_segment([Pos2::new(cx - r, cy), Pos2::new(cx + r, cy)], s); // horizontal
    painter.line_segment([Pos2::new(cx, cy - r), Pos2::new(cx, cy + r)], s); // vertical
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

/// Index of the grade keyframe on `track` (0=bright,1=contrast,2=sat,3=opacity) whose frame
/// equals `t`, or `None` if that track has no key at `t`. Used to resolve a grade POSE (the
/// up-to-four coincident diamonds `add_grade_key` writes at the same frame) into the per-track
/// `(track, idx)` pairs that `move_grade_key`/`delete_grade_key` take, so a move/delete acts on
/// the whole pose rather than only the topmost (sat) diamond the pointer hit-tests to.
fn grade_key_idx_at(project: &model::Project, track: u8, t: i64) -> Option<usize> {
    let kfs: &[model::Kf] = match track {
        0 => &project.bright_kf,
        1 => &project.contrast_kf,
        2 => &project.sat_kf,
        3 => &project.opacity_kf,
        _ => return None,
    };
    kfs.iter().position(|k| k.t == t)
}

/// Draw the timeline. `selected` = the PRIMARY (last-clicked) clip index — drives the properties
/// panel + split/lift; stays a plain `usize`. `selection` = the MULTI-SELECT set (P3 editing): the
/// clips Shift/Ctrl-clicked, plus the primary; a distinct highlight is drawn for its members and
/// app.rs's copy/cut/ripple-delete act on it. A PLAIN click sets `selected` and resets `selection`
/// to just that clip; Shift/Ctrl-click toggles a clip in/out of `selection`. `playhead` = current
/// frame (mutated by ruler/lane clicks). `hist` = undo stack (a pre-edit snapshot is pushed BEFORE
/// each mutating timeline gesture so Ctrl+Z reverts trims/moves/slips/rolls/transitions, not only
/// split/delete). `ppf` = pixels per frame (zoom; owned + mutated by app.rs handle_keys).
/// `x_scroll` = horizontal scroll OFFSET in FRAMES (mutated here by mouse-wheel pan and the
/// keep-playhead-visible clamp). `snap` = snapping enabled (Ctrl+P toggle in app.rs); when false,
/// clip moves/trims/drops do not snap to edges/markers.
///
/// P3 DRAG MODES (in addition to the existing move/trim):
///   * ALT + body-drag   = SLIP: re-time the source under the fixed timeline window (model.slip).
///   * shared-boundary drag = ROLL: slide the cut between two abutting same-track clips (model.roll).
///
/// UNDO GESTURE EDGES (push exactly once per gesture, never every frame of a drag):
///   * clip body-move / trim-start / trim-end / slip / roll -> push on `drag_started()`
///   * pool-drop                                -> push on `dnd_release_payload` (the commit frame)
///   * transition add / cycle / remove          -> push on the committing click
///   * marker add                               -> pushed in app.rs (M key) before the push to
///                                                 project.markers; NOT here.
///   * multi-select toggle                      -> NOT an undo gesture (selection is not project state).
pub fn timeline_ui(
    ui: &mut egui::Ui,
    project: &mut model::Project,
    selected: &mut usize,
    selection: &mut Vec<usize>,
    playhead: &mut i64,
    hist: &mut model::History,
    ppf: &mut f32,
    x_scroll: &mut f32,
    snap: bool,
    zoom_fit: bool,
) {
    let full = ui.available_rect_before_wrap();
    let painter = ui.painter().clone();

    let ruler_h = 18.0;
    let track_h = 40.0;
    let gap = 4.0;
    let left = full.left() + 8.0;
    let ruler_top = full.top() + 8.0;
    // Keyframe strip sits directly under the ruler; lanes start below the strip (slice B).
    let strip_top = ruler_top + ruler_h; // flush under the ruler
    let strip_center = strip_top + KF_STRIP_H * 0.5;
    let top = strip_top + KF_STRIP_H + gap; // lanes start below the keyframe strip
    let lane_w = full.width() - 16.0;
    let lanes_bottom = top + 3.0 * (track_h + gap);
    let total = project.total_frames().max(1);
    let clip_area_w = (lane_w - LANE_X_OFF).max(1.0);

    // ---- ZOOM-TO-FIT (wave P1): app.rs sets `zoom_fit` on the `0` key. We own the only place
    // that knows the on-screen clip-area width, so the fit ppf (clip_area_w / total) is computed
    // HERE, clamped to the zoom range, and the scroll reset to 0. Mutates *ppf in place. ----------
    if zoom_fit {
        let fit = (clip_area_w / total as f32).clamp(MIN_PPF, MAX_PPF);
        *ppf = fit;
        *x_scroll = 0.0;
    }

    // ---- horizontal scroll / pan (wave P1) ----------------------------------------------------
    // `*x_scroll` is the scroll offset in FRAMES: the first frame drawn flush at LANE_X_OFF. The
    // clip area is `clip_area_w` px wide, holding `visible_frames` frames at the current zoom; the
    // offset is clamped so we never pan past the project end (or before frame 0). At low zoom where
    // the whole project fits, max_scroll collapses to 0 (no pan needed / possible).
    //
    // Mouse-wheel: Ctrl+wheel ZOOMS about the pointer (optional, Shotcut "scroll zoom"); a plain
    // wheel PANS. Both are hover-gated to the timeline panel so they never fight another panel's
    // scroll. Pan translates raw scroll-delta pixels into frames (delta_px / ppf), picking the
    // dominant axis so a vertical wheel still pans the (horizontal) timeline.
    let tl_panel_rect = Rect::from_min_max(full.min, Pos2::new(full.right(), lanes_bottom + 4.0));
    let pointer_over_tl = ui
        .ctx()
        .pointer_hover_pos()
        .map(|p| tl_panel_rect.contains(p))
        .unwrap_or(false);
    // Did the user PAN via the wheel this frame? If so, skip the keep-playhead-visible clamp below
    // for this frame so a deliberate look-away pan isn't immediately yanked back to the playhead.
    let mut wheel_panned = false;
    if pointer_over_tl {
        let (scroll_delta, ctrl, ptr_x) = ui.ctx().input(|i| {
            let d = i.raw_scroll_delta;
            // pick the dominant axis so a vertical wheel still pans horizontally
            let dom = if d.x.abs() >= d.y.abs() { d.x } else { d.y };
            let px = i.pointer.hover_pos().map(|p| p.x);
            (dom, i.modifiers.command || i.modifiers.ctrl, px)
        });
        if scroll_delta.abs() > 0.0 {
            if ctrl {
                // Ctrl+wheel ZOOM about the pointer: keep the frame under the cursor fixed by
                // adjusting the scroll offset after rescaling ppf. zoom factor scales with the
                // wheel delta (a soft exponential so a notch ~= one zoom step).
                let factor = (1.0 + scroll_delta.signum() * 0.12).max(0.1);
                let old_ppf = *ppf;
                let new_ppf = (old_ppf * factor).clamp(MIN_PPF, MAX_PPF);
                if let Some(px) = ptr_x {
                    // frame currently under the cursor at old zoom
                    let f_under = (px - left - LANE_X_OFF) / old_ppf + *x_scroll;
                    // re-solve scroll so that same frame stays under the cursor at the new zoom
                    *x_scroll = f_under - (px - left - LANE_X_OFF) / new_ppf;
                }
                *ppf = new_ppf;
            } else {
                // wheel-up / swipe-right moves the content; subtract so the gesture feels natural
                *x_scroll -= scroll_delta / *ppf;
                wheel_panned = true;
            }
        }
    }

    // From here on the rest of the widget reads the zoom as a plain `f32` (shadowing the &mut
    // param); all draw/hit-test uses this single value so they agree for the whole frame.
    let ppf: f32 = *ppf;
    let visible_frames = clip_area_w / ppf;
    let max_scroll = (total as f32 - visible_frames).max(0.0);

    // Keep the playhead visible: if the (clamped) playhead would fall outside the visible window,
    // nudge the scroll so it sits just inside the near edge. Runs every frame so a step/scrub/seek
    // (Left/Right/Home/End/markers) that pushes the playhead off-screen pans to follow it. A small
    // margin keeps the head off the very edge. Skipped when the whole project fits (max_scroll==0).
    if max_scroll > 0.0 && !wheel_panned {
        let ph_f = (*playhead).clamp(0, total - 1) as f32;
        let margin = (visible_frames * 0.1).clamp(1.0, 30.0);
        if ph_f < *x_scroll + margin {
            *x_scroll = (ph_f - margin).max(0.0);
        } else if ph_f > *x_scroll + visible_frames - margin {
            *x_scroll = ph_f - visible_frames + margin;
        }
    }
    // Final clamp of the scroll offset into [0, max_scroll].
    *x_scroll = x_scroll.clamp(0.0, max_scroll);
    // Local copy threaded into every frame_to_x / x_to_frame this frame so draw + hit-test agree.
    let scroll = *x_scroll;

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
    let ruler_x_max = left + lane_w;
    // First tick at/after the left edge of the visible window (scroll-aware): round the scroll
    // offset down to the previous multiple of `step` so labels stay aligned to whole steps as we
    // pan. Iterate forward by `step` frames, mapping each through the scroll-aware frame_to_x.
    let first_tick = ((scroll as i64) / step) * step;
    let mut f = first_tick.max(0);
    loop {
        let x = frame_to_x(left, f as f32, ppf, scroll);
        if x > ruler_x_max || f > total + step {
            break;
        }
        // skip ticks that fall left of the clip area (can happen for the first rounded-down tick)
        if x < left + LANE_X_OFF - 0.5 {
            f += step;
            continue;
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
            let frame = x_to_frame(left, pos.x, ppf, scroll).clamp(0, (total - 1).max(0));
            *playhead = frame;
        }
    }

    // ---- keyframe strip (slice B): background + GRADE keyframe diamonds + click-to-seek ----
    // The strip spans the clip area width (left + LANE_X_OFF .. left + lane_w), one band tall,
    // directly under the ruler. We draw a faint background, then a color-coded diamond per grade
    // keyframe at frame_to_x. Click-to-seek: a click within KF_HIT_PX of a diamond snaps the
    // playhead to that key's frame. The strip interact (below) is registered AFTER `scrub_resp`
    // so it is the last-registered widget over the band (egui resolves overlap in favour of the
    // last registration) — a click on the band is handled by the strip handler, which snaps to a
    // near diamond or, failing that, scrubs to the clicked frame itself (so empty-strip clicks
    // still scrub).
    let strip_rect = Rect::from_min_max(
        Pos2::new(left + LANE_X_OFF, strip_top),
        Pos2::new(left + lane_w, strip_top + KF_STRIP_H),
    );
    painter.rect_filled(strip_rect, CornerRadius::ZERO, theme::BASE.gamma_multiply(0.85));

    // All four grade tracks with their PINNED track index (0=bright,1=contrast,2=sat,3=opacity)
    // and color. The track snapshot is taken as (idx, frame) pairs UP FRONT so the interaction
    // loop below can mutate the project (move/delete) without holding a borrow of the tracks —
    // and a move/delete this frame just takes effect next repaint, which is fine for a drag.
    let strip_x_max = left + lane_w;
    let grade_meta: [(u8, Color32); 4] = [
        (0, KF_COL_BRIGHT),
        (1, KF_COL_CONTRAST),
        (2, KF_COL_SAT),
        (3, KF_COL_OPACITY),
    ];
    // Snapshot every grade key as (track, idx, frame) so we can both draw and interact without
    // re-borrowing project.bright_kf/... inside the loop that mutates project.
    let mut grade_keys: Vec<(u8, usize, i64, Color32)> = Vec::new();
    for (track, col) in grade_meta.iter() {
        let kfs: &[model::Kf] = match track {
            0 => &project.bright_kf,
            1 => &project.contrast_kf,
            2 => &project.sat_kf,
            _ => &project.opacity_kf,
        };
        for (idx, kf) in kfs.iter().enumerate() {
            grade_keys.push((*track, idx, kf.t, *col));
        }
    }

    // Draw every on-screen grade diamond (off-screen ones skipped). Drawn first; the per-diamond
    // interactions registered below sit on top.
    for &(_, _, t, col) in grade_keys.iter() {
        let kx = frame_to_x(left, t as f32, ppf, scroll);
        if kx < left + LANE_X_OFF || kx > strip_x_max {
            continue;
        }
        draw_kf_diamond(&painter, kx, strip_center, KF_DIAMOND_R, col);
    }

    // Strip-wide scrub handler, registered FIRST (before the per-diamond rects below) so the
    // diamonds — registered last — win the pointer on overlap (egui resolves overlapping
    // interactions in favour of the last-registered widget). A click landing on a diamond is then
    // consumed by that diamond (this `strip_resp.clicked()` is FALSE for it), while a click on the
    // bare band scrubs the playhead here. This preserves the old empty-strip click-to-scrub.
    let strip_resp = ui.interact(strip_rect, ui.id().with("tl_kf_strip"), Sense::click());
    if strip_resp.clicked() {
        if let Some(pos) = strip_resp.interact_pointer_pos() {
            *playhead = x_to_frame(left, pos.x, ppf, scroll).clamp(0, (total - 1).max(0));
        }
    }

    // ---- per-diamond interaction (slice B): drag = move (pose), right-click = delete (pose),
    // plain click = seek ----------------------------------------------------------------------
    // Each on-screen diamond gets its own click_and_drag() interact rect (a small square around
    // the diamond), registered AFTER `strip_resp` above so it wins the pointer on overlap (egui:
    // last-registered widget wins).
    //
    // COMMIT-ON-RELEASE (important): `move_grade_key` re-sorts the track by `t`, which can change
    // the dragged key's INDEX. The egui drag is tracked by the widget id `("kf_grade", track,
    // idx)`; if we mutated (and re-sorted) every frame, that index — and therefore the id — would
    // shift mid-drag and egui would drop the active drag (the classic sortable-keyframe jitter).
    // So during a drag we only DRAW a ghost diamond at the live pointer (no model change, so the
    // index/id stay stable for the whole gesture) and commit the move with a single
    // `move_grade_key` on `drag_stopped()` (release). Delete fires on a secondary-click (right
    // mouse) ONLY — keyboard Delete is NOT read here because app.rs already binds Delete to the
    // clip lift (Genesis::handle_keys, the `k.lift` branch) and `key_pressed` returns true for every reader in
    // the same frame, so a Delete tap meant for a keyframe would ALSO delete the selected clip
    // (silent data loss, and the keyframe delete would not be captured in history). Right-click is
    // the unambiguous, collision-free delete. A plain primary click (no drag) seeks the playhead
    // to the key's frame (preserves the old per-diamond click-to-seek). The hit rect is a touch
    // larger than the drawn diamond so it is comfortably grabbable.
    //
    // GRADE POSE (important): `add_grade_key` keys bright/contrast/sat at the SAME frame `t`, so
    // the three color-coded diamonds always render stacked at one (kx, strip_center). egui's
    // hit-test awards an overlapping pointer to the LAST-registered widget (the sat diamond,
    // track 2), so a drag/delete that touched the visible stack would move/delete ONLY sat and
    // tear the previously-coincident grade pose apart (bright/contrast left behind, un-grabbable
    // under sat). So we treat a grade edit as a POSE keyed on the dragged key's ORIGINAL frame
    // `t`: a move/delete is applied to EVERY grade track that has a key at that frame, not just
    // (track, idx). This mirrors the PiP pose handling below. We carry the original frame and
    // resolve matching (track, idx) pairs when we apply (model ops take (track, idx)).
    let hit_r = (KF_DIAMOND_R + 2.0).max(KF_HIT_PX);
    let mut grade_move: Option<(i64, i64)> = None; // (orig_t, new_t) — POSE move, committed on release
    let mut grade_delete: Option<i64> = None;      // orig_t — POSE delete
    let mut grade_seek: Option<i64> = None;        // frame to seek to on a plain click
    for &(track, idx, t, col) in grade_keys.iter() {
        let kx = frame_to_x(left, t as f32, ppf, scroll);
        if kx < left + LANE_X_OFF || kx > strip_x_max {
            continue; // don't register interactions for off-screen diamonds
        }
        let drect = Rect::from_center_size(Pos2::new(kx, strip_center), Vec2::splat(hit_r * 2.0));
        // stable per-(track,idx) id so egui tracks each diamond's drag independently
        let resp = ui.interact(drect, ui.id().with(("kf_grade", track, idx)), Sense::click_and_drag());
        if resp.dragged() {
            // live preview only: draw a ghost diamond following the pointer (clamped to the strip)
            if let Some(pos) = resp.interact_pointer_pos() {
                let gx = pos.x.clamp(left + LANE_X_OFF, strip_x_max);
                draw_kf_diamond(&painter, gx, strip_center, KF_DIAMOND_R + 1.0, col);
            }
        } else if resp.drag_stopped_by(egui::PointerButton::Primary) {
            // commit the POSE move once, on a real primary-button RELEASE, from the final pointer
            // x. `drag_stopped_by(Primary)` is false when egui aborts the drag on Escape (it clears
            // the drag without a button release), so Escape-to-cancel no longer commits the move.
            if let Some(pos) = resp.interact_pointer_pos() {
                grade_move = Some((t, x_to_frame(left, pos.x, ppf, scroll)));
            }
        } else if resp.secondary_clicked() {
            grade_delete = Some(t); // right-click: delete the whole pose at this frame
        } else if resp.clicked() {
            grade_seek = Some(t); // plain primary click with no drag: seek to this key
        }
    }
    // Apply at most one grade edit this frame (a single pointer can only act on one diamond).
    // Delete takes precedence over move/seek so a right-click during a tiny drag still deletes.
    // Both move and delete act on the WHOLE pose: every grade track (0..=3) that has a key at the
    // dragged key's ORIGINAL frame. Delete resolves matching indices fresh per track (each track
    // is independent, so no cross-track index shift). Move re-sorts inside move_grade_key, but we
    // resolve each track's matching idx independently right before moving it, so a re-sort of one
    // track never invalidates another track's index.
    // Undo coverage (skeptic #1): a committed grade-keyframe POSE move/delete is a real mutation
    // that originates in this (Team A) file, so push a pre-edit snapshot BEFORE applying it — once
    // per committed edit, guarded so a plain seek (no mutation) never pushes a dead undo step.
    if grade_delete.is_some() || grade_move.is_some() {
        hist.push(project);
    }
    if let Some(orig_t) = grade_delete {
        for track in 0u8..=3 {
            if let Some(idx) = grade_key_idx_at(project, track, orig_t) {
                project.delete_grade_key(track, idx);
            }
        }
    } else if let Some((orig_t, nt)) = grade_move {
        let new_t = nt.clamp(0, (total - 1).max(0));
        for track in 0u8..=3 {
            if let Some(idx) = grade_key_idx_at(project, track, orig_t) {
                project.move_grade_key(track, idx, new_t);
            }
        }
    } else if let Some(f) = grade_seek {
        *playhead = f.clamp(0, (total - 1).max(0));
    }

    // ---- clips: draw body, name chip, selection border + handle interaction ----
    // Clip interactions are registered AFTER the scrub rect so they take pointer priority
    // (egui resolves overlapping interactions by the last-registered widget under the cursor).
    //
    // PiP keyframe tick edits (slice B) are COLLECTED here and applied AFTER the clip loop so we
    // never mutate `project.pip_kf` while the loop borrows `project` to draw clips. Because
    // `add_pip_key` writes one entry per param (0..3) at the SAME (clip, t_local), a single tick
    // represents up to four coincident `pip_kf` entries; a move/delete acts on the WHOLE pose
    // (every entry matching that clip + original t_local) so the four params stay together. We
    // address them by (clip, old_t_local) rather than a flat index for exactly this reason —
    // model::move_pip_key/delete_pip_key take a flat index, so we resolve matching indices when
    // we apply.
    let mut pip_move: Option<(usize, i64, i64)> = None; // (clip, old_t_local, new_t_local)
    let mut pip_delete: Option<(usize, i64)> = None;    // (clip, old_t_local)

    // P3 editing: snapshot the modifier state ONCE for this frame. `multi_mod` (Shift OR Ctrl held)
    // routes a clip click to the additive multi-select toggle; `alt_mod` (Alt held) routes a body
    // drag to a SLIP instead of a move. One input() borrow keeps the clip loop borrow-free.
    let (multi_mod, alt_mod) = ui.ctx().input(|i| {
        let m = &i.modifiers;
        (m.shift || m.command || m.ctrl, m.alt)
    });
    for i in 0..project.clips.len() {
        let (start, len, track) = {
            let c = &project.clips[i];
            (c.t0 as f32, c.len as f32, c.track)
        };
        let row = row_of(track);
        let x = frame_to_x(left, start, ppf, scroll);
        let w = (len * ppf).max(6.0);
        let y = top + row as f32 * (track_h + gap);
        let rect = Rect::from_min_size(Pos2::new(x, y + 1.0), Vec2::new(w, track_h - 2.0));

        // LOCK ENFORCEMENT (wave P1): a clip on a locked track must not be trimmed, moved, or
        // (by the app's S key) split. We compute this once per clip; the drag cascade below skips
        // every mutating branch when the SOURCE track is locked (it still selects on click so the
        // user can inspect a locked clip). `is_locked` is a model read (Team C added track_lock).
        let src_locked = project.is_locked(track);

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

        // Snapping is honoured only when the app's snap toggle (Ctrl+P) is on; otherwise the raw
        // (frame-rounded) value is used directly. We branch by selecting which value feeds the
        // model op below, keeping the snap_edges() build out of the hot path when snap is off.
        if src_locked {
            // Locked source: ignore all trim/move drags (a no-op), but keep click-to-select live.
        } else if lresp.dragged() {
            // left-edge trim: move t0 to the pointer (snapped), holding the right edge fixed.
            // UNDO: push once at the drag START edge so Ctrl+Z reverts the whole trim gesture.
            *selected = i;
            if lresp.drag_started() {
                hist.push(project);
            }
            if let Some(pos) = lresp.interact_pointer_pos() {
                let raw = x_to_frame(left, pos.x, ppf, scroll);
                let nt0 = if snap {
                    let edges = snap_edges(project, i);
                    snap_frame(raw, &edges, ppf)
                } else {
                    raw
                };
                project.trim_start(i, nt0);
            }
        } else if rresp.dragged() {
            // right-edge trim: new length from pointer x (snapped to nearby edges).
            // UNDO: push once at the drag START edge.
            *selected = i;
            if rresp.drag_started() {
                hist.push(project);
            }
            if let Some(pos) = rresp.interact_pointer_pos() {
                let raw_end = x_to_frame(left, pos.x, ppf, scroll);
                let snapped_end = if snap {
                    let edges = snap_edges(project, i);
                    snap_frame(raw_end, &edges, ppf)
                } else {
                    raw_end
                };
                // Guard against the snapped end landing at/left of the clip start (an earlier
                // edge or frame 0 in the snap set): never pass a non-positive length. The
                // real MIN_CLIP floor is enforced in `trim_end`.
                let new_len = (snapped_end - project.clips[i].t0).max(1);
                project.trim_end(i, new_len);
            }
        } else if body.dragged() {
            // BODY DRAG. The GESTURE MODE is decided ONCE at the drag-start edge and stashed in
            // egui temp memory so it stays stable for the whole drag (the live modifier can wander
            // mid-drag without re-classifying it). Two modes (ROLL is a SEPARATE dedicated hot-zone
            // pass after the clip loop, so it never collides with the move/trim here):
            //   * SLIP  (Alt held at drag start): re-time the source under the fixed timeline window.
            //   * MOVE  (default): reposition the clip (cross-track) — the existing behaviour.
            // UNDO: a single pre-edit snapshot pushed on the drag-start edge for whichever mode.
            *selected = i;
            // `mode` = 1 SLIP, 0 MOVE; `anchor` = (origin_x, origin_t0_or_src). Both stashed at start.
            let mode_id = body.id.with("drag_mode");
            let anchor_id = body.id.with("drag_anchor");
            if body.drag_started() {
                hist.push(project);
                if let Some(pos) = body.interact_pointer_pos() {
                    if alt_mod {
                        // SLIP: anchor the original src_in (and remember this is a slip gesture).
                        let origin_src = project.clips[i].src_in;
                        ui.data_mut(|d| {
                            d.insert_temp(mode_id, 1u8);
                            d.insert_temp(anchor_id, (pos.x, origin_src));
                        });
                    } else {
                        // MOVE: anchor the original t0 (existing behaviour).
                        let origin_t0 = project.clips[i].t0;
                        ui.data_mut(|d| {
                            d.insert_temp(mode_id, 0u8);
                            d.insert_temp(anchor_id, (pos.x, origin_t0));
                        });
                    }
                }
            }
            let mode: u8 = ui.data(|d| d.get_temp(mode_id)).unwrap_or(0u8);
            if let Some(pos) = body.interact_pointer_pos() {
                let anchor: Option<(f32, i64)> = ui.data(|d| d.get_temp(anchor_id));
                if let Some((origin_x, a)) = anchor {
                    if mode == 1 {
                        // SLIP: source delta from horizontal pointer motion. Set src_in to
                        // (origin_src + moved) via slip's delta = target - current. t0/len held.
                        let moved = ((pos.x - origin_x) / ppf).round() as i64;
                        let target_src = (a + moved).max(0);
                        let cur_src = project.clips[i].src_in;
                        project.slip(i, target_src - cur_src);
                    } else {
                        // MOVE (default): the existing absolute-mapping cross-track move.
                        let moved = ((pos.x - origin_x) / ppf).round() as i64;
                        let raw = (a + moved).max(0);
                        let ns = if snap {
                            let edges = snap_edges(project, i);
                            snap_frame(raw, &edges, ppf).max(0)
                        } else {
                            raw
                        };
                        project.clips[i].t0 = ns;
                        // Destination track from the pointer Y (cross-track move). Only commit to a
                        // NON-locked destination; mirrors the pool-drop track mapping.
                        if let Some(dest_row) = row_at_y(pos.y, top, track_h, gap) {
                            let dest_track = track_of_row(dest_row);
                            if dest_track != project.clips[i].track && !project.is_locked(dest_track) {
                                project.clips[i].track = dest_track;
                            }
                        }
                    }
                }
            }
        }
        // Click selection (P3 multi-select): a plain click sets the primary + collapses the set to
        // this clip; a Shift/Ctrl-click toggles it in/out of the multi-select set. Routed through
        // `apply_clip_click`. Fires for the body OR either trim edge (clicking a handle still
        // selects). The drag branches above already set `*selected = i` for an active drag, so this
        // only runs on a true (non-drag) click.
        if body.clicked() || lresp.clicked() || rresp.clicked() {
            apply_clip_click(i, multi_mod, selected, selection);
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

        // ---- per-clip PiP KEYFRAME ticks (slice B): small orange marks, draggable + deletable --
        // For every PiP key bound to THIS clip, draw a thin orange tick at the key's absolute
        // frame (clip.t0 + key.t_local). Drawn after the gradient/fade but BEFORE the thumbnails/
        // waveform/border/chip so the ticks read on the body without hiding the clip content (per
        // the slice spec). `add_pip_key` stores 4 params at the SAME t_local, so the four ticks
        // coincide at one x and render as a single tick (mirrors MojoMedia drawing only param 0).
        //
        // Interaction: collect the DISTINCT on-screen t_local values for this clip, then register
        // ONE click_and_drag() rect per distinct tick (a thin vertical strip over the tick). These
        // are registered AFTER the clip body/edge interactions above, so a press on a tick wins the
        // pointer over the clip body (egui: last-registered wins overlap). Drag moves the pose
        // (t_local = frame - clip.t0, via move_pip_key on every matching entry); right-click
        // deletes the pose (keyboard Delete is NOT read — it collides with app.rs's clip razor;
        // see the per-tick handler below); a plain click falls through to the body's normal
        // select-clip (we don't seek per-tick, matching the per-clip pose model).
        //
        // COMMIT-ON-RELEASE (same reason as the grade diamonds): the tick's egui id is
        // `("kf_pip", i, tl)` keyed on the ORIGINAL t_local; mutating t_local every frame would
        // change `tl` (we rebuild `tlocals` from the now-moved store next repaint) and drop the
        // active drag. So during a drag we only draw a ghost tick at the pointer and commit the
        // move once on `drag_stopped()`. Edits are stashed in `pip_move`/`pip_delete` and applied
        // after the clip loop (so we never mutate `pip_kf` while the clip loop reads `project`).
        let tick_h = rect.height();
        let mut tlocals: Vec<i64> = Vec::new();
        for key in project.pip_kf.iter().filter(|k| k.clip == i) {
            if !tlocals.contains(&key.t_local) {
                tlocals.push(key.t_local);
            }
        }
        for &tl in tlocals.iter() {
            let kx = frame_to_x(left, start + tl as f32, ppf, scroll);
            // only draw/interact with ticks that fall within this clip's body rect
            if kx < rect.min.x || kx > rect.max.x {
                continue;
            }
            painter.line_segment(
                [Pos2::new(kx, rect.min.y), Pos2::new(kx, rect.min.y + tick_h)],
                Stroke::new(1.5, KF_COL_PIP),
            );
            // a thin grab strip (a few px wide, full clip height) centered on the tick
            let trect = Rect::from_center_size(
                Pos2::new(kx, rect.center().y),
                Vec2::new((KF_DIAMOND_R + 2.0) * 2.0, rect.height()),
            );
            let resp = ui.interact(trect, ui.id().with(("kf_pip", i, tl)), Sense::click_and_drag());
            if resp.dragged() {
                // live preview only: a brighter ghost tick following the pointer (clamped to body)
                if let Some(pos) = resp.interact_pointer_pos() {
                    let gx = pos.x.clamp(rect.min.x, rect.max.x);
                    painter.line_segment(
                        [Pos2::new(gx, rect.min.y), Pos2::new(gx, rect.min.y + tick_h)],
                        Stroke::new(2.0, KF_COL_PIP),
                    );
                }
            } else if resp.drag_stopped_by(egui::PointerButton::Primary) {
                // commit on a real primary-button RELEASE only — `drag_stopped_by(Primary)` is
                // false when egui aborts the drag on Escape, so Escape-to-cancel no longer commits.
                if let Some(pos) = resp.interact_pointer_pos() {
                    // Clamp the committed pointer x to the clip body (mirroring the ghost above)
                    // BEFORE mapping to a clip-local frame, so a release past the right edge can't
                    // store t_local > clip.len — which the next repaint would skip drawing/
                    // registering (line ~`kx > rect.max.x → continue`), leaving the tick invisible
                    // and un-grabbable. (move_pip_key only clamps the low end to >= 0.)
                    let nx = pos.x.clamp(rect.min.x, rect.max.x);
                    let new_local = x_to_frame(left, nx, ppf, scroll) - project.clips[i].t0;
                    pip_move = Some((i, tl, new_local));
                }
            } else if resp.secondary_clicked() {
                // right-click deletes the pose. Keyboard Delete is NOT read here: app.rs binds
                // Delete to the clip lift (Genesis::handle_keys `k.lift` branch) and `key_pressed` fires for every
                // reader the same frame, so a Delete meant for a keyframe would also delete the
                // selected clip. Right-click is the unambiguous, collision-free delete.
                pip_delete = Some((i, tl));
            }
        }

        // ---- per-clip visuals (slice C): drawn ON the body, UNDER the border + name chip ----
        // VIDEO clips (track 0=V1, 1=V2) get in/out thumbnails; AUDIO clips (track 2) get the
        // envelope waveform. Fetches are memoised + bounded (in/out thumb only, one envelope
        // per media) so the single serial worker is not hammered during a repaint.
        if track == 2 {
            draw_clip_waveform(&painter, project, i, rect);
        } else if w >= MIN_THUMB_CLIP_W {
            draw_clip_thumbs(ui.ctx(), &painter, project, i, rect);
        }

        // Border telegraphs selection state (P3 multi-select):
        //   * PRIMARY clip (i == selected)            -> white 1px (panel target / split-lift focus)
        //   * MULTI-SELECTED (in `selection`, not primary) -> orange 2px (part of the copy/cut set)
        //   * unselected                              -> black 1px
        // The primary takes precedence over the multi-select color so the panel-target clip always
        // reads as white even when it is also a set member (it usually is).
        let in_multi = selection.contains(&i);
        let (border, bw) = if i == *selected {
            (Color32::WHITE, 1.0)
        } else if in_multi {
            (MULTISEL_BORDER, 2.0)
        } else {
            (Color32::BLACK, 1.0)
        };
        painter.rect_stroke(rect, corner, Stroke::new(bw, border), StrokeKind::Inside);

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

    // ---- apply the collected PiP keyframe edit (slice B) -------------------------------------
    // A pose is up to four `pip_kf` entries sharing (clip, t_local); we resolve the matching flat
    // indices here and act on ALL of them so the four params move/delete together. Delete is
    // processed in reverse index order so earlier removals don't shift the indices of later ones.
    // Delete takes precedence over move (a right-click during a tiny drag still deletes). At most
    // one pose is edited per frame (a single pointer touches one tick).
    // Undo coverage (skeptic #1): mirror the grade-keyframe path — a committed PiP-tick POSE
    // move/delete is a real mutation originating in this file, so push a pre-edit snapshot BEFORE
    // applying it, guarded so a no-op frame never pushes a dead undo step.
    if pip_delete.is_some() || pip_move.is_some() {
        hist.push(project);
    }
    if let Some((clip, old_local)) = pip_delete {
        let mut idxs: Vec<usize> = project
            .pip_kf
            .iter()
            .enumerate()
            .filter(|(_, k)| k.clip == clip && k.t_local == old_local)
            .map(|(j, _)| j)
            .collect();
        idxs.sort_unstable();
        for &j in idxs.iter().rev() {
            project.delete_pip_key(j);
        }
    } else if let Some((clip, old_local, new_local)) = pip_move {
        // Move every entry of this pose to the new clip-local frame (model clamps to >= 0).
        let idxs: Vec<usize> = project
            .pip_kf
            .iter()
            .enumerate()
            .filter(|(_, k)| k.clip == clip && k.t_local == old_local)
            .map(|(j, _)| j)
            .collect();
        for &j in idxs.iter() {
            project.move_pip_key(j, new_local);
        }
    }

    // ---- ROLL EDIT pass (P3 editing): drag a shared INTERNAL cut to slide it -----------------
    // A ROLL hot-zone is a thin DRAG-only strip (±ROLL_HIT_PX) centered on every EXACT internal
    // shared cut — a frame where one same-track clip ends and another begins (left.end()==right.t0).
    // Registered AFTER the clip loop so it wins the pointer over the clip body/trim edges AT THAT
    // EXACT cut (egui: last-registered wins overlap). It senses DRAG ONLY, so a plain CLICK on the
    // cut still falls through to the clip beneath (select); only a press-AND-drag rolls. This keeps
    // roll reachable without a tool-mode toggle and without stealing trims away from clip ENDS that
    // have open space beside them (a cut with no abutting partner registers NO roll zone, so its
    // trim handle is untouched). Cross-track: only video+audio same-track pairs; a cut between two
    // clips on different tracks is not a shared cut and gets no zone.
    //
    // Implementation: scan each track's clips for abutting (left.end()==right.t0) same-track pairs
    // (this is the EXACT-cut subset of `boundaries()`, but we need the precise frame + both indices
    // and only EXACT abutment, so we compute it inline). For each, register a drag strip; on
    // drag_started push history + stash (origin_x, left_i, right_i); each drag frame map the pointer
    // delta to frames and call roll_edit, re-anchoring origin_x by the APPLIED delta so the cut
    // tracks the pointer without drift even when clamped. Collect at most one roll per frame.
    {
        // Build the list of exact internal cuts: (track, boundary_frame, left_i, right_i).
        let mut cuts: Vec<(u8, i64, usize, usize)> = Vec::new();
        for track in 0u8..3 {
            // indices on this track sorted by t0 for a left-to-right abutment scan
            let mut order: Vec<usize> = (0..project.clips.len())
                .filter(|&k| project.clips[k].track == track)
                .collect();
            order.sort_by_key(|&k| project.clips[k].t0);
            for w in order.windows(2) {
                let (l, r) = (w[0], w[1]);
                if project.clips[l].end() == project.clips[r].t0 {
                    cuts.push((track, project.clips[r].t0, l, r));
                }
            }
        }
        for (track, bf, li, ri) in cuts {
            let bx = frame_to_x(left, bf as f32, ppf, scroll);
            if bx < left + LANE_X_OFF || bx > left + lane_w {
                continue; // cut scrolled out of view
            }
            let row = row_of(track);
            let lane_y = top + row as f32 * (track_h + gap);
            let zone = Rect::from_center_size(
                Pos2::new(bx, lane_y + (track_h - 2.0) * 0.5 + 1.0),
                Vec2::new(ROLL_HIT_PX * 2.0, (track_h - 2.0).max(2.0)),
            );
            // Stable id keyed on (track, boundary frame). Sense DRAG only so clicks pass through.
            let resp = ui.interact(zone, ui.id().with(("tl_roll", track, bf)), Sense::drag());
            let resp = resp.on_hover_text("Roll edit (drag to slide the cut)");
            if resp.dragged() {
                let anchor_id = resp.id.with("roll_anchor");
                if resp.drag_started() {
                    // Refuse a roll if EITHER side sits on a locked track (advisory lock, like move).
                    if !project.is_locked(track) {
                        hist.push(project);
                        if let Some(pos) = resp.interact_pointer_pos() {
                            ui.data_mut(|d| d.insert_temp(anchor_id, (pos.x, li as i64, ri as i64)));
                        }
                    }
                }
                if let Some(pos) = resp.interact_pointer_pos() {
                    let stash: Option<(f32, i64, i64)> = ui.data(|d| d.get_temp(anchor_id));
                    if let Some((origin_x, a, b)) = stash {
                        // Map the pointer delta to frames, apply the (clamped) roll, then re-anchor
                        // origin_x by the APPLIED delta so the cut tracks the pointer without drift.
                        let moved = ((pos.x - origin_x) / ppf).round() as i64;
                        if moved != 0 {
                            let applied = project.roll_edit(a as usize, b as usize, moved);
                            if applied != 0 {
                                let new_origin_x = origin_x + applied as f32 * ppf;
                                ui.data_mut(|d| d.insert_temp(anchor_id, (new_origin_x, a, b)));
                            }
                        }
                    }
                }
                // Visual: a bright accent bar on the cut while rolling so the gesture reads.
                painter.line_segment(
                    [Pos2::new(bx, lane_y), Pos2::new(bx, lane_y + track_h)],
                    Stroke::new(2.0, theme::ACCENT),
                );
            } else if resp.hovered() {
                // Hover affordance: a faint accent bar inviting the roll drag.
                painter.line_segment(
                    [Pos2::new(bx, lane_y), Pos2::new(bx, lane_y + track_h)],
                    Stroke::new(1.5, theme::ACCENT.gamma_multiply(0.5)),
                );
            }
        }
    }

    // ---- per-boundary TRANSITIONS (wave 8, slice C-trans-ui) --------------------------------
    // For each VIDEO track (0 = V1, 1 = V2 — A1/audio gets no transitions), ask the model for the
    // same-track clip boundaries (project.boundaries(track) -> (out_clip, in_clip, boundary_frame))
    // and draw a TRANSITION AFFORDANCE on that track's lane at the boundary x:
    //   * a filled mediumpurple bowtie ("X") if a transition already lives there
    //     (project.transition_at(track, boundary).is_some()), or
    //   * a faint "+" hint otherwise.
    // Each affordance registers a TIGHT click() hit rect (only ~TRANS_HIT_PX wide around the
    // boundary x, centered vertically on the lane) so it never steals the lane scrub or a
    // clip-body drag for clicks elsewhere on the lane. These rects are registered AFTER the clip
    // body/edge widgets above and after the scrub rect, so egui awards an overlapping pointer-CLICK
    // to them (last-registered wins) only when the pointer is actually within the tight hit zone.
    // The rect senses click only (not drag), so a PRESS-AND-DRAG starting on a boundary is still
    // claimed by the clip body beneath it (which senses drag) — dragging moves the clip; a plain
    // click adds/cycles the transition. The two never collide on one gesture.
    //
    // Interaction (mirrors MojoMedia's per-boundary "Bndry>/Trans>" cycling, surfaced inline):
    //   * left-click an EMPTY boundary  -> add_transition(track, boundary, TRANS_NEW_DUR, 0)  (crossfade)
    //   * left-click an EXISTING marker -> cycle kind: remove + re-add with kind=(kind+1)%8
    //   * right-click an EXISTING marker-> remove it (resolve its index in project.transitions)
    // Tooltip shows the kind name (Crossfade/.../Dissolve). Edits are COLLECTED here and applied
    // after both video-track loops so we never mutate project.transitions while iterating the
    // boundaries Vec it (indirectly) describes. At most one transition edit fires per frame (one
    // pointer touches one boundary).
    enum TransEdit {
        Add { track: u8, center: i64 },              // empty boundary clicked: add a default crossfade
        Cycle { track: u8, center: i64, kind: i32 }, // existing marker clicked: advance kind 0..7
        Remove { track: u8, center: i64 },           // existing marker right-clicked: delete
    }
    let mut trans_edit: Option<TransEdit> = None;
    // Only the two VIDEO tracks host transitions (track 0 = V1, 1 = V2). Audio (2) is skipped.
    for track in [0u8, 1u8] {
        let row = row_of(track);
        let lane_y = top + row as f32 * (track_h + gap);
        let lane_cy = lane_y + (track_h - 2.0) * 0.5 + 1.0; // vertical center of the clip body rect
        // Snapshot boundaries up front (Vec of (out, in, boundary_frame)); the per-boundary
        // interaction may stash an edit but does not mutate project, so this borrow is released
        // before we apply anything.
        let bounds = project.boundaries(track);
        for (_out_i, _in_i, bf) in bounds {
            let bx = frame_to_x(left, bf as f32, ppf, scroll);
            // skip boundaries scrolled out of the clip area
            if bx < left + LANE_X_OFF || bx > left + lane_w {
                continue;
            }
            // `kind` is owned (Option<i32>), so the transition_at borrow ends on this line; the
            // rest of the loop reads only `kind` and never touches `project` mutably.
            let kind: Option<i32> = project.transition_at(track, bf).map(|tr| tr.kind);
            if kind.is_some() {
                draw_trans_marker(&painter, bx, lane_cy, TRANS_R, TRANS_PURPLE);
            } else {
                draw_trans_hint(&painter, bx, lane_cy, TRANS_R);
            }
            // Tight hit zone: a thin strip (±TRANS_HIT_PX) around the boundary x, the lane's
            // body height tall. Centered on the boundary so a click just left/right of it still
            // registers, but a click well away from the boundary falls through to the scrub/clip.
            let hit = Rect::from_center_size(
                Pos2::new(bx, lane_cy),
                Vec2::new(TRANS_HIT_PX * 2.0, (track_h - 2.0).max(2.0)),
            );
            // Stable id keyed on (track, boundary_frame). boundary_frame shifts only when clips
            // move, in which case the affordance legitimately belongs to the new boundary.
            let resp = ui.interact(
                hit,
                ui.id().with(("tl_trans", track, bf)),
                Sense::click(),
            );
            // Tooltip: name the active transition, or invite adding one.
            let resp = match kind {
                Some(k) => resp.on_hover_text(format!("Transition: {} (click to cycle, right-click to remove)", trans_name(k))),
                None => resp.on_hover_text("Add transition (crossfade)"),
            };
            if resp.secondary_clicked() {
                if kind.is_some() {
                    trans_edit = Some(TransEdit::Remove { track, center: bf });
                }
            } else if resp.clicked() {
                match kind {
                    Some(k) => trans_edit = Some(TransEdit::Cycle { track, center: bf, kind: k }),
                    None => trans_edit = Some(TransEdit::Add { track, center: bf }),
                }
            }
        }
    }
    // Apply the single collected transition edit (after both loops, so project.transitions is no
    // longer indirectly borrowed by boundaries()). add_transition replaces a same track+center
    // record, so the Cycle path (remove-by-index then add) and the Add path both keep one record
    // per boundary, matching the pinned API contract.
    // UNDO (wave P1): a transition add/cycle/remove is a single committed click gesture (collected
    // in the Option above, so at most one per frame). Push a pre-edit snapshot BEFORE mutating so
    // Ctrl+Z reverts the transition change. The None arm pushes nothing (no edit this frame).
    if trans_edit.is_some() {
        hist.push(project);
    }
    match trans_edit {
        Some(TransEdit::Add { track, center }) => {
            project.add_transition(track, center, TRANS_NEW_DUR, 0);
        }
        Some(TransEdit::Cycle { track, center, kind }) => {
            let next = (kind + 1).rem_euclid(TRANS_KIND_N);
            // Resolve the SAME record the marker was drawn for: by the window-contains /
            // nearest-center predicate (matching `transition_at`), NOT by exact center. After a
            // clip move the boundary frame `center` (= the drawn `bf`) can differ from the stored
            // transition's `center`; we must re-add at the record's TRUE center so the cycle
            // mutates it in place (add_transition dedups on track+center) instead of pushing a
            // duplicate at the stale boundary frame. Preserve the window length; only kind cycles.
            let existing = trans_idx_containing(project, track, center)
                .and_then(|idx| project.transitions.get(idx))
                .map(|tr| (tr.center, tr.dur));
            let (real_center, dur) = existing.unwrap_or((center, TRANS_NEW_DUR));
            if let Some(idx) = trans_idx_containing(project, track, center) {
                project.remove_transition(idx);
            }
            project.add_transition(track, real_center, dur, next);
        }
        Some(TransEdit::Remove { track, center }) => {
            // Resolve by the same window-contains predicate the marker was drawn with, so a
            // boundary that drifted from the stored center is still removable.
            if let Some(idx) = trans_idx_containing(project, track, center) {
                project.remove_transition(idx);
            }
        }
        None => {}
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
            // a thin insertion marker at the drop frame (snapped iff snap on), so the user sees
            // where it lands.
            if let Some(pos) = ptr {
                let raw = x_to_frame(left, pos.x, ppf, scroll);
                let snapped = if snap {
                    let edges = snap_edges(project, usize::MAX); // no clip to exclude during a drop
                    snap_frame(raw, &edges, ppf)
                } else {
                    raw
                };
                let ix = frame_to_x(left, snapped as f32, ppf, scroll);
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
                // Block drops onto a locked track (Team C added track_lock; wave P1 enforces it).
                if !project.is_locked(track) {
                    let raw = x_to_frame(left, pos.x, ppf, scroll);
                    let t0 = if snap {
                        let edges = snap_edges(project, usize::MAX);
                        snap_frame(raw, &edges, ppf).max(0)
                    } else {
                        raw.max(0)
                    };
                    // UNDO (wave P1): the drop is the commit gesture; push a pre-edit snapshot once
                    // here (the only frame dnd_release_payload fires) so Ctrl+Z removes the clip.
                    hist.push(project);
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
        let mx = frame_to_x(left, m as f32, ppf, scroll);
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

    // ---- playhead: vertical line at left + LANE_X_OFF + (playhead - scroll)*ppf ----
    let ph = (*playhead).clamp(0, (total - 1).max(0));
    *playhead = ph;
    let px = frame_to_x(left, ph as f32, ppf, scroll);
    painter.line_segment(
        [Pos2::new(px, ruler_top), Pos2::new(px, lanes_bottom)],
        Stroke::new(1.0, theme::ACCENT),
    );
    // playhead head marker (small triangle-ish tab at the top of the ruler)
    let head = Rect::from_min_size(Pos2::new(px - 3.0, ruler_top), Vec2::new(6.0, 5.0));
    painter.rect_filled(head, CornerRadius::same(1), theme::ACCENT);
}
