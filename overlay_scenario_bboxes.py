import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple


Box = Tuple[int, int, int, int]
FrameDetections = Dict[int, List[Tuple[int, Box]]]
DEFAULT_SCENARIO_FPS = 4.28274


def parse_summary_file(summary_path: str) -> Tuple[int, int, List[int], List[int], int | None]:
    start_frame = None
    end_frame = None
    vehicle_ids: List[int] = []
    highlight_ids: List[int] = []
    ego_id: int | None = None
    active_section: str | None = None

    with open(summary_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Start Frame:"):
                start_frame = int(line.split(":", 1)[1].strip())
                active_section = None
                continue

            if line.startswith("End Frame:"):
                end_frame = int(line.split(":", 1)[1].strip())
                active_section = None
                continue

            if line.startswith("Ego ID:"):
                ego_id = int(line.split(":", 1)[1].strip())
                active_section = None
                continue

            if line == "Highlight IDs:":
                active_section = "highlight"
                continue

            if line == "Neighbor IDs:":
                active_section = "neighbor"
                continue

            if line == "Vehicles:":
                active_section = "vehicles"
                continue

            if active_section and line.startswith("-"):
                target_id = int(line[1:].strip())
                if active_section == "vehicles":
                    vehicle_ids.append(target_id)
                elif active_section == "highlight":
                    highlight_ids.append(target_id)
                continue

            active_section = None

    if start_frame is None or end_frame is None:
        raise ValueError(
            f"Could not read frame range from summary file: {summary_path}"
        )

    if end_frame < start_frame:
        raise ValueError(
            f"Invalid frame range in {summary_path}: {start_frame} to {end_frame}"
        )

    if not vehicle_ids:
        raise ValueError(f"No vehicle IDs found in summary file: {summary_path}")

    return start_frame, end_frame, vehicle_ids, highlight_ids, ego_id


def iter_annotation_items(
    annotation_path: str, chunk_size: int = 1024 * 1024
) -> Iterable[dict]:
    decoder = json.JSONDecoder()

    with open(annotation_path, "r", encoding="utf-8") as handle:
        buffer = ""

        while '"items"' not in buffer:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(
                    f'Could not find the "items" array in {annotation_path}'
                )
            buffer += chunk
            if len(buffer) > chunk_size * 4:
                buffer = buffer[-chunk_size * 4 :]

        # Some CVAT exports include nested "...\"items\": []" sections (for example
        # categories.points.items) before the top-level frame list. We must lock onto
        # the first items-array that starts with JSON objects.
        search_start = 0
        items_key_index = -1
        bracket_index = -1
        while True:
            candidate_index = buffer.find('"items"', search_start)
            while candidate_index == -1:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(
                        f'Could not find the top-level "items" array in {annotation_path}'
                    )
                buffer += chunk
                candidate_index = buffer.find('"items"', search_start)

            candidate_bracket = buffer.find("[", candidate_index)
            while candidate_bracket == -1:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(
                        f'Could not find the start of the "items" array in {annotation_path}'
                    )
                buffer += chunk
                candidate_bracket = buffer.find("[", candidate_index)

            next_char_index = candidate_bracket + 1
            while next_char_index >= len(buffer) or buffer[next_char_index].isspace():
                if next_char_index < len(buffer) and buffer[next_char_index].isspace():
                    next_char_index += 1
                    continue
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(
                        f'Could not inspect "items" array contents in {annotation_path}'
                    )
                buffer += chunk

            # Frame items array should contain objects (or be empty in edge cases).
            # Skip obvious non-frame arrays such as [] from metadata sections.
            if buffer[next_char_index] in "{":
                items_key_index = candidate_index
                bracket_index = candidate_bracket
                break

            search_start = candidate_bracket + 1

        buffer = buffer[bracket_index + 1 :]

        while True:
            buffer = buffer.lstrip()
            while not buffer:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(
                        f"Unexpected end of file while reading {annotation_path}"
                    )
                buffer += chunk
                buffer = buffer.lstrip()

            if buffer[0] == "]":
                return

            if buffer[0] == ",":
                buffer = buffer[1:]
                continue

            while True:
                try:
                    item, end_index = decoder.raw_decode(buffer)
                    buffer = buffer[end_index:]
                    yield item
                    break
                except json.JSONDecodeError:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        preview = buffer[:120].replace("\n", " ")
                        raise ValueError(
                            "Unexpected end of file while decoding an annotation item "
                            f"near: {preview!r}"
                        )
                    buffer += chunk


def load_frame_detections(
    annotation_path: str, track_ids: List[int], start_frame: int, end_frame: int
) -> Tuple[FrameDetections, Dict[int, int]]:
    target_ids = set(track_ids)
    track_hits = {track_id: 0 for track_id in track_ids}
    detections: FrameDetections = {}

    for frame_index, item in enumerate(iter_annotation_items(annotation_path)):
        if frame_index > end_frame:
            break

        if frame_index < start_frame:
            continue

        frame_matches: List[Tuple[int, Box]] = []
        for annotation in item.get("annotations", []):
            attributes = annotation.get("attributes") or {}
            track_id = attributes.get("track_id")
            if track_id not in target_ids:
                continue

            bbox = annotation.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x, y, w, h = bbox
            left = int(round(x))
            top = int(round(y))
            right = int(round(x + w))
            bottom = int(round(y + h))
            frame_matches.append((int(track_id), (left, top, right, bottom)))
            track_hits[int(track_id)] += 1

        detections[frame_index] = frame_matches

    return detections, track_hits


def color_for_track(track_id: int) -> Tuple[int, int, int]:
    return (
        64 + (track_id * 37) % 192,
        64 + (track_id * 67) % 192,
        64 + (track_id * 97) % 192,
    )


def label_for_track(track_id: int, highlight_track_ids: List[int], ego_track_id: int | None) -> str:
    if ego_track_id == track_id:
        return f"EGO {track_id}"
    if track_id in highlight_track_ids:
        return f"FOCUS {track_id}"
    return f"VEH {track_id}"


def style_for_track(
    track_id: int,
    highlight_track_ids: List[int],
    ego_track_id: int | None,
    default_thickness: int,
) -> Tuple[Tuple[int, int, int], int]:
    if ego_track_id == track_id:
        return (40, 40, 255), max(default_thickness + 2, default_thickness * 2)
    if track_id in highlight_track_ids:
        return (0, 215, 255), max(default_thickness + 1, default_thickness * 2 - 1)
    return color_for_track(track_id), max(default_thickness, default_thickness + 1)


def default_output_path(summary_path: str, full_video: bool, zoom: bool) -> str:
    summary_dir = os.path.dirname(summary_path)
    summary_name = os.path.splitext(os.path.basename(summary_path))[0]
    if summary_name.endswith("_summary"):
        summary_name = summary_name[: -len("_summary")]

    if full_video and zoom:
        suffix = "_full_zoom_boxed.mp4"
    elif full_video:
        suffix = "_full_boxed.mp4"
    elif zoom:
        suffix = "_zoom_boxed.mp4"
    else:
        suffix = "_boxed.mp4"
    return os.path.join(summary_dir, f"{summary_name}{suffix}")


def annotation_frame_to_source_frame(
    annotation_frame: int, source_fps: float, annotation_fps: float
) -> int:
    return int(round(annotation_frame * (source_fps / annotation_fps)))


def draw_detections(
    cv2,
    frame,
    frame_index: int,
    detections: List[Tuple[int, Box]],
    legend_track_ids: List[int],
    highlight_track_ids: List[int],
    ego_track_id: int | None,
) -> None:
    frame_height, frame_width = frame.shape[:2]
    font_scale = max(0.5, frame_width / 3200.0)
    line_thickness = max(2, int(round(frame_width / 1200.0)))
    text_thickness = max(1, line_thickness - 1)
    padding = max(4, line_thickness * 2)

    for track_id, (left, top, right, bottom) in detections:
        color, box_thickness = style_for_track(
            track_id,
            highlight_track_ids,
            ego_track_id,
            line_thickness,
        )

        left = max(0, min(frame_width - 1, left))
        right = max(0, min(frame_width - 1, right))
        top = max(0, min(frame_height - 1, top))
        bottom = max(0, min(frame_height - 1, bottom))

        if right <= left or bottom <= top:
            continue

        cv2.rectangle(frame, (left, top), (right, bottom), color, box_thickness)

        label = label_for_track(track_id, highlight_track_ids, ego_track_id)
        (label_width, label_height), label_baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        label_left = left
        label_top = max(0, top - label_height - label_baseline - (padding // 2))
        label_right = min(frame_width - 1, label_left + label_width + padding)
        label_bottom = min(frame_height - 1, label_top + label_height + label_baseline + (padding // 2))

        cv2.rectangle(
            frame,
            (label_left, label_top),
            (label_right, label_bottom),
            color,
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (label_left + (padding // 2), label_bottom - label_baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (15, 15, 15),
            text_thickness,
            cv2.LINE_AA,
        )

    banner = f"Frame {frame_index}"
    (banner_width, banner_height), banner_baseline = cv2.getTextSize(
        banner, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
    )
    banner_left = padding
    banner_top = padding
    banner_right = banner_left + banner_width + (padding * 2)
    banner_bottom = banner_top + banner_height + banner_baseline + (padding * 2)

    cv2.rectangle(
        frame,
        (banner_left, banner_top),
        (banner_right, banner_bottom),
        (30, 30, 30),
        thickness=-1,
    )
    cv2.putText(
        frame,
        banner,
        (banner_left + padding, banner_bottom - banner_baseline - padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    legend_title = "Vehicles"
    (title_width, title_height), title_baseline = cv2.getTextSize(
        legend_title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
    )
    row_height = title_height + title_baseline + padding
    legend_width = title_width
    for track_id in legend_track_ids:
        label = label_for_track(track_id, highlight_track_ids, ego_track_id)
        (label_width, _,) = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )[0]
        legend_width = max(legend_width, label_width + row_height)

    legend_left = padding
    legend_top = banner_bottom + padding
    legend_height = row_height * (len(legend_track_ids) + 1) + padding
    legend_right = min(frame_width - 1, legend_left + legend_width + (padding * 3))
    legend_bottom = min(frame_height - 1, legend_top + legend_height)

    cv2.rectangle(
        frame,
        (legend_left, legend_top),
        (legend_right, legend_bottom),
        (30, 30, 30),
        thickness=-1,
    )

    cv2.putText(
        frame,
        legend_title,
        (legend_left + padding, legend_top + title_height + padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    row_y = legend_top + row_height + padding
    swatch_size = max(10, row_height - padding)
    for track_id in legend_track_ids:
        color, _ = style_for_track(
            track_id,
            highlight_track_ids,
            ego_track_id,
            line_thickness,
        )
        swatch_top = row_y - title_height
        swatch_bottom = swatch_top + swatch_size
        swatch_left = legend_left + padding
        swatch_right = swatch_left + swatch_size

        cv2.rectangle(
            frame,
            (swatch_left, swatch_top),
            (swatch_right, swatch_bottom),
            color,
            thickness=-1,
        )
        cv2.putText(
            frame,
            label_for_track(track_id, highlight_track_ids, ego_track_id),
            (swatch_right + padding, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
        row_y += row_height


def compute_zoom_window(
    detections: List[Tuple[int, Box]],
    frame_width: int,
    frame_height: int,
    padding: int,
    min_size: int,
    previous_window: Tuple[int, int, int, int] | None,
) -> Tuple[int, int, int, int] | None:
    if not detections:
        return previous_window

    left = min(box[0] for _, box in detections)
    top = min(box[1] for _, box in detections)
    right = max(box[2] for _, box in detections)
    bottom = max(box[3] for _, box in detections)

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    target_width = max((right - left) + (2 * padding), min_size)
    target_height = max((bottom - top) + (2 * padding), min_size)

    aspect_ratio = frame_width / frame_height if frame_height else 1.0
    if target_width / target_height < aspect_ratio:
        target_width = target_height * aspect_ratio
    else:
        target_height = target_width / aspect_ratio

    if previous_window is not None:
        prev_left, prev_top, prev_right, prev_bottom = previous_window
        prev_center_x = (prev_left + prev_right) / 2.0
        prev_center_y = (prev_top + prev_bottom) / 2.0
        prev_width = prev_right - prev_left
        prev_height = prev_bottom - prev_top
        smoothing = 0.35
        center_x = (smoothing * center_x) + ((1.0 - smoothing) * prev_center_x)
        center_y = (smoothing * center_y) + ((1.0 - smoothing) * prev_center_y)
        target_width = (smoothing * target_width) + ((1.0 - smoothing) * prev_width)
        target_height = (smoothing * target_height) + ((1.0 - smoothing) * prev_height)

    target_width = min(target_width, frame_width)
    target_height = min(target_height, frame_height)

    crop_left = int(round(center_x - (target_width / 2.0)))
    crop_top = int(round(center_y - (target_height / 2.0)))
    crop_right = int(round(crop_left + target_width))
    crop_bottom = int(round(crop_top + target_height))

    if crop_left < 0:
        crop_right -= crop_left
        crop_left = 0
    if crop_top < 0:
        crop_bottom -= crop_top
        crop_top = 0
    if crop_right > frame_width:
        shift = crop_right - frame_width
        crop_left -= shift
        crop_right = frame_width
    if crop_bottom > frame_height:
        shift = crop_bottom - frame_height
        crop_top -= shift
        crop_bottom = frame_height

    crop_left = max(0, crop_left)
    crop_top = max(0, crop_top)
    crop_right = min(frame_width, max(crop_left + 2, crop_right))
    crop_bottom = min(frame_height, max(crop_top + 2, crop_bottom))
    return crop_left, crop_top, crop_right, crop_bottom


def apply_zoom_window(
    cv2,
    frame,
    zoom_window: Tuple[int, int, int, int] | None,
    output_width: int,
    output_height: int,
):
    if zoom_window is None:
        return frame

    left, top, right, bottom = zoom_window
    cropped = frame[top:bottom, left:right]
    if cropped.size == 0:
        return frame
    return cv2.resize(cropped, (output_width, output_height), interpolation=cv2.INTER_LINEAR)


def render_video(
    video_path: str,
    output_path: str,
    detections_by_frame: FrameDetections,
    track_ids: List[int],
    highlight_track_ids: List[int],
    ego_track_id: int | None,
    start_frame: int,
    end_frame: int,
    full_video: bool,
    codec: str,
    output_fps: float | None,
    annotation_fps: float,
    zoom: bool,
    zoom_padding: int,
    zoom_min_size: int,
) -> Tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required to render the output video. "
            "Install it with `pip install opencv-python` or use a Python environment "
            "that already provides `cv2`."
        ) from exc

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    if annotation_fps <= 0:
        raise ValueError("Annotation FPS must be greater than zero.")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer_fps = output_fps or source_fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    mapped_start_frame = annotation_frame_to_source_frame(
        start_frame, source_fps, annotation_fps
    )
    mapped_end_frame = annotation_frame_to_source_frame(
        end_frame, source_fps, annotation_fps
    )

    if total_frames > 0 and mapped_start_frame >= total_frames:
        raise ValueError(
            f"Mapped start frame {mapped_start_frame} is outside the video range "
            f"(0 to {total_frames - 1})"
        )

    if total_frames > 0 and mapped_end_frame >= total_frames:
        raise ValueError(
            f"Mapped end frame {mapped_end_frame} is outside the video range "
            f"(0 to {total_frames - 1})"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*codec),
        writer_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: {output_path}. "
            f"Try a different codec with --codec."
        )

    written_frames = 0
    zoom_window = None
    if full_video:
        source_frame_to_annotation = {
            annotation_frame_to_source_frame(frame_index, source_fps, annotation_fps): frame_index
            for frame_index in range(start_frame, end_frame + 1)
        }

        current_source_frame = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            annotation_frame = source_frame_to_annotation.get(current_source_frame)
            if annotation_frame is not None:
                detections = detections_by_frame.get(annotation_frame, [])
                draw_detections(
                    cv2,
                    frame,
                    annotation_frame,
                    detections,
                    track_ids,
                    highlight_track_ids,
                    ego_track_id,
                )
                if zoom:
                    zoom_window = compute_zoom_window(
                        detections,
                        width,
                        height,
                        zoom_padding,
                        zoom_min_size,
                        zoom_window,
                    )
                    frame = apply_zoom_window(cv2, frame, zoom_window, width, height)

            writer.write(frame)
            written_frames += 1
            current_source_frame += 1
    else:
        for annotation_frame in range(start_frame, end_frame + 1):
            source_frame = annotation_frame_to_source_frame(
                annotation_frame, source_fps, annotation_fps
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            success, frame = capture.read()
            if not success:
                break

            detections = detections_by_frame.get(annotation_frame, [])
            draw_detections(
                cv2,
                frame,
                annotation_frame,
                detections,
                track_ids,
                highlight_track_ids,
                ego_track_id,
            )
            if zoom:
                zoom_window = compute_zoom_window(
                    detections,
                    width,
                    height,
                    zoom_padding,
                    zoom_min_size,
                    zoom_window,
                )
                frame = apply_zoom_window(cv2, frame, zoom_window, width, height)

            writer.write(frame)
            written_frames += 1

    capture.release()
    writer.release()

    return written_frames, total_frames


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay scenario bounding boxes from default.json onto a video."
        )
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to a scenario summary text file.",
    )
    parser.add_argument(
        "--annotations",
        default="default.json",
        help="Path to the annotation JSON file.",
    )
    parser.add_argument(
        "--video",
        default="DJI_0003.MP4",
        help="Path to the source video.",
    )
    parser.add_argument(
        "--output",
        help="Path to the output MP4. Defaults next to the summary file.",
    )
    parser.add_argument(
        "--full-video",
        action="store_true",
        help=(
            "Write the full video and draw boxes only during the summary frame range. "
            "By default, the script writes only the selected frame span."
        ),
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec for OpenCV VideoWriter, for example mp4v or avc1.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        help=(
            "FPS for the rendered output. Defaults to 4 for clipped scenario output "
            "and to the source video FPS for --full-video."
        ),
    )
    parser.add_argument(
        "--annotation-fps",
        type=float,
        default=DEFAULT_SCENARIO_FPS,
        help=(
            "FPS of the annotated/scenario timeline. This is used to map summary frame "
            "numbers onto the original source video."
        ),
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Crop around the active scenario vehicles and resize back to video size.",
    )
    parser.add_argument(
        "--zoom-padding",
        type=int,
        default=160,
        help="Extra pixels around the tracked vehicles when --zoom is enabled.",
    )
    parser.add_argument(
        "--zoom-min-size",
        type=int,
        default=500,
        help="Minimum crop size in pixels when --zoom is enabled.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    start_frame, end_frame, track_ids, highlight_track_ids, ego_track_id = parse_summary_file(args.summary)
    output_path = args.output or default_output_path(
        args.summary,
        args.full_video,
        args.zoom,
    )

    print(
        f"Summary frames: {start_frame} to {end_frame} | "
        f"Vehicles: {', '.join(str(track_id) for track_id in track_ids)}"
    )
    if ego_track_id is not None or highlight_track_ids:
        print(
            f"Highlighting ego={ego_track_id if ego_track_id is not None else 'n/a'} | "
            f"focus IDs: {', '.join(str(track_id) for track_id in highlight_track_ids) or 'none'}"
        )
    print("Loading only the requested detections from the annotation file...")
    detections_by_frame, track_hits = load_frame_detections(
        args.annotations,
        track_ids,
        start_frame,
        end_frame,
    )

    output_fps = args.output_fps
    if output_fps is None and not args.full_video:
        output_fps = DEFAULT_SCENARIO_FPS

    missing_tracks = [track_id for track_id, hits in track_hits.items() if hits == 0]
    if missing_tracks:
        print(
            "Warning: no detections were found in the selected frame range for "
            f"track IDs: {', '.join(str(track_id) for track_id in missing_tracks)}"
        )

    written_frames, total_frames = render_video(
        video_path=args.video,
        output_path=output_path,
        detections_by_frame=detections_by_frame,
        track_ids=track_ids,
        highlight_track_ids=highlight_track_ids,
        ego_track_id=ego_track_id,
        start_frame=start_frame,
        end_frame=end_frame,
        full_video=args.full_video,
        codec=args.codec,
        output_fps=output_fps,
        annotation_fps=args.annotation_fps,
        zoom=args.zoom,
        zoom_padding=args.zoom_padding,
        zoom_min_size=args.zoom_min_size,
    )

    if args.full_video:
        print(
            f"Finished writing full video overlay to {output_path} "
            f"({written_frames} frames written, source reports {total_frames} frames)."
        )
    else:
        print(
            f"Finished writing clipped overlay to {output_path} "
            f"({written_frames} frames written at {output_fps or DEFAULT_SCENARIO_FPS:.2f} fps)."
        )


if __name__ == "__main__":
    main()
