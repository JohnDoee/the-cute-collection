import concurrent.futures
import json
import logging
import math
import os
import pickle
import shlex
import shutil
import subprocess
import tempfile
import uuid
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from re import sub

import click
import cv2
import ffmpeg
import librosa
import numpy as np
import pysubs2
from annoy import AnnoyIndex
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist, cosine, pdist
from skimage.metrics import structural_similarity

logger = logging.getLogger(__name__)

# HOP_LENGTH = 1024
# HOP_LENGTH = 512
HOP_LENGTH = None


class Video:
    _cv2_video = None
    _cv2_video_info = None
    _ffmpeg_probe = None

    def __init__(self, filepath):
        self.filepath = filepath

    @property
    def video_capture(self):
        if not self._cv2_video:
            self._cv2_video = cv2.VideoCapture(self.filepath)

        return self._cv2_video

    @property
    def video_info(self):
        if not self._cv2_video_info:
            cap = self.video_capture
            self._cv2_video_info = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
            self._cv2_video_info["duration"] = Decimal(
                self._cv2_video_info["frame_count"]
            ) / Decimal(self._cv2_video_info["fps"])

        return self._cv2_video_info

    @property
    def probe(self):
        if self._ffmpeg_probe is None:
            self._ffmpeg_probe = ffmpeg.probe(self.filepath)
        return self._ffmpeg_probe

    @property
    def subtitle_streams(self):
        return [s for s in self.probe["streams"] if s["codec_type"] == "subtitle"]

    @property
    def audio_streams(self):
        return [s for s in self.probe["streams"] if s["codec_type"] == "audio"]

    @property
    def chapters(self):
        chapters = (
            json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-loglevel",
                        "error",
                        "-hide_banner",
                        "-of",
                        "json",
                        "-show_chapters",
                        self.filepath,
                    ],
                    stdin=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            )
            or {}
        )
        return chapters.get("chapters", [])

    def create_ffmpeg(self):
        return ffmpeg.input(self.filepath)

    def extract_subtitle_metadata(self, track_id):
        stream = self.subtitle_streams[track_id]
        metadata = []
        for k, v in stream.get("tags", {}).items():
            if k in ["language", "title"]:
                metadata.append((k, v))
        return metadata

    def extract_audio_metadata(self, track_id):
        stream = self.audio_streams[track_id]
        metadata = []
        for k, v in stream.get("tags", {}).items():
            if k in ["language", "title"]:
                metadata.append((k, v))
        return metadata


def estimate_audio_shift_points_from_subtitles(
    x_1_chroma,
    x_2_chroma,
    fs,
    video_file,
    track_id,
    n_chroma,
    adjust_delay=None,
    framerate_align=None,
    max_ms_cutoff=110,
    external_subtitle_file=None,
):
    # subtitle, subtitle_format = extract_subtitle_data(video_file, track_id, framerate_align)
    if external_subtitle_file is not None:
        subtitle, subtitle_format = import_subtitle_data(
            external_subtitle_file, framerate_align
        )
    else:
        subtitle, subtitle_format = extract_subtitle_data(
            video_file, track_id, framerate_align
        )
    index = AnnoyIndex(n_chroma, "euclidean")
    for i, c in enumerate(x_2_chroma):
        index.add_item(i, c)
    index.build(10)

    def align_subtitle(subtitle_line, min_datapoint_percent=0.2, min_avg=0.05, n=10):
        start_i = int(subtitle_line.start * fs / HOP_LENGTH / 1000)
        end_i = int(subtitle_line.end * fs / HOP_LENGTH / 1000)
        if end_i >= len(x_1_chroma):
            return []
        found_indexes = {}
        for i in range(end_i - start_i):
            source_vector = list(x_1_chroma[start_i + i])
            for vector_index in index.get_nns_by_vector(source_vector, n):
                target_vector = index.get_item_vector(vector_index)
                found_indexes.setdefault(vector_index - i, []).append(
                    cosine(source_vector, target_vector)
                )
        candidates = []
        for k, v in sorted(
            found_indexes.items(), key=lambda x: len(x[1]), reverse=True
        ):
            if len(v) < (end_i - start_i) * min_datapoint_percent:
                v = v + found_indexes.get(k + 1, []) + found_indexes.get(k - 1, [])
                if len(v) < (end_i - start_i) * min_datapoint_percent:
                    break
            if np.average(v) > min_avg:
                continue

            candidates.append(k)
        return candidates

    subtitle_matches = []
    for i, s in enumerate(subtitle):
        candidates = align_subtitle(s)
        if candidates:
            subtitle_matches.append(
                [
                    int(
                        librosa.frames_to_time(candidate, sr=fs, hop_length=HOP_LENGTH)
                        * 1000
                    )
                    - s.start
                    for candidate in candidates
                ]
            )
        else:
            subtitle_matches.append(None)

    def generate_best_chains(subtitle_matches):
        subtitle_groups = []
        for i, subtitle_match in enumerate(subtitle_matches):
            if subtitle_match is None:
                continue
            ts_diff = subtitle_match[0]
            if (
                not subtitle_groups
                or abs(
                    ts_diff - np.median([t for (t, _, _) in subtitle_groups[-1][:20]])
                )
                > max_ms_cutoff
            ):
                print(f"Creating new group with {ts_diff=}")
                subtitle_groups.append([(ts_diff, i, subtitle_match)])
            else:
                subtitle_groups[-1].append((ts_diff, i, subtitle_match))

        group_size_cutoff = 5
        has_modified = True
        while has_modified:
            has_modified = False
            previous_subtitle_group = None
            for i, subtitle_group in enumerate(subtitle_groups):
                if len(subtitle_group) > group_size_cutoff:
                    ts_diff = np.median([t for (t, _, _) in subtitle_group[:20]])
                    if i >= len(subtitle_groups):
                        next_subtitle_group = subtitle_groups[i + 1]
                    else:
                        next_subtitle_group = None
                    if previous_subtitle_group is not None:
                        for entry in list(previous_subtitle_group[::-1]):
                            matched_ts = sorted(
                                [(abs(ts - ts_diff), ts) for ts in entry[2]]
                            )
                            if matched_ts[0][0] <= max_ms_cutoff:
                                print(f"Moving entry forward {entry}")
                                previous_subtitle_group.pop(-1)
                                subtitle_group.insert(
                                    0, (matched_ts[0][1], entry[1], entry[2])
                                )
                                has_modified = True
                            else:
                                print(f"Breaking at {entry}")
                                break
                    if next_subtitle_group is not None:
                        for entry in list(next_subtitle_group):
                            matched_ts = sorted(
                                [(abs(ts - ts_diff), ts) for ts in entry[2]]
                            )
                            if matched_ts[0][0] <= max_ms_cutoff:
                                print(f"Moving entry back {entry}")
                                next_subtitle_group.pop(0)
                                subtitle_group.append(
                                    (matched_ts[0][1], entry[1], entry[2])
                                )
                                has_modified = True
                            else:
                                print(f"Breaking at back {entry}")
                                break

                    previous_subtitle_group = None
                else:
                    previous_subtitle_group = subtitle_group

            to_remove_groups = []
            for i, subtitle_group in enumerate(subtitle_groups):
                if len(subtitle_group) == 0:
                    to_remove_groups.append(i)

            for i in to_remove_groups[::-1]:
                print(f"Removing group {i}")
                del subtitle_groups[i]

        has_modified = True
        while has_modified:
            has_modified = False
            to_remove_groups = []
            for i, subtitle_group in enumerate(subtitle_groups):
                if len(subtitle_group) <= 2:
                    to_remove_groups.append(i)

            for i in to_remove_groups[::-1]:
                del subtitle_groups[i]
                print(f"Deleting {i}")
                has_modified = True

            for i, subtitle_group in enumerate(subtitle_groups):
                if not subtitle_group or i == 0:
                    continue
                previous_subtitle_group = subtitle_groups[i - 1]
                previous_ts_diff = np.median(
                    [t for (t, _, _) in previous_subtitle_group]
                )
                ts_diff = np.median([t for (t, _, _) in subtitle_group])
                if abs(previous_ts_diff - ts_diff) < max_ms_cutoff:
                    print(f"Merging into {i}")
                    subtitle_groups[i] = previous_subtitle_group + subtitle_groups[i]
                    subtitle_groups[i - 1] = []
                    has_modified = True

        return subtitle_groups

    audio_shift_points, sync_buckets, delete_buckets = [], [], []

    previous_start_timestamp, previous_end_timestamp = None, None
    for subtitle_group in generate_best_chains(subtitle_matches):
        delta = np.median([t for (t, _, _) in subtitle_group if t is not None]) / 1000
        subtitle_group_indexes = set([i for (_, i, _) in subtitle_group])
        timestamps = []
        for i, subtitle_line in enumerate(subtitle):
            if i not in subtitle_group_indexes:
                continue
            timestamps += [subtitle_line.start, subtitle_line.end]
        start_timestamp = min(timestamps) / 1000
        end_timestamp = max(timestamps) / 1000

        slice_buffer_length = 51200 // HOP_LENGTH
        x_1_start_i = int((start_timestamp * fs) / HOP_LENGTH)
        x_1_end_i = (
            int((end_timestamp * fs) / HOP_LENGTH)
            - slice_buffer_length
            - slice_buffer_length
        )
        min_slice_length = 204800 // HOP_LENGTH
        if x_1_end_i - x_1_start_i > min_slice_length:
            x_2_start_i = int(((start_timestamp + delta) * fs) / HOP_LENGTH)
            x_2_end_i = int(((end_timestamp + delta) * fs) / HOP_LENGTH)

            x_1_chroma_slice = x_1_chroma[
                x_1_start_i
                + slice_buffer_length : x_1_start_i
                + slice_buffer_length
                + min_slice_length
            ]
            x_2_chroma_slice = x_2_chroma[
                x_2_start_i : x_2_start_i
                + min_slice_length
                + slice_buffer_length
                + slice_buffer_length
            ]

            C = cdist(x_1_chroma_slice, x_2_chroma_slice, metric="cosine")
            C = np.nan_to_num(C, copy=False)

            smallest_i, smallest_value = None, None
            for i in range(len(x_2_chroma_slice) - len(x_1_chroma_slice)):
                cost_diagonal = np.flip(np.diagonal(C, offset=i))
                total_cost = np.sum(cost_diagonal)
                if smallest_value is None or smallest_value > total_cost:
                    smallest_value = total_cost
                    smallest_i = i
            print(f"Additional buffer change: {smallest_i - slice_buffer_length}")
            delta = librosa.frames_to_time(
                [(x_2_start_i - x_1_start_i) + (smallest_i - slice_buffer_length)],
                sr=fs,
                hop_length=HOP_LENGTH,
            )[0]
            # delta += librosa.frames_to_time([smallest_i - slice_buffer_length], sr=fs, hop_length=HOP_LENGTH)[0]
            delta += adjust_delay or 0

        print(f"sync points {start_timestamp} {end_timestamp} {delta=}")
        if not audio_shift_points and start_timestamp > 0 and delta > 0:
            audio_shift_points.append((0.0, delta, delta))
            delete_buckets.append((0, start_timestamp - 0.001))

        if previous_end_timestamp is not None:
            audio_shift_points.append(
                (
                    previous_end_timestamp + 0.01 + 100_000_000,
                    previous_end_timestamp + 0.01,
                    -100_000_000,
                )
            )
            delete_buckets.append(
                (previous_end_timestamp + 0.001, start_timestamp - 0.001)
            )

        sync_buckets.append((start_timestamp, end_timestamp, delta))
        audio_shift_points.append(
            (start_timestamp - 0.01 + delta, start_timestamp - 0.01, delta)
        )
        previous_start_timestamp, previous_end_timestamp = (
            start_timestamp,
            end_timestamp,
        )
    if previous_end_timestamp is not None:
        delete_buckets.append((previous_end_timestamp + 0.001, 100_000))

    return audio_shift_points, sync_buckets, delete_buckets


def estimate_audio_shift_points(
    x_1_chroma,
    x_2_chroma,
    fs,
    max_cost_matrix_size=200_000_000,
    only_delta=False,
    adjust_delay=None,
    sliding_window_size=300,
):
    expected_matrix_size = len(x_1_chroma) * len(x_2_chroma)

    print(f"Expected cost-matrix size is {expected_matrix_size=}")

    if expected_matrix_size > max_cost_matrix_size:
        print(
            f"Since our cost-matrix is bigger than max allowed cost {max_cost_matrix_size=} we will slice it."
        )
        chroma_slice_size = int(math.sqrt(max_cost_matrix_size))
        chroma_slice_step = int(chroma_slice_size * 0.8)
    else:
        print("Memory can fully fit our cost-matrix")
        chroma_slice_size = 100_000_000
        chroma_slice_step = 100_000_000

    all_diffs = None
    all_timestamps = None

    for i in range(
        0, max([len(x_1_chroma), len(x_2_chroma)]), chroma_slice_step
    ):  # TODO
        start_i = max(
            min(
                len(x_1_chroma) - chroma_slice_step,
                len(x_2_chroma) - chroma_slice_step,
                i,
            ),
            0,
        )

        x_1_chroma_slice = x_1_chroma[start_i : start_i + chroma_slice_size]
        x_2_chroma_slice = x_2_chroma[start_i : start_i + chroma_slice_size]
        if start_i:
            wp_offset = librosa.frames_to_time([start_i], sr=fs, hop_length=HOP_LENGTH)[
                0
            ]
        else:
            wp_offset = 0
        print(
            f"Doing chroma slices x1={len(x_1_chroma_slice)} x2={len(x_2_chroma_slice)} {wp_offset=}"
        )

        C = cdist(x_2_chroma_slice, x_1_chroma_slice, metric="cosine")
        C = np.nan_to_num(C, copy=False)
        D, wp = librosa.sequence.dtw(C=C)
        wp_s = np.flip(
            librosa.frames_to_time(wp, sr=fs, hop_length=HOP_LENGTH) + wp_offset, axis=0
        )

        diffs = []
        timestamps = []

        t1_already_seen = set()
        t2_already_seen = set()

        for t1, t2 in wp_s:
            should_skip = t1 in t1_already_seen or t2 in t2_already_seen
            t1_already_seen.add(t1)
            t2_already_seen.add(t2)
            if should_skip:
                continue

            diff = np.round(t1 - t2, 3)
            diffs.append(diff)
            timestamps.append((t1, t2))

        if all_timestamps:
            at1, at2 = all_timestamps[-1]
            t1, t2 = timestamps[0]

            cutoff_t = t1 + ((at1 - t1) / 2)
            for timestamp_index, (t1, t2) in enumerate(timestamps):
                if t1 > cutoff_t:
                    break
            for all_timestamp_index, (t1, t2) in enumerate(all_timestamps[::-1]):
                if t1 <= cutoff_t:
                    break
            all_timestamps = (
                all_timestamps[:-all_timestamp_index] + timestamps[timestamp_index:]
            )
            all_diffs = all_diffs[:-all_timestamp_index] + diffs[timestamp_index:]
        else:
            all_timestamps = timestamps
            all_diffs = diffs

    if only_delta:
        min_abs_diff = 0.03
    else:
        min_abs_diff = 0.06
    # sliding_window_size = 300 TODO: make it change near the end to detect changes while keeping it higher before.
    shift_points = []
    last_most_common = None
    for i, v in enumerate(
        sliding_window_view(all_diffs, window_shape=sliding_window_size)
    ):
        most_common, most_common_count = Counter(v).most_common(1)[0]
        if (
            last_most_common is None
            or abs(most_common - last_most_common) > min_abs_diff
        ):
            j = list(v).index(most_common)
            t1, t2 = all_timestamps[i + j]
            if adjust_delay:
                adjusted_most_common = most_common + adjust_delay
            else:
                adjusted_most_common = most_common
            shift_points.append((t1, t2, adjusted_most_common))
            last_most_common = most_common
            print(
                f"Found sync point source_timestamp={t2} target_timestamp={t1} delta={most_common} delta_count={most_common_count} delta_average={np.average(v)} delta_median={np.median(v)}"
            )

    new_shift_points = []
    is_first_point = True
    for t1, t2, delta in shift_points:
        x_2_compare_point = int((t1 * fs) / HOP_LENGTH)
        x_1_compare_point = int((t2 * fs) / HOP_LENGTH)

        step_back = min(
            int((max(abs(delta * 0.3), 1) * fs) / HOP_LENGTH),
            x_1_compare_point,
            x_2_compare_point,
        )  # TODO: do not float into previous shift point
        range_end = min(
            int((5 * fs) / HOP_LENGTH),
            len(x_2_chroma) - x_2_compare_point,
            len(x_1_chroma) - x_1_compare_point,
        )

        C = cdist(
            x_2_chroma[x_2_compare_point - step_back : x_2_compare_point + range_end],
            x_1_chroma[x_1_compare_point - step_back : x_1_compare_point + range_end],
            metric="cosine",
        )
        C = np.nan_to_num(C, copy=False)
        cost_diagonal = np.flip(np.diagonal(C))
        print(
            f"Trying to align range source_timestamp={t2} target_timestamp={t1} {delta=} {step_back=} {range_end=} {x_1_compare_point=} {x_2_compare_point=}"
        )
        max_cost = np.max(cost_diagonal[:range_end]) * 1.15
        print(f"{max_cost=}")
        for i, cost in enumerate(cost_diagonal):
            if cost > max_cost:
                print(f"Found breakpoint at additional delta {i - range_end}")
                seconds = ((i - range_end) * HOP_LENGTH) / fs
                new_shift_points.append((t1 - seconds, t2 - seconds, delta))
                break
        else:
            if is_first_point:
                new_shift_points.append((t1 - min(t1, t2), t2 - min(t1, t2), delta))
                print(
                    "No breakpoint found, moving all the way back (if this is the first point)"
                )
            else:
                new_shift_points.append((t1, t2, delta))
                print(
                    "No breakpoint found, assume cost problems et.al. and just adding the current"
                )

        is_first_point = False

    zero_shifting_delta = 8.0
    t1, t2, delta = new_shift_points[0]
    if 0 < t1 < zero_shifting_delta or 0 < t2 < zero_shifting_delta:
        print(f"Zero-shifting initial delta {t1=} {t2=} {delta}")
        shift_delta = min(t1, t2)
        t1 -= shift_delta
        t2 -= shift_delta
        new_shift_points[0] = (t1, t2, delta)

    return new_shift_points


class TrackMappingParamType(click.ParamType):
    name = "trackmapping"

    def convert(self, value, param, ctx):
        converted_mapping = []
        for mapping in value.split(","):
            try:
                mapping = [int(v) for v in mapping.split(":")]
            except ValueError:
                self.fail(
                    "Failed to convert mappings to numbers. Syntax is file:track e.g. 0:1"
                )

            if len(mapping) != 2:
                self.fail("Mappings must be 2 long. Syntax is file:track e.g. 0:1")

            converted_mapping.append(mapping)
        return converted_mapping


TRACK_MAPPING = TrackMappingParamType()


class MetadataMappingParamType(click.ParamType):
    name = "metadatamapping"

    def convert(self, value, param, ctx):
        value = value.split("=", 2)
        if len(value) != 3:
            self.fail("Missing arguments, syntax is track_id=key=value")

        track_id, key, value = value

        try:
            track_id = int(track_id)
        except ValueError:
            self.fail("Track ID must be an ID")

        return (track_id, key, value)


METADATA_MAPPING = MetadataMappingParamType()


class IntegerRangeParamType(click.ParamType):
    name = "integerrange"

    def convert(self, value, param, ctx):
        value = value.split("-")
        if len(value) != 2:
            self.fail("Missing arguments, syntax is start_second-end_second")

        start_second, end_second = value

        try:
            start_second, end_second = int(start_second), int(end_second)
        except ValueError:
            self.fail("Range must be integers")

        return (start_second, end_second)


INTEGER_RANGE = IntegerRangeParamType()


def generate_chroma_cqt(
    source_file,
    audio_track_id,
    target_file,
    n_chroma=12,
    framerate_align=None,
    preserve_silence=False,
):
    target_wav_file = target_file.with_suffix(".wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_file),
        "-map",
        f"0:a:{audio_track_id}",
        "-ar",
        "22050",
    ]

    # -filter:a "atempo=2.0"
    if framerate_align:
        cmd += [
            "-filter:a",
            f"atempo={framerate_align[1] / framerate_align[0]}",
        ]

    cmd += [str(target_wav_file)]

    subprocess.check_call(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.PIPE,
    )

    src, fs = librosa.load(str(target_wav_file))
    y = src
    if not preserve_silence:
        _, (ltrim, rtrim) = librosa.effects.trim(src)
        print(
            f"Trimming {(len(src) - rtrim) / fs}s silence from end from {source_file.name}"
        )
        y = y[:rtrim]
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=fs, hop_length=HOP_LENGTH, n_chroma=n_chroma
    ).T
    target_file.write_bytes(pickle.dumps((chroma, fs)))
    target_wav_file.unlink()
    return chroma, fs


def find_and_align_chapter(x_1_chroma, x_2_chroma, fs, min_match_value=60):
    """Find x_1_chroma in x_2_chroma"""
    best_equals = 0
    C = cdist(x_1_chroma[42:], x_2_chroma, metric="cosine")
    C = np.nan_to_num(C, copy=False)
    smallest_sum = None
    location = None
    smallest_found = 9999999
    for i, v in enumerate(
        sliding_window_view(C.T, window_shape=len(x_1_chroma[42:]), axis=0)
    ):
        s = np.sum(np.diagonal(v))
        if smallest_found > s:
            # print(s)
            smallest_found = s
        if s > min_match_value:
            continue
        if smallest_sum is None or s < smallest_sum:
            smallest_sum = s
            location = i

    if smallest_sum is None:
        return None

    return librosa.frames_to_time(
        [location, location + len(x_1_chroma)], sr=fs, hop_length=HOP_LENGTH
    )


def humanize_seconds(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{(s):06.3f}"


def turn_audio_shift_points_to_audio_segments(audio_shift_points):
    sync_buckets = []
    delete_buckets = []

    if len(audio_shift_points) > 1:
        for i in range(len(audio_shift_points)):
            local_audio_shift_points = audio_shift_points[i : i + 2]
            if len(local_audio_shift_points) > 1:
                (st1, st2, sdelta), (et1, et2, edelta) = local_audio_shift_points

                from_delete_time = (et2 + edelta) - sdelta
                # from_delete_delay = max(0.0, et2 - from_delete_time)
                if from_delete_time < et2:
                    delete_buckets.append((from_delete_time, et2))

                # sync_buckets.append(((st2 - min(from_delete_delay, to_delete_max_reuse)), st2 + (et1 - st1), sdelta))
                sync_buckets.append(
                    (st2, min(st2 + (et1 - st1), et2), sdelta)
                )  # TODO: make sure the et2 stuff is correct

    t1, t2, delta = audio_shift_points[-1]
    # sync_buckets.append((t2 - min(from_delete_delay, to_delete_max_reuse), 1_000_000, delta))
    sync_buckets.append((t2, 1_000_000, delta))
    print(f"Delete buckets {delete_buckets}")

    return sync_buckets, delete_buckets


def find_good_frame_breakpoint(video, current_frame):  # do binary search instead?
    compare_frame_size = (32, 32)
    frame_cache = {}

    def get_frame(frame_no):
        if frame_no not in frame_cache:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            frame_cache[frame_no] = cv2.cvtColor(
                cv2.resize(video.read()[1], compare_frame_size), cv2.COLOR_BGR2GRAY
            )
        return frame_cache[frame_no]

    best_score = 1.0
    best_frame = current_frame
    for frame_no in frame_generator(current_frame):
        score = structural_similarity(get_frame(frame_no), get_frame(frame_no + 1))
        if score < best_score:
            best_score = score
            best_frame = frame_no
        if score < 0.65:
            return frame_no - current_frame
    return best_frame - current_frame


def frame_generator(start_i):
    for i in range(1, 300):
        if i >= start_i:
            continue
        yield start_i + i
        yield start_i - i


def estimate_frame_diff(
    source_video, target_video, current_source_frame, current_target_frame
):
    frame_index_size = (64, 64)
    compare_frame_count = 5
    spread_frame_count = 14
    ret, source_frame = source_video.read()
    ret, target_frame = target_video.read()

    sy, sx, sz = source_frame.shape
    ty, tx, tz = target_frame.shape

    s_aspect = sx / sy
    t_aspect = tx / ty

    source_frames = []
    target_frames = []

    source_from_frame = current_source_frame - (compare_frame_count // 2)
    source_to_frame = source_from_frame + compare_frame_count

    source_video.set(cv2.CAP_PROP_POS_FRAMES, source_from_frame)
    for _ in range(source_from_frame, source_to_frame):
        frame_no = source_video.get(cv2.CAP_PROP_POS_FRAMES)
        source_frames.append(
            (
                cv2.cvtColor(
                    cv2.resize(source_video.read()[1], frame_index_size),
                    cv2.COLOR_BGR2GRAY,
                ),
                frame_no,
            )
        )

    target_from_frame = current_target_frame - spread_frame_count
    target_to_frame = target_from_frame + (spread_frame_count * 2)

    target_video.set(cv2.CAP_PROP_POS_FRAMES, target_from_frame)
    for _ in range(target_from_frame, target_to_frame):
        frame_no = target_video.get(cv2.CAP_PROP_POS_FRAMES)
        target_frame = target_video.read()[1]
        if s_aspect > t_aspect:
            new_tx = (tx / sx) * sy
            slice_each_x = int((tx - new_tx) / 2)
            target_frame = target_frame[0:ty, slice_each_x : (tx - slice_each_x)]

        target_frame = cv2.resize(target_frame, frame_index_size)
        target_frames.append((cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY), frame_no))
    best_diff = 0
    best_frame_diff = None
    for i in range(len(target_frames) - len(source_frames)):
        v = target_frames[i : i + len(source_frames)]
        diffs = []
        frame_nos = []
        for sf, tf in zip(source_frames, v):
            sf, sfn = sf
            tf, tfn = tf
            frame_nos.append((sfn, tfn))
            diffs.append(structural_similarity(sf, tf, multichannel=False))
        diffs = np.square(np.array(diffs) * 100)
        if sum(diffs) > best_diff:
            best_diff = sum(diffs)
            best_frame_diff = (target_from_frame + i) - source_from_frame

    return best_frame_diff


def frame_align_video(source_video, target_video, line_start, delta):
    print(f"Frame aligning video at {line_start=} {delta=}")
    source_frame_no = math.ceil((line_start * source_video.video_info["fps"]) / 1000)
    target_frame_no = math.ceil(
        ((line_start + delta) * target_video.video_info["fps"]) / 1000
    )

    good_breakpoint = find_good_frame_breakpoint(
        source_video.video_capture, source_frame_no
    )

    frame_diff = estimate_frame_diff(
        source_video.video_capture,
        target_video.video_capture,
        source_frame_no + good_breakpoint,
        target_frame_no + good_breakpoint,
    )
    # frame_diff_delta = math.ceil((frame_diff / target_video.video_info["fps"]) * 1000)
    best_target_frame = source_frame_no + frame_diff
    best_target_frame_time = math.ceil(
        best_target_frame * 1000 / target_video.video_info["fps"]
    )
    # actual_delta = line.start + frame_diff_delta
    actual_delta = best_target_frame_time - line_start
    if actual_delta != delta:
        print(
            f"Sign delta is different {delta=} {actual_delta=} {line_start=} {best_target_frame=} {best_target_frame_time=} {humanize_seconds((line_start + actual_delta)/1000)}"
        )

    return actual_delta


def frame_align_sync_bucket(source_video, target_video, sync_bucket):
    start_timestamp, end_timestamp, delta = sync_bucket
    delta = round(delta * 1000)
    actual_end_timestamp = min(
        end_timestamp, float(source_video.video_info["duration"])
    )
    line_start = int(((actual_end_timestamp - start_timestamp) * 1000) / 2)
    new_delta = frame_align_video(source_video, target_video, line_start, delta)
    if delta != new_delta:
        print(f"Changed delta from {delta=} {new_delta=}")
    return (start_timestamp, end_timestamp, new_delta / 1000)


def extract_subtitle_data(video_file, track_id, framerate_align=None):
    track_stream = video_file.subtitle_streams[track_id]
    codec_name_mapping = {
        "ass": "ass",
        "subrip": "srt",
    }
    subtitle_format = codec_name_mapping[track_stream["codec_name"]]
    subtitles_data = (
        video_file.create_ffmpeg()[f"s:{track_id}"]
        .output("pipe:", format=subtitle_format)
        .run(capture_stdout=True, quiet=True)[0]
        .decode("utf-8")
    )
    subtitles = pysubs2.SSAFile.from_string(subtitles_data)
    if framerate_align is not None:
        subtitles.transform_framerate(framerate_align[0], framerate_align[1])
    return subtitles, subtitle_format


def import_subtitle_data(external_subtitle_file, framerate_align=None):
    subtitle_format = external_subtitle_file.split(".")[-1]
    subtitles = pysubs2.load(external_subtitle_file)
    if framerate_align is not None:
        subtitles.transform_framerate(framerate_align[0], framerate_align[1])
    return subtitles, subtitle_format


def extract_and_sync_subtitles(
    video_file,
    track_id,
    video_duration,
    only_delta,
    audio_shift_points,
    subtitle_sync_buckets,
    subtitle_delete_buckets,
    output_file,
    subtitle_cutoff,
    sync_non_dialogue_to_video,
    output_video_file,
    framerate_align=None,
    external_subtitle_file=None,
    subtitle_min_font_size=None,
):
    video_align_cache = {}
    if external_subtitle_file is not None:
        subtitle, subtitle_format = import_subtitle_data(
            external_subtitle_file, framerate_align
        )
    else:
        subtitle, subtitle_format = extract_subtitle_data(
            video_file, track_id, framerate_align
        )
    # TODO: if target video is missing some stuff, make sure we do not shift into that part and have double subtitles.

    new_subtitles = []
    video_duration = int(video_duration * 1000)
    subtitle_sync_buckets = [
        (round(t1 * 1000), round(t2 * 1000), round(delta * 1000))
        for (t1, t2, delta) in subtitle_sync_buckets
    ]
    subtitle_delete_buckets = [
        (round(t1 * 1000), round(t2 * 1000)) for (t1, t2) in subtitle_delete_buckets
    ]
    print(f"Sync buckets {subtitle_sync_buckets=} {subtitle_delete_buckets=}")
    to_delete_lines = set()
    for i, line in enumerate(subtitle):
        is_dialogue = True
        if (
            sync_non_dialogue_to_video
            and line.start >= sync_non_dialogue_to_video[0] * 1000
            and line.start <= sync_non_dialogue_to_video[1] * 1000
            and subtitle.format == "ass"
            and line.type == "Dialogue"
            and (
                "\\pos(" in line.text or "\\move(" in line.text
            )  # a bit lazy way to figure out if sign
        ):
            is_dialogue = False
        skip_sync = False
        for t1, t2 in subtitle_delete_buckets:
            if (
                not only_delta
                and line.start > t1
                and line.start < t2
                or line.end > t1
                and line.end < t2
            ):
                print(f"DELETING LINE {line}")
                to_delete_lines.add(i)
                skip_sync = True
                break
        if skip_sync:
            continue
        for (
            t1,
            t2,
            delta,
        ) in subtitle_sync_buckets:  # TODO: sync to new time and move to at least 0.00
            current_line_length = line.end - line.start
            if current_line_length < 1000:
                min_line_length = current_line_length
            elif current_line_length < 5000:
                min_line_length = int(current_line_length * 0.75)
            else:
                min_line_length = 5000
            if (line.start >= t1 and line.start <= t2) or (
                (max(t1 + delta, 0) - delta) >= line.start
                and (max(t2 + delta, 0) - delta) <= line.start
            ):  # TODO: what does the extra part fix
                if not only_delta and line.end >= t1 and line.end > t2:
                    print(
                        f"WARNING, we are floating outside bounds with end: {line.start} {line.end} - {line}"
                    )
                # print(f"Matching with t1={humanize_seconds(t1/1000)} t2={humanize_seconds(t2/1000)} {delta=} {line=} start")
                if not is_dialogue:
                    if line.start not in video_align_cache:
                        video_align_cache[line.start] = frame_align_video(
                            video_file, output_video_file, line.start, delta
                        )
                    delta = video_align_cache[line.start]
                line.start += delta
                if only_delta:
                    line.end += delta
                else:
                    line.end = min(
                        line.end + delta, t2
                    )  # TODO: allow floating outside if it doesn't hit anything
                if not only_delta and line.end - line.start < min_line_length:
                    print(f"WARNING, line is too short {line} - {min_line_length=}")
                break
            elif (line.end >= t1 and line.end <= t2) or (
                (max(t1 + delta, 0) - delta) >= line.end
                and (max(t2 + delta, 0) - delta) <= line.end
            ):  # TODO: what does the extra part fix
                if not only_delta and line.start >= t1 and line.start > t2:
                    print(
                        f"WARNING, we are floating outside bounds with start: {line.start} {line.end} - {line}"
                    )
                # print(f"Matching with t1={humanize_seconds(t1/1000)} t2={humanize_seconds(t2/1000)} {delta=} {line=} end")
                if not is_dialogue:
                    if line.start not in video_align_cache:
                        video_align_cache[line.start] = frame_align_video(
                            video_file, output_video_file, line.start, delta
                        )
                    delta = video_align_cache[line.start]
                if only_delta:
                    line.start += delta
                else:
                    line.start = min(
                        line.start + delta, t1
                    )  # TODO: allow floating outside if it doesn't hit anything
                line.end += delta
                if not only_delta and line.end - line.start < min_line_length:
                    print(f"WARNING, line is too short {line} - {min_line_length=}")
                break
        else:
            print(f"Unable to find place for {line}")
            to_delete_lines.add(i)

    for i, line in enumerate(subtitle):
        if i in to_delete_lines:
            continue
        if subtitle_cutoff is not None and line.start > int(subtitle_cutoff * 1000):
            print(f"Removing line {line} because it is after {subtitle_cutoff=}")
            to_delete_lines.add(i)
        elif line.start > video_duration:
            print(f"Removing line {line} because it starts after end")
            to_delete_lines.add(i)
        elif line.end < 0:
            print(f"Removing line {line} because it ends after the video")
            to_delete_lines.add(i)
        elif line.end > video_duration:
            print(f"Moving line {line} end to end of video because it ended after")
            line.end = video_duration
        elif line.start < 0:
            print(
                f"Moving line {line} start to beginning of video because it started before"
            )
            line.start = 0

    for i in sorted(to_delete_lines, reverse=True):
        print(f"Deleting line {i}")
        del subtitle[i]

    if subtitle_min_font_size is not None:
        for style_name, style in subtitle.styles.items():
            if style.fontsize < subtitle_min_font_size:
                print(f"Setting font size from {style.fontsize} for {style_name}")
                style.fontsize = subtitle_min_font_size

    output_file = output_file.with_suffix("." + subtitle_format)
    subtitle.save(output_file)
    return output_file


def extract_and_sync_audio(
    video_file,
    track_id,
    output_video_duration,
    audio_shift_points,
    sync_buckets,
    delete_buckets,
    audio_output_file,
):
    audio_stream = video_file.audio_streams[track_id]
    audio_file_list = []
    segment_id = 1
    already_added_delta = 0.0
    for t1, t2, delta in sync_buckets:
        expected_delta = delta - already_added_delta
        # delay_cmd = []

        if expected_delta > 0:
            print(
                f"Adding {expected_delta=} with absolute {delta=} silence audio from {t1=}"
            )

            silence_segment_output_file = audio_output_file.with_suffix(
                f".s.{segment_id}.mkv"
            )
            audio_file_list.append(silence_segment_output_file)
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=channel_layout={audio_stream['channel_layout']}:sample_rate={audio_stream['sample_rate']}",
                "-t",
                str(expected_delta),
                "-c:a",
                audio_stream["codec_name"],
                str(silence_segment_output_file),
            ]
            subprocess.check_call(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif expected_delta < 0:
            print(f"Removing {expected_delta=} with absolute {delta=} audio from {t1=}")
            pass  # we should cut the beginning of this track and cut from t2 to tt2

        print(f"Copying audio segment {t1=} {t2=} with cut {expected_delta=}")
        segment_output_file = audio_output_file.with_suffix(f".{segment_id}.mkv")
        audio_file_list.append(segment_output_file)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_file.filepath,
            "-map",
            f"a:{track_id}",
            "-c",
            "copy",
            # ] + delay_cmd + [
            "-ss",
            str(t1),
            "-t",
            str(t2 - t1),
            str(segment_output_file),
        ]
        subprocess.check_call(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        already_added_delta += expected_delta  # TODO: measure actual length of file instead of assuming it is 100% correct.
        segment_id += 1

    input_file = audio_output_file.with_suffix(".txt")
    input_file.write_text("\n".join(f"file '{p.name}'" for p in audio_file_list))
    actual_audio_output_file = audio_output_file.with_suffix(f".mkv")
    print(f"Combining audio segments for {str(actual_audio_output_file)}")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(input_file),
        "-c",
        "copy",
        str(actual_audio_output_file),
    ]
    subprocess.check_call(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    return actual_audio_output_file


def extract_and_sync_chapters(video_file, video_duration, audio_shift_points):
    chapters = []
    for chapter in video_file.chapters:
        start_time = float(chapter["start_time"])
        title = chapter["tags"]["title"]
        for t1, t2, delta in audio_shift_points:
            if t2 < 5.0:
                t2 = 0
            if start_time >= t2:
                new_start_time = max(start_time + delta, 0.0)
                if new_start_time - 5.0 > video_duration:
                    print(
                        f"Skipping chapter {start_time=} {title=} because it floats after end"
                    )
                    break
                new_start_time = min(new_start_time, video_duration)
                if new_start_time < 3.0:
                    new_start_time = 0
                chapters.append((new_start_time, title))
                break
        else:
            print(f"Unable to find place for {start_time=} {title=}")
    return chapters


@click.command()
@click.argument("file", type=click.Path(exists=True), nargs=-1, required=True)
@click.option(
    "--only-generate-chroma", is_flag=True, help="Quit after chroma is generated"
)
@click.option(
    "--sync-using-subtitle-audio",
    is_flag=True,
    help="Extract audio from source where subtitles are and align with target. Good when video is partial or re-arranged. Bad for audio syncs.",
)
@click.option("--skip-subtitles", is_flag=True, help="Do not align subtitles.")
@click.option("--skip-shift-point", type=str, help="List of sync points to skip")
@click.option(
    "--subtitle-cutoff",
    type=float,
    help="Subtitle cutoff where everything after is removed",
)
@click.option(
    "--only-delta",
    is_flag=True,
    help="Only do delta shifts, not group alignment (warning, subtitles might overlap?)",
)
@click.option(
    "--align-framerate",
    is_flag=True,
    help="Align source framerate to target video framerate, when speedup/slowdown used as technique to change framerate.",
)
@click.option(
    "--align-frames-too",
    is_flag=True,
    help="Align using frames when the delta is discovered.",
)
@click.option(
    "--preserve-silence",
    is_flag=True,
    help="Preserve silence at the end of the video instead of trimming it",
)
@click.option(
    "--temp-folder",
    type=click.Path(),
    default="milk-temp",
    help="Temp folder to store various files in.",
)
@click.option(
    "--audio-tracks", type=TRACK_MAPPING, help="Specify audio tracks to compare with."
)
@click.option(
    "--adjust-shift-point",
    type=str,
    help="Maually adjust an audio shift point.",
    multiple=True,
)
@click.option(
    "--adjust-delay", type=float, help="Maually adjust delay."
)  # TODO: define audio track to do it to
@click.option(
    "--sync-non-dialogue-to-video",
    type=INTEGER_RANGE,
    help="Sync non-dialogue using frames instead of audio, good for e.g. remastered where they audio might be re-aligned",
)
@click.option(
    "--chapter-source",
    type=int,
    help="Input file index where to extract chapters from.",
)
@click.option(
    "--chapter-beginning",
    type=str,
    help="Name of chapter from the beginning of the file",
)
@click.option(
    "--chapter-segment-file",
    type=click.Path(exists=True),
    help="Files to try to automatically generate chapters from.",
    multiple=True,
)
@click.option(
    "--chapter-segment-name-start",
    type=str,
    help="Name start of chapter given in --chapter-segment-file, handled in same order",
    multiple=True,
)
@click.option(
    "--chapter-segment-name-end",
    type=str,
    help="Name end of chapter given in --chapter-segment-file, handled in same order",
    multiple=True,
)
@click.option(
    "--chapter-segment-required",
    is_flag=True,
    help="Error out if not all chapters are found",
)
@click.option(
    "--metadata-audio-track",
    type=METADATA_MAPPING,
    help="Set metadata for an audio track, syntax track_id=key=value",
    multiple=True,
)
@click.option(
    "--metadata-subtitle-track",
    type=METADATA_MAPPING,
    help="Set metadata for a subtitle track, syntax track_id=key=value",
    multiple=True,
)
@click.option(
    "--subtitle-min-font-size",
    type=int,
    help="Set the minimum font size",
)
@click.option(
    "--input-external-subtitle-track",
    type=click.Path(exists=True),
    help="External subtitle track, presumed to be part of first input file",
)
@click.option(
    "--output-video-file-index", type=int, help="Which file to pull video track from"
)
@click.option(
    "--output-audio-mapping",
    type=TRACK_MAPPING,
    help="Which audio tracks to include in output",
)
@click.option(
    "--output-subtitle-mapping",
    type=TRACK_MAPPING,
    help="Which subtitle tracks to include in output",
)
@click.option("--output", type=click.Path(exists=False), help="Output file.")
@click.option(
    "--output-subtitle", type=click.Path(exists=False), help="Output subtitle file."
)
def main(
    file,
    only_generate_chroma,
    sync_using_subtitle_audio,
    skip_subtitles,
    skip_shift_point,
    subtitle_cutoff,
    only_delta,
    align_framerate,
    align_frames_too,
    preserve_silence,
    temp_folder,
    audio_tracks,
    adjust_shift_point,
    adjust_delay,
    sync_non_dialogue_to_video,
    chapter_source,
    chapter_beginning,
    chapter_segment_file,
    chapter_segment_name_start,
    chapter_segment_name_end,
    chapter_segment_required,
    metadata_audio_track,
    metadata_subtitle_track,
    subtitle_min_font_size,
    input_external_subtitle_track,
    output_video_file_index,
    output_audio_mapping,
    output_subtitle_mapping,
    output,
    output_subtitle,
):
    # logging.basicConfig(level=logging.DEBUG)
    global HOP_LENGTH

    if output_video_file_index is None:
        output_video_file_index = len(file) - 1

    if output_audio_mapping is None:
        output_audio_mapping = [[output_video_file_index, 0]]

    if output_subtitle_mapping is None:
        output_subtitle_mapping = [[0, 0]]

    if skip_shift_point:
        skip_shift_point = sorted(
            [int(i) for i in skip_shift_point.split(",")], reverse=True
        )
    else:
        skip_shift_point = []

    if metadata_audio_track is None:
        metadata_audio_track = []

    if metadata_subtitle_track is None:
        metadata_subtitle_track = []

    if chapter_source is None:
        chapter_source = output_video_file_index

    if adjust_shift_point is None:
        adjust_shift_point = []

    adjust_shift_points = []
    for point in adjust_shift_point:
        point = point.split(":")
        adjust_shift_points.append(
            (int(point[0]), int(point[1]), int(point[2]), float(point[3]))
        )

    if sync_using_subtitle_audio:
        n_chroma = 36
        HOP_LENGTH = 512
    else:
        n_chroma = 12
        HOP_LENGTH = 1024

    mapped_metadata_audio_track = {}
    for track_id, key, value in metadata_audio_track:
        mapped_metadata_audio_track.setdefault(track_id, {})[key] = value

    mapped_metadata_subtitle_track = {}
    for track_id, key, value in metadata_subtitle_track:
        mapped_metadata_subtitle_track.setdefault(track_id, {})[key] = value

    chapter_segment_name_start = list(chapter_segment_name_start or [])
    chapter_segment_name_start += [""] * len(chapter_segment_file)

    chapter_segment_name_end = list(chapter_segment_name_end or [])
    chapter_segment_name_end += [""] * len(chapter_segment_file)

    chapter_segment_files = {}
    if chapter_segment_file:
        for i, csf in enumerate(chapter_segment_file):
            chapter_segment_files[Path(csf)] = (
                chapter_segment_name_start[i],
                chapter_segment_name_end[i],
            )

    temp_folder = Path(temp_folder)
    temp_folder.mkdir(exist_ok=True)
    files = [Path(f) for f in file]
    video_files = [Video(f) for f in file]
    output_video_file = video_files[output_video_file_index]
    output_video_duration = float(output_video_file.video_info["duration"])

    if subtitle_cutoff and subtitle_cutoff < 0:
        subtitle_cutoff += output_video_duration
        print(f"Setting cutoff from negative to {subtitle_cutoff}")

    sync_audio_track_mapping = [0] * len(files)
    if audio_tracks is not None:
        for file_id, audio_track_id in audio_tracks:
            sync_audio_track_mapping[file_id] = audio_track_id

    chromas = {}

    framerate_aligns = {}
    if align_framerate:
        target_framerate = output_video_file.video_info["fps"]
        for i, (f, video_file) in enumerate(zip(files, video_files)):
            if i == output_video_file_index:
                continue
            video_framerate = video_file.video_info["fps"]
            if target_framerate != video_framerate:
                framerate_aligns[i] = (video_framerate, target_framerate)

    with ThreadPoolExecutor(max_workers=8) as executor:
        jobqueue = {}
        for i, (f, audio_track_id) in enumerate(zip(files, sync_audio_track_mapping)):
            framerate_align = framerate_aligns.get(i)
            framerate_align_filename = (
                framerate_align
                and f".{str(framerate_align[0]).replace('.', '_')}-{str(framerate_align[1]).replace('.', '_')}"
                or ""
            )
            preserve_silence_filename = preserve_silence and ".ps" or ""

            audio_chroma_output_file = temp_folder / (
                f.stem
                + f"{framerate_align_filename}{preserve_silence_filename}.{HOP_LENGTH}.{n_chroma}.{audio_track_id}.chroma"
            )
            if audio_chroma_output_file.exists():
                print(f"Loading chroma from {audio_chroma_output_file}")
                chromas[f] = pickle.loads(audio_chroma_output_file.read_bytes())
            else:
                print(f"Extracting chroma from {f.name}")
                future = executor.submit(
                    generate_chroma_cqt,
                    f,
                    sync_audio_track_mapping[i],
                    audio_chroma_output_file,
                    n_chroma=n_chroma,
                    framerate_align=framerate_align,
                    preserve_silence=preserve_silence,
                )
                jobqueue[future] = f

        for f in chapter_segment_files:
            preserve_silence_filename = preserve_silence and ".ps" or ""
            audio_chroma_output_file = temp_folder / (
                f.stem + f"{preserve_silence_filename}.{HOP_LENGTH}.{n_chroma}.c.chroma"
            )
            if audio_chroma_output_file.exists():
                print(f"Loading chroma from {audio_chroma_output_file}")
                chromas[f] = pickle.loads(audio_chroma_output_file.read_bytes())
            else:
                print(f"Extracting chroma from {f.name}")
                future = executor.submit(
                    generate_chroma_cqt,
                    f,
                    0,
                    audio_chroma_output_file,
                    n_chroma=n_chroma,
                    preserve_silence=preserve_silence,
                )
                jobqueue[future] = f

        for future in concurrent.futures.as_completed(jobqueue):
            chromas[jobqueue[future]] = future.result()

    if only_generate_chroma:
        print("Done generating chroma")
        quit(0)

    x_2_chroma, fs = chromas[
        files[output_video_file_index]
    ]  # x_2_chroma is always target video, the one we align everything with

    chapter_timestamps = []
    for f, (start_name, end_name) in chapter_segment_files.items():
        print(f"Looking for chapter matching {f}")
        x_1_chroma, fs = chromas[f]
        chapter_specifications = find_and_align_chapter(
            x_1_chroma, x_2_chroma, fs, min_match_value=60 + (n_chroma * 3)
        )
        if chapter_specifications is None:
            if chapter_segment_required:
                print("Did not find required chapters")
                quit(1)
            else:
                continue

        chapter_timestamps.append((chapter_specifications[0], start_name))
        chapter_timestamps.append((chapter_specifications[1], end_name))

    if chapter_timestamps:
        min_chapter_delay = 4.0
        previous_chapter_time = None

        new_chapter_timestamps = []
        for chapter_timestamp, name in sorted(chapter_timestamps):
            if chapter_timestamp < 3.0:
                chapter_timestamp = 0.0
            if (
                previous_chapter_time is not None
                and chapter_timestamp - previous_chapter_time < min_chapter_delay
            ):
                print(
                    f"Skipping chapter '{name}' because it is too close to previous chapter"
                )
                continue

            if output_video_duration - chapter_timestamp < min_chapter_delay:
                print(
                    f"Skipping chapter '{name}' because it is too close to end of video"
                )
                continue

            previous_chapter_time = chapter_timestamp
            new_chapter_timestamps.append((chapter_timestamp, name))

        chapter_timestamps = new_chapter_timestamps

        if chapter_beginning and chapter_timestamps[0][0] > 0.0:
            print("Injecting chapter at beginning")
            chapter_timestamps.insert(0, (0.0, chapter_beginning))

        print("Found chapters:")
        for chapter_timestamp, name in chapter_timestamps:
            print(f" {humanize_seconds(chapter_timestamp)} - {name}")

    attachment_files = set()
    subtitle_files = []
    audio_files = []

    for i, (f, video_file) in enumerate(zip(files, video_files)):
        if i == output_video_file_index:
            continue

        x_1_chroma, fs = chromas[f]
        if sync_using_subtitle_audio:
            for j, (file_index, track_id) in enumerate(output_subtitle_mapping):
                if file_index != i:
                    continue
                break
            else:
                print(
                    "No subtitle track found to sync with, please specify a mapping to use this feature"
                )
                quit(1)

            (
                audio_shift_points,
                sync_buckets,
                delete_buckets,
            ) = estimate_audio_shift_points_from_subtitles(
                x_1_chroma,
                x_2_chroma,
                fs,
                video_file,
                j,
                n_chroma,
                adjust_delay=adjust_delay,
                framerate_align=framerate_aligns.get(i),
                external_subtitle_file=input_external_subtitle_track,
            )
        else:
            audio_shift_points = estimate_audio_shift_points(
                x_1_chroma,
                x_2_chroma,
                fs,
                only_delta=only_delta,
                adjust_delay=adjust_delay,
                sliding_window_size=300,
            )

            for skip_point in skip_shift_point:
                if len(audio_shift_points) - 1 >= skip_point:
                    print(f"Skipping shift point {audio_shift_points[skip_point]}")
                    del audio_shift_points[skip_point]

            for point in adjust_shift_points:
                if point[0] != i:
                    continue
                print(f"Changing shift point {point=} {audio_shift_points[point[1]]}")
                p = list(audio_shift_points[point[1]])
                p[point[2]] = point[3]
                audio_shift_points[point[1]] = tuple(p)
                print(audio_shift_points)

            sync_buckets, delete_buckets = turn_audio_shift_points_to_audio_segments(
                audio_shift_points
            )
        print(
            f"Found audio shift points {audio_shift_points=} {sync_buckets=} {delete_buckets=}"
        )

        if align_frames_too:
            print("Frame aligning buckets")
            sync_buckets = [
                frame_align_sync_bucket(video_file, output_video_file, sync_bucket)
                for sync_bucket in sync_buckets
            ]

        if not skip_subtitles:
            for j, (file_index, track_id) in enumerate(output_subtitle_mapping):
                if file_index != i:
                    continue

                subtitle_output_file = temp_folder / (f.stem + f".{track_id}.unknown")
                subtitle_output_file = extract_and_sync_subtitles(
                    video_file,
                    track_id,
                    output_video_duration,
                    only_delta,
                    audio_shift_points,
                    sync_buckets,
                    delete_buckets,
                    subtitle_output_file,
                    subtitle_cutoff,
                    sync_non_dialogue_to_video,
                    output_video_file,
                    framerate_align=framerate_aligns.get(i),
                    external_subtitle_file=input_external_subtitle_track,
                    subtitle_min_font_size=subtitle_min_font_size,
                )

                if input_external_subtitle_track:
                    subtitle_files.append((j, subtitle_output_file, []))
                elif subtitle_output_file:
                    subtitle_metadata = video_file.extract_subtitle_metadata(track_id)
                    subtitle_files.append((j, subtitle_output_file, subtitle_metadata))
                    attachment_files.add(i)

        for j, (file_index, track_id) in enumerate(output_audio_mapping):
            if file_index != i:
                continue

            audio_output_file = temp_folder / (f.stem + f".{track_id}.unknown")
            audio_output_file = extract_and_sync_audio(
                video_file,
                track_id,
                output_video_duration,
                audio_shift_points,
                sync_buckets,
                delete_buckets,
                audio_output_file,
            )

            if audio_output_file:
                audio_metadata = video_file.extract_audio_metadata(track_id)
                audio_files.append((j, audio_output_file, audio_metadata))

        if not chapter_timestamps and i == chapter_source:
            chapter_timestamps = extract_and_sync_chapters(
                video_file, output_video_duration, audio_shift_points
            )

    if output_subtitle:
        if subtitle_files:
            subtitle_file = subtitle_files[0][1]
            subtitle_output_file = Path(output_subtitle).with_suffix(
                subtitle_file.suffix
            )
            print(f"Wrote the first subtitle track to {subtitle_output_file}")
            shutil.copy(subtitle_file, subtitle_output_file)
        else:
            print("No subtitle file to output, skipping")

    if not output:
        print("No output defined, quitting here")
        quit(1)

    print("Combining everything")
    temp_output_file = temp_folder / (f.stem + f".temp.mkv")

    ffmpeg_inputs = []
    ffmpeg_options = []
    subtitle_track_count = 0
    audio_track_count = 0

    for f in files:
        ffmpeg_inputs.append(str(f))

    for i, (f, video_file) in enumerate(zip(files, video_files)):
        if i != output_video_file_index:
            continue

        if not skip_subtitles:
            for (
                file_index,
                track_id,
            ) in output_subtitle_mapping:  # TODO, inject at correct point
                if file_index != i:
                    continue
                ffmpeg_options += ["-map", f"{i}:s:{track_id}", "-c", "copy"]
                subtitle_track_count += 1
                attachment_files.add(i)

        for (
            file_index,
            track_id,
        ) in output_audio_mapping:  # TODO, inject at correct point
            if file_index != i:
                continue
            ffmpeg_options += ["-map", f"{i}:a:{track_id}", "-c", "copy"]

            for tag_key, tag_value in mapped_metadata_audio_track.get(0, {}).items():
                ffmpeg_options += [f"-metadata:s:a:{0}", f"{tag_key}={tag_value}"]
            audio_track_count += 1

    for i in sorted(attachment_files):
        ffmpeg_options += [
            "-map",
            f"{i}:d?",
            "-c",
            "copy",
            "-map",
            f"{i}:t?",
            "-c",
            "copy",
        ]

    ffmpeg_options += ["-map", f"{output_video_file_index}:v", "-c", "copy"]
    if not chapter_timestamps:
        ffmpeg_options += ["-map_chapters", str(chapter_source)]
    else:
        ffmpeg_options += ["-map_chapters", "-1"]

    for j, subtitle_file, subtitle_metadata in sorted(subtitle_files):
        input_index = len(ffmpeg_inputs)
        ffmpeg_options += ["-map", f"{input_index}:s", "-c", "copy"]
        seen_tag_keys = set()
        for tag_key, tag_value in subtitle_metadata:
            seen_tag_keys.add(tag_key)
            tag_value = mapped_metadata_subtitle_track.get(j, {}).get(
                tag_key, tag_value
            )
            ffmpeg_options += [
                f"-metadata:s:s:{subtitle_track_count}",
                f"{tag_key}={tag_value}",
            ]
        for tag_key, tag_value in mapped_metadata_subtitle_track.get(j, {}).items():
            if tag_key in seen_tag_keys:
                continue
            ffmpeg_options += [
                f"-metadata:s:s:{subtitle_track_count}",
                f"{tag_key}={tag_value}",
            ]
        ffmpeg_inputs.append(str(subtitle_file))
        subtitle_track_count += 1

    for j, audio_file, audio_metadata in sorted(audio_files):
        input_index = len(ffmpeg_inputs)
        ffmpeg_options += ["-map", f"{input_index}:a", "-c", "copy"]
        seen_tag_keys = set()
        for tag_key, tag_value in audio_metadata:
            seen_tag_keys.add(tag_key)
            tag_value = mapped_metadata_audio_track.get(j, {}).get(tag_key, tag_value)
            ffmpeg_options += [
                f"-metadata:s:a:{audio_track_count}",
                f"{tag_key}={tag_value}",
            ]
        for tag_key, tag_value in mapped_metadata_audio_track.get(j, {}).items():
            if tag_key in seen_tag_keys:
                continue
            ffmpeg_options += [
                f"-metadata:s:a:{audio_track_count}",
                f"{tag_key}={tag_value}",
            ]
        # for tag_key, tag_value in audio_metadata:
        #     tag_value = mapped_metadata_audio_track.get(j, {}).get(tag_key, tag_value)
        #     ffmpeg_options += [f"-metadata:s:a:{audio_track_count}", f"{tag_key}={tag_value}"]
        ffmpeg_inputs.append(str(audio_file))
        audio_track_count += 1

    if output_subtitle_mapping:
        ffmpeg_options += ["-disposition:s:0", "default"]

    cmd = ["ffmpeg", "-y"]
    for ffmpeg_input in ffmpeg_inputs:
        cmd += ["-i", ffmpeg_input]
    cmd += ffmpeg_options
    cmd += [str(temp_output_file)]
    print(f"Running: {shlex.join(cmd)}")
    subprocess.check_call(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    cmd = ["mkvmerge", "--no-global-tags", "-o", str(output)]

    if chapter_timestamps:
        chapter_file = temp_folder / (f.stem + f".chapters.txt")
        chapter_file.write_text(
            "\n".join(
                f"CHAPTER{i:02}={humanize_seconds(chapter_timestamp)}\nCHAPTER{i:02}NAME={name}"
                for (i, (chapter_timestamp, name)) in enumerate(chapter_timestamps)
            )
        )
        cmd += ["--chapters", str(chapter_file)]

    cmd += [str(temp_output_file)]
    print(f"Creating final MKV file: {output}")
    subprocess.check_call(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    os.unlink(temp_output_file)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
