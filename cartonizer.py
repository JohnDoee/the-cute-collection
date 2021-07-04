#!/usr/bin/env python

import re
import shlex
import sys
from pathlib import Path
from pprint import pprint

import click
import ffmpeg
import guessit

KNOWN_SUBTITLE_EXTENSIONS = [".ass"]
KNOWN_EXTENSIONS = [".mp4", ".mkv", ".ogm", ".avi"]
VIDEO_MAPPING = {
    ("hevc", "Main 10"): "HEVC 10-bit",
    ("hevc", "Rext"): "HEVC 12-bit",
    ("h264", "High"): "h264",
    ("h264", "Main"): "h264",
    ("h264", "High 10"): "h264 10-bit",
}
VIDEO_RESOLUTION_MAPPING = {
    1080: "1080p",
    1088: "1080p",
}
AUDIO_MAPPING = {"flac": "FLAC", "aac": "AAC", "dts": "DTS-HDMA", "ac3": "AC3"}
SOURCE_MAPPING = {"Blu-ray": "BD", "DVD": "DVD"}

OP_CHAPTER_NAMES = ["OP", "Episode"]
ED_CHAPTER_NAMES = ["ED", "Preview"]

CHROMA_GENERATE_PARAM = "--only-generate-chroma"


def map_episode_files(paths):
    episode_mapping = {}
    for path in paths:
        for f in path.iterdir():
            if not f.is_file():
                continue
            if not f.suffix.lower() in KNOWN_EXTENSIONS:
                continue
            info = guessit.guessit(f.name)
            episode = info.get("episode")
            if episode is None:
                episode = info.get("episode_title")
                if episode is not None:
                    episode = int(episode.split(" ")[0].split("v")[0])

            if episode is None:
                re_episode = re.findall("第(\d+)話", f.name)
                if not re_episode:
                    continue
                episode = int(re_episode[0])
            if isinstance(episode, list):
                episode = episode[-1]
            episode_mapping.setdefault(episode, []).append(f)
    return episode_mapping


@click.group()
def cli():
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True), nargs=-1, required=True)
@click.option("--input-subtitle-path", type=click.Path(exists=True))
@click.option("--op-ed-path", type=click.Path(exists=True), multiple=True)
@click.option("--group", type=str)
@click.option("--source", type=str)
@click.option("--audio", type=str)
@click.option("--title", type=str)
@click.option("--dual-audio", is_flag=True)
@click.option("--skip-chapters", is_flag=True)
@click.option("--pre-generate-chroma", is_flag=True)
@click.option("--skip-copy-oped", is_flag=True)
@click.option("--additional-params", type=str)
@click.option("--folder-name", type=str)
@click.option("--file-name-template", type=str)
@click.option("--output-subtitles-path", type=click.Path())
def sync(
    path,
    input_subtitle_path,
    op_ed_path,
    group,
    source,
    audio,
    title,
    dual_audio,
    skip_chapters,
    pre_generate_chroma,
    skip_copy_oped,
    additional_params,
    folder_name,
    file_name_template,
    output_subtitles_path,
):
    command_path = (
        f"{sys.executable} {(Path(__file__).parent / 'milksync.py').absolute()}"
    )

    if output_subtitles_path:
        output_subtitles_path = Path(output_subtitles_path)
        output_subtitles_path.mkdir(parents=True, exist_ok=True)

    external_subtitles = {}
    if input_subtitle_path is not None:
        for s in Path(input_subtitle_path).iterdir():
            if s.suffix.lower() not in KNOWN_SUBTITLE_EXTENSIONS:
                continue
            external_subtitles[s.stem] = s

    paths = [Path(p) for p in path]
    if op_ed_path:
        op_ed_paths = [Path(p) for p in op_ed_path]
    else:
        op_ed_paths = []

    episode_mapping = map_episode_files(paths)
    first_episode = sorted(episode_mapping.items())[0][1]
    probe_result = ffmpeg.probe(first_episode[-1])
    release_name = {
        "show_name": title or guessit.guessit(first_episode[0].name)["title"],
    }
    if source:
        release_name["source"] = source
    else:
        release_name["source"] = SOURCE_MAPPING[
            guessit.guessit(first_episode[-1].name)["source"]
        ]
    if audio:
        release_name["audio"] = audio
    for stream in probe_result["streams"]:
        if stream["codec_type"] == "video" and "video" not in release_name:
            key = (stream["codec_name"], stream["profile"])
            if key not in VIDEO_MAPPING:
                click.echo(f"Unknown video key {key=}")
                quit(1)
            release_name["video"] = VIDEO_MAPPING[key]
            release_name["video_resolution"] = VIDEO_RESOLUTION_MAPPING.get(
                stream["coded_height"],
                f"{stream['coded_width']}x{stream['coded_height']}",
            )
        elif stream["codec_type"] == "audio" and "audio" not in release_name:
            key = stream["codec_name"]
            if key not in AUDIO_MAPPING:
                click.echo(f"Unknown audio key {key=}")
                quit(1)
            release_name["audio"] = AUDIO_MAPPING[key]

    if not folder_name:
        folder_name = f"{group and '[' + group + '] ' or ''}{release_name['show_name']} ({release_name['source']} {release_name['video_resolution']} {release_name['video']} {release_name['audio']}{dual_audio and ' Dual-Audio' or ''})"
    if not file_name_template:
        file_name_template = f"{group and '[' + group + '] ' or ''}{release_name['show_name']} - %s ({release_name['source']} {release_name['video_resolution']} {release_name['video']} {release_name['audio']}{dual_audio and ' Dual-Audio' or ''})"
    click.echo(f"Folder name: {folder_name}")
    click.echo(f"File name template: {file_name_template}")

    copy_files = []

    endings = []
    openings = []
    for op_ed_path in op_ed_paths:
        for f in op_ed_path.iterdir():
            if "NCOP" in f.name:
                click.echo(f"Found OP {f.name}")
                openings.append(f)
            elif "NCED" in f.name:
                click.echo(f"Found ED {f.name}")
                endings.append(f)

    op_ed_chapter_command = []
    if openings or endings:
        for i, opening in enumerate(sorted(openings, key=lambda f: f.name), 1):
            if not skip_chapters:
                op_ed_chapter_command.append(
                    f"--chapter-segment-file '{str(opening)}' --chapter-segment-name-start '{OP_CHAPTER_NAMES[0]}' --chapter-segment-name-end '{OP_CHAPTER_NAMES[1]}'"
                )
            name = "NCOP"
            if len(openings) > 1:
                name += str(i)
            if not skip_copy_oped:
                copy_files.append(
                    (str(opening), f"{folder_name}/{file_name_template % name}.mkv")
                )

        for i, ending in enumerate(sorted(endings, key=lambda f: f.name), 1):
            if not skip_chapters:
                op_ed_chapter_command.append(
                    f"--chapter-segment-file '{str(ending)}' --chapter-segment-name-start '{ED_CHAPTER_NAMES[0]}' --chapter-segment-name-end '{ED_CHAPTER_NAMES[1]}'"
                )
            name = "NCED"
            if len(endings) > 1:
                name += str(i)
            if not skip_copy_oped:
                copy_files.append(
                    (str(ending), f"{folder_name}/{file_name_template % name}.mkv")
                )

    op_ed_chapter_command = "".join([f"  {cmd} \\\n" for cmd in op_ed_chapter_command])
    episode_num_length = max(max(len(str(k)) for k in episode_mapping.keys()), 2)

    output_file = []

    chroma_files = []
    for files in episode_mapping.values():
        for f in files:
            chroma_files.append(f"'{str(f)}'")
    if pre_generate_chroma:
        output_file.append("echo 'Generating chroma'")
        output_file.append(
            f"{command_path} {CHROMA_GENERATE_PARAM} {' '.join(chroma_files)}"
        )
    output_file.append(f"mkdir -p '{folder_name}'")
    if additional_params:
        additional_params = f"  {additional_params} \\\n"
    else:
        additional_params = ""

    for episode, files in sorted(episode_mapping.items()):
        if len(files) < 2:
            click.echo(f"Skipping episode {episode}")
            continue
        external_subtitle = ""
        if files[0].stem in external_subtitles:
            external_subtitle = f"  --input-external-subtitle-track {shlex.quote(str(external_subtitles[files[0].stem]))} \\\n"
        output_file.append("echo ''")
        output_file.append(f"echo 'Handling episode {episode}'")
        if output_subtitles_path:
            output_subtitles = f"  --output-subtitle {shlex.quote(str(output_subtitles_path / files[-1].with_suffix('.subtitle').name))} \\\n"
        else:
            output_subtitles = ""
        files = "".join([f"  {shlex.quote(str(f))} \\\n" for f in files])
        output_file.append(
            f"{command_path} \\\n{files}{op_ed_chapter_command}{external_subtitle}{additional_params}{output_subtitles}  --output '{folder_name}/{file_name_template % str(episode).zfill(episode_num_length)}.mkv'"
        )
    output_file.append("echo ''")
    output_file.append("echo 'Copying files'")
    for (src_f, dst_f) in copy_files:
        if not src_f.lower().endswith(".mkv"):
            click.echo("Copy file is not an mkv")
            quit(1)
        output_file.append(f"cp {shlex.quote(src_f)} {shlex.quote(dst_f)}")

    Path("create_release.sh").write_text("\n".join(output_file))
    click.echo("Release file created, run: bash create_release.sh")


@cli.command()
@click.argument("subbed_path", type=click.Path(exists=True), required=True)
@click.argument("unsubbed_path", type=click.Path(exists=True), required=True)
@click.option("--additional-params", type=str)
def ocr(
    subbed_path,
    unsubbed_path,
    additional_params,
):
    paths = [Path(subbed_path), Path(unsubbed_path)]
    command_path = (
        f"{sys.executable} {(Path(__file__).parent / 'cowocr.py').absolute()}"
    )

    output_file = []

    episode_mapping = map_episode_files(paths)

    for episode, files in sorted(episode_mapping.items()):
        if len(files) < 2:
            click.echo(f"Skipping episode {episode}")
            continue
        output_file.append("echo ''")
        output_file.append(f"echo 'Handling episode {episode}'")
        files = "".join([f"  {shlex.quote(str(f))} \\\n" for f in files])
        output_file.append(
            f"{command_path} \\\n{files}  extract-subtitles \\\n  {additional_params or ''}"
        )
        output_file.append(f"{command_path} \\\n{files}  create-report")

    output_file.append("")
    Path("ocr_release.sh").write_text("\n".join(output_file))
    print("OCR script file created, run: bash ocr_release.sh")


if __name__ == "__main__":
    cli()
