# The Cute Collection

This is a bunch of tools to sync subtitles and audio tracks automatically.

## Requirements

* Linux
* 10GB of available memory
* ffmpeg and ffprobe in environment path
* Python 3.8+ (might work with earlier, not tested though)
* Tesseract in environment path if you want to OCR

## Installation instructions

```bash
cd ~
git clone https://github.com/JohnDoee/the-cute-collection.git the-cute-collection
cd the-cute-collection

python3 -m venv .env

.env/bin/pip install -U setuptools pip wheel
.env/bin/pip install ffmpeg-python click guessit opencv-python librosa pysubs2 scikit-image jinja2 lxml tqdm pyxdameraulevenshtein textblob jinja2 pytesseract lxml
```

## Cartonizer

Generate a script for milksync to do bulk operations instead of one by one.

### How to use for automatic subtitle sync (milksync)

The most basic usage is:

`~/the-cute-collection/.env/bin/python ~/the-cute-collection/cartonizer.py sync path-to-subbed path-to-unsubbed`

Make sure the files are correctly matched, sometimes it takes multiple files it should not.

This will generate a bash script called `create_release.sh`, you just have to run it with `bash create_release.sh` and wait.

A description of all arguments and how and when to use them.

#### --op-ed-path

Look for OP and ED in the given path and use them to auto-generate chapters and copy them to the result path.

Example: `--op-ed-path Unsubbed-Files/NC-OP-ED-Folder/` - looks for OP ED in the specified folder.

#### --group

Put a group name in the result folder name.

Example: `--group Horse` - Sets group name to Horse and prefixes the folder name and files with it.

#### --source

Specify the source of the video track, e.g. BD for bluray. Will be auto-detected if not specified.

Example: `--source VHS` - sets the source to the text string VHS.

#### --audio

Same as source but for the audio track.

Example: `--audio Opus` - sets the audio source to the text string Opus.

#### --title

Sets the title of the release.

Example: `--title Big Buck Bunny` - sets the title to the text string Big Buck Bunny.

#### --dual-audio

Marks the release as Dual-Audio

Example: `--dual-audio` - Adds the text Dual-Audio to the release.

#### --skip-chapters

Skips adding chapters based on OP-ED specified with `--op-ed-path`.
This is useful if you want to copy the NC-OP-ED files but copy the chapters from a release.

Example: `--skip-chapters` - Instructs milksync to not assign chapters from OP & ED.

#### --pre-generate-chroma

Pre-generate chromas, this can sometimes speed up the total speed but is not recommended

Example: `--pre-generate-chroma` - Adds a line to the script that pre-generates chromas.

#### --skip-copy-oped

Skips copying OP-ED specified with `--op-ed-path`. This is useful if you created the files yourself just to assign the chapters.

Example: `--skip-copy-oped` - Cartonizer does not add the line to copy the files to the release folder.

#### --additional-params

Pass additional arguments to `milksync.py`.

Example: `--additional-params '--chapter-beginning Intro'` - Tells milksync to add a chapter to the beginning fo the file.

See milksync arguments for more.

#### --folder-name

Instead of auto-generating a foldername, use this name.

Example: `--folder-name 'Happy Bunnies Riding The Wave (DVD)'`

#### --file-name-template

Instead of auto-generating a file name template, use this template.
Must have a %s where the episode number is placed.

Example: `--folder-name 'Happy Bunnies Riding The Wave (DVD) %s'`

### How to use for ocr (cowocr)

The most basic usage is:

`~/the-cute-collection/.env/bin/python ~/the-cute-collection/cartonizer.py ocr path-to-subbed path-to-unsubbed`

Make sure the files are correctly matched, sometimes it takes multiple files it should not.

This will generate a bash script called `ocr_release.sh`, you just have to run it with `bash ocr_release.sh` and wait.

A description of all arguments and how and when to use them.

#### --additional-params

Pass additional arguments to `cowocr.py`.

Example: `--additional-params '--threads 1 --run-subregions-in-parallel'` - Tells cowocr to use 2 threads and run every subregion in parallel.

See cowocr arguments for more.

### FAQ

#### There is no OP/ED to assign chapters from (or it fails to use existing OP/ED) what do I do?

The easiest way right now is to extract them manually, if you have a file named `Big Buck Bunny 01.mkv` and the chapters are like this in the file:

* Opening: starts at 00:01:27.062 and stops at 00:02:57.123
* Ending: starts at 00:22:11.362 and stops at 00:23:33.333

Extract them with:
```
mkdir extracted
ffmpeg -i 'Big Buck Bunny 01.mkv' -ss 00:01:27.062 -to 00:02:57.123 -map a:0 extracted/NCOP-01.mkv
ffmpeg -i 'Big Buck Bunny 01.mkv' -ss 00:22:11.362 -to 00:23:33.333 -map a:0 extracted/NCED-01.mkv
```

These can then be used with `--op-ed-path extracted/ --skip-copy-oped`

## Milksync

Compare audio tracks between two files and take subtitles and audio tracks from one and add to another. The intention is to remove the tedious work of manually aligning subtitles to a new files and give a far more exact
result.

### How to use

The most basic usage is:

`~/the-cute-collection/.env/bin/python ~/the-cute-collection/milksync.py path-to-subbed/episode-01.mkv path-to-unsubbed/episode-01.mkv --output merged-episode-01.mkv`

This will take video and audio from the last file and put subtitles from the first file into the merged file.

The command prints out information about what is going on, e.g. where chapters are placed and how much subtitles are moved.
Make sure to check the result, especially around the breakpoints. WARNINGs can also be a hint about what might be wrong with the resulting file.

Remember, you can always modify a command and just run it again to see what happens. Second time is normally faster than the first too.
Sometimes experimenting can help you on your way.

A description of all arguments and how and when to use them.

#### --only-generate-chroma

Only extract audio from the file and generate index, this can sometimes be used to speed up the overall progress, not recommended.

Example: `--only-generate-chroma` - Quits after extracting chroma

#### --sync-using-subtitle-audio

Use the audio where the subtitles run to sync a specific line. Good when video is partial or re-arranged. Bad for audio syncs.

Example: `--sync-using-subtitle-audio` - Enable the sync feature.

#### --skip-subtitles

Do not copy any subtitles, can be used for e.g. dubs only releases or subtitles from another source that are not to be synced this way.

Example: `--skip-subtitles` - No subtitles copied

#### --skip-shift-point

The script prints out the points it uses to shift the subtitles, sometimes one or more of them might be bad or you want to see what happens with them removed. They are index based and you have to count it yourself from the milksync output.

Generally not used.

Example: `--skip-shift-point 2,3` - Skips shift point 2 and 3.

#### --subtitle-cutoff

If the subtitles start too early or run too long, this command can cut off subtitles to prevent this. The command takes a number in seconds that can be both positive (count from beginning of video result file) and negative (count from end of video result file)

Example: `--subtitle-cutoff -50` - The last 50 second of the result will not have any subtitles.
Example: `--subtitle-cutoff 30` - The first 30 second of the result will not have any subtitles.

#### --only-delta

Instead of putting subtitles into buckets and adjusting them to fit in it, just modify the timestamp on the subtitles.
This one is very useful if one input runs faster or slower than the other. This can often be seen in the milksync output as a lot of sync points that either decrease or increase in delta.

Example: `--only-delta` - Enable delta mode instead of subtitle bucket mode.

#### --align-framerate

Align source framerate to target video framerate, when speedup/slowdown used as technique to change framerate.

Example: `--align-framerate` - Enable the feature and change source framerate to target framerate.

#### --align-frames-too

When using `--only-delta` it can be helpful to look at frames too to find a better difference.

Example: `--only-delta --align-frames-too` - Enables frame alignment.

#### --preserve-silence

When extracting chroma from the files the audio at the end is trimmed to prevent silence from blocking alignment, this disables that feature.

Example: `--preserve-silence` - Preserves silence.

#### --temp-folder

Where to save temporary files, this includes extracted audio and subtitle tracks including chroma generated from audio files.

Example: `--temp-folder '/tmp/milk-temp/'` - Saves temp files to the specified folder

#### --audio-tracks

Define which audio tracks to use for syncing audio tracks. Milksync only works when the audio tracks in the input files are the same, i.e. same language. If you take e.g. english and japanese audio tracks and try to use then the results will vary quite a bit and likely not be very good.

Input file index is defined as the order they are given to milksync.

Example: `--audio-tracks 0:1,1:0` - Use audio track 1 from input file 0 and audio track 0 from input file 1.

#### --adjust-shift-point

Manually change a shift point. Can be used to see if the auto detect is not good enough or just modify it to work correctly. This is mostly used for debugging.

Example: `--adjust-shift-point 0.3:10.3:1.3:11.3` - Set the first shift point to the specified values. Order is same as printed in milksync.

#### --adjust-delay

Manually adjust the delay to all points. Can be used for debugging.

Example: `--adjust-delay 0.3` - Adds 0.3 second to every subtitle

#### --sync-non-dialogue-to-video

Sometimes the audio has been resynced to the video which means the speech subtitles and the sign subtitles must be synced independently.
This flag tries to align signs to the video and speech to the audio, can be useful when the target is e.g. remastered. It can be very slow and quality can vary, the result is printed and you should check if the signs are positioned correctly.

Example: `--sync-non-dialogue-to-video 0-1000` - Enables this feature for the given range of seconds.

#### --chapter-source

Specify which source file index to pull chapters from, these are synced in the same way as the audio tracks.

If nothing chapter-related is specified, they are pulled from video source, i.e. last file.

Example: `--chapter-source 0` - Take chapters from input file 0.

#### --chapter-beginning

Add a chapter to the beginning of the result. This means every part of the result will be part of a chapter.

Example: `--chapter-beginning Beginning` - The first chapter at 00:00 is named Beginning.

#### --chapter-segment-file

Source file to generate chapter from, this is a part of the video that is sought for in the file. Useful for e.g. openings or endings.

This is used in conjunction with `--chapter-segment-name-start` and `--chapter-segment-name-end`. Order matters and each `--chapter-segment-file` must have a `--chapter-segment-name-start` and `--chapter-segment-name-end`.

Example: `--chapter-source NCED-01.mkv` - Match content of NCED-01.mkv to the result video and add chapters if found.

#### --chapter-segment-name-start

Name of the chapter starting where the beginning of `--chapter-segment-file` is matched.

Example: `--chapter-segment-file End` - Names the chapter that matches the beginning of `--chapter-segment-file` End.

#### --chapter-segment-name-end

Name of the chapter starting where the end of `--chapter-segment-file` is matched.

Example: `--chapter-segment-file 'After End'` - Names the chapter that matches the end of `--chapter-segment-file` After End.

#### --chapter-segment-required

Enforces that every chapter segment must be matched.

Example: `--chapter-segment-required` - If a chapter segment is not matched, it will quit with an error.

#### --metadata-audio-track

Manually set metadata for an audio track, this is passed directly to ffmpeg. These matches the output mapping and no the input mapping.

Example: `--metadata-audio-track 0=language=jpn --metadata-audio-track 0=title='Japanese' --metadata-audio-track 1=language=fra --metadata-audio-track 1=title='Bad french'` - Sets the first output audio track metadata to japanese with a matching title and the second audio track to french with a matching title.

#### --metadata-subtitle-track

Manually set metadata for a subtitle track, this is passed directly to ffmpeg. These matches the output mapping and no the input mapping.

Example: `--metadata-subtitle-track 0=language=jpn --metadata-subtitle-track 0=title='Japanese' --metadata-subtitle-track 1=language=fra --metadata-audsubtitleio-track 1=title='Bad french'` - Sets the first output subtitle track metadata to japanese with a matching title and the second subtitle track to french with a matching title.

#### --subtitle-min-font-size

Increase font-size to minimum this. Sometimes subtitles are unreadable on the source.

Example: `--subtitle-min-font-size 26` - Sets the font-size to, minimum, 26.

#### --input-external-subtitle-track

Use a specific external subtitle in output, it is assumed it matches video input 0.

Example `--input-external-subtitle-track subtitles.ass` - Assumes the subtitle matches input 0 and syncs it to output.

#### --output-video-file-index

Which file to pull video data from, this is normally the last specified file and is normally not used.

Example: `--output-video-file-index 1` - Pull video data from the second input file.

#### --output-audio-mapping

Define which audio tracks the output has and where to pull them from. Defaults to using only first audio from the last input file, same source as video.

Example: `--output-audio-mapping 0:0,1:2` - Takes the first audio track from the first input file and the third audio track from the second input file. The result file first audio track is 0:0 and the second is 1:2.

#### --output-subtitle-mapping

Define which subtitle tracks the output has and where to pull them from. Defaults to using only first subtitle from the first input file.

Example: `--output-subtitle-mapping 1:1,1:0` - Takes the first and the second subtitle track from the second input file. The order is as specified, i.e. the tracks are flipped.

#### --output

Where to save the result.

Example: `--output Result-EP01.mkv` - Saves the complete file to Result-EP01.mkv

#### --output-subtitle

Save the synced subtitles.

Example: `--output-subtitle Result-EP01.ass` - Saves the subtitle file to Result-EP01.ass

## CowOCR

Compare two video tracks and look for differences. The intention is to find differences as they will indicate e.g. subititles and signs.

The output is an .ass file and a report that can be used to verify and correct the output.

### How it works

The base assumtion that CowOCR relies on is to find the differences between the source and destination video. To do this it goes through a few steps.

The initial differences are found by running ORB algorithm against both source and target video, keypoints found at source not found at target is assumed to be differences.

We now have a region we can assume is different, we look for text in that one. Threshold algorithm is run against the source and matching white areas are extracted.

To make sure what is part of the text the color of all found area is extracted and grouped using k-means. Areas with colors close enough to the majority color found are considered part of the text. Additionally the border color is used in the same way.

A bruteforce is performed here to find the best text mask by cycling through the colors.

With text found and a mask matching the text (where it is in the picture) it is now time to figure out when it starts and ends. This is done by looping through the frames before and after the current frame and see if the colors match the extracted text, i.e. is the same text in the frames before and after the current frame.

### How to use

The most basic usage to extract-subtitles is:

`~/the-cute-collection/.env/bin/python ~/the-cute-collection/cowocr.py path-to-subbed/episode-01.mkv path-to-unsubbed/episode-01.mkv extract-subtitles`

This will compare the two video files and try to extract the subtitles.

After the subtitles are extracted, a report plus an .ass file can be created from the output with the create-report command.

`~/the-cute-collection/.env/bin/python ~/the-cute-collection/cowocr.py path-to-subbed/episode-01.mkv path-to-unsubbed/episode-01.mkv create-report`

The report and subtitle is, per default, located in the cow-temp folder which is created relative to where the command was executed.
In this example, in the folder that contains the path-to-subbed and path-to-unsubbed folder.

Verify the subtitles and we're all done, now we just need them merged. For this we can use milksync with just one additional parameter, `--input-subtitle-path cow-temp/` - that will pull the subtitles from the .ass file instead of the source video file.

This is likely not how the actual workflow will be. See further down for an actual workflow.

A description of all arguments and how and when to use them.

### extract-subtitles arguments

This command extracts the subtitles from the video

#### --threads

How many threads to extract subtitles with. Unless specified it runs a subtitle region at a time.

Example: `--threads 1` - Use only one thread

#### --tesseract-data-path

Path to tesseract data path.

Example: `--threads tess-data/` - Read data from the tess-data folder.

#### --frame-diff

When comparing source and target video it can sometimes be necessary to specify frame differences. It should be sufficient to rely
on the auto detection though.

Example: `--frame-diff 8` - The target is 8 frames ahead of the source.

#### --frame-range

Specify frames on the source to extract subtitles from, can be useful to e.g. skip OP/ED

Example: `--frame-range 1000-5000` - Extracts subtitles from frame 1000 to 5000

#### --ignore-diff-fps

As it uses frame differences to find subtitles the FPS must be the same. Sometimes it can be ignored (e.g. if the source just runs faster but has the same frames). This option makes it ignore the criteria

Example: `--ignore-diff-fps` - Ignores FPS differences

#### --run-subregions-in-parallel

Run extraction for each subtitle region in parallel. Each thread will run for each subtitle regions so the total number
of threads will be threads times subtitle region count.

Example: `--run-subregions-in-parallel` - Run every subtitle region in parallel

#### --fix-broken-frame-alignment

Sometimes frames drift a bit differently so while the FPS is the same, one of the video files might have ghost frames or other annoyances. This tries to alleviate that issue.

Example: `--fix-broken-frame-alignment` - Enable frame alignment fix.

#### --debug-frame

While you are editing the subtitle region configuration it is necessary to try and extract a specific frame to see the result.
This is the command for that. It will run the current subtitle region configuration for the given frame and save an array of outputs
to the temp-folder/debug.

It will also print out what the various files contain.

Example: `--debug-frame 1000` - Try to extract subtitles from source frame 1000.

#### --debug-subregion

In combination with --debug-frame it will use a specific subtitle region. If not specified, defaults to the first subtitle region.

Example: `--debug-subregion bottom` - Extract using the sutitle region named bottom.

### create-report arguments

This command turns the extracted subtitles into a report and an .ass file

A report contains information for each subtitle region, this explanation is for the default config.
The report is an html file you should open in your webbrowser, e.g. `cow-temp/Episode 1.avi-report/index.html`. In that report each region has two sections, "subtitle lines" and "missing regions".

The "subtitle lines" are the found lines and these are reflected in the .ass file.
With the bottom subtitles there area few things:

- A start and end timestamp of the subtitle
- Start and end frame and the initial discovery frame.
- The subtitle text
- Four frames used to check if timing is correct, before first frame, first frame, last frame and after last frame. If before first or after last contains the matching text, then timing is off.

The "missing regions" part contains images of stuff where there are differences between source and target but it was unable to discover what exactly. Sometimes it is short words or un-ocrable subtitles.

A subtitle-region scan does not yield the same type of results as it is unable to merge subtitle lines in the same way. It also contains green squares for matched text under "subtitle signs" section.

Make sure to browse through the "missing regions" section, no tool is perfect.

#### --output-report-path

Where to save the report generated. Defaults to the temp-dir.

Example: `--output-report-path /mnt/sdd/subtitle-temp-reports` - Save the report to the specified path.

#### --output-subtitle-path

Where to save the .ass subtitle file is saved. Defaults to the temp-dir.

Example: `--output-report-path /mnt/sdd/subtitle-temp-subs` - Save the subtitles to the specified path.

### subtitle_regions.json

This file is generated in the temp folder when the command is first run. Any video that uses a specific temp folder will use the same
subtitle region file. A description of all available options can be found here.

#### name

Name of the subtitle region. Used with e.g. --debug-subregion parameter.

#### scan_mode

Specify how to find subtitles in a region, there are two choices, `bottom_center` and `search_slice`.

`bottom_center` looks for subtitles in the middle of the region and assumes there is, max, one subtitle in the given region.
Useful for normal subtitles at the bottom of the screen.

`search_slice` looks around for differences that contains text, useful for e.g. signs. Cannot merge similar regions and can create a lot of duplicate lines.

#### y, h, x, w, margin

Specifies the dimension of a subtitle region, it starts at `x`, `y` and ends at `x+w`, `y+h`. If you run with --debug-frame it will show where the regions are.

The `margin` is part of the region that cannot contain subtitles and any object that are part of it will be removed, useful for `bottom_center` scan mode where normal subtitles are not in the margin.

#### area_min, area_max, area_min_density

Minimum `area_min` and maximum `area_max` number of pixels a letter can contain. Minimum density `area_min_density` a letter has.

These can be useful to remove things that most certainly cannot be letters.

#### max_w, max_h

Maximum size of a letter in pixels.

#### min_stroke_width, max_stroke_width

Minimum and maximum stroke width a letter can have. These are measured at the thickest spot of a letter.

Examples could be, a long thin line will have a width of 1px while a circle will have a width of its radius.

#### border_size

Assumed size of border.

This will often be either 1 or 2, it depends a bit on how the "Threshold" debug image looks, e.g. does it consume lots of the border or not.

See "How it works" to understand what it is useful for.

#### max_text_diff, max_border_diff

Maximum difference for text and border to be assumed part of the same text line.

This depends a bit on how well the text is marked and extracted, if it finds too few letters it might be smart to turn them up and vice-versa if it finds too much.

See "How it works" to understand what it is useful for.

#### percent_good_border

How much of a border of a given figure must be good to be assumed part of the text.

See "How it works" to understand what it is useful for.

#### edge_threshold

Used in relation with finding the differnence between source and target frames. Should probably not be touched.

See "How it works" to understand what it might be useful for.

#### threshold_mode, threshold_value

Method and value to look for threshold with. There are two modes `adaptive` and `static`.

`adaptive` finds out which pixel should be black and which should be white depending on the pixels around it. Can be useful if the inner text on the subtitles varies but is always bright. An example `threshold_value` for this could be 27, that will prevent most noise too.

`static` is an absolute way of finding them, useful if the inner subtitle text is always bright and same color. An example `threshold_value` could be 200, which is the brightness cutoff.

See "How it works" to understand what it is useful for and https://docs.opencv.org/4.5.2/d7/d4d/tutorial_py_thresholding.html for information about thresholds generally.

#### ass_style_name

Style name to use with text found here.

#### invert_mode

Not implemented, no effect.

### A realistic workflow

In this example we have a set of 12 episodes we want to OCR, the source is 640x480 which matches the default subtitle region.
Source files are located in `source-video` and the target files are in `target-video`.

First we run cartonizer to create a batch script.

`~/the-cute-collection/.env/bin/python ~/the-cute-collection/cartonizer.py ocr source-video target-video --additional-params '--threads 1 --run-subregions-in-parallel'`

This creates a file named `ocr_release.sh` and it will be the script we run when we have modified `subtitle_regions.json` enough.

We open up the `ocr_release.sh` file and find OCR of the first episode. We need temp folder and configuration created first before we can OCR it all.

```
~/the-cute-collection/.env/bin/python ~/the-cute-collection/cowocr.py \
  'source-video/Episode 1.mkv' \
  'target-video/Episode 1.mkv' \
  extract-subtitles \
  --threads 1 --run-subregions-in-parallel
```

That is the command that extracts subtitles from the first episode, it will be the one we use for modifying `subtitle_regions.json`.

Lets see how good the default config is by running it against part of the episode, 5000 frames should suffice (that is 3.5 minutes at 23.976 fps).

```
~/the-cute-collection/.env/bin/python ~/the-cute-collection/cowocr.py \
  'source-video/Episode 1.mkv' \
  'target-video/Episode 1.mkv' \
  extract-subtitles \
  --threads 1 --run-subregions-in-parallel \
  --frame-range 5000-10000 # framerange we use to see
```

After it is done, create a report and see the result with

```
~/the-cute-collection/.env/bin/python ~/the-cute-collection/cowocr.py \
  'source-video/Episode 1.mkv' \
  'target-video/Episode 1.mkv' \
  create-report
```

The report is in the cow-temp folder in this example including an .ass file and the `subtitle_regions.json` file.

To modify and test changes to `subtitle_regions.json` we find a good subtitle frame number in the report and use that.

```
~/the-cute-collection/.env/bin/python ~/the-cute-collection/cowocr.py \
  'source-video/Episode 1.mkv' \
  'target-video/Episode 1.mkv' \
  extract-subtitles \
  --threads 1 --run-subregions-in-parallel \
  --debug-frame 13754 --debug-subregion bottom
```

We can then run the initial 5000 frames again and see if the result is good enough. If it is, then just run the whole `ocr_release.sh`.

When it is done the .ass in the cow-temp folder must be modified and the report followed. I do this by loading the subtitle file and source episode file into Aegisub.

The subtitles must now be synced with the video, chapters added and other stuff. This can be done with MilkSync.

`~/the-cute-collection/.env/bin/python ~/the-cute-collection/cartonizer.py sync source-video target-video --additional-params '--external-subtitles cow-temp/'`

Then run `create_release.sh` and you got a fully synced video.

# License

AGPL