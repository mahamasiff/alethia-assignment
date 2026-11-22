# Write-up

## 1. Your timings

How long did your first working version take on the full 50 files, and how long did your final version take? Which single change saved the most time? *(100 words)*

Because of time I couldn't run all 50 files.I ran as much was possible based on the estimate of total runtime I got from AI so these numbers come from what I did run scaled assumed the other files are similar. My first working version ran every frame in FP32. On a 1,270 second sample it took 381.6 s, about 3x real time, so around 39 hours for 50 files. My final version (FP16, 5 frames per second) did files 0 and 1 (5.22 video hours) in 447.9 s, about 42x real time, so roughly 3 hours for 50 files. The biggest saving was using 5 frames per second instead of every frame, about 5x faster on its own. FP16 gave about 2.4x.


## 2. Your pipeline

Describe what your script does, step by step, from opening a Parquet file to writing an output line. *(150 words)*

First it reads the metadata of every Parquet file to get the list of videos. Then it starts one process per GPU. In each process a reader thread reads the mp4 column one row group at a time, and the processes take videos from a shared list one by one, so if one GPU is faster it just takes more videos. Two decoder threads open each video in memory with PyAV, fix the rotation if the video has a rotation tag, keep 5 frames per second and resize them to the size YOLO would use anyway. The frames of one video go to YOLO11l in batches of 32, in FP16 with conf 0.4. For each second i check which objects showed up in at least half of its frames, and the people count is the median person count. Each process writes its lines to its own part file, and at the end the parts are merged into one results.jsonl.


## 3. Where the time goes

Which part of your final version takes the most time? How did you measure this? *(150 words)*

The YOLO inference on the GPU takes the most time. I didn't put a timer on each step, I measured it from the fps that run.py logs per GPU and by changing one thing at a time. On the same sample, running every frame went from 50.5 fps per GPU in FP32 to 113.9 fps in FP16. Only the GPU part changed, and the whole run got more than 2x faster, so most of the time was the model. In the final run (FP16, 5 frames per second) each GPU still did about 106 to 108 analysed fps, almost the same as FP16 on every frame. If decoding was the slow part, this number would have dropped a lot, because the decoder still has to decode every frame even when we keep only 5 per second. So the GPU is still the limit.

## 4. Checking correctness

How did you check that your results stayed correct after you made the script faster? *(100 words)*

I made a reference by running FP32 on every frame for the first 40 seconds of 32 videos (1,270 seconds) and saved the detections for each frame. With check.py I used those saved frames to test 1 to 10 frames per second and FP16 against the reference using the grader's metrics, without running the GPU again. 1 frame per second failed, so I picked 5, which still had some margin. Then I scored the real final run on the same seconds and got F1 0.9819, exact 0.9543, within 1 0.9984, the same as the test predicted. I also checked rotation on 64 videos with 0 mismatches.


## 5. Working with AI

Describe one suggestion from an AI tool that was wrong or that you decided not to use. How did you find out it was wrong, or why did you reject it? *(100 words)*

Some of the problems were misunderstandings about the project. like, AI assumed an object is present in a second if it appeared in any frame of that second. The instructions say it has to be in at least half of the frames, so with the AI's version one wrong detection in a single frame would add an object that isn't really there.


## 6. Next steps

If you had one more week, what would you change, and why? *(100 words)*

First I'd process all 50 files. At the speed I measured (about 42x real time on the 2 file run) it would take roughly 3 hours for about 130 hours of video, which I didn't have time for. Second I'd test in the grader's environment. My runs used Python 3.13 and PyAV 19 but the grader uses 3.11 and PyAV 18.1.1, so I'd run the same checks against the reference there, since decoding and rotation could behave differently between versions. I'd also build a bigger reference with more videos and full lengths, test videos that actually have a rotation tag, and time each stage separately.


