#!/bin/bash


read -p 'Blob pattern to the directory: ' directory

for f in $directory; do
    # echo $f
    ffmpeg -framerate 2 -pattern_type glob -i "$f/*.png" -c:v libx264 -pix_fmt yuv420p "$f/out.mp4"
done