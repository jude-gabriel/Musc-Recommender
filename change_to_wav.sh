#!/bin/bash
mkdir wav_songs
for file in *.mid; do timidity "$file" -Ow -o "${file%.mid}".wav;done
mv *.wav wav_songs/ 

