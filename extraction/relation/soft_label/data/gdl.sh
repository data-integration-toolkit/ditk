#!/bin/bash

#----------------Initialize----------------
Green_font_prefix="\033[32m" && Red_font_prefix="\033[31m" && Green_background_prefix="\033[42;37m" && Red_background_prefix="\033[41;37m" && Font_color_suffix="\033[0m"

Info="${Green_font_prefix}[INFO]${Font_color_suffix}"
Error="${Red_font_prefix}[ERROR]${Font_color_suffix}"

fileid=$1
filename=$2

if [ $# -ne 2 ]; then
    echo -e "${Error} Input Argument Error"
    exit 1
fi

echo -e "${Info} Downloading ${filename} with ID: ${fileid} from Google Cloud"

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

echo -e "${Info} Download Success!"
rm -f cookie

echo ""
