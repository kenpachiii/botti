#!/bin/bash

if [ -d "./data" ] 
then
    echo "Directory /data exists." 
else
    echo "Directory /data does not exists. Creating."
    mkdir -p data/okx 
fi

exclude=$(date +'%Y-%m-%d')
rsync -azP -e "ssh -i ~/.ssh/botti" --ignore-existing --exclude=${exclude}.lzma root@188.166.23.165:/mnt/volume_ams3_01/okx/ data/okx/