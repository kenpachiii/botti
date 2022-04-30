#!/bin/bash

if [ -d "/dump" ] 
then
    echo "Directory /dump exists." 
else
    echo "Directory /dump does not exists. Creating."
    mkdir dump
fi

scp -i id_rsa root@188.166.23.165:~/botti/dump/* dump/
scp -i id_rsa root@188.166.23.165:~/botti/botti.db dump/botti.db.dump

