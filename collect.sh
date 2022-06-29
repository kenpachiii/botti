#!/bin/bash

symbol=BTC-USD-SWAP 

BASE=https://static.okx.com/cdn/okex/traderecords/trades/daily/
DIRECTORY=./data/okx/trades/$symbol

dates=()

start=2021-10-01
end=$(gdate +%Y-%m-%d)

d=$(gdate +%Y-%m-%d -d "$start - 1 days")
until [[ $d == $(gdate +%Y-%m-%d -d "$end - 1 days") ]]; do 
    d=$(gdate +%Y-%m-%d -d "$d +$i days")
    dates+=($d)
done

for date in "${dates[@]}"; do
    folder=$(echo $date | sed "s/-//g")

    if [ ! -d "$DIRECTORY" ]; then
       mkdir -p $DIRECTORY
    fi

    # current=$(find "$DIRECTORY" -type f)
    # if printf '%s\0' "${current[@]}" | grep -Fxq $DIRECTORY/$date.zip; then
    #     continue
    # fi

    if unzip -tq $DIRECTORY/$date.zip | grep -qs 'No errors detected'; then
        continue
    fi

    echo ${BASE}${folder}/${symbol}-trades-${date}.zip
    curl -C - ${BASE}${folder}/${symbol}-trades-${date}.zip -o $DIRECTORY/$date.zip
    printf "\n"

done




