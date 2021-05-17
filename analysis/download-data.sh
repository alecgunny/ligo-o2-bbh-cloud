#! /bin/bash -e

FILEID=$1

URL="https://docs.google.com/uc?export=download&id=$FILEID"
confirm=$(
    wget \
        --quiet \
        --save-cookies /tmp/cookies.txt \
        --keep-session-cookies \
        --no-check-certificate \
        $URL \
        -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p'
)
wget --quiet --load-cookies /tmp/cookies.txt "${URL}&confirm=${confirm}" -O data.zip
rm -rf /tmp/cookies.txt

unzip data.zip
rm data.zip