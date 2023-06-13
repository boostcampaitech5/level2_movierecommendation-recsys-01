#!/bin/sh

# which python
pwd

if [ ! -d "data" ]; then
    echo "data folder not exists. please volumes options."
else
    echo "data folder exists."
fi

# ls -l

python code/main.py
