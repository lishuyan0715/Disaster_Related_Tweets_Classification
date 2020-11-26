#! /bin/bash

if [ ! -f $MODEL_FILE ];         then echo "Missing file: ${MODEL_FILE}"; exit 1; fi


CPUS=`nproc --all`
#WORKERS=$((4 * $CPUS + 1))
WORKERS=3
export PYTHONUNBUFFERED=TRUE

exec gunicorn api \
     -w $WORKERS \
     -b 0.0.0.0:5001 \
     --timeout 1800 \
     --log-file "-"
     --enable-stdio-inheritance \
     --reload \
     --log-level "debug"
