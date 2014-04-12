#!/usr/bin/env bash

set -e

LOG_FILE=${LOG_FILE:-"$HOME/perftest.log"}
OUT_FILE=${OUT_FILE:-"$HOME/perftest.out"}
BRANDES=${BRANDES:-"./brandes"}

touch "$LOG_FILE"
exec >  >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

echo "TEST BEGIN | `date` | $# graphs"
$BRANDES
for graph in "$@"; do
  echo -ne "`basename $graph` \t "
  ts=$(date +%s%N)
  $BRANDES $graph $OUT_FILE &>/dev/null
  te=$(date +%s%N)
  echo "$((($te-$ts)/1000000))"
done
echo "TEST END"

