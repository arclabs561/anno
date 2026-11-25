#!/bin/sh
echo "$$" >> "$4/pids"
if [ -f "$4/hang-start" ]; then
    sleep 20 &
    echo "$!" > "$4/helper-pid"
    wait
    exit
fi
printf '"ready"\n'
if [ -f "$4/block-input" ]; then exec sleep 20; fi
while IFS= read -r line; do
    case "$line" in
        *hang*) exec sleep 20 ;;
        *crash*) exit 2 ;;
        *broken*) printf 'invalid frame\n' ;;
        *reject*) printf '{"error":"invalid span"}\n' ;;
        *) printf '{"clusters":[[[0,1]]]}\n' ;;
    esac
done
