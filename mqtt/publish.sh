#!/bin/bash

for file in data-file/*.csv; do
	if [ "$count" -eq 0 ]; then
		count=$((count + 1))
        continue
    fi
  echo "Processing $file..."
  node index.js -f "$file" -b "$BROKER_URL" -t "$TOPIC" -s 20
done
