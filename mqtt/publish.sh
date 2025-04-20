#!/bin/sh

for file in data-file/*.csv; do
  echo "Processing $file..."
  node index.js -f "$file" -b "$BROKER_URL" -t "$TOPIC"
done
