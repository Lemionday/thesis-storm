#!/bin/bash

commands=(
  "docker compose up -d building_7 building_8"
  "docker compose up -d building_3 building_6"
  "docker compose up -d building_4 building_5"
  "docker compose down building_4 building_6"
  "docker compose down building_3 building_5"
  "docker compose down"
  )

# Loop through each command
for cmd in "${commands[@]}"; do
  echo ">> Executing: $cmd"
  eval "$cmd"

  echo "⏳ Sleeping for 10 minutes..."
  sleep 600  # 20 minutes = 1200 seconds
done

echo "✨ All commands completed. Rest well, script traveler."
