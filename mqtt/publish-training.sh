#!/bin/bash

commands=(
  "docker compose up -d building_1 building_2" # 1 2
  "docker compose up -d building_3 building_6" # 1 2 3 6
  "docker compose up -d building_4 building_5" # 1 2 3 4 5 6
  "docker compose down building_1 building_6" # 2 3 4 6
  "docker compose down building_2 building_4" # 3 6
  "docker compose up -d building_8 building_6" # 3 5 6 8
  "docker compose up -d building_9" # 3 5 6 8 9
  "docker compose up -d building_7" # 3 5 6 8 9 7
  "docker compose down"
  "docker compose up -d building_1 building_2"
  "docker compose up -d building_3 building_4 building_6 building_8"
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
