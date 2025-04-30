import csv
import random
import signal
import subprocess
import sys
import time


def down_all():
    print("\n🛑 Ctrl+C detected! Bringing down all buildings...")
    cmd = "docker compose down"
    subprocess.run(cmd, shell=True)
    print("🏚️ All buildings have been taken down. Exiting gracefully.")
    sys.exit(0)


# Attach the signal handler
signal.signal(signal.SIGINT, lambda sig, frame: down_all())

# Settings
# building_1 to building_10
buildings = [f"building_{i}" for i in range(1, 11)]
iterations = 20  # How many random actions to perform
delay_range = (5, 10)  # Random delay between actions, in seconds

current_buildings = set()

for _ in range(iterations):
    action = random.choice(["up", "down"])
    if len(current_buildings) < 3:
        action = "up"

    if action == "up":
        available_buildings = list(set(buildings) - current_buildings)
        if not available_buildings:
            print("🏙️ All buildings are already spawned!")
            continue

        building = random.choice(available_buildings)

        cmd = f"docker compose up -d {building}"

        current_buildings.add(building)
    else:
        building = random.choice(list(current_buildings))

        cmd = f"docker compose down {building}"

        current_buildings.remove(building)

    print(f"🔧 Executing: {cmd}")
    print(current_buildings)
    with open("building_publishers.csv", "a", newline="") as f:
        write = csv.writer(f)
        write.writerow(current_buildings)

    subprocess.run(cmd, shell=True)

    # Wait a random time before the next action
    delay = random.randint(*delay_range)
    if len(current_buildings) < 3:
        delay = 1
    print(f"⏳ Waiting {delay} minutes...")
    time.sleep(delay * 60)
