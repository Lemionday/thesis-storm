provider "google" {
  project = "smart-poet-454914-t5"
  region  = var.region
  zone    = var.zone
}

resource "google_compute_instance" "manager" {
  name         = "storm-manager"
  machine_type = var.manager_machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  network_interface {
    network = google_compute_network.storm_network.name

    network_ip = var.manager_internal_ip
    access_config {}
  }

  metadata_startup_script = join("\n",
    [
      file("./scripts/setup_docker.sh"),
      file("./scripts/setup_wireguard.sh"),
      file("./scripts/setup_storm.sh"),
      file("./scripts/setup_storm_manager.sh")
    ]
  )

  metadata = {
    ssh-keys = "storm:${file("./secrets/storm.pub")}"
  }

  tags = ["mysql-server", "storm", "wireguard"]
}

resource "google_compute_instance" "worker" {
  count        = 5
  name         = "worker-${count.index}"
  machine_type = var.worker_machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  network_interface {
    network    = google_compute_network.storm_network.name
    network_ip = var.worker_internal_ips[count.index]
  }

  metadata_startup_script = join("\n",
    [
      file("./scripts/setup_docker.sh"),
      file("./scripts/setup_storm.sh"),
      file("./scripts/setup_storm_supervisor.sh"),
      file("./scripts/setup_docker_exporter.sh")
    ]
  )

  metadata = {
    ssh-keys = "storm:${file("./secrets/storm.pub")}"
  }

  tags = ["storm", "wireguard", "docker-access"]
}

# resource "google_compute_instance" "mqtt_publisher" {
#   name         = "mqtt-publisher"
#   machine_type = var.worker_machine_type
#   zone         = var.zone
#
#   boot_disk {
#     initialize_params {
#       image = "debian-cloud/debian-11"
#     }
#   }
#
#   network_interface {
#     network    = google_compute_network.storm_network.name
#     network_ip = var.mqtt_publisher_internal_ips[0]
#   }
#
#   metadata_startup_script = join("\n",
#     [
#       file("./scripts/setup_docker.sh"),
#       file("./scripts/setup_storm.sh"),
#     ]
#   )
#
#   metadata = {
#     ssh-keys = "storm:${file("./secrets/storm.pub")}"
#   }
#
#   tags = ["mqtt-publisher"]
# }
#
# resource "google_compute_instance" "forecaster" {
#   name         = "forecaster"
#   machine_type = var.worker_machine_type
#   zone         = "asia-southeast2-a"
#
#   boot_disk {
#     initialize_params {
#       image = "debian-cloud/debian-11"
#       size  = 30
#     }
#   }
#
#   network_interface {
#     network = "default"
#     access_config {}
#   }
#
#   metadata_startup_script = join("\n",
#     [
#       file("./scripts/setup_forecaster.sh"), <<-EOT
#       #!/bin/bash
#       apt-get update
#       apt-get install -y wireguard
#
#       cat <<EOF > /etc/wireguard/wg0.conf
#       ${local.peer01_wg0_conf}
#       EOF
#
#       # Enable IP forwarding
#       echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
#       sysctl -p
#
#       # Start WireGuard
#       systemctl enable wg-quick@wg0
#       wg-quick up wg0
#   EOT
#     ]
#   )
#
#   metadata = {
#     ssh-keys = "storm:${file("./secrets/storm.pub")}"
#   }
#
#   tags = ["forecaster"]
# }

# resource "google_compute_instance" "mosquitto" {
#   name         = "mosquitto-vm"
#   machine_type = "e2-micro"
#
#   boot_disk {
#     initialize_params {
#       image = "debian-cloud/debian-11"
#     }
#   }
#
#   network_interface {
#     network = "default"
#     access_config {}
#   }
#
#   metadata_startup_script = join("\n",
#     [
#       file("./scripts/setup_docker.sh"),
#       file("./scripts/setup_wireguard.sh"),
#       "apt install -y wireguard mosquitto mosquitto-clients",
#     ]
#   )
#
#   tags = ["wireguard"]
# }
