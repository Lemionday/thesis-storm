resource "google_compute_network" "storm_network" {
  name = "storm-network"
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh"
  network = google_compute_network.storm_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "allow_wireguard" {
  name    = "allow-wireguard"
  network = google_compute_network.storm_network.name

  direction = "INGRESS"
  priority  = 1000

  allow {
    protocol = "udp"
    ports    = ["51820"]
  }

  source_ranges = ["0.0.0.0/0"] # Adjust to restrict access to specific IP ranges
  target_tags   = ["wireguard"] # Ensure your instance has this network tag
}

# resource "google_compute_firewall" "storm_firewall" {
#   name    = "storm-allow"
#   network = google_compute_network.storm_network.name
#
#   allow {
#     protocol = "tcp"
#     ports    = ["6627", "6628", "8080", "8000", "2181", "6700-6703"]
#   }
#
#   source_ranges = [var.subnet_cidr]
#   target_tags   = ["storm"]
# }
#
# # 1. Create a Cloud Router
# resource "google_compute_router" "nat_router" {
#   name    = "nat-router"
#   network = google_compute_network.storm_network.name
#   region  = var.region
# }
#
# # 2. Create a NAT Gateway
# resource "google_compute_router_nat" "nat_gateway" {
#   name                               = "nat-config"
#   router                             = google_compute_router.nat_router.name
#   region                             = var.region
#   nat_ip_allocate_option             = "AUTO_ONLY"
#   source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
# }
#
# resource "google_compute_firewall" "docker_firewall" {
#   name    = "allow-docker"
#   network = google_compute_network.storm_network.name
#
#   allow {
#     protocol = "tcp"
#     ports    = ["2375"]
#   }
#
#   target_tags   = ["docker-access"]
#   source_ranges = [var.subnet_cidr]
# }
#
# resource "google_compute_firewall" "allow_mysql" {
#   name    = "allow-mysql-ingress"
#   network = google_compute_network.storm_network.name
#
#   allow {
#     protocol = "tcp"
#     ports    = ["3306"]
#   }
#
#   direction = "INGRESS"
#
#   source_ranges = [var.subnet_cidr]
#
#   target_tags = ["mysql-server"]
# }
#
