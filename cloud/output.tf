locals {
  peer01_wg0_conf = templatefile("${path.module}/templates/wg0.conf.tftpl", {
    client_private_key = file("./secrets/peer01_private.key")
    client_private_ip  = "10.0.0.2",
    server_public_key  = file("./secrets/server_public.key")
    server_public_ip   = google_compute_instance.manager.network_interface[0].access_config[0].nat_ip
  })

  peer02_wg0_conf = templatefile("${path.module}/templates/wg0.conf.tftpl", {
    client_private_key = file("./secrets/peer02_private.key")
    client_private_ip  = "10.0.0.3",
    server_public_key  = file("./secrets/server_public.key")
    server_public_ip   = google_compute_instance.manager.network_interface[0].access_config[0].nat_ip
  })
}

output "peer01_wg0_conf" {
  description = "Generated WireGuard configuration for peer 01"
  value       = local.peer01_wg0_conf
}

output "peer02_wg0_conf" {
  description = "Generated WireGuard configuration for peer 02"
  value       = local.peer02_wg0_conf
}

output "private_ips" {
  value = [for instance in google_compute_instance.worker : instance.network_interface[0].network_ip]
}
