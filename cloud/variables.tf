variable "subnet_cidr" {
  description = "CIDR range for the subnet"
  type        = string
  default     = "10.148.0.0/24"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "asia-southeast1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "asia-southeast1-a"
}

variable "manager_machine_type" {
  description = "Machine type for manager node"
  type        = string
  default     = "e2-standard-2"
}

variable "worker_machine_type" {
  description = "Machine type for worker nodes"
  type        = string
  default     = "e2-micro"
}

variable "initial_worker_count" {
  description = "Initial number of worker nodes"
  type        = number
  default     = 2
}

variable "manager_internal_ip" {
  default = "10.148.0.20"
}

variable "worker_internal_ips" {
  default = ["10.148.0.21", "10.148.0.22", "10.148.0.23", "10.148.0.24", "10.148.0.25"]
}

variable "mqtt_publisher_internal_ips" {
  default = ["10.148.0.11", "10.148.0.12", "10.148.0.13", "10.148.0.14", "10.148.0.15"]
}
