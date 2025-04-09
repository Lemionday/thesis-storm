package main

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"gopkg.in/yaml.v3"
)

type DockerScaler struct {
	dockerComposeFilePath string
	serviceName           string
}

func NewDockerScaler() *DockerScaler {
	DOCKER_COMPOSE_FILE_PATH := os.Getenv("DOCKER_COMPOSE_FILE_PATH")
	if DOCKER_COMPOSE_FILE_PATH == "" {
		log.Fatal("DOCKER_COMPOSE_FILE_PATH is emptys")
	}

	SERVICE_NAME := os.Getenv("SERVICE_NAME")
	if SERVICE_NAME == "" {
		log.Fatal("SERVICE_NAME is empty")
	}

	return &DockerScaler{dockerComposeFilePath: DOCKER_COMPOSE_FILE_PATH, serviceName: SERVICE_NAME}
}

func (s *DockerScaler) SetNumber(machines int) (int, error) {
	err := updateReplicas(s.dockerComposeFilePath, s.serviceName, machines)
	if err != nil {
		fmt.Println("Failed to update replicas:", err)
		return 0, err
	}

	fmt.Println("Updated replicas. Starting docker compose...")
	err = dockerComposeUp()
	if err != nil {
		fmt.Println("Docker Compose failed:", err)
	}

	return s.Number()
}

func (s *DockerScaler) Number() (int, error) {
	cmd := exec.Command(
		"sh",
		"-c",
		fmt.Sprintf("docker compose ps | grep %s | wc -l", s.serviceName),
	)

	var out bytes.Buffer
	cmd.Stdout = &out

	err := cmd.Run()
	if err != nil {
		return 0, err
	}

	// Clean output and convert to int
	output := strings.TrimSpace(out.String())
	var count int
	_, err = fmt.Sscanf(output, "%d", &count)
	if err != nil {
		return 0, err
	}

	return count, nil
}

type ComposeFile struct {
	Services map[string]Service `yaml:"services"`
}

type Service struct {
	Deploy Deploy `yaml:"deploy,omitempty"`
}

type Deploy struct {
	Replicas int `yaml:"replicas,omitempty"`
}

func updateReplicas(filename, serviceName string, newReplicas int) error {
	// Step 1: Read YAML
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	var compose ComposeFile
	err = yaml.Unmarshal(data, &compose)
	if err != nil {
		return err
	}

	// Step 2: Update replicas
	service, ok := compose.Services[serviceName]
	if !ok {
		return fmt.Errorf("service %s not found", serviceName)
	}

	service.Deploy.Replicas = newReplicas
	compose.Services[serviceName] = service

	// Step 3: Write YAML back
	newData, err := yaml.Marshal(&compose)
	if err != nil {
		return err
	}
	err = os.WriteFile(filename, newData, 0644)
	if err != nil {
		return err
	}

	return nil
}

func dockerComposeUp() error {
	cmd := exec.Command("docker-compose", "up", "-d")
	cmd.Dir = "../"
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
