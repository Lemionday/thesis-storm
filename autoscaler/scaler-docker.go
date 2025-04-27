package main

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/client"
)

const serviceName = "storm-supervisor"

type DockerScaler struct {
	serviceName string
	cli         *client.Client
}

func NewDockerScaler() *DockerScaler {
	cli, err := client.NewClientWithOpts(client.FromEnv)
	if err != nil {
		log.Fatalf("Failed to create Docker client: %v", err)
	}
	return &DockerScaler{
		serviceName: serviceName,
		cli:         cli,
	}
}

func (s *DockerScaler) SetNumber(machines int) (int, error) {
	running, err := s.Number()
	if err != nil {
		log.Println("Failed to get replicas", err)
		return 0, err
	}

	if running == machines {
		return machines, nil
	}

	err = dockerComposeUpScaleReplicas(machines)
	if err != nil {
		log.Println("Failed to update replicas:", err)
		return 0, err
	}

	return s.Number()
}

func (s *DockerScaler) Number() (int, error) {
	cmd := exec.Command(
		"sh",
		"-c",
		fmt.Sprintf("docker compose ps | grep %s | wc -l", s.serviceName),
	)
	cmd.Dir = "../"

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

func dockerComposeUpScaleReplicas(replicas int) error {
	filters := filters.NewArgs()
	filters.Add("label", "com.docker.compose.service=storm-supervisor")

	cmd := exec.Command(
		"sh",
		"-c",
		fmt.Sprintf("docker compose up -d --scale %s=%d", serviceName, replicas),
	)
	cmd.Dir = "../"
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
