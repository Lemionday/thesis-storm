package main

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
)

const serviceName = "storm-supervisor"

type DockerScaler struct {
	serviceName string
}

func NewDockerScaler() *DockerScaler {
	return &DockerScaler{
		serviceName: serviceName,
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

	// time.Sleep(10 * time.Second)

	rebalanceStormTopologyInContainer("nimbus", "iot-smarthome", 10, machines, "")
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

func dockerComposeUpScaleReplicas(replicas int) error {
	cmd := exec.Command(
		"docker-compose",
		"up",
		"-d",
		"--scale",
		fmt.Sprintf("%s=%d", serviceName, replicas),
	)
	cmd.Dir = "../"
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
