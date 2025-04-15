package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/stdcopy"
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

func rebalanceStormTopologyInContainer(
	containerName, topologyName string,
	waitTime, numWorkers int,
	componentParallelism string,
) error {
	ctx := context.Background()
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return fmt.Errorf("failed to create Docker client: %w", err)
	}
	defer cli.Close()

	containerInfo, err := cli.ContainerInspect(ctx, containerName)
	if err != nil {
		return fmt.Errorf("failed to inspect container %s: %w", containerName, err)
	}

	command := []string{"storm", "rebalance", topologyName}
	if waitTime > 0 {
		command = append(command, "-w", fmt.Sprintf("%d", waitTime))
	}
	if numWorkers > 0 {
		command = append(command, "-n", fmt.Sprintf("%d", numWorkers))
	}
	if componentParallelism != "" {
		command = append(command, "-e", componentParallelism)
	}

	execConfig := container.ExecOptions{
		Cmd:          command,
		AttachStdout: true,
		AttachStderr: true,
		Tty:          false,
	}

	resp, err := cli.ContainerExecCreate(ctx, containerInfo.ID, execConfig)
	if err != nil {
		return fmt.Errorf("failed to create exec: %w", err)
	}

	attach, err := cli.ContainerExecAttach(ctx, resp.ID, container.ExecStartOptions{Tty: false})
	if err != nil {
		return fmt.Errorf("failed to attach exec: %w", err)
	}
	defer attach.Close()

	_, err = stdcopy.StdCopy(os.Stdout, os.Stderr, attach.Reader)

	if err != nil && err != io.EOF {
		return fmt.Errorf("failed to copy output: %w", err)
	}

	inspect, err := cli.ContainerExecInspect(ctx, resp.ID)
	if err != nil {
		return fmt.Errorf("failed to inspect exec: %w", err)
	}

	if inspect.ExitCode != 0 {
		return fmt.Errorf("command execution failed with exit code: %d", inspect.ExitCode)
	}

	fmt.Println("Topology rebalanced successfully.")
	return nil
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

	time.Sleep(10 * time.Second)

	rebalanceStormTopologyInContainer("nimbus", "iot-smarthome", 0, 0, "")
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
