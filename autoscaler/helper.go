package main

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/stdcopy"
)

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
