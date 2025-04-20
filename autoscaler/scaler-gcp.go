package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/docker/cli/cli/connhelper"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/client"
)

type GCPScaler struct {
	serviceName string
	addresses   []string
}

func NewGCPScaler() *GCPScaler {
	return &GCPScaler{
		serviceName: "thesis-storm-storm-supervisor-1",
		addresses: []string{
			"10.148.0.21",
			"10.148.0.22",
			"10.148.0.23",
			"10.148.0.24",
			"10.148.0.25",
		},
	}
}

func (s *GCPScaler) SetNumber(machines int) (int, error) {
	running, err := s.Number()
	if err != nil {
		log.Println("Failed to get replicas", err)
		return 0, err
	}

	if running == machines {
		return machines, nil
	}

	hasError := false
	var mu sync.Mutex
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	for i, addr := range s.addresses {
		if i < MIN_REPLICAS {
			continue
		}

		cli, err := connectRemoteHost(addr)
		if err != nil {
			fmt.Printf("%v\n", err)
			continue
		}
		defer cli.Close()

		// Exceed number of containers get shut down
		if i >= machines {
			go func() {
				err := StopContainer(ctx, cli, s.serviceName)
				if err != nil {
					mu.Lock()
					hasError = true
					mu.Unlock()
				}
			}()
			continue
		}

		go func() {
			err := StartContainer(ctx, cli, s.serviceName)
			if err != nil {
				mu.Lock()
				hasError = true
				mu.Unlock()
			}
		}()
	}

	<-ctx.Done()

	if hasError {
		log.Println("failed scale supervisors")
		return 0, nil
	}

	log.Printf("%d instances running\n", machines)
	time.Sleep(10 * time.Second)

	rebalanceStormTopologyInContainer("nimbus", "iot-smarthome", 10, machines, "")
	return machines, nil
}

func (s *GCPScaler) Number() (int, error) {
	var mu sync.Mutex
	var wg sync.WaitGroup

	replicas := 0
	for _, addr := range s.addresses {
		wg.Add(1)
		go func() {
			defer wg.Done()

			cli, err := connectRemoteHost(addr)
			if err != nil {
				fmt.Printf("%v\n", err)
				return
			}
			defer cli.Close()

			ctx := context.Background()

			container, err := cli.ContainerInspect(ctx, s.serviceName)
			if err != nil {
				fmt.Fprintf(
					os.Stderr,
					"⚠️  Could not inspect container '%s': %v\n",
					addr,
					err,
				)
				return
			}

			if container.State != nil && container.State.Running {
				fmt.Printf("✅ Container '%s' is running!\n", addr)
				mu.Lock()
				replicas = replicas + 1
				mu.Unlock()
			} else {
				fmt.Printf("❌ Container '%s' is NOT running.\n", addr)
			}
		}()
	}

	wg.Wait()

	return replicas, nil
}

// Start the container on remote host
func StartContainer(ctx context.Context, cli *client.Client, containerName string) error {
	err := cli.ContainerStart(ctx, containerName, container.StartOptions{})
	if err != nil {
		return fmt.Errorf("failed to start container '%s': %w", containerName, err)
	}
	return nil
}

// Stop the container on remote host
func StopContainer(
	ctx context.Context,
	cli *client.Client,
	containerName string,
) error {
	containers, err := cli.ContainerList(
		ctx,
		container.ListOptions{
			Filters: filters.NewArgs(
				filters.KeyValuePair{Key: "name", Value: "thesis-storm-storm-supervisor-1"},
			),
		},
	)
	if err != nil {
		return fmt.Errorf("failed to list containers %v", err)
	}

	if len(containers) == 0 {
		return nil
	}

	err = cli.ContainerStop(ctx, containers[0].ID, container.StopOptions{})
	if err != nil {
		return fmt.Errorf("failed to stop container '%s': %w", containerName, err)
	}
	return nil
}

func connectRemoteHost(addr string) (*client.Client, error) {
	remoteHost := fmt.Sprintf("ssh://storm@%s", addr)
	helper, err := connhelper.GetConnectionHelper(remoteHost)
	if err != nil {
		return nil, fmt.Errorf("Error getting connection helper: %v\n", err)
	}

	// Create an HTTP client with the SSH dialer
	httpClient := &http.Client{
		Transport: &http.Transport{
			DialContext: helper.Dialer,
		},
	}
	cli, err := client.NewClientWithOpts(
		client.WithHost(remoteHost),
		client.WithHTTPClient(httpClient),
		client.WithAPIVersionNegotiation(),
		client.WithTimeout(1*time.Second),
	)
	if err != nil {
		return nil, fmt.Errorf("❌ Failed to connect to Docker daemon: %v\n", err)
	}

	return cli, nil
}
