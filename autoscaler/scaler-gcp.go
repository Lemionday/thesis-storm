package main

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"
)

type GCPScaler struct {
	serviceName string
	sshConfig   *ssh.ClientConfig
	addresses   []string
}

func NewGCPScaler() *GCPScaler {
	sshFilePath := os.Getenv("SSH_FILE_PATH")
	if sshFilePath == "" {
		log.Fatal("SSH_FILE_PATH is empty")
	}

	key, err := os.ReadFile(sshFilePath)
	if err != nil {
		log.Fatalf("unable to read private key: %v", err)
	}

	// Create the Signer for this private key.
	signer, err := ssh.ParsePrivateKey(key)
	if err != nil {
		log.Fatalf("unable to parse private key: %v", err)
	}
	// SSH client configuration
	config := &ssh.ClientConfig{
		User: "storm",
		Auth: []ssh.AuthMethod{
			ssh.PublicKeys(signer),
		},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(), // For testing purposes only
		Timeout:         2 * time.Second,
	}
	return &GCPScaler{
		sshConfig:   config,
		serviceName: serviceName,
		addresses: []string{
			"10.148.0.21:22",
			"10.148.0.22:22",
			"10.148.0.23:22",
			"10.148.0.24:22",
			"10.148.0.25:22",
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

	err = dockerComposeUpScaleReplicas(machines)
	if err != nil {
		log.Println("Failed to update replicas:", err)
		return 0, err
	}

	// time.Sleep(10 * time.Second)

	rebalanceStormTopologyInContainer("nimbus", "iot-smarthome", 10, machines, "")
	return s.Number()
}

func (s *GCPScaler) Number() (int, error) {
	var mu sync.Mutex
	var wg sync.WaitGroup

	replicas := 0
	for _, addr := range s.addresses {
		wg.Add(1)
		go func() {
			defer wg.Done()

			client, err := ssh.Dial("tcp", addr, s.sshConfig)
			if err != nil {
				fmt.Printf("failed to dial: %v\n", err)
				return
			}
			defer client.Close()

			// Create a new session
			session, err := client.NewSession()
			if err != nil {
				fmt.Printf("failed to create session: %v\n", err)
				return
			}
			defer session.Close()

			var output bytes.Buffer
			session.Stdout = &output

			// === Compose Service Check ===
			cmd := fmt.Sprintf(
				"docker inspect -f '{{.State.Running}}' %s",
				serviceName,
			)
			err = session.Run(cmd)
			if err != nil {
				fmt.Printf("%v\n", err)
				return
			}

			mu.Lock()
			replicas = replicas + 1
			mu.Unlock()
		}()
	}

	wg.Wait()

	return replicas, nil
}
