package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"golang.org/x/crypto/ssh"
)

type SwarmScaler struct {
	serviceName string
	connString  string
	sshConfig   *ssh.ClientConfig
}

func NewSwarmScaler() *SwarmScaler {
	remoteServerAddress := os.Getenv("REMOTE_SERVER_ADDRESS")
	if remoteServerAddress == "" {
		log.Fatal("REMOTE_SERVER_ADDRESS is empty")
	}

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
		User: "lemonday",
		Auth: []ssh.AuthMethod{
			ssh.PublicKeys(signer),
		},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(), // For testing purposes only
	}
	return &SwarmScaler{
		connString:  fmt.Sprintf("%s:22", remoteServerAddress),
		sshConfig:   config,
		serviceName: serviceName,
	}
}

func (s *SwarmScaler) SetNumber(machines int) (int, error) {
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

func (s *SwarmScaler) Number() (int, error) {
	client, err := ssh.Dial("tcp", s.connString, s.sshConfig)
	if err != nil {
		return 0, fmt.Errorf("failed to dial: %v", err)
	}
	defer client.Close()

	// Create a new session
	session, err := client.NewSession()
	if err != nil {
		return 0, fmt.Errorf("failed to create session: %v", err)
	}
	defer session.Close()

	// Run the command to get Docker service replicas
	output, err := session.CombinedOutput(
		"docker service ls --filter name=storm-supervisor --format '{{.Replicas}}'",
	)
	if err != nil {
		log.Fatalf("failed to run command: %v", err)
	}

	// Parse the output
	var replicas int
	_, err = fmt.Sscanf(strings.TrimSpace(string(output)), "%d", &replicas)
	if err != nil {
		return 0, err
	}

	return replicas, nil
}

func dockerSwarmUpScaleReplicas(replicas int) error {
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
