package main

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"strings"

	"golang.org/x/crypto/ssh"
)

type SystemdScaler struct {
	remoteHost    string
	config        *ssh.ClientConfig
	serviceStates map[string]bool
}

func NewSystemdScaler() Scaler {
	remoteUser := os.Getenv("REMOTE_USER")
	remoteHostsStr := os.Getenv("REMOTE_HOSTS")    // Should include port, e.g., "192.168.1.100:22"
	privateKeyPath := os.Getenv("SSH_PRIVATE_KEY") // Optional env var for path

	if remoteUser == "" || remoteHostsStr == "" {
		log.Fatal("REMOTE_USER and REMOTE_HOSTS environment variables must be set.")
	}

	remoteHosts := strings.Split(remoteHostsStr, ",")
	serviceStates := make(map[string]bool)
	for _, remoteHost := range remoteHosts {
		serviceStates[remoteHost] = false
	}

	if privateKeyPath == "" {
		privateKeyPath = os.Getenv("HOME") + "/.ssh/id_rsa"
	}

	key, err := os.ReadFile(privateKeyPath)
	if err != nil {
		log.Fatalf("Unable to read private key: %v", err)
	}

	signer, err := ssh.ParsePrivateKey(key)
	if err != nil {
		log.Fatalf("Unable to parse private key: %v", err)
	}

	config := &ssh.ClientConfig{
		User: remoteUser,
		Auth: []ssh.AuthMethod{
			ssh.PublicKeys(signer),
		},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
	}

	return &SystemdScaler{config: config, serviceStates: serviceStates}
}

func (s *SystemdScaler) SetNumber(machines int) (int, error) {
	running := 0
	for remoteHost, state := range s.serviceStates {
		if state {
			running = running + 1

			if running >= machines {
				s.stopService(remoteHost, "storm-supervisor")
			}
		}
	}
	return running, nil
}

func (s *SystemdScaler) execute(remoteHost string, cmd func(session *ssh.Session) error) error {
	client, err := ssh.Dial("tcp", remoteHost, s.config)
	if err != nil {
		log.Fatalf("Failed to dial: %v", err)
	}
	defer client.Close()

	session, err := client.NewSession()
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	defer session.Close()

	return cmd(session)
}

func (s *SystemdScaler) startService(remoteHost string, serviceName string) error {
	return s.execute(remoteHost, func(session *ssh.Session) error {
		s.serviceStatus(remoteHost, serviceName)
		if s.serviceStates[remoteHost] {
			return nil
		}

		cmd := fmt.Sprintf("sudo systemctl start %s", serviceName)
		if err := session.Run(cmd); err != nil {
			log.Printf("Failed to run command: %v\n", err)
			return err
		}
		fmt.Printf("Start service %s on %s successfully", serviceName, remoteHost)
		return nil
	})
}

func (s *SystemdScaler) stopService(remoteHost string, serviceName string) error {
	return s.execute(remoteHost, func(session *ssh.Session) error {
		s.serviceStatus(remoteHost, serviceName)
		if !s.serviceStates[remoteHost] {
			return nil
		}

		cmd := fmt.Sprintf("sudo systemctl stop %s", serviceName)
		if err := session.Run(cmd); err != nil {
			log.Printf("Failed to run command: %v\n", err)
			return err
		}
		fmt.Printf("Stop service %s on %s successfully", serviceName, remoteHost)
		return nil
	})
}

func (s *SystemdScaler) serviceStatus(remoteHost string, serviceName string) error {
	return s.execute(remoteHost, func(session *ssh.Session) error {
		var stdout bytes.Buffer
		session.Stdout = &stdout

		cmd := fmt.Sprintf("sudo systemctl is-active %s", serviceName)
		if err := session.Run(cmd); err != nil {
			log.Printf("Failed to run command: %v\n", err)
			return err
		}

		status := strings.TrimSpace(stdout.String())
		log.Printf("Service %s on remote host %s status: %s\n", serviceName, remoteHost, status)

		if status == "active" {
			s.serviceStates[remoteHost] = true
		} else {
			s.serviceStates[remoteHost] = false
		}

		return nil
	})
}

func (s *SystemdScaler) Number() (int, error) {
	running := 0
	for remoteHost := range s.serviceStates {
		s.serviceStatus(remoteHost, "storm-supervisor")
		if s.serviceStates[remoteHost] {
			running = running + 1
		}
	}
	return running, nil
}
