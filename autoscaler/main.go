package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
)

type Scaler interface {
	SetNumber(int) (int, error)
	Number() (int, error)
}

var (
	scaler       Scaler
	MIN_REPLICAS int
	MAX_REPLICAS int
)

func readIntFromBody(r *http.Request) (int, error) {
	// Read the entire body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return 0, fmt.Errorf("failed to read body: %w", err)
	}
	defer r.Body.Close() // Important to close the body

	// Convert the body to a string
	bodyStr := string(body)

	// Attempt to convert the string to an integer
	num, err := strconv.Atoi(bodyStr)
	if err != nil {
		return 0, fmt.Errorf("failed to parse integer from body: %w", err)
	}

	return num, nil
}

func scaleHandler(w http.ResponseWriter, r *http.Request) {
	replicas, err := readIntFromBody(r)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error reading integer: %v", err), http.StatusBadRequest)
		return
	}

	if replicas < MIN_REPLICAS || replicas > MAX_REPLICAS {
		http.Error(w, "Invalid parameters", http.StatusBadRequest)
		return
	}

	running, err := scaler.SetNumber(replicas)
	if err != nil {
		http.Error(w, "Failed to scale containers", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "%d", running)
}

func getRunningHandler(w http.ResponseWriter, r *http.Request) {
	running, err := scaler.Number()
	if err != nil {
		http.Error(w, "Failed to get number of supervisors", http.StatusInternalServerError)
		return
	}
	fmt.Fprintf(w, "%d", running)
}

func main() {
	port, err := parsePortFromEnv("PORT", 8080) // default 8080
	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1) // Exit with error code
	}

	MIN_REPLICAS = parseIntFromEnv("MIN_REPLICAS", 1)
	MAX_REPLICAS = parseIntFromEnv("MAX_REPLICAS", 5)

	ENVIRONMENT := os.Getenv("ENVIRONMENT")
	if ENVIRONMENT == "AWS" {
		scaler = NewSystemdScaler()
	} else if ENVIRONMENT == "Docker" {
		scaler = NewDockerScaler()
	} else if ENVIRONMENT == "Swarm" {
		scaler = NewSwarmScaler()
	} else {
		log.Fatal("Unknown ENVIRONMENT")
	}

	http.Handle("POST /scale", logRequest(http.HandlerFunc(scaleHandler)))
	http.Handle("GET /scale", logRequest(http.HandlerFunc(getRunningHandler)))

	fmt.Println("🔌 Listening on", fmt.Sprintf("http://localhost:%d/scale", port))
	// log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", port), nil))
	runContainer()
}

func parsePortFromEnv(envVarName string, defaultPort int) (int, error) {
	portStr := os.Getenv(envVarName)
	if portStr == "" {
		return defaultPort, nil // Return default if env var is not set
	}

	port, err := strconv.Atoi(portStr)
	if err != nil {
		return 0, fmt.Errorf("invalid port value in %s: %w", envVarName, err)
	}

	if port < 1 || port > 65535 {
		return 0, fmt.Errorf("port number out of range in %s", envVarName)
	}

	return port, nil
}

// parseIntFromEnv parses an integer from an environment variable, using a default if not found.
func parseIntFromEnv(key string, defaultValue int) int {
	valueStr := os.Getenv(key)
	if valueStr == "" {
		return defaultValue // Return default value if not set.
	}

	valueInt, err := strconv.Atoi(valueStr)
	if err != nil {
		log.Printf("error parsing %s: %v, use default: %d\n", key, err, defaultValue)
		return defaultValue
	}

	return valueInt
}

func logRequest(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		host := r.Host
		method := r.Method
		path := r.URL.Path

		log.Printf("Received request from %s: %s %s", host, method, path)

		next.ServeHTTP(w, r)
	})
}
