package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/client"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/common/expfmt"
)

const (
	statsInterval = 1 * time.Second
)

// DockerMonitor struct encapsulates the Docker client and stats data
type DockerMonitor struct {
	client *client.Client
	logger *slog.Logger
}

// NewDockerMonitor creates a new DockerMonitor instance
func NewDockerMonitor(logger *slog.Logger) (*DockerMonitor, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %w", err)
	}
	return &DockerMonitor{
		client: cli,
		logger: logger,
	}, nil
}

// Close closes the Docker client
func (dm *DockerMonitor) Close() {
	if dm.client != nil {
		dm.client.Close()
	}
}

func fetchAndTransform(url string, index int, sm *supervisorMetrics) {
	resp, err := http.Get(url)
	if err != nil {
		log.Printf("Failed to fetch from %s: %v", url, err)
		return
	}
	defer resp.Body.Close()

	parser := expfmt.TextParser{}
	metricFamilies, err := parser.TextToMetricFamilies(resp.Body)
	if err != nil {
		log.Printf("Failed to parse metrics from %s: %v", url, err)
		return
	}

	// Extract the metric you're interested in
	if mf, ok := metricFamilies["cxp_memory_percentage"]; ok {
		for _, m := range mf.Metric {
			supervisorID := fmt.Sprintf("%d", index)
			if m.Gauge != nil {
				value := m.Gauge.GetValue()
				sm.MemoryPercent.WithLabelValues(supervisorID).Set(value)
			}
		}
	} else {
		log.Printf("Metric cxp_memory_percentage not found in %s", url)
	}
}

// GetContainerStats retrieves and monitors stats for matching containers
func (dm *DockerMonitor) collectSupervisorMetrics(
	ctx context.Context,
	conf *config,
	m *supervisorMetrics,
) {
	if conf.environment == "cloud" {
		for i, endpoint := range conf.endpoints {
			fetchAndTransform(endpoint, i+1, m)
		}

		return
	}
	containers, err := dm.client.ContainerList(
		ctx,
		container.ListOptions{
			Filters: filters.NewArgs(
				filters.Arg("label", "com.docker.compose.project=storm-docker"),
				filters.Arg("label", "com.docker.compose.service=storm-supervisor"),
			),
		},
	)
	if err != nil {
		log.Printf("Failed to list containers: %v", err)
		return
	}

	// fmt.Printf("%+v\n", containers)

	IDs := []string{}
	var wg sync.WaitGroup
	for _, c := range containers {
		containerName := c.Names[0]
		supervisorID := string(containerName[len(containerName)-1])
		IDs = append(IDs, supervisorID)
		go dm.monitorContainerStats(ctx, c.ID, containerName, m, &wg)
	}

	for i := range 5 {
		found := false
		currentID := strconv.Itoa(i + 1)
		for _, id := range IDs {
			if currentID == id {
				found = true
			}
		}

		if found {
			continue
		}

		m.CPUPercent.WithLabelValues(currentID).Set(0)
		m.MemoryPercent.WithLabelValues(currentID).Set(0)
	}
}

func (dm *DockerMonitor) monitorContainerStats(
	ctx context.Context,
	containerID, containerName string,
	m *supervisorMetrics,
	wg *sync.WaitGroup,
) {
	wg.Add(1)
	defer wg.Done()
	// log.Printf("Monitoring stats for container: %s (%s)\n", containerName, containerID[:12])

	stats, err := dm.client.ContainerStats(ctx, containerID, false)
	if err != nil {
		dm.logger.Error("Failed getting container stats", slog.String("err", err.Error()))
	}
	defer stats.Body.Close()

	decoder := json.NewDecoder(stats.Body)
	var stat container.StatsResponse
	if err := decoder.Decode(&stat); err != nil && err != io.EOF {
		dm.logger.Error("Failed parsing container stats response", slog.String("err", err.Error()))
	}

	supervisorID := string(containerName[len(containerName)-1])

	m.CPUPercent.WithLabelValues(supervisorID).
		Set(float64(dm.calculateCPUPercent(&stat.CPUStats, &stat.PreCPUStats)))
	m.MemoryPercent.WithLabelValues(supervisorID).
		Set(float64(dm.calculateMemPercent(&stat.MemoryStats)))
}

type supervisorMetrics struct {
	MemoryPercent   *prometheus.GaugeVec
	MemoryLimitHits *prometheus.GaugeVec
	CPUPercent      *prometheus.GaugeVec
}

func NewSupervisorMetrics(reg prometheus.Registerer) *supervisorMetrics {
	m := &supervisorMetrics{
		CPUPercent: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "storm_supervisor_cpu_percent",
			Help: "Supervisor's cpu usage",
		}, []string{"SupervisorID"}),
		MemoryPercent: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "storm_supervisor_memory_percent",
			Help: "Supervisor's memory usage (percent)",
		}, []string{"SupervisorID"}),
		// MemoryLimitHits: prometheus.NewGaugeVec(prometheus.GaugeOpts{
		// 	Name: "storm_supervisor_memory_limit_hits",
		// 	Help: "Number of times supervisor hit memory limit",
		// }, []string{"SupervisorID"}),
	}

	reg.MustRegister(m.CPUPercent)
	reg.MustRegister(m.MemoryPercent)
	// reg.MustRegister(m.MemoryLimitHits)

	return m
}

func (dm *DockerMonitor) calculateCPUPercent(
	cpuStats *container.CPUStats,
	preCPUStats *container.CPUStats,
) float64 {
	var (
		cpuPercent = 0.0
		cpuDelta   = float64(
			cpuStats.CPUUsage.TotalUsage,
		) - float64(
			preCPUStats.CPUUsage.TotalUsage,
		)
		systemDelta = float64(cpuStats.SystemUsage) - float64(preCPUStats.SystemUsage)
	)

	if systemDelta > 0.0 && cpuDelta > 0.0 {
		cpuPercent = (cpuDelta / systemDelta) * float64(len(cpuStats.CPUUsage.PercpuUsage)) * 100.0
	}
	return cpuPercent
}

func (dm *DockerMonitor) calculateMemPercent(memStats *container.MemoryStats) float64 {
	memPercent := 0.0
	if memStats.Limit != 0 {
		memPercent = float64(memStats.Usage) / float64(memStats.Limit) * 100.0
	}
	return memPercent
}

// func (dm *DockerMonitor) formatBytes(b int64) string {
// 	const unit = 1024
// 	if b < unit {
// 		return fmt.Sprintf("%d B", b)
// 	}
// 	div, exp := int64(unit), 0
// 	for n := b / unit; n >= unit; n /= unit {
// 		div *= unit
// 		exp++
// 	}
// 	return fmt.Sprintf("%.1f %ciB", float64(b)/float64(div), "KMGTPE"[exp])
// }
