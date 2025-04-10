package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"log/slog"
	"sync"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/client"
	"github.com/prometheus/client_golang/prometheus"
)

const (
	statsInterval = 1 * time.Second
)

// DockerMonitor struct encapsulates the Docker client and stats data
type DockerMonitor struct {
	client          *client.Client
	statsMap        map[string]*ContainerStatsData
	statsMutex      *sync.Mutex
	containerPrefix string
	logger          *slog.Logger
}

// ContainerStatsData struct to hold the collected metrics for a container
type ContainerStatsData struct {
	ContainerName   string
	CPUPercent      float64
	MemoryUsage     uint64
	MemoryLimitHits uint64
	MemoryPercent   float64
}

// NewDockerMonitor creates a new DockerMonitor instance
func NewDockerMonitor(
	prefix string,
	mutex *sync.Mutex,
	logger *slog.Logger,
) (*DockerMonitor, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %w", err)
	}
	return &DockerMonitor{
		client:          cli,
		statsMap:        make(map[string]*ContainerStatsData),
		statsMutex:      mutex,
		containerPrefix: prefix,
		logger:          logger,
	}, nil
}

// Close closes the Docker client
func (dm *DockerMonitor) Close() {
	if dm.client != nil {
		dm.client.Close()
	}
}

// GetContainerStats retrieves and monitors stats for matching containers
func (dm *DockerMonitor) GetContainerStats(ctx context.Context) {
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

	var wg sync.WaitGroup

	for _, c := range containers {
		containerName := c.Names[0]
		dm.statsMap[containerName] = &ContainerStatsData{ContainerName: containerName}
		wg.Add(1)
		go dm.monitorContainerStats(ctx, c.ID, containerName, &wg)
	}

	log.Println("Initial container monitoring started.")
	wg.Wait()
}

// GetStatsForContainers retrieves the current stats for all monitored containers
func (dm *DockerMonitor) GetStatsForContainers() map[string]*ContainerStatsData {
	dm.statsMutex.Lock()
	defer dm.statsMutex.Unlock()
	statsCopy := make(map[string]*ContainerStatsData)
	for name, data := range dm.statsMap {
		statsCopy[name] = &ContainerStatsData{
			ContainerName:   name,
			CPUPercent:      data.CPUPercent,
			MemoryUsage:     data.MemoryUsage,
			MemoryLimitHits: data.MemoryLimitHits,
			MemoryPercent:   data.MemoryPercent,
		}
	}
	return statsCopy
}

func (dm *DockerMonitor) monitorContainerStats(
	ctx context.Context,
	containerID, containerName string,
	wg *sync.WaitGroup,
) {
	defer wg.Done()

	log.Printf("Monitoring stats for container: %s (%s)\n", containerName, containerID[:12])

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

	dm.statsMutex.Lock()
	if data, ok := dm.statsMap[containerName]; ok {
		data.CPUPercent = float64(stat.CPUStats.CPUUsage.TotalUsage)
		data.MemoryUsage = stat.MemoryStats.Usage
		// data.MemoryLimitHits = stat.MemoryStats.Failcnt
		data.MemoryPercent = dm.calculateMemPercent(&stat.MemoryStats)
	}
	dm.statsMutex.Unlock()

	// stats, err := dm.client.ContainerStats(ctx, containerID, true)
	// if err != nil {
	// 	dm.logger.Error("Failed getting container stats", slog.String("err", err.Error()))
	// }
	// defer stats.Body.Close()
	// for {
	// 	if err := decoder.Decode(&stat); err != nil {
	// 		if err == io.EOF {
	// 			break
	// 		}
	// 		dm.logger.Error("Failed parsing container stats response", err)
	// 	}
	//
	// 	memUsage := stat.MemoryStats.Usage
	// 	dm.logger.Info("Stats", slog.String("name", containerName), slog.Uint64("memory", memUsage))
	// }
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
		MemoryLimitHits: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "storm_supervisor_memory_limit_hits",
			Help: "Number of times supervisor hit memory limit",
		}, []string{"SupervisorID"}),
	}

	reg.MustRegister(m.CPUPercent)
	reg.MustRegister(m.MemoryPercent)
	reg.MustRegister(m.MemoryLimitHits)

	return m
}

func collectSupervisorMetrics(
	m *supervisorMetrics,
	dm *DockerMonitor,
) {
	dm.statsMutex.Lock()
	defer dm.statsMutex.Unlock()

	for supervisor, stat := range dm.statsMap {
		supervisorID := string(supervisor[len(supervisor)-1])

		m.CPUPercent.WithLabelValues(supervisorID).Set(float64(stat.CPUPercent))
		m.MemoryPercent.WithLabelValues(supervisorID).Set(float64(stat.MemoryPercent))
		m.MemoryLimitHits.WithLabelValues(supervisorID).Set(float64(stat.MemoryLimitHits))
	}

	// Reset stats data map
	dm.statsMap = make(map[string]*ContainerStatsData)
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
