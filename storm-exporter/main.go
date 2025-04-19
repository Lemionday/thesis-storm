package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type config struct {
	addr        string
	stormUIHost string
	refreshRate int64
	environment string
	endpoints   []string
}

const (
	containerNamePrefix = "storm-supervisor"
)

func main() {
	conf := loadConfig()

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	ticker := time.NewTicker(time.Duration(conf.refreshRate) * time.Second)
	defer ticker.Stop()

	dockerMonitor, err := NewDockerMonitor(logger)
	if err != nil {
		log.Fatalf("Failed to initialize Docker monitor: %v", err)
	}
	defer dockerMonitor.Close()

	// dockerMonitor.GetContainerStats(ctx)

	// var wg sync.WaitGroup
	//
	// Create a non-global registry
	reg := prometheus.NewRegistry()
	clusterMetric := NewClusterMetrics(reg)
	topologyMetrics := NewTopologyMetrics(reg)
	spoutMetrics := NewSpoutMetrics(reg)
	boltMetrics := NewBoltMetrics(reg)
	supervisorMetrics := NewSupervisorMetrics(reg)
	go func() {
		for {
			select {
			case <-ticker.C:
				go collectClusterMetrics(clusterMetric, conf.stormUIHost, logger)

				go func() {
					topologies, err := FetchAndDecode[struct {
						Topologies []topologySummary `json:"topologies,omitempty"`
					}](
						fmt.Sprintf("http://%s/api/v1/topology/summary", conf.stormUIHost),
					)
					if err != nil {
						logger.Error(err.Error())
						return
					}

					for _, topo := range topologies.Topologies {
						collectTopologyMetrics(topologyMetrics, topo)

						data, err := FetchAndDecode[struct {
							Spouts []spoutSummary `json:"spouts"`
							Bolts  []boltSummary  `json:"bolts"`
						}](
							fmt.Sprintf(
								"http://%s/api/v1/topology/%s?window=600",
								conf.stormUIHost,
								topo.ID,
							),
						)
						if err != nil {
							logger.Error(err.Error())
							continue
						}
						collectSpoutMetrics(spoutMetrics, data.Spouts, topo.Name, topo.ID)
						collectBoltMetrics(boltMetrics, data.Bolts, topo.Name, topo.ID)
					}
				}()

				go dockerMonitor.collectSupervisorMetrics(
					context.Background(),
					conf,
					supervisorMetrics,
				)
				// logger.Info("Updated topologies's metrics")
			}
		}
	}()

	http.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{Registry: reg}))

	log.Println("Listening on", conf.addr)
	log.Fatal(http.ListenAndServe(conf.addr, nil))
}

func loadConfig() *config {
	var conf config
	conf.addr = os.Getenv("EXPORTER_LISTEN_ADDR")
	if conf.addr == "" {
		conf.addr = ":8080"
	}

	conf.stormUIHost = os.Getenv("STORM_UI_HOST")
	if conf.stormUIHost == "" {
		conf.stormUIHost = "localhost:8081"
	}

	refreshRateStr := os.Getenv("REFRESH_RATE")
	if refreshRateStr == "" {
		conf.refreshRate = 5
	} else {
		var err error
		conf.refreshRate, err = strconv.ParseInt(refreshRateStr, 10, 64)
		if err != nil {
			log.Fatal(err)
		}
	}

	conf.environment = os.Getenv("ENVIRONMENT")
	if conf.environment == "cloud" {
		for _, endpoint := range strings.Split(os.Getenv("END_POINTS"), ",") {
			conf.endpoints = append(
				conf.endpoints,
				fmt.Sprintf("http://%s:8000/metrics", strings.Trim(endpoint, " ")),
			)
		}
		fmt.Println(conf.endpoints)
	}
	return &conf
}
