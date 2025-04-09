package main

import (
	"fmt"
	"log/slog"

	"github.com/prometheus/client_golang/prometheus"
)

type clusterMetrics struct {
	WorkersTotal  prometheus.Gauge
	WorkersInUsed prometheus.Gauge
	Supervisors   prometheus.Gauge
}

func NewClusterMetrics(reg prometheus.Registerer) *clusterMetrics {
	m := &clusterMetrics{
		Supervisors: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "storm_cluster_supervisors_total",
			Help: "Number of supervisors in cluster",
		}),
		WorkersTotal: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "storm_cluster_worker_processes_total",
			Help: "Number of worker processes in cluster",
		}),
		WorkersInUsed: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "storm_cluster_worker_processes_used",
			Help: "Number of worker processes used in cluster",
		}),
	}

	reg.MustRegister(m.Supervisors)
	reg.MustRegister(m.WorkersTotal)
	reg.MustRegister(m.WorkersInUsed)

	return m
}

type ClusterSummary struct {
	SlotsUsed   int `json:"slotsUsed,omitempty"`
	SlotsTotal  int `json:"slotsTotal,omitempty"`
	Supervisors int `json:"supervisors,omitempty"`
}

func collectClusterMetrics(m *clusterMetrics, stormUIHost string, logger *slog.Logger) {
	summary, err := FetchAndDecode[ClusterSummary](
		fmt.Sprintf("http://%s/api/v1/cluster/summary", stormUIHost),
	)
	if err != nil {
		logger.Error(err.Error())
		return
	}

	m.Supervisors.Set(float64(summary.Supervisors))
	m.WorkersTotal.Set(float64(summary.SlotsTotal))
	m.WorkersInUsed.Set(float64(summary.SlotsUsed))

	logger.Info("Updated cluster's metrics")
}
