package main

import (
	"context"
	"fmt"
	"time"

	run "cloud.google.com/go/run/apiv2"
	runpb "cloud.google.com/go/run/apiv2/runpb"
	"google.golang.org/api/option"
	"google.golang.org/protobuf/types/known/durationpb"
)

func runContainer() {
	ctx := context.Background()

	client, err := run.NewServicesClient(
		ctx,
		option.WithCredentialsFile("./credentials/smart-poet-454914-t5-3758cbdab057.json"),
	)
	if err != nil {
		panic(fmt.Sprintf("Run client error: %v", err))
	}
	defer client.Close()

	projectID := "smart-poet-454914-t5"
	location := "asia-southeast1"
	serviceID := "zookeeper-instance"
	image := "zookeeper:3.7"

	req := &runpb.CreateServiceRequest{
		Parent: fmt.Sprintf("projects/%s/locations/%s", projectID, location),
		Service: &runpb.Service{
			Name: fmt.Sprintf(
				"projects/%s/locations/%s/services/%s",
				projectID,
				location,
				serviceID,
			),
			Template: &runpb.RevisionTemplate{
				Containers: []*runpb.Container{
					{
						Image: image,
						Ports: []*runpb.ContainerPort{
							{ContainerPort: 2181}, // default ZooKeeper port
						},
						Resources: &runpb.ResourceRequirements{
							Limits: map[string]string{
								"cpu":    "1",
								"memory": "512Mi",
							},
						},
					},
				},
				Timeout: durationpb.New(60 * time.Second),
			},
		},
		ServiceId: serviceID,
	}

	op, err := client.CreateService(ctx, req)
	if err != nil {
		panic(fmt.Sprintf("Service creation failed: %v", err))
	}

	// Wait for deployment
	resp, err := op.Wait(ctx)
	if err != nil {
		panic(fmt.Sprintf("Waiting for deployment failed: %v", err))
	}

	fmt.Printf("ZooKeeper deployed: %s\n", resp.Name)
}
