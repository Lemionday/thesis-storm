push-topology:
	gcloud compute scp ./stormsmarthome/target/Storm-IOTdata-1.0-SNAPSHOT-jar-with-dependencies.jar storm@storm-manager:~/thesis-storm/stormsmarthome/target

copy-ssh-key:
	gcloud compute scp ./cloud/secrets/storm storm@storm-manager:/home/storm/.ssh/storm


