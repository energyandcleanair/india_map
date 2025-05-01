
To run this code, you need the Google Cloud CLI `gcloud` set up and
authenticated locally against the project you wish to run this against.

> [!WARNING]
> Most of these scripts make assumptions about how the infrastructure is set up
> (like regions, bucket names, etc.) - with hardcoded values for each of these.
>
> Take care that when running data related tests that the VM and the storage is
> running in the same region to avoid data costs.

### Running experiments

Many of these experiments are designed to be thrown away, but can be reran if
needed. To run them in the cloud, we recommend you use a spot VM. A spot VM
lives for at most 24 hours (and might be shut down any time within those
24 hours).

> [!WARNING]
> No data will be persisted on the VM with the configuration below when it is
> automatically shut down.

To create a VM that is suitable for testing:
```
gcloud compute instances create instance-20250409-082242 \
    --zone=europe-west1-b \
    --machine-type=n2-standard-8 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --max-run-duration=86400s \
    --local-ssd-recovery-timeout=1 \
    --service-account=716476042137-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --create-disk=auto-delete=yes,boot=yes,device-name=instance-20250409-082242,image=projects/debian-cloud/global/images/debian-12-bookworm-v20250311,mode=rw,size=10,type=pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any \
    --metadata=startup-script='
#!/bin/bash
set -eux

# Create mount point
mkdir -p /mnt/disks/local-ssd

# Format the local SSD (if not already)
if ! blkid /dev/nvme0n1; then
  mkfs.ext4 -F /dev/nvme0n1
fi

# Mount it
mount /dev/nvme0n1 /mnt/disks/local-ssd

# Optional: add to fstab so it mounts again if the VM reboots
echo "/dev/nvme0n1 /mnt/disks/local-ssd ext4 defaults 0 0" >> /etc/fstab
' \
    --local-ssd=interface=NVME
```

Then, `ssh` into the VM just created (for example, using `gcloud compute ssh`).

Then, run the following commands to get the environment set up:
```
sudo apt update
sudo apt install -y git python-is-python3 python3-pip pipx tmux
pipx install poetry
git clone https://github.com/energyandcleanair/india_map
cd india_map
export PATH=$PATH:/home/panda/.local/bin/
poetry install
```

Then, run the script you want:
```
poetry run python <path to script>
```

