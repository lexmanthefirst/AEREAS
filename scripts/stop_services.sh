#!/usr/bin/env bash
# Stop AEREAS infrastructure containers.
# Usage: bash scripts/stop_services.sh

set -euo pipefail

info() { echo -e "\033[0;32m[+]\033[0m $*"; }

for name in acrev_db acrev_minio; do
    if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
        docker stop "$name"
        info "Stopped $name"
    else
        info "$name not running — skipping."
    fi
done

info "All services stopped."
