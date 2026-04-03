Write-Host "Starting PostgreSQL DataBase (v18.1+)..."
docker run --rm -d `
  --name acrev_db `
  -e POSTGRES_USER=acrev_user `
  -e POSTGRES_PASSWORD=acrev_password `
  -e POSTGRES_DB=acrev_db `
  -e PGDATA=/var/lib/postgresql/data `
  -p 5435:5432 `
  -v acrev_pgdata:/var/lib/postgresql `
  postgres:18.1-bookworm

Write-Host "Starting MinIO Storage..."
docker run --rm -d `
  --name acrev_minio `
  -e MINIO_ROOT_USER=minioadmin `
  -e MINIO_ROOT_PASSWORD=minioadmin `
  -p 9010:9000 `
  -p 9011:9001 `
  -v minio_data:/data `
  minio/minio:latest server /data --console-address ":9001"