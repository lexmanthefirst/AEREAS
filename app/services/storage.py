import io
import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile
from typing import Optional

from app.core.config import settings
from app.utils.logger import logger


class StorageService:
    """S3 compatible Minio storage service"""
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION
        )
        self.bucket = settings.S3_BUCKET_NAME
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self.s3_client.create_bucket(Bucket=self.bucket)
                logger.info("Created bucket: %s", self.bucket)
            except Exception as e:
                logger.warning("Could not create bucket %s: %s", self.bucket, e)
    
    def upload_file(self, file: UploadFile, key: str) -> str:
        """
        Upload a file to object storage.
        
        Args:
            file: FastAPI UploadFile object
            key: S3 object key (path)
            
        Returns:
            The object key
        """
        try:
            self.s3_client.upload_fileobj(
                file.file,
                self.bucket,
                key,
                ExtraArgs={'ContentType': file.content_type or 'application/octet-stream'}
            )
            logger.info("Uploaded file %s to %s", key, self.bucket)
            return key
        except ClientError as e:
            logger.error("Failed to upload file %s: %s", key, e)
            raise
    
    def upload_bytes(self, data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes to storage."""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type
            )
            return key
        except ClientError as e:
            logger.error("Failed to upload bytes %s: %s", key, e)
            raise
    
    def get_file(self, key: str) -> io.BytesIO:
        """
        Download a file from object storage.
        
        Args:
            key: S3 object key
            
        Returns:
            BytesIO object containing file data
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            return io.BytesIO(response['Body'].read())
        except ClientError as e:
            logger.error("Failed to download file %s: %s", key, e)
            raise
    
    def delete_file(self, key: str) -> bool:
        """Delete a file from storage."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            logger.error("Failed to delete file %s: %s", key, e)
            return False
    
    def get_presigned_url(self, key: str, expiration: int = 3600) -> Optional[str]:
        """Generate a presigned URL for file download."""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error("Failed to generate presigned URL for %s: %s", key, e)
            return None


# Singleton instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get or create the storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
