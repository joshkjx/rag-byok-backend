from typing import Optional
import boto3
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
BUCKET_NAME = os.getenv("DOCUMENTS_BUCKET_NAME")

class FileIOClient:
    def __init__(self, client):
        self._client = client

    async def upload(self, path: str, content: bytes, content_type: str, bucket: Optional[str] = None):
        bucket_name = bucket if bucket else BUCKET_NAME
        if not bucket_name:
            raise ValueError("DOCUMENTS_BUCKET_NAME environment variable is not set")

        try:  # storing the document in s3
            await asyncio.to_thread(
                self._client.put_object,
                    Bucket=bucket_name,
                    Key=path,
                    Body=content,
                    ContentType=content_type,
            )
            return {"success": True, "path": path}
        except:
            raise

    async def download(self, path: str, bucket: Optional[str] = None) -> bytes:
        bucket_name = bucket if bucket else BUCKET_NAME
        if not bucket_name:
            raise ValueError("DOCUMENTS_BUCKET_NAME environment variable is not set")
        try:
            response = await asyncio.to_thread(
                self._client.get_object,
                Bucket=bucket_name,
                Key=path
            )
            file_bytes = await asyncio.to_thread(response["Body"].read)
            return file_bytes
        except:
            raise

    async def delete(self, path: str, bucket: Optional[str] = None):
        bucket_name = bucket if bucket else BUCKET_NAME
        if not bucket_name:
            raise ValueError("DOCUMENTS_BUCKET_NAME environment variable is not set")
        try:
            await asyncio.to_thread(
                self._client.delete_object,
                Bucket=bucket_name,
                Key=path
            )
            return {"success": True, "path": path}
        except:
            raise

_CLIENT : FileIOClient | None = None

async def get_file_client() -> FileIOClient:
    global _CLIENT
    if not _CLIENT:
        client = boto3.client("s3",
                              endpoint_url=os.getenv('S3_ENDPOINT'),
                              aws_access_key_id=os.getenv('AWS_S3_ACCESS_KEY_ID'),
                              aws_secret_access_key=os.getenv('AWS_S3_SECRET_ACCESS_KEY'),
                              region_name=os.getenv('AWS_S3_REGION_NAME'))
        _CLIENT = FileIOClient(client)
    return _CLIENT