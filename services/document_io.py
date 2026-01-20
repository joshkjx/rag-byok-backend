import uuid

from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException, APIRouter, UploadFile, File, Form
from services.file_management_utils import get_file_client, FileIOClient
from pydantic import BaseModel
import logging

from services.doc_processing import load_and_split_uploaded_document
from services.vectorstore import create_vectorstore, VStore
import services.db_utils as db
import services.auth_service as auth

ACCEPTABLE_FILETYPES = ["application/pdf", "document/pdf"]

router = APIRouter(prefix='/api/documents', tags=['document_utils'])
logger = logging.getLogger("uvicorn.error")

class DeleteDocumentRequest(BaseModel):
    document_id: int

@router.get('/get')
async def get_user_docs(user = Depends(auth.get_current_user), db = Depends(db.get_db_session)): # Dependency injection should handle the cookie extraction
    return await retrieve_document_list(user, db)

@router.post('/upload')
async def upload_new_document(user = Depends(auth.get_current_user),
                              db_session: AsyncSession = Depends(db.get_db_session),
                              file: UploadFile = File(...),
                              file_content_type: str = Form(...),
                              client: FileIOClient = Depends(get_file_client)):
    try:
        upload_info = await _upload_document(
                           uid=user,
                           file=file,
                           file_content_type=file_content_type,
                           client=client,
                           db_session=db_session)
    except:
        raise

    return upload_info

@router.delete('/delete')
async def delete_document(request:DeleteDocumentRequest,
                          user = Depends(auth.get_current_user),
                          db_session: AsyncSession = Depends(db.get_db_session),
                          client: FileIOClient = Depends(get_file_client)):

    document_id = request.document_id

    await _delete_document(uid=user,
                           document_id = document_id,
                           db_session= db_session,
                           client=client)
    return {"success": True}

async def get_user_vectorstore(user_id: int = None, db_session: AsyncSession = None) -> VStore:
    if not user_id:
        raise HTTPException(status_code=404, detail={
            "message": "No valid user id provided.",
            "error_code": "INVALID_USER_ID",
        })
    collection = await db.retrieve_user_vectorstore_ids(user_id, db_session) # gets collection name from db, if it exists
    collection = collection[0].collection_name if collection else None

    if not collection: # create if not exist
        collection = "vstore_user_{}".format(str(user_id))
        try:
            await db.add_user_vectorstore(user_id = user_id,
                                              vectorstore_name = collection,
                                              session=db_session)
        except:
            raise

    vstore = create_vectorstore(vectorstore_name=collection)
    return vstore


async def retrieve_document_list(uid:int, db_session: AsyncSession) -> list:
    results = await db.retrieve_user_documents(uid, db_session)
    if not results:
        return []
    return [
        {
            "document_id": document.document_id,
            "original_filename": document.original_filename,
            "storage_path": document.storage_path,
            "vector_db_id": document.vector_db_id,
        }
        for document in results
    ]

async def _upload_document(uid:int, file: UploadFile, file_content_type: str, db_session: AsyncSession, client: FileIOClient) -> dict:
    """
    Uploads a file, splits it, vectorizes splits, and updates DocumentInfo db
    """
    splits: list = []
    content_type = file_content_type
    if content_type not in ACCEPTABLE_FILETYPES:
        raise HTTPException(400, detail={
            "message": "Uploaded file is not of a supported file type",
            "error_code": "UNSUPPORTED_FILE_TYPE",})

    file_ext = ".pdf" if content_type in ["application/pdf","document/pdf"] else None

    filename = file.filename

    document = await db.retrieve_document_info(uid, filename, db_session) # Duplication check
    if document:
        document_id = document.document_id
        doc_exists = True
    else:
        for attempt in range(5): # For/Else retry loop in case non-unique doc_id
            uuid_value = uuid.uuid4()
            document_id: int = int(uuid_value) % 214748367 # we need to hash it to a 32bit integer so it plays nicely with db
            existing = await db.get_document_by_id(uid=uid,document_id=document_id,session=db_session)
            if not existing:
                break
        else:
            raise HTTPException(500, detail="Failed to generate unique document ID")
        doc_exists = False

    storage_path = f"{str(uid)}/documents/{str(document_id)}{file_ext}"

    content = await file.read()

    if client: # Attempt File Upload
        try:
            await client.upload(
            path=storage_path,
            content=content,
            content_type=content_type,
        )
        except:
            raise

    if content: # split files
        splits: list[Document] = load_and_split_uploaded_document(content=content,
                                                                  source=file.filename,
                                                                  document_id = document_id)

    if not splits:
        raise HTTPException(400, detail={
            "message": "Error while splitting uploaded file",
            "error_code": "DOC_SPLIT_FAILED",
        })

    vectorstore_collection: VStore = await get_user_vectorstore(uid, db_session) # builds VStore instance by searching for collection name

    try:
        await vectorstore_collection.update_collection_with_splits(splits, document_id) # Update collection with new embeddings
    except:
        raise

    if doc_exists: # update DocumentInfo db
        await db.update_user_document(uid, document_id, db_session)
    else:
        vstore_collection_name: str = vectorstore_collection.name
        if vstore_collection_name:
            await db.add_user_document(
                userid=uid,
                document_id=document_id,
                filepath=storage_path,
                original_name=filename,
                vector_db=vstore_collection_name,
                session=db_session,)

    return {"filename": filename,
            "filetype": file_ext,}

async def _delete_document(uid:int, document_id:int, db_session:AsyncSession, client: FileIOClient):
    document = await db.get_document_by_id(uid, document_id, db_session)
    if not document:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Document not found",
                "error_code": "DOC_NOT_FOUND",
            }
        )

    await db.delete_user_document(uid, document_id, db_session)

    if client:
        try:
            await client.delete(document.storage_path)
        except Exception as e:
            logger.error(
                f"Failed to delete document {document_id} due to {e}",
                exc_info=True,
                extra={
                    "event_type": "storage_deletion_failed",
                    "uid": uid,
                    "document_id": document_id,
                    "storage_path": document.storage_path,
                }
            )
