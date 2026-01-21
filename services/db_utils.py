import os

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, select, delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

DB_SESSION_URL = os.getenv("DB_SESSION_URL")
engine = create_async_engine(DB_SESSION_URL, pool_pre_ping=False, echo=True, connect_args={
        "prepared_statement_cache_size": 0,
        "statement_cache_size": 0
    })

AsyncSessionFactory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
Base = declarative_base()

class User(Base):
    __tablename__ = "user"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    def __repr__(self):
        return f"<User(user_id = {self.user_id}, username = {self.username})>"

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    refresh_token = Column(String)
    user_id = Column(Integer, ForeignKey("user.user_id"), primary_key=True)
    expires = Column(DateTime(timezone=True), nullable=False)
    username = Column(String, ForeignKey("user.username"), nullable=False)
    def __repr__(self):
        return f"<RefreshToken(user_id = {self.user_id}, expiry = {self.expires}, refresh_token = {self.refresh_token})>"

class DocumentInfo(Base):
    __tablename__ = "document_info"
    user_id = Column(Integer, ForeignKey("user.user_id"), primary_key=True)
    document_id = Column(Integer, primary_key=True) # Document ID, not unique, composite key with user id
    storage_path = Column(String, nullable=False, unique=True)
    original_filename = Column(String, nullable=False)
    vector_db_id = Column(String, ForeignKey("vector_stores.collection_name"))
    last_updated = Column(DateTime, nullable=False)

class VectorDB(Base):
    __tablename__ = "vector_stores"
    user_id = Column(Integer, ForeignKey("user.user_id"), primary_key=True)
    collection_name = Column(String, primary_key=True, unique=True)

class UserSecrets(Base):
    __tablename__ = "user_secrets"
    user_id = Column(Integer, ForeignKey("user.user_id"), primary_key=True)
    provider = Column(String, nullable=False)
    secret_key = Column(String, nullable=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db_session():
    async with AsyncSessionFactory() as session:
        yield session

async def get_user_by_username(username: str, session: AsyncSession):
    query = select(User).where(User.username == username)
    result = await session.execute(query)
    user = result.scalar_one_or_none()
    return user

async def get_refresh_token_by_uid(user_id: int, session: AsyncSession):
    query = select(RefreshToken).where(RefreshToken.user_id == user_id)
    result = await session.execute(query)
    refresh_token = result.scalar_one_or_none()
    return refresh_token

async def add_or_update_refresh_token(refresh_token: str, user_id: int, username: str, expires:datetime, session: AsyncSession):
    refresh_token_entry = RefreshToken(refresh_token=refresh_token,
                                       username=username,
                                       user_id=user_id,
                                       expires=expires)
    try:
        await session.merge(refresh_token_entry)
        await session.commit()
    except SQLAlchemyError:
        await session.rollback()
        raise

async def add_user(username: str, hashed_password: str, session: AsyncSession):
    user_entry = User(username=username,
                      hashed_password=hashed_password)
    session.add(user_entry)
    try:
        await session.commit()
        return user_entry
    except SQLAlchemyError:
        await session.rollback()
        raise

async def update_user_password(user_id: int, hashed_password_existing: str, hashed_password_new: str,session: AsyncSession):
    user = await session.get(User, user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")

    if hashed_password_existing != user.hashed_password:
        raise ValueError(f"Old Password does not match")
    user.hashed_password = hashed_password_new

    await session.commit()
    return user

async def retrieve_user_documents(userid: int, session: AsyncSession):
    query = select(DocumentInfo).where(DocumentInfo.user_id == userid)
    result = await session.execute(query)
    return result.scalars().all()

async def retrieve_document_info(uid: int, file_name: str,session: AsyncSession):
    stmt = select(DocumentInfo).where(DocumentInfo.user_id == uid,
                                      DocumentInfo.original_filename == file_name)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()

async def retrieve_user_vectorstore_ids(userid: int, session: AsyncSession):
    query = select(VectorDB).where(VectorDB.user_id == userid)
    result = await session.execute(query)
    return result.scalars().all()

async def add_user_vectorstore(user_id: int, vectorstore_name: str, session: AsyncSession):
    col_entry = VectorDB(user_id=user_id, collection_name=vectorstore_name)
    session.add(col_entry)
    try:
        await session.commit()
    except SQLAlchemyError:
        await session.rollback()
        raise

async def get_document_by_id(uid: int, document_id: int, session: AsyncSession):
    query = select(DocumentInfo).where(DocumentInfo.document_id == document_id, DocumentInfo.user_id == uid)
    result = await session.execute(query)
    return result.scalar_one_or_none()

async def add_user_document(userid: int,
                            document_id: int,
                            filepath: str,
                            original_name: str,
                            vector_db: str,
                            session: AsyncSession):
    doc_entry = DocumentInfo(
        user_id=userid,
        document_id=document_id,
        storage_path=filepath,
        original_filename=original_name,
        vector_db_id=vector_db,
        last_updated=datetime.now(),
    )
    session.add(doc_entry)
    try:
        await session.commit()
    except SQLAlchemyError:
        await session.rollback()
        raise

async def delete_user_document(userid: int, document_id: int, session: AsyncSession):
    stmt = delete(DocumentInfo).where(DocumentInfo.document_id == document_id, DocumentInfo.user_id == userid)
    result = await session.execute(stmt)
    if result:
        await session.commit()

async def update_user_document(user_id: int,document_id:int, session: AsyncSession):
    """
    DB function to update the last_modified of a document. No change to other fields because storage location and collections name should not change.
    """
    stmt = select(DocumentInfo).where(
        DocumentInfo.user_id == user_id,
        DocumentInfo.document_id == document_id
    )
    result = await session.execute(stmt)
    result = result.scalar_one_or_none()
    if result is None:
        raise ValueError(f"Document {document_id} not found")
    result.last_updated = datetime.now()

    await session.commit()
    return result