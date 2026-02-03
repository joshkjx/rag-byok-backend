from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List,Optional
from dotenv import load_dotenv
import services.doc_processing as doctools
import chromadb
import os

load_dotenv()

class VStore:
    def __init__(self, name:str, client, embeddings_model:HuggingFaceEmbeddings): # Currently chroma db is assumed - will refactor when scaling up becomes a possibility.
        """
        class container for a vectorstore collection
        :param name: the name of the collection
        :param client: the API reference to the chromadb client
        :param embeddings_model: the reference to the HuggingFaceEmbeddings object used for vector embedding
        """
        self.name = name
        self.client = client
        self.embeddings = embeddings_model
        self.collection = None

    def build(self, doc_splits: Optional[List[Document]]):
        """
        Builds the vectorstore collection in the chroma persistent client
        :param doc_splits: List of Documents that have been chunked
        :return: langchain Chroma object.
        """
        existing_collections = [col.name for col in self.client.list_collections()]
        if self.name in existing_collections:
            print(f"Collection for {self.name} already exists, loading from file...")
            self.collection = Chroma(
                client=self.client,
                embedding_function=self.embeddings,
                collection_name=self.name,
            )
        else:
            print(f"Creating ChromaDB collection for {self.name}...")
            if not doc_splits:
                self.collection = Chroma(
                    client=self.client,
                    embedding_function=self.embeddings,
                    collection_name=self.name,
                )
            else:
                self.collection = Chroma.from_documents(
                    client=self.client,
                    embedding=self.embeddings,
                    collection_name=self.name,
                    documents=doc_splits
                )
        return self.collection

    def set_retriever(self):
        """
        grabs the collection and sets it as a retriever
        """
        return self.collection.as_retriever(search_type="similarity",search_kwargs={'k': 3})

    async def update_collection_with_splits(self,splits:List[Document], doc_id: int):
        """
        Checks if the source exists in vectorstore, then uploads it to the collection. If source already exists, existing vectors are discarded.
        :param: splits (List[Document]) the uploaded file splits to be embedded. All splits assumed to belong to same document.
        :param: doc_id (int) the id of the document
        :return: True if successful, False otherwise.
        """
        success = await self._create_or_update_file_in_collection(splits, doc_id)
        return success

    async def _add_splits_to_vectorstore(self, splits:List[Document]):
        """
        Adds a file to the vectorstore
        :param splits (List[Document]) the uploaded file splits to be embedded. All splits assumed to belong to same document.
        :return: true if split was successful, false otherwise
        """
        if splits:
            await self.collection.aadd_documents(splits)
            return True
        return False

    async def _create_or_update_file_in_collection(self, splits:List[Document], doc_id:int):
        """
        Deletes embeddings with the same document id as the given splits and regenerates them with the new info.
        :param splits (List[Document]): the uploaded file splits to be embedded. All splits assumed to belong to same document.
        :param doc_id (int) the id of the document
        :return: True if successful, False otherwise.
        """
        current = self.collection.get(where={"document_id": doc_id}) # Check if embeddings for this document id exist
        current_ids = current['ids']

        if current_ids:
            await self.collection.adelete(ids=current_ids) # deletes existing embeddings tagged to the source name

        success = await self._add_splits_to_vectorstore(splits)
        return success

    async def delete_documents(self, doc_id:int):
        """
        Deletes embeddings with the given document id
        :param doc_id: (int) document id - retrieved from documents db
        :return: True if successful deletion, False otherwise.
        """
        relevant = self.collection.get(where={"document_id": doc_id})
        relevant_ids = relevant['ids']
        if relevant_ids:
            success = await self.collection.adelete(ids=relevant_ids)
        else:
            success = False
        return success


def initialise_chroma_client():
    """
    Initialises the chroma persistent client
    :return: (ClientAPI) reference to the chromadb client
    """

    if os.getenv('ENVIRONMENT') == 'dev':
        persist_path = os.getenv("CHROMADB_PATH")
        return chromadb.PersistentClient(path=persist_path)
    else:
        return chromadb.HttpClient(
            host = os.getenv("CHROMADB_HOST", "localhost"),
            port=int(os.getenv("CHROMADB_PORT", 8000)),
        )

ACTIVE_CLIENT = initialise_chroma_client()
embed_model = os.getenv("EMBEDDING_MODEL")
COMPUTED_EMBEDDINGS = doctools.initialise_embeddings_model(embed_model)

def get_chroma_client():
    return ACTIVE_CLIENT

def get_embeddings() -> HuggingFaceEmbeddings:
    return COMPUTED_EMBEDDINGS

def create_vectorstore(doc_splits: Optional[List[Document]] = None, vectorstore_name:str = None) -> VStore:
    """
    Builds a vectorstore and sets its collection as retriever
    :param vectorstore_name: Collection name, for separation and uniqueness
    :param doc_splits: list of chunked documents for embedding (Optional)
    :return: VStore object
    """
    client = get_chroma_client()
    embeddings = get_embeddings()
    vstore = VStore(
        name = vectorstore_name,
        client=client,
        embeddings_model=embeddings)
    vstore.build(doc_splits)

    return vstore