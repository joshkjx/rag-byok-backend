import os
import uuid, json

from dotenv import load_dotenv
from langgraph.graph import StateGraph,add_messages,START,END
from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, ToolMessage, message_to_dict, messages_to_dict
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from services.dependencies.engine_manager import get_engine_manager
from services.vectorstore import VStore
import services.assets.prompts as prompts
import services.auth_service as auth
import services.db_utils as db
from pydantic import BaseModel,Field
from typing import Literal, Annotated, Tuple, Optional
from typing_extensions import TypedDict

import logging

class State(TypedDict):
    messages: Annotated[list, add_messages]

load_dotenv()

SUPPORTED_PROVIDERS = ["anthropic", "openai"]

GRADE_PROMPT = prompts.GRADE_PROMPT
REWRITE_PROMPT = prompts.REWRITE_PROMPT
GENERATE_ANSWER_PROMPT = prompts.GENERATE_ANSWER_PROMPT
ROUTER_PROMPT = prompts.ROUTER_PROMPT
DIRECT_RESPONSE_PROMPT = prompts.DIRECT_RESPONSE_PROMPT

logger = logging.getLogger("uvicorn.error")
router = APIRouter(prefix='/api/agent', tags=['agent'])

class UserSettings(BaseModel):
    provider: str
    model_name: str
    temperature: float | None = None
    api_key: str | None

class InferenceRequest(BaseModel):
    question:str
    settings: UserSettings
    conversation_state:State | None = None

@router.post("/query")
async def query(query_req: InferenceRequest,
                user_id: int = Depends(auth.get_current_user),
                db_session = Depends(db.get_db_session),
                engine_manager = Depends(get_engine_manager)):

    logger.info(f"Query received from user: {user_id}")
    logger.info(f"Question: {query_req.question}")
    # Get user settings
    user_settings: UserSettings = query_req.settings

    # Get or Create engine
    engine = await engine_manager.get_or_create_engine(user_id, user_settings, db_session)

    async def responses_generator():

        async for stage in engine.query_streaming(
            query_req.question,
            conversation_state=query_req.conversation_state,
        ):
            if stage['type'] == 'final_state':
                logger.info(f"State stored as of Final state: {stage['state']}")

            if 'messages' in stage:
                stage['messages'] = [
                    message_to_dict(msg) for msg in stage['messages'] # converting messages to dict for serialization
                ]

            yield f"data: {json.dumps(stage)}\n\n"

    return StreamingResponse(responses_generator(), media_type="text/event-stream")

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant."
    )

class InferenceEngine:
    def __init__(self, vectorstore:VStore, provider:str, modelname:str, api_key:str):
        """
        An engine with the full RAG pipeline
        :param vectorstore: a VStore instance
        :param provider: Name of provider. Lowercase, must be supported
        :param modelname: name of model api.
        """
        self.collection = vectorstore
        self.retriever = self.collection.set_retriever()
        self.provider = provider.lower()
        self.modelname = modelname
        self.api_key = api_key
        self.response_model = self.init_model()
        self.grader_model = self.init_model()


        self.retrieve_tool = StructuredTool.from_function(
            func = lambda query: self._retrieve_documents(query),
            name = "retrieve_documents",
            description = "Search the document database for relevant information."
        )

        # Set up checkpointer variables
        self.db_type = os.getenv('CHECKPOINTER_DB_TYPE')
        self.db_conn_string = os.getenv('CHECKPOINTER_DB_CONN_STRING')
        self.checkpointer = None
        self.graph = None

    async def __aenter__(self):
        """Enter async context"""
        self.checkpointer = await self._init_db(
            db_type=self.db_type,
            conn_string= self.db_conn_string)
        self.graph = self._build_graph().compile(checkpointer=self.checkpointer)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context"""
        if self.checkpointer:
            await self.checkpointer.__aexit__(exc_type, exc_val, exc_tb)

    async def _init_db(self, db_type: str = "sqlite", conn_string: str = "checkpoints.sqlite"):
        if db_type == "sqlite":
            session = AsyncSqliteSaver.from_conn_string(conn_string)

        elif db_type == "postgres":
            session = AsyncPostgresSaver.from_conn_string(conn_string)

        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        await session.setup()
        return session if session else None

    def init_model(self):
        if self.provider in SUPPORTED_PROVIDERS:
            return init_chat_model(model_provider=self.provider,
                                 model=self.modelname,
                                 temperature=0,
                                 streaming=True,
                                   api_key=self.api_key
                                )
        else:
            print("Provider not supported.")
            raise NotImplementedError

    def _retrieve_documents(self, query:str):
        docs = self.retriever.invoke(query)
        return "\n".join([doc.page_content for doc in docs])

    def _build_conversation_state(self, message:str, conversation_state:Optional[State]=None) -> State:
        """
        Helper function for building conversation state for continued conversations
        """
        if conversation_state:
            state = conversation_state.copy()
            state["messages"].append(HumanMessage(content=message))
        else:
            state = State(messages=[HumanMessage(content=message)])
        return state

    def query(self, message: str, conversation_state: Optional[State]=None, thread_id: str = None):
        """
        Query RAG agent with optional conversation history.
        :param message: User's question
        :param conversation_state: Previous conversation state for followup
        :return: final State
        """
        if not thread_id:
            thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        state = self._build_conversation_state(message, conversation_state)
        result = self.graph.invoke(state, config)
        return {
            "type": "final_state",
            "thread_id": thread_id,
            "messages": result["messages"],
        }

    async def query_streaming(self,message: str, conversation_state: Optional[State]=None, thread_id: str = None):
        """
        Query RAG agent with optional conversation history. Streaming version.
        :param message: User's question
        :param conversation_state: Previous conversation state for followup
        :return: final State
        """
        if not thread_id:
            thread_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": thread_id}}

        state = self._build_conversation_state(message, conversation_state)

        async for event in self.graph.astream_events(state, config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                node_name = event.get('metadata', {}).get('langgraph_node')
                if node_name == "generate_answer" or node_name == "direct_answer":
                    yield {
                        "type": "token",
                        "content": content,
                        "step": event.get("name", "unknown"),
                    }
            elif kind == "on_chain_start":
                yield {
                    "type": "step_start",
                    "step": event.get("name"),
                }

        final_snapshot = await self.graph.aget_state(config)
        yield {
            "type": "final_state",
            "thread_id": thread_id,
            "messages": final_snapshot.values["messages"],
            "state":messages_to_dict(state['messages'])
        }

    def _get_step_type(self, node:str)-> str:
        """Categorize nodes for frontend handling."""
        step_types = {
            "agent": "thinking",
            "retrieve": "retrieval",
            "grade_documents": "evaluation",
            "rewrite_question": "rewrite",
            "generate_answer": "final_answer",
        }
        return step_types.get(node, "unknown")

    def route_question(self, state: State) -> State:
        """Decide whether to retrieve or answer directly"""
        messages = [{"role":"system", "content": ROUTER_PROMPT}] + state["messages"]
        response = self.response_model.bind_tools([self.retrieve_tool]).invoke(messages)
        # Force empty content if no tool was called
        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            response.content = ""

        return {"messages": [response]}

    def grade_documents(self, state: State) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question."""
        question, context = self._get_current_question_and_context(state)

        prompt = GRADE_PROMPT.format(
            question=question,
            context=context,)
        response = self.grader_model.with_structured_output(GradeDocuments).invoke([{"role": "user", "content": prompt}])

        score = response.binary_score

        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"

    def direct_answer(self, state: State) -> State:
        """Answer directly without retrieval"""
        system_prompt = DIRECT_RESPONSE_PROMPT
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = self.response_model.invoke(messages)
        return {"messages": [response]}

    def rewrite_question(self,state: State) -> State:
        """Rewrite the original user question."""

        question, _ = self._get_current_question_and_context(state)
        prompt = REWRITE_PROMPT.format(question=question)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=response.content)]}

    def generate_answer(self,state: State) -> State:
        """Generate the answer using context only from currently active question."""

        question, context = self._get_current_question_and_context(state)
        prompt = GENERATE_ANSWER_PROMPT.format(question=question, context=context)

        message = self._prepare_message_content(state,prompt) # This will modify the prompt depending on whether user wants to send conversation history or not.
        response = self.response_model.invoke(message)
        return {"messages": [response]}

    def _get_current_question_and_context(self, state: State) -> Tuple[str, str]:
        messages = state["messages"] # Check to find the latest human message - prevent retrieval leak from previous interactions
        question_idx = None
        for i in range(len(messages)-1, -1, -1): # iterate backward through message history to find the last human message
            if isinstance(messages[i], HumanMessage):
                question_idx = i
                break

        question = messages[question_idx].content

        recent_tool_messages = [m for m in messages[question_idx:] if isinstance(m, ToolMessage)]  # Robustness check to see if the model has called the tool.
        context = recent_tool_messages[-1].content if recent_tool_messages else ""
        return question, context

    def _build_graph(self) -> StateGraph:
        """
        Build the graph for the agent - does not compile yet; we want to compile later.
        :return: Uncompiled StateGraph
        """
        workflow = StateGraph(State)

        workflow.add_node("router",self.route_question)
        workflow.add_node("retrieve", ToolNode([self.retrieve_tool]))
        workflow.add_node("rewrite_question", self.rewrite_question)
        workflow.add_node("generate_answer",self.generate_answer)
        workflow.add_node("direct_answer", self.direct_answer)

        workflow.add_edge(START, "router")
        workflow.add_conditional_edges("router",
                                       tools_condition,
                                       {
                                           "tools": "retrieve",
                                           END: "direct_answer",
                                       },)
        workflow.add_conditional_edges("retrieve", self.grade_documents)
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "router")
        return workflow # Returns uncompiled for compilation with checkpointer

    def _prepare_message_content(self, state:State, content: str) -> list:
        """Helper function to build a message for extensibility based on State parameters. Planned future functionality."""
        return [{"role": "user", "content": content}]


