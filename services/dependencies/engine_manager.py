from typing import Dict, TYPE_CHECKING
from services.document_io import get_user_vectorstore

if TYPE_CHECKING:
    from services.inference_engine import InferenceEngine


class EngineManager:
    """
    Manages InferenceEngine instances per user
    """
    def __init__(self):
        self.engines : Dict[str, "InferenceEngine"] = {}

    async def get_or_create_engine(self, user_id: int, user_settings, db_session) -> "InferenceEngine":
        """Get cached engine or create a new one if it doesn't exist"""
        new_settings = user_settings

        if str(user_id) in self.engines:
            # check to see if anything has changed
            current_engine = self.engines[str(user_id)]

            if new_settings.provider == current_engine.provider and new_settings.model_name == current_engine.modelname and new_settings.api_key == current_engine.api_key:
                return current_engine  # early return if nothing changed, passes cached engine

            await current_engine.__aexit__(None, None, None) # cleanup if need to generate new engine

        engine = await self._create_engine(user_id, user_settings, db_session) # create a brand-new engine if one doesn't exist

        return engine

    async def _create_engine(self, user_id: int, user_settings, db_session) -> "InferenceEngine":
        from services.inference_engine import InferenceEngine
        vstore = await get_user_vectorstore(user_id, db_session)
        if vstore:
            engine = InferenceEngine(
                vectorstore=vstore,
                provider=user_settings.provider,
                modelname=user_settings.model_name,
                api_key=user_settings.api_key,
            )
            await engine.__aenter__()
            self.engines[str(user_id)] = engine
        return engine


    async def cleanup(self):
        """Cleanup all engines on shutdown"""
        for engine in self.engines.values():
            await engine.__aexit__(None, None, None)

_engine_manager: EngineManager | None = None

def get_engine_manager():
    """Dependency to inject engine manager"""
    if _engine_manager is None:
        raise RuntimeError("Engine manager not initialized")
    return _engine_manager