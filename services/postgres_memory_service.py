import datetime
from typing import List

from sqlalchemy import Column, Integer, String, Text, DateTime, Index, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from overrides import override
from google.genai import types as genai_types

from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session

Base = declarative_base()


class MemoryEntryModel(Base):
    __tablename__ = "memory_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    app_name = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    session_id = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    author = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("idx_memory_app_user", "app_name", "user_id"),)


class PostgresMemoryService(BaseMemoryService):
    """
    Long-term memory backed by Postgres using SQLAlchemy async.
    """

    def __init__(self, db_url: str):
        self._engine = create_async_engine(db_url, echo=False, future=True)
        self._session_factory = sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )
        self._initialized = False
        # you can call `await self.init()` from your app startup

    async def init(self) -> None:
        if self._initialized:
            return
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._initialized = True

    @override
    async def add_session_to_memory(self, session: Session) -> None:
        """
        Ingest a finished session into long-term memory.
        You can customize what you store (e.g. only user messages, only summaries, etc.)
        """
        if not self._initialized:
            await self.init()

        if not session.events:
            return

        async with self._session_factory() as db:
            for event in session.events:
                # Extract plain text from Content parts
                txt_parts: List[str] = []
                if event.content and event.content.parts:
                    for p in event.content.parts:
                        text = getattr(p, "text", None)
                        if text:
                            txt_parts.append(text)

                content_text = "".join(txt_parts).strip()
                if not content_text:
                    continue

                # Convert timestamp to datetime if it's a float/int (Unix timestamp)
                if event.timestamp:
                    if isinstance(event.timestamp, (int, float)):
                        ts = datetime.datetime.fromtimestamp(event.timestamp, tz=datetime.timezone.utc)
                    else:
                        ts = event.timestamp
                else:
                    ts = datetime.datetime.now(datetime.timezone.utc)

                db.add(
                    MemoryEntryModel(
                        app_name=session.app_name,
                        user_id=session.user_id,
                        session_id=session.id,
                        content=content_text,
                        author=event.author,
                        timestamp=ts,
                    )
                )

            await db.commit()

    @override
    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        """
        Very basic search: SQL LIKE.
        You can upgrade this later to Postgres full-text search or pgvector.
        """
        if not self._initialized:
            await self.init()

        async with self._session_factory() as db:
            stmt = (
                select(MemoryEntryModel)
                .where(
                    MemoryEntryModel.app_name == app_name,
                    MemoryEntryModel.user_id == user_id,
                    MemoryEntryModel.content.ilike(f"%{query}%"),
                )
                .order_by(MemoryEntryModel.id.desc())
                .limit(20)
            )
            rows = (await db.execute(stmt)).scalars().all()

        memories: List[MemoryEntry] = []
        for row in rows:
            content_obj = genai_types.Content(
                role="user" if (row.author or "").lower() == "user" else "model",
                parts=[genai_types.Part(text=row.content)],
            )
            memories.append(
                MemoryEntry(
                    content=content_obj,
                    author=row.author,
                    timestamp=row.timestamp.isoformat() if row.timestamp else None,
                )
            )

        return SearchMemoryResponse(memories=memories)
