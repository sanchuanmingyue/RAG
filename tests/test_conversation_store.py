"""Persistence checks for SQLite-backed conversations."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from backend.conversation_store import ConversationStore
from backend.memory import AgentMemory
from backend.services import AgentSessionStore


class ConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "conversations.db"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_exchange_state_and_artifact_survive_store_restart(self) -> None:
        store = ConversationStore(self.db_path, user_id="tester")
        conversation_id = store.create_conversation()
        memory = AgentMemory(current_paper_id="paper-a")
        artifact = {
            "artifact_id": "classification-v1",
            "paper_count": 3,
            "category_count": 2,
            "categories": [
                {"rank": 1, "label": "视觉语言模型", "count": 2},
                {"rank": 2, "label": "检索增强生成", "count": 1},
            ],
        }
        memory.remember_result(
            "已将 3 篇论文分为两类。",
            [{"text": "source", "metadata": {"paper_id": "paper-a"}}],
            intent="corpus_analysis",
            user_query="按研究背景分类",
            rewritten_query="对当前论文库按研究背景进行分类",
            artifacts={"corpus_analysis": artifact},
        )

        store.save_exchange(
            conversation_id,
            user_content="按研究背景分类",
            assistant_content="已将 3 篇论文分为两类。",
            memory=memory,
            intent="corpus_analysis",
            rewritten_query="对当前论文库按研究背景进行分类",
            assistant_result={"category_count": 2},
            paper_id="paper-a",
        )

        reopened = ConversationStore(self.db_path, user_id="tester")
        messages = reopened.load_messages(conversation_id)
        restored = reopened.load_memory(conversation_id)
        conversations = reopened.list_conversations()

        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["result"], {"category_count": 2})
        self.assertEqual(restored.current_paper_id, "paper-a")
        self.assertEqual(restored.state.last_intent, "corpus_analysis")
        self.assertEqual(restored.state.active_artifact_id, "classification-v1")
        self.assertEqual(restored.state.active_artifact["category_count"], 2)
        self.assertEqual(len(restored.history), 2)
        self.assertEqual(conversations[0]["message_count"], 2)
        self.assertEqual(conversations[0]["title"], "按研究背景分类")

    def test_artifact_lineage_and_clear(self) -> None:
        store = ConversationStore(self.db_path)
        conversation_id = store.create_conversation()
        memory = AgentMemory()
        memory.remember_result(
            "初次分类",
            [],
            intent="corpus_analysis",
            user_query="分类",
            artifacts={
                "corpus_analysis": {
                    "artifact_id": "classification-v1",
                    "categories": [],
                }
            },
        )
        store.save_memory(conversation_id, memory)

        memory.remember_result(
            "细化分类",
            [],
            intent="corpus_analysis",
            user_query="再详细一点",
            artifacts={
                "corpus_analysis": {
                    "artifact_id": "classification-v2",
                    "refined_from_artifact_id": "classification-v1",
                    "categories": [],
                }
            },
        )
        store.save_memory(conversation_id, memory)

        artifacts = store.list_artifacts(conversation_id)
        self.assertEqual([item["id"] for item in artifacts], ["classification-v2", "classification-v1"])
        self.assertEqual(artifacts[0]["parent_artifact_id"], "classification-v1")

        store.clear_conversation(conversation_id)
        self.assertEqual(store.load_messages(conversation_id), [])
        self.assertEqual(store.list_artifacts(conversation_id), [])
        self.assertIsNone(store.load_memory(conversation_id).state.last_intent)
        self.assertEqual(store.list_conversations()[0]["title"], "新会话")

    def test_api_session_memory_is_loaded_from_sqlite(self) -> None:
        store = ConversationStore(self.db_path)
        sessions = AgentSessionStore(store)
        with sessions.session("api-session") as memory:
            memory.remember_result(
                "回答",
                [],
                intent="qa",
                user_query="第一问",
                rewritten_query="完整的第一问",
            )
            sessions.persist_exchange(
                "api-session",
                user_content="第一问",
                assistant_content="回答",
                intent="qa",
                memory=memory,
            )

        restarted_sessions = AgentSessionStore(ConversationStore(self.db_path))
        with restarted_sessions.session("api-session") as restored:
            self.assertEqual(restored.state.last_intent, "qa")
            self.assertEqual(restored.state.last_user_query, "第一问")
            self.assertEqual(len(restored.history), 2)


if __name__ == "__main__":
    unittest.main()
