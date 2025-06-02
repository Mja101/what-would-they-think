import unittest
import tempfile
import os
import pandas as pd
from app import data_utils

class TestDataUtils(unittest.TestCase):
    def test_replace_links(self):
        text = "Check this out: http://example.com and https://test.com"
        replaced = data_utils.replace_links(text)
        self.assertEqual(replaced, "Check this out: [LINK] and [LINK]")

    def test_parse_whatsapp_chat_basic(self):
        # Create a fake WhatsApp export
        chat = (
            "12/06/24, 9:00 am - Bob: Hello!\n"
            "12/06/24, 9:01 am - Jane: Hi Bob!\n"
            "12/06/24, 9:05 am - Bob: How are you?\n"
            "12/06/24, 10:10 am - Jane: Good, thanks!\n"
        )
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as tmp:
            tmp.write(chat)
            tmp_path = tmp.name

        df = data_utils.parse_whatsapp_chat(tmp_path, remove_media=False, convert_emojis=False, max_gap_minutes=30)
        os.remove(tmp_path)

        self.assertEqual(list(df['sender']), ["Bob", "Jane", "Bob", "Jane"])
        self.assertEqual(list(df['message']), ["Hello!", "Hi Bob!", "How are you?", "Good, thanks!"])
        self.assertIn('conversation_id', df.columns)
        # Should split into 2 conversations due to time gap > 30 min
        self.assertEqual(df['conversation_id'].nunique(), 2)

    def test_parse_whatsapp_chat_continuation(self):
        chat = (
            "12/06/24, 9:00 am - Bob: Hello!\n"
            "This is a continuation.\n"
            "12/06/24, 9:01 am - Jane: Hi!\n"
        )
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as tmp:
            tmp.write(chat)
            tmp_path = tmp.name

        df = data_utils.parse_whatsapp_chat(tmp_path)
        os.remove(tmp_path)

        self.assertIn("Hello! This is a continuation.", list(df['message']))

    def test_parse_whatsapp_chat_remove_media(self):
        chat = (
            "12/06/24, 9:00 am - Bob: <Media omitted>\n"
            "12/06/24, 9:01 am - Jane: Hi!\n"
        )
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as tmp:
            tmp.write(chat)
            tmp_path = tmp.name

        df = data_utils.parse_whatsapp_chat(tmp_path, remove_media=True)
        os.remove(tmp_path)

        self.assertNotIn("<Media omitted>", df['message'].values)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['sender'], "Jane")

    def test_save_parsed_chat(self):
        df = pd.DataFrame({
            "datetime": ["2024-06-12 09:00:00"],
            "sender": ["Bob"],
            "message": ["Hello!"],
            "conversation_id": [0]
        })
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name

        data_utils.save_parsed_chat(df, tmp_path)
        loaded = pd.read_csv(tmp_path)
        os.remove(tmp_path)

        self.assertIn("sender", loaded.columns)
        self.assertEqual(loaded.iloc[0]["sender"], "Bob")

if __name__ == "__main__":
    unittest.main()
