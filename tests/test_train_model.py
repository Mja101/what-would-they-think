import unittest
import tempfile
import os
import pandas as pd
from app.train_model import load_and_preprocess_data

class TestTrainModel(unittest.TestCase):
    def test_load_and_preprocess_data(self):
        # Create a fake CSV with conversation data
        df = pd.DataFrame({
            "message": ["Hello!", "Hi Bob!", "How are you?", "Good, thanks!"],
            "sender": ["Bob", "Jane", "Bob", "Jane"],
            "conversation_id": [0, 0, 0, 0]
        })
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        tokenized_dataset, tokenizer = load_and_preprocess_data(tmp_path)
        os.remove(tmp_path)

        # Check that the dataset is not empty and has the expected fields
        self.assertTrue(len(tokenized_dataset) > 0)
        self.assertIn("input_ids", tokenized_dataset[0])
        self.assertIn("labels", tokenized_dataset[0])
        # Check that the tokenizer has the special speaker tokens
        self.assertTrue(any("<Bob>" in t for t in tokenizer.additional_special_tokens))
        self.assertTrue(any("<Jane>" in t for t in tokenizer.additional_special_tokens))

if __name__ == "__main__":
    unittest.main()