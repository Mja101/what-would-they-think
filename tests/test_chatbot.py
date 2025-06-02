import unittest
from unittest.mock import patch, MagicMock
from app import chatbot

class TestChatbot(unittest.TestCase):
    @patch("app.chatbot.AutoTokenizer")
    @patch("app.chatbot.AutoModelForCausalLM")
    def test_load_model(self, mock_model_cls, mock_tokenizer_cls):
        """Test that load_model returns a tokenizer and model."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        # Patch model_path in app.constants since chatbot.py imports it from there
        with patch("app.constants.model_path", "dummy_path"):
            tokenizer, model = chatbot.load_model()
            self.assertEqual(tokenizer, mock_tokenizer)
            self.assertEqual(model, mock_model)

    @patch("app.chatbot.load_model")
    @patch("app.chatbot.emoji.emojize")
    def test_generate_response(self, mock_emojize, mock_load_model):
        """Test generate_response constructs prompt and extracts response."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load_model.return_value = (mock_tokenizer, mock_model)

        # Simulate encoding and model output
        mock_inputs = MagicMock()
        mock_inputs.shape = (1, 10)
        mock_tokenizer.encode.return_value = mock_inputs

        # The output that generate returns should be passed to decode
        fake_output = MagicMock()
        mock_model.generate.return_value = fake_output
        mock_tokenizer.decode.return_value = ": Hello\nBob: Did you go gym or did you go home on Tuesday??\n"
        mock_emojize.side_effect = lambda x, language=None: x  # passthrough

        chat_history = [('<Bob>', 'Hello')]
        current_sender = '<Bob>'

        response = chatbot.generate_response(chat_history, current_sender)
        self.assertEqual(response, 'Did you go gym or did you go home on Tuesday??')

        prompt_arg = mock_tokenizer.encode.call_args[0][0]
        self.assertIn("<Bob>: Hello\nBob:", prompt_arg)

if __name__ == "__main__":
    unittest.main()