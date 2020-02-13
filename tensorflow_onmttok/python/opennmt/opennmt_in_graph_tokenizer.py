import copy

from tensorflow_onmttok.python.ops.onmttok_ops import detokenize, tokenize

try:
    from opennmt.tokenizers import tokenizer
except ImportError:
    pass


def register_opennmt_in_graph_tokenizer():
    @tokenizer.register_tokenizer
    class OpenNMTInGraphTokenizer(tokenizer.Tokenizer):
        def __init__(self, **kwargs):
            self._config = copy.deepcopy(kwargs)

        @property
        def in_graph(self):
            return True

        def _tokenize_tensor(self, text):
            return tokenize(text, **self._config)

        def _tokenize_string(self, text):
            tokens = self._tokenize_tensor(text).numpy()
            return [t.decode('utf-8') for t in tokens]

        def _detokenize_tensor(self, tokens):
            return detokenize(tokens, **self._config)

        def _detokenize_string(self, tokens):
            text = self._detokenize_tensor(tokens).numpy()
            return text[0].decode('utf-8')
