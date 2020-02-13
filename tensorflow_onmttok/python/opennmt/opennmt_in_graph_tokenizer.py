from tensorflow_onmttok.python.ops.onmttok_ops import detokenize, tokenize

try:
    from opennmt.tokenizers import tokenizer, OpenNMTTokenizer
except ImportError:
    pass


def register_opennmt_in_graph_tokenizer():
    @tokenizer.register_tokenizer
    class OpenNMTInGraphTokenizer(OpenNMTTokenizer):
        @property
        def in_graph(self):
            return True

        def _tokenize_tensor(self, text):
            return tokenize(text, **self._config)

        def _detokenize_tensor(self, tokens):
            return detokenize(tokens, **self._config)
