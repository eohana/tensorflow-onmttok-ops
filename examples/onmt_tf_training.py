import argparse
import logging
import opennmt as onmt
import tensorflow as tf

from tensorflow_onmttok import register_opennmt_in_graph_tokenizer

tf.get_logger().setLevel(logging.INFO)
register_opennmt_in_graph_tokenizer()  # <-- registers the in graph tokenizer


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run", choices=["train", "translate"],
                        help="Run type.")
    parser.add_argument("--src", required=True,
                        help="Path to the source file.")
    parser.add_argument("--tgt",
                        help="Path to the target file.")
    parser.add_argument("--src_vocab", required=True,
                        help="Path to the source vocabulary.")
    parser.add_argument("--tgt_vocab", required=True,
                        help="Path to the target vocabulary.")
    parser.add_argument("--model_dir", default="checkpoint",
                        help="Directory where checkpoint are written.")
    args = parser.parse_args()

    # See http://opennmt.net/OpenNMT-tf/configuration.html for a complete specification
    # of the configuration.
    config = {
        "model_dir": args.model_dir,
        "data": {
            "source_vocabulary": args.src_vocab,
            "target_vocabulary": args.tgt_vocab,
            "train_features_file": args.src,
            "train_labels_file": args.tgt,
        }
    }

    model = onmt.models.TransformerBase()
    runner = onmt.Runner(model, config, auto_config=True)

    if args.run == "train":
        runner.train()
    elif args.run == "translate":
        runner.infer(args.src)


if __name__ == "__main__":
    main()
