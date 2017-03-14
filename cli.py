import argparse
import os

import run
import demo


def get_parser():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    parser.add_argument('--squad_dir', default=os.path.join(home, 'data', 'squad'))
    parser.add_argument('--glove_dir', default=os.path.join(home, 'data', 'glove'))
    parser.add_argument('--out_dir', default='out')
    parser.add_argument("--fresh", action='store_true')
    parser.add_argument('-d', "--draft", action='store_true')
    parser.add_argument('-t', "--train", action='store_true')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--model', default='cbow')
    parser.add_argument('--device_type', default='cpu')
    parser.add_argument('--num_devices', type=int, default=1)

    parser.add_argument('--word_count_th', default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--init_lr', type=float, default=0.5)
    parser.add_argument('--max_context_size', type=int, default=150)
    parser.add_argument('--max_ques_size', type=int, default=15)

    parser.add_argument('--save_model_secs', type=int, default=120)
    parser.add_argument('--summary_period', type=int, default=50)
    parser.add_argument('--val_period', type=int, default=1000)
    parser.add_argument('--queue_capacity', type=int, default=100000)
    parser.add_argument('--min_after_dequeue', type=int, default=1000)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default="0.0.0.0")
    return parser


def auto(config, data_type):
    # auto-complete the configurations for the ease of using the command line arguments
    config.data_type = data_type
    config.squad_path = os.path.join(config.squad_dir, "{}_small.json".format(data_type))
    config.glove_path = os.path.join(config.glove_dir, "glove.6B.50d.txt")
    config.metadata_path = os.path.join(config.out_dir, "metadata_{}.json".format(data_type))
    config.data_path = os.path.join(config.out_dir, "data_{}.json".format(data_type))
    config.common_path = os.path.join(config.out_dir, "common.json")
    config.emb_metadata_path = os.path.join(config.out_dir, "words.tsv")
    config.supervised = True
    if data_type == 'train':
        config.is_train = True
    else:
        config.save_model_secs = 0
        config.is_train = False


def main():
    parser = get_parser()
    config = parser.parse_args()
    if config.train:
        auto(config, 'train')
        val_config = parser.parse_args()
        auto(val_config, 'val')
        val_config.train = False
        run.train(config, val_config=val_config)
    else:
        if config.serve:
            auto(config, 'serve')
            config.fresh = True
            demo.demo(config)
        else:
            auto(config, 'test')
            run.test(config)

if __name__ == "__main__":
    main()

