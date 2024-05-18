import argparse
import pandas as pd
from process_text import TextProcessor, process_df
from parse_tar import open_tar, get_path_files

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process text files to create datasets.')
    parser.add_argument('--taiga', action='store_true', help='Use this flag to process files from Taiga. If not set, will process files from Twitter.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    taiga = args.taiga
    if taiga:
        open_tar('social.tar')
        file_paths = get_path_files('data', 'texts')
        group_sizes = {1: 1000, 2: 1000, 3: 1000, 4: 742}
        processor = TextProcessor(max_group_sizes=group_sizes)
        processor.read_and_process_files_taiga(file_paths)
    else:
        file_paths = get_path_files('twitter', 'twitter')
        group_sizes = {1: 0, 2: 0, 3: 0, 4: 258}
        processor = TextProcessor(max_group_sizes=group_sizes)
        processor.read_and_process_files_twitter(file_paths)
    df = processor.groups_to_dataframe()
    df = process_df(df)
    df.to_csv('parsed_prompts.csv', index=False, encoding='utf-8')
