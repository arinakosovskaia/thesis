import pandas as pd
import json
import argparse

def create_message(row):
    label = 'Toxic' if row['label'] == 1 else 'Not toxic'
    return {
        "messages": [
            {"role": "system", "content": "Detect if the following comment is toxic."},
            {"role": "user", "content": row['text']},
            {"role": "assistant", "content": label}
        ]
    }

def convert_to_jsonl(input_file, output_file):
    df = pd.read_csv(input_file)
    with open(output_file, 'w', encoding='utf-8') as file:
        for index, row in df.iterrows():
            json_object = create_message(row)
            file.write(json.dumps(json_object, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CSV file to JSONL for OpenAI fine-tuning.')
    parser.add_argument('--input_file', type=str, required=True, help='The path to the input CSV file.')
    parser.add_argument('--output_file', type=str, required=True, help='The path to the output JSONL file.')
    args = parser.parse_args()
    convert_to_jsonl(args.input_file, args.output_file)
