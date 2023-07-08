import argparse
import json
def create_parser_disease_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'aug_dataset/')
    parser.add_argument('--epochs', type=int, default = 2)
    parser.add_argument('--lr', type=float, default = 1e-3)
    parser.add_argument('--es', type=int, default = 1)
    parser.add_argument('--batch_size', type=int, default = 16)
    parser.add_argument('--file2read', type=int, default = -1)
    parser.add_argument('--output', type=str, default = 'output_file')
    parser.add_argument('--experiment_tag', type=str, default = 'alg')
    
     
    
    return parser

    