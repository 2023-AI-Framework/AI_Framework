import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    
    parser.add_argument('--batch_size', required=True, type=int, help='batch size')
    parser.add_argument('--model', required=True, type=str, help='model name', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--num_workers', required=True, type=int)
    parser.add_argument('--gradient_accumulation_steps', required=True, type=str, help='GA')
    parser.add_argument('--mixed_precision', required=True, type=str, help='mixed_precision')
    
    parser.add_argument('--tag', required=False, default=None, type=str, help='tag')
    
    
    args = parser.parse_args()
    return args