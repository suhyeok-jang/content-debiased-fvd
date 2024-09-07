from cleanfid import fid
import argparse

def main(args):
    #기존 제거
    fid.remove_custom_stats(args.dataset, mode="clean")
    fid.make_custom_stats(args.dataset, f'/home/jsh/content-debiased-fvd/{args.dataset}/cliptoimages/real_1', mode="clean")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the latent representation of video clips by VideoMAE')
    parser.add_argument("--dataset", type=str, choices=['UCF-101', 'kinetics400'])

    args = parser.parse_args()

    main(args)