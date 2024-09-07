import warnings
warnings.filterwarnings('ignore')
import os
import argparse

from cdfvd import fvd

def main(args):
    
    for i,f_name in enumerate(['fvd_128to224.tsv', 'fvd_256to224.tsv','fvd_direc_224.tsv']):
        if i == 0:
            real_stat_path = '/home/jsh/content-debiased-fvd/stats/new/fvd_128/kinetics400/real_1/128_128_2048/videomae_feature.pkl'
            fake_stat_path = '/home/jsh/content-debiased-fvd/stats/new/fvd_128/kinetics400/fake/128_128_2048/mix_fixed/0.2/videomae_feature.pkl'
        elif i == 1:
            real_stat_path = '/home/jsh/content-debiased-fvd/stats/new/fvd_128/kinetics400/real_1/256_128_2048/videomae_feature.pkl'
            fake_stat_path = '/home/jsh/content-debiased-fvd/stats/new/fvd_128/kinetics400/fake/256_128_2048/mix_fixed/0.2/videomae_feature.pkl'
        elif i == 2:
            real_stat_path = '/home/jsh/content-debiased-fvd/stats/new/fvd_128/kinetics400/real_1/224_128_2048/videomae_feature.pkl'
            fake_stat_path = '/home/jsh/content-debiased-fvd/stats/new/fvd_128/kinetics400/fake/224_128_2048/mix_fixed/0.2/videomae_feature.pkl'
          
        outfile = os.path.join('/home/jsh/content-debiased-fvd/stats/new/fvd_128/kinetics400', f_name)
        
        with open(outfile, 'a') as f:
            print("idx\tdistortion_type\tseverity\tfvd\tincrease_ratio", file=f, flush=True)

            evaluator = fvd.cdfvd('videomae', device='cpu')
            
            evaluator.load_real_stats(real_stat_path)
            evaluator.load_fake_stats(fake_stat_path)
            
            fvd_score = evaluator.compute_fvd_from_stats()
            
            print("{}\t{}\t{}\t{:.2f}\t{:.1f}".format(
                        i, 'mix_fixed', 0.2, fvd_score, 0.0), file=f, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantify the VideoMAE FVD temporal sensitivity with distortion methods')
    parser.add_argument("--real_stat_path", type=str)
    parser.add_argument("--fake_stat_root", type=str)
    args = parser.parse_args()
    
    main(args)