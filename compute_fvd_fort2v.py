import warnings
warnings.filterwarnings('ignore')
import os
import argparse

from cdfvd import fvd

def main(args):
    
    outfile = os.path.join('/home/jsh/content-debiased-fvd/stats/fvd_128/UCF-101', 'fvd_256to224.tsv')
    
    with open(outfile, 'a') as f:
        print("idx\tdistortion_type\tfvd\tincrease_ratio", file=f, flush=True)

        evaluator = fvd.cdfvd('videomae', device='cuda')
        evaluator.load_real_stats(args.real_stat_path)

        i = 0
        evaluator.load_fake_stats(args.real_stat_path.replace('real_1','real_2'))
        # evaluator.load_fake_stats(args.real_stat_path.replace('256_16_2048','256_128_2048'))
        ref_fvd_score = evaluator.compute_fvd_from_stats()
        
        print(ref_fvd_score)
        
        print("{}\t{}\t{:.2f}\t{:.1f}".format(
                i, 'Another 2048 clips', ref_fvd_score, 0.0), file=f, flush=True)
        
        evaluator.empty_fake_stats()
        
        for generation_type in ['videocrafter_freenoise','videocrafter_fifo']:
            evaluator.load_fake_stats(args.real_stat_path.replace('real_1', generation_type).replace('_x1',''))
            fvd_score = evaluator.compute_fvd_from_stats()
            evaluator.empty_fake_stats()
            
            # 증가 비율 계산
            increase_ratio = (fvd_score - ref_fvd_score) / ref_fvd_score * 100
            
            # 결과를 TSV에 기록
            print("{}\t{}\t{:.2f}\t{:.1f}".format(
                i, generation_type, fvd_score, increase_ratio), file=f, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantify the VideoMAE FVD temporal sensitivity with distortion methods')
    parser.add_argument("--real_stat_path", type=str)
    # parser.add_argument("--generation_type", type=str)
    args = parser.parse_args()
    
    main(args)