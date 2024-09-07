import warnings
warnings.filterwarnings('ignore')
import os
import argparse

from cdfvd import fvd

def main(args):
    
    outfile = os.path.join(os.path.dirname(os.path.dirname(args.fake_stat_root)), 'fvd_direc_378.tsv')
    
    with open(outfile, 'a') as f:
        print("idx\tdistortion_type\tseverity\tfvd\tincrease_ratio", file=f, flush=True)

        evaluator = fvd.cdfvd('videomae', device='cpu')
        evaluator.load_real_stats(args.real_stat_path)
        
        i = 0
        evaluator.load_fake_stats(args.real_stat_path.replace('real_1','real_2'))
        ref_fvd_score = evaluator.compute_fvd_from_stats()
        
        print("{}\t{}\t{}\t{:.2f}\t{:.1f}".format(
                i, 'real_2(ref)', None, ref_fvd_score, 0.0), file=f, flush=True)
        
        evaluator.empty_fake_stats()

        for root, dirs, files in os.walk(args.fake_stat_root):
            if len(files) > 0:  # 파일이 존재하는 경우
                fake_stat_dir = root

                # fake_stat_dir 경로가 존재하면 FID 계산
                if os.path.exists(fake_stat_dir):
                    i += 1
                    fake_stat_path = os.path.join(fake_stat_dir,'videomae_feature.pkl')
                    evaluator.load_fake_stats(fake_stat_path)
                    fvd_score = evaluator.compute_fvd_from_stats()
                    evaluator.empty_fake_stats()
                    
                    # 증가 비율 계산
                    increase_ratio = (fvd_score - ref_fvd_score) / ref_fvd_score * 100

                    # distortion_type은 fake 하위의 첫 번째 디렉토리 이름
                    relative_path = os.path.relpath(fake_stat_dir, args.fake_stat_root)
                    path_parts = relative_path.split(os.sep)

                    # distortion_type 설정
                    distortion_type = path_parts[0] if len(path_parts) > 0 else 'Unknown'

                    # severity 설정 (두 번째 부분이 있는 경우 severity로 사용, 없으면 None)
                    severity = path_parts[1] if len(path_parts) > 1 else None

                    # 결과를 TSV에 기록
                    print("{}\t{}\t{}\t{:.2f}\t{:.1f}".format(
                        i, distortion_type, severity, fvd_score, increase_ratio), file=f, flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantify the VideoMAE FVD temporal sensitivity with distortion methods')
    parser.add_argument("--real_stat_path", type=str)
    parser.add_argument("--fake_stat_root", type=str)
    args = parser.parse_args()
    
    main(args)