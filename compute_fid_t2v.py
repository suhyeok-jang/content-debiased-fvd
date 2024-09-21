from cleanfid import fid
import os
import argparse

def main(args):
    # 기준 real_1 경로와 출력 파일 경로
    real_image_dir = f'/home/jsh/content-debiased-fvd/{args.dataset}/cliptoimages/real_1/{args.frames}'
    outfile = os.path.join(f'/home/jsh/content-debiased-fvd/{args.dataset}/cliptoimages', 'fid.tsv')
    
    with open(outfile, 'a') as f:
        print("idx\tdistortion_type\tfid\tincrease_ratio", file=f, flush=True)

        # i=0
        # real_2_image_dir = real_image_dir.replace('real_1','real_2')
        # if os.path.exists(real_2_image_dir):
        #     ref_fid_score = fid.compute_fid(real_2_image_dir, dataset_name=args.dataset, mode="clean", dataset_split="custom")
        #      # 결과를 TSV에 기록
        #     print("{}\t{}\t{:.2f}\t{:.2f}".format(
        #         i, 'real_2', ref_fid_score, 0.0), file=f, flush=True)
        # else:
        #     raise FileNotFoundError(f"Reference directory {real_2_image_dir} not found.")
        i=0
        for generation_type in ['videocrafter_fifo']:
            fake_image_dir = real_image_dir.replace('real_1', generation_type)
            if os.path.exists(fake_image_dir):
                i+=1
                
                if args.frames == 16:
                    ref_fid_score = 12.32
                else:
                    ref_fid_score = 6.71
                    
                fid_score = fid.compute_fid(fake_image_dir, dataset_name=args.dataset, mode="clean", dataset_split="custom")

                # 증가 비율 계산
                increase_ratio = (fid_score - ref_fid_score) / ref_fid_score * 100

                # 결과를 TSV에 기록
                print("{}\t{}\t{:.2f}\t{:.2f}".format(
                    i,  generation_type, fid_score, increase_ratio), file=f, flush=True)

    print("모든 FID 계산이 완료되었습니다.")

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"

    parser = argparse.ArgumentParser(description='Compute the latent representation of video clips by VideoMAE')
    parser.add_argument("--dataset", type=str, choices=['kinetics400', 'UCF-101'])
    parser.add_argument("--frames", type=int, default = 16)
    

    args = parser.parse_args()

    main(args)