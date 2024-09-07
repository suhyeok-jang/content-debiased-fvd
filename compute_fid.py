from cleanfid import fid
import os
import argparse

def main(args):
    # 기준 real_1 경로와 출력 파일 경로
    real_image_dir = f'/home/jsh/content-debiased-fvd/{args.dataset}/cliptoimages/real_1'
    outfile = os.path.join(f'/home/jsh/content-debiased-fvd/{args.dataset}/cliptoimages', 'fid.tsv')
    
    with open(outfile, 'a') as f:
        print("idx\tdistortion_type\tseverity\tfid\tincrease_ratio", file=f, flush=True)

        # ref_fid_score를 real_2와 비교하여 계산
        fake_image_dir = f'/home/jsh/content-debiased-fvd/{args.dataset}/cliptoimages/fake/mix_fixed/0.2'
        if os.path.exists(fake_image_dir):
            fid_score = fid.compute_fid(fake_image_dir, dataset_name=args.dataset, mode="clean", dataset_split="custom")
             # 결과를 TSV에 기록
            print("{}\t{}\t{}\t{:.2f}\t{:.2f}".format(
                0, 'mix_fixed', .2, fid_score, 0.0), file=f, flush=True)
        else:
            raise FileNotFoundError(f"Reference directory {fake_image_dir} not found.")

        # # fake 이미지들의 하위 디렉터리를 탐색
        # fake_base_dir = f'/home/jsh/content-debiased-fvd/{args.dataset}/cliptoimages/fake'

        # i = 0
        # for root, dirs, files in os.walk(fake_base_dir):
        #     if len(files) > 0:  # 파일이 존재하는 경우
        #         fake_image_dir = root

        #         # fake_image_dir 경로가 존재하면 FID 계산
        #         if os.path.exists(fake_image_dir):
        #             i += 1
        #             fid_score = fid.compute_fid(fake_image_dir, dataset_name=args.dataset, mode="clean", dataset_split="custom")

        #             # 증가 비율 계산
        #             increase_ratio = (fid_score - ref_fid_score) / ref_fid_score * 100

        #             # distortion_type은 fake 하위의 첫 번째 디렉토리 이름
        #             relative_path = os.path.relpath(fake_image_dir, fake_base_dir)
        #             path_parts = relative_path.split(os.sep)

        #             # distortion_type 설정
        #             distortion_type = path_parts[0] if len(path_parts) > 0 else 'Unknown'

        #             # severity 설정 (두 번째 부분이 있는 경우 severity로 사용, 없으면 None)
        #             severity = path_parts[1] if len(path_parts) > 1 else None

        #             # 결과를 TSV에 기록
        #             print("{}\t{}\t{}\t{:.2f}\t{:.2f}".format(
        #                 i, distortion_type, severity, fid_score, increase_ratio), file=f, flush=True)

    print("모든 FID 계산이 완료되었습니다.")

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "6,7"

    parser = argparse.ArgumentParser(description='Compute the latent representation of video clips by VideoMAE')
    parser.add_argument("--dataset", type=str, choices=['kinetics400', 'UCF-101'])

    args = parser.parse_args()

    main(args)