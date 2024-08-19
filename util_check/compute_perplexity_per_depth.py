import torch
import pandas as pd

def calculate_perplexity(flatten_ind, codebook_size, device='cpu'):
    # b는 flatten_ind의 길이로 설정
    b = flatten_ind.shape[0]

    # encoding tensor 초기화
    encoding = torch.zeros(b, codebook_size, device=device)
    encoding.scatter_(1, flatten_ind.reshape([-1, 1]), 1)
    
    # avg_probs 계산
    avg_probs = torch.mean(encoding, dim=0)
    
    # perplexity 계산
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-12)))

    return perplexity

# 예시: flatten 된 인덱스와 codebook_size를 사용하여 perplexity 계산
if __name__ == "__main__":
    df_full_sorted = pd.read_csv('util_check/slot_util/codebook_distribution.csv')
    codebook_size = df_full_sorted['Value'].max() + 1

    df_full_sorted = df_full_sorted.dropna()
    flatten_ind = torch.tensor(df_full_sorted['Value'].repeat(df_full_sorted['Count']).values, dtype=torch.long)
    device = 'cpu'

    perplexity = calculate_perplexity(flatten_ind, codebook_size, device)
    print(f"Perplexity: {perplexity.item()}")