import tqdm
import numpy as np

import metrics


def evaluate(model, test_loader, shots, device, foldername=""):
    ssd_total = 0
    mad_total = 0
    prd_total = 0
    cos_sim_total = 0
    snr_noise = 0
    snr_recon = 0
    snr_improvement = 0
    eval_points = 0
    
    restored_sig = []
    with tqdm(test_loader) as it:
        for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
            clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
            
            if shots > 1:
                output = 0
                for i in range(shots):
                    output+=model.denoising(noisy_batch)
                output /= shots
            else:
                output = model.denoising(noisy_batch) #B,1,L
            clean_batch = clean_batch.permute(0, 2, 1)
            noisy_batch = noisy_batch.permute(0, 2, 1)
            output = output.permute(0, 2, 1) #B,L,1
            out_numpy = output.cpu().detach().numpy()
            clean_numpy = clean_batch.cpu().detach().numpy()
            noisy_numpy = noisy_batch.cpu().detach().numpy()
            
            
            eval_points += len(output)
            ssd_total += np.sum(metrics.SSD(clean_numpy, out_numpy))
            mad_total += np.sum(metrics.MAD(clean_numpy, out_numpy))
            prd_total += np.sum(metrics.PRD(clean_numpy, out_numpy))
            cos_sim_total += np.sum(metrics.COS_SIM(clean_numpy, out_numpy))
            snr_noise += np.sum(metrics.SNR(clean_numpy, noisy_numpy))
            snr_recon += np.sum(metrics.SNR(clean_numpy, out_numpy))
            snr_improvement += np.sum(metrics.SNR_improvement(noisy_numpy, out_numpy, clean_numpy))
            restored_sig.append(out_numpy)
            
            it.set_postfix(
                ordered_dict={
                    "ssd_total": ssd_total/eval_points,
                    "mad_total": mad_total/eval_points,
                    "prd_total": prd_total/eval_points,
                    "cos_sim_total": cos_sim_total/eval_points,
                    "snr_in": snr_noise/eval_points,
                    "snr_out": snr_recon/eval_points,
                    "snr_improve": snr_improvement/eval_points,
                },
                refresh=True,
            )
    
    restored_sig = np.concatenate(restored_sig)
    
    #np.save(foldername + '/denoised.npy', restored_sig)
    
    print("ssd_total: ",ssd_total/eval_points)
    print("mad_total: ", mad_total/eval_points,)
    print("prd_total: ", prd_total/eval_points,)
    print("cos_sim_total: ", cos_sim_total/eval_points,)
    print("snr_in: ", snr_noise/eval_points,)
    print("snr_out: ", snr_recon/eval_points,)
    print("snr_improve: ", snr_improvement/eval_points,)