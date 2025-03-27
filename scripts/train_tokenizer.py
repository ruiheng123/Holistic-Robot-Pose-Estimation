import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from lib.core.function import farward_loss, validate
from lib.dataset.const import INITIAL_JOINT_ANGLE
from lib.models.full_net import get_rootNetwithRegInt_model
from lib.models.tokenizer import VectorQuantizeTokenizer
from lib.utils.urdf_robot import URDFRobot
from lib.utils.utils import set_random_seed, create_logger, get_dataloaders, get_scheduler, resume_run, save_checkpoint, copy_and_rename
from torchnet.meter import AverageValueMeter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from lib.utils.utils import cast

def train_vq_tokenizer(args):
    
    torch.autograd.set_detect_anomaly(True)
    set_random_seed(808)

    save_folder = os.path.join('experiments',  args.exp_name)
    log_folder = os.path.join(save_folder,  'log_tokenizer')
    writer = SummaryWriter(log_dir=log_folder)
    
    tokenizer_ckpt_folder = os.path.join(save_folder, 'tokenizer_pretrained')
    copy_and_rename(args.config_path, save_folder, "config.yaml")
    if not os.path.exists(tokenizer_ckpt_folder):
        os.makedirs(tokenizer_ckpt_folder)


    urdf_robot_name = args.urdf_robot_name
    robot = URDFRobot(urdf_robot_name)

    device_id = args.device_id
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    ds_iter_train, test_loader_dict = get_dataloaders(args)


# 提取每个 DataLoader 的 dataset
    init_param_dict = {
        "robot_type" : urdf_robot_name,
        "pose_params": INITIAL_JOINT_ANGLE,
        "cam_params": np.eye(4,dtype=float),
        "init_pose_from_mean": True
    }

    tokenizer = VectorQuantizeTokenizer(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            encoder_num_blocks=args.encoder_num_blocks,
            num_joints=args.num_joints,
            encoder_token_inter_dim=args.encoder_token_inter_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            encoder_hidden_inter_dim=args.encoder_hidden_inter_dim,
            encoder_dropout=args.encoder_dropout,
            token_num=args.token_num,
            token_class_num=args.token_class_num,
            token_dim=args.token_dim,
            ema_decay=args.ema_decay,
            decoder_num_blocks=args.decoder_num_blocks,
            decoder_hidden_dim=args.decoder_hidden_dim,
            decoder_hidden_inter_dim=args.decoder_hidden_inter_dim,
            decoder_token_inter_dim=args.decoder_token_inter_dim,
            decoder_p_dropout=args.decoder_p_dropout,
            stage= "tokenizer"
        )
    tokenizer.to(device)
    tokenizer.float()
    tokenizer = torch.nn.DataParallel(tokenizer, device_ids=device_id, output_device=device_id[0])

    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-4, weight_decay=args.tokenizer_weight_decay)

    start_epoch, last_epoch, end_epoch = 0, -1, args.n_tokenizer_epochs
        
    lr_scheduler = get_scheduler(args, optimizer, last_epoch)

    for epoch in range(start_epoch, end_epoch + 1):
        print(f'In epoch {epoch} ----------------- (script: supervised training for tokenizer) ---------------------')
        
        tokenizer.train()
        iterator = tqdm(ds_iter_train, dynamic_ncols=True)
        losses = AverageValueMeter()
        losses_reconstruct = AverageValueMeter()
        losses_codebook_latent = AverageValueMeter()
        for batchid, sample in enumerate(iterator):
            optimizer.zero_grad()
            # gt_keypoints2d_original = cast(sample["keypoints_2d_original"],device).float()
            gt_keypoints2d = cast(sample["other"]["keypoints_2d"],device).float()
            gt_keypoints3d = cast(sample["other"]["keypoints_3d"],device).float()
            gt_keypoints3d_homo = torch.cat([gt_keypoints3d, torch.ones_like(gt_keypoints3d[..., :1])], dim=-1)  # [B, N, 4]

            TCO = cast(sample["TCO"],device).float()
            TCO_inv = torch.linalg.inv(TCO)
            gt_transf_keypoints3d = (TCO_inv.unsqueeze(1) @ gt_keypoints3d_homo.unsqueeze(-1))[..., :3, 0]
            # gt_root_uv = gt_keypoints2d[:,args.reference_keypoint_id,0:2]
            # valid_mask = cast(sample["valid_mask"],device).float()
            # valid_mask_crop = cast(sample["other"]["valid_mask_crop"],device).float()
            # if vq_target == "3d_point":
            recounstruct_transf, _, e_latent_loss = tokenizer(gt_transf_keypoints3d, train=True)
            loss_3d_int = torch.norm(recounstruct_transf - gt_transf_keypoints3d, dim = 2)
            
            # recounstruct, _, e_latent_loss = tokenizer(gt_keypoints3d, train=True)
            # loss_3d_int = torch.norm(recounstruct - gt_keypoints3d, dim = 2)
            if batchid == 0:
                print(f"gt_keypoints3d_val[0]: {gt_transf_keypoints3d[0]}, recounstrtuct[0]: {recounstruct_transf[0]}")

            loss_3d_int = cast(loss_3d_int,device)
            loss_3d_int = torch.mean(loss_3d_int)
            loss = loss_3d_int + args.latent_loss_weight * e_latent_loss
            iterator.set_postfix(loss=f"{loss.item():.4f}", recon=f"{loss_3d_int.item():.4f}")
            if epoch > 25: 
                loss = loss_3d_int + 0.2 * e_latent_loss
            loss_dict = {
                "loss": loss,
                "loss_3d_int": loss_3d_int,
                "loss_latent": e_latent_loss
            }
            # elif vq_target == "2d_point":
            # gt_2d_with_mask = torch.cat([gt_keypoints2d, valid_mask_crop.unsqueeze(-1)], dim=2)
            # mask_3d = valid_mask_crop.unsqueeze(2).expand(-1, -1, 3)
            # gt_2d_with_mask = gt_2d_with_mask * mask_3d


            # recounstruct, _, e_latent_loss = tokenizer(gt_2d_with_mask, train=True)
            
            # # loss_2d = torch.norm((recounstruct - gt_keypoints2d)/args.image_size, dim = 1)
            # # loss_2d = cast(loss_2d,device) * valid_mask_crop.unsqueeze(-1)
            # # loss_2d = torch.sum(loss_2d) / torch.sum(valid_mask_crop[:,args.reference_keypoint_id] != 0)
            # loss_2d = F.smooth_l1_loss(recounstruct, gt_keypoints2d)
            # loss = loss_2d + args.latent_loss_weight * e_latent_loss
            # loss_dict = {
            #     "loss": loss,
            #     "loss_2d": loss_2d,
            #     "loss_latent": e_latent_loss
            # }
            loss.backward()
            if args.tokenizer_clip_gradient is not None:
                clipping_value = args.tokenizer_clip_gradient
                torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), clipping_value)
            optimizer.step()
            losses.add(loss.detach().cpu().numpy())
            # if vq_target == "3d_point":
            #     losses_reconstruct.add(loss_3d_int.detach().cpu().numpy())
            # elif vq_target == "2d_point":
            losses_reconstruct.add(loss_3d_int.detach().cpu().numpy())
            losses_codebook_latent.add(e_latent_loss.detach().cpu().numpy())
            if (batchid+1) % 100 == 0: 
                writer.add_scalar('Tokenizer/loss', losses.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Tokenizer/loss_reconstruct', losses_reconstruct.mean , epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Tokenizer/loss_latent', losses_codebook_latent.mean , epoch * len(ds_iter_train) + batchid + 1)
                losses.reset()
                losses_reconstruct.reset()
                losses_codebook_latent.reset()
            writer.add_scalar('LR/learning_rate_opti', optimizer.param_groups[0]['lr'], epoch * len(ds_iter_train) + batchid + 1)
            if args.use_schedule:
                lr_scheduler.step()
                for pgid in range(len(optimizer.param_groups)):
                    writer.add_scalar(f'Tokenizer/learning_rate_opti_{pgid}', optimizer.param_groups[pgid]['lr'], epoch * len(ds_iter_train) + batchid + 1)
        if args.use_schedule:
            lr_scheduler.step()


        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(tokenizer_ckpt_folder, f'epoch_{epoch+1}_tokenizer.pk')
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': tokenizer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_last_epoch':last_epoch,
                        }, save_path)
        print("Validate start!")
        with torch.no_grad():
            tokenizer.eval()
            for dsname, loader in test_loader_dict.items():
                val_iterator = tqdm(loader, dynamic_ncols=True, desc=f"Validating {dsname}: {epoch}")
                val_losses = AverageValueMeter()
                for idx, val_sample in enumerate(val_iterator):

                    
                    gt_keypoints3d_val = cast(val_sample["other"]["keypoints_3d"], device).float()
                    
                    gt_keypoints3d_homo_val = torch.cat([gt_keypoints3d_val, torch.ones_like(gt_keypoints3d_val[..., :1])], dim=-1)  # [B, N, 4]

                    TCO_val = cast(val_sample["TCO"],device).float()
                    TCO_inv_val = torch.linalg.inv(TCO_val)
                    
                        # print(TCO_val)
                        # print(TCO_inv_val)
                    gt_transf_keypoints3d_val = (TCO_inv_val.unsqueeze(1) @ gt_keypoints3d_homo_val.unsqueeze(-1))[..., :3, 0]
                    if dsname in ["azure", "kinect", "realsense", "orb"]:
                        gt_transf_keypoints3d_val[..., [0, 1, 2]] = gt_transf_keypoints3d_val[..., [2, 0, 1]]
                        # gt_transf_keypoints3d_val[..., 0] = -gt_transf_keypoints3d_val[..., 0]
                    recounstruct_transf_val, _, e_latent_loss = tokenizer(gt_transf_keypoints3d_val, train=False)
                    loss_3d_val = torch.norm(recounstruct_transf_val - gt_transf_keypoints3d_val, dim = 2)
                    # recounstruct_val, _, e_latent_loss = tokenizer(gt_keypoints3d_val, train=False)
                    if idx == 0:
                        print(f"gt_keypoints3d_val[0]: {gt_transf_keypoints3d_val[0]}, recounstruct_val[0]: {recounstruct_transf_val[0]}")
                    # loss_3d_val = torch.norm(recounstruct_val - gt_keypoints3d_val, dim = 2)

                    loss_3d_val = cast(loss_3d_val, device)
                    loss_3d_val = torch.mean(loss_3d_val)
                    val_iterator.set_postfix(error=f"{loss_3d_val.item():.4f}")
                    val_losses.add(loss_3d_val.detach().cpu().numpy())
                    if (idx+1) % 20 == 0: 
                        writer.add_scalar(f'Tokenizer/val_{dsname}_3d_error', val_losses.mean , epoch * len(loader) + idx + 1)
                        val_losses.reset()
        print("Validate end!")  
        
        