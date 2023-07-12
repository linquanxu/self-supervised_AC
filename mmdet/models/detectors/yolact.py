# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_head
from .single_stage import SingleStageDetector
import torch.nn.functional as F
import cv2
import time


@DETECTORS.register_module()
class YOLACT(SingleStageDetector):
    """Implementation of `YOLACT <https://arxiv.org/abs/1904.02689>`_"""

    def __init__(self,
                 backbone,###for seg
                 backbone1,##for critic
                 backbone_pred,##for seg and prediction
                 neck,
                 bbox_head,
                 segm_head,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(YOLACT, self).__init__(backbone, backbone1, backbone_pred, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)
        self.segm_head = build_head(segm_head)
        self.mask_head = build_head(mask_head)
        self.cls_loss = torch.nn.L1Loss()
        self.predict_file = open('predicted_final.txt', 'w')
        self.save_flg = 0
        self.m = 550
        self.n = 550

    def imnormalize_(self, img, to_rgb=True):
        """Inplace normalize an image with mean and std.

        Args:
            img (ndarray): Image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.

        Returns:
            ndarray: The normalized image.
        """
        # cv2 inplace normalization does not accept uint8
        mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
        std = np.array([58.40, 57.12, 57.38], dtype=np.float32)
        assert img.dtype != np.uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    def imdenormalize(self, img, to_bgr=True):
        mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
        std = np.array([58.40, 57.12, 57.38], dtype=np.float32)
        assert img.dtype != np.uint8
        mean = mean.reshape(1, -1).astype(np.float64)
        std = std.reshape(1, -1).astype(np.float64)
        img = cv2.multiply(img, std)  # make a copy
        cv2.add(img, mean, img)  # inplace
        if to_bgr:
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
        return img

    

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        feat = self.extract_feat(img)
        bbox_outs = self.bbox_head(feat)
        prototypes = self.mask_head.forward_dummy(feat[0])
        return (bbox_outs, prototypes)
    
    def crop_seg(self, img, gt_bbox, gt_mask):
        new_img = img * gt_mask
        xmin, ymin, xmax, ymax = int(np.round(gt_bbox[0][0].item())), int(np.round(gt_bbox[0][1].item())), \
            int(np.round(gt_bbox[0][2].item())), int(np.round(gt_bbox[0][3].item()))
        new_img =new_img[:,ymin:ymax,xmin:xmax]
        new_img = F.interpolate(
                new_img.unsqueeze(0), (550, 550),
                mode='bilinear',
                align_corners=False).squeeze(0)
        # self.vison_img(img, new_img)
        return new_img
        
    def train_eye(self, img, img_metas, gt_bboxes, gt_masks=None):
            bs, c, h, w = img.size()
            # batch_imgs = []
            cls_target = []
            for batch in range(bs):
                # b_img = img[batch]
                # b_mask = gt_masks[batch]
                # if 0: ## corp mask for prediction net
                #     gt_bbox = gt_bboxes[batch]
                #     b_img_mask = self.crop_seg(b_img,gt_bbox,b_mask)
                # else:
                #     b_img_mask = b_img * b_mask
                
                # batch_imgs.append(b_img_mask)
                cls_target.append(float(img_metas[batch]['pre_value'].strip()))

            # batch_imgs = torch.stack(batch_imgs)
            x = self.backbone1(img)
            x = x[-1]
            x = self.gap(x)
            fsize = x.shape[1]
            x = x.view(-1, fsize)
            x = self.cfc(x)
            #x = self.extract_feat(batch_imgs,0) ###for training critic. make cirtic image could be crop.
            cls_target = torch.tensor(cls_target, dtype=torch.float32).view(x.shape[0], 1).cuda()
            classifiy_loss = self.cls_loss(x, cls_target)
            a_losses = {}
            a_losses['cls_loss'] = classifiy_loss
            return a_losses     
        
    def train_predict_net(self, img, img_metas, gt_bboxes, gt_masks=None):
            bs, c, h, w = img.size()
            batch_imgs = []
            cls_target = []
            for batch in range(bs):
                b_img = img[batch]
                b_mask = gt_masks[batch]
                if 1: ## corp mask for prediction net
                    gt_bbox = gt_bboxes[batch]
                    b_img_mask = self.crop_seg(b_img,gt_bbox,b_mask)
                else:
                    b_img_mask = b_img * b_mask
                
                batch_imgs.append(b_img_mask)
                cls_target.append(float(img_metas[batch]['pre_value'].strip()))

            batch_imgs = torch.stack(batch_imgs)
            x = self.backbone1(batch_imgs)
            x = x[-1]
            x = self.gap(x)
            fsize = x.shape[1]
            x = x.view(-1, fsize)
            x = self.cfc(x)
            cls_target = torch.tensor(cls_target, dtype=torch.float32).view(x.shape[0], 1).cuda()
            classifiy_loss = self.cls_loss(x, cls_target)
            a_losses = {}
            a_losses['cls_loss'] = classifiy_loss
            return a_losses    
    
    
    
    def train_seg(self,img,img_metas,gt_bboxes,gt_labels,gt_bboxes_ignore,gt_masks):
        x = self.extract_feat(img)
        cls_score, bbox_pred, coeff_pred = self.bbox_head(x)
        bbox_head_loss_inputs = (cls_score, bbox_pred) + (gt_bboxes, gt_labels,
                                                            img_metas)
        losses, sampling_results = self.bbox_head.loss(
            *bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        segm_head_outs = self.segm_head(x[0])
        loss_segm = self.segm_head.loss(segm_head_outs, gt_masks, gt_labels)
        losses.update(loss_segm)

        mask_pred, mask_pred_feature = self.mask_head(x[0], coeff_pred, gt_bboxes, img_metas,
                                                        sampling_results)
        loss_mask = self.mask_head.loss(mask_pred, gt_masks, gt_bboxes,
                                        img_metas, sampling_results, mask_pred_feature)
        losses.update(loss_mask)
        return losses,mask_pred,sampling_results,bbox_pred

    def train_ac(self,img,img_metas,gt_masks,mask_pred,sampling_results,losses,bbox_preds,gt_bboxes):
        bs, c, h, w = img.size()
        all_bbox_preds = torch.cat([b.permute(0, 2, 3, 1).reshape(bs, -1, 4) for b in bbox_preds], -2)
        
        critc_pred = []
        pred_gt = []
        for b in range(bs):
            gt_bbox = gt_bboxes[b]
            cur_p = mask_pred[b]
            num_pos = cur_p.shape[0]
            cur_p = F.interpolate(
                cur_p.unsqueeze(0), (550, 550),
                mode='bilinear',
                align_corners=False).squeeze(0) > 0.5
            list_img = []
            pre_value = []
            list_img_critic = []
            top1 = torch.nonzero(sampling_results[b].pos_inds == sampling_results[b].top1)[0][0]
            img_t = img[b]
            cur_p_s = cur_p[top1]
            gt_masks_t = gt_masks[b]
            # self.show_imgs(img_t,cur_p_s,gt_masks_t)
            # self.save_flg +=1 
            
            ### for critic net
            img_critic = img[b]
            if 1:
                sample_re = sampling_results[b]
                pos_inds = sample_re.pos_inds
                anchors = sample_re.pos_bboxes[top1]
                anchors = anchors.reshape((1,4))
                bbox_pred = all_bbox_preds[b][pos_inds]
                bbox_pred = bbox_pred[top1]
                bbox_pred = bbox_pred.reshape((1,4))
                bboxes = self.bbox_head.bbox_coder.decode( anchors, bbox_pred, max_shape=(550,550,3))
                img_c = self.crop_seg(img_t,bboxes,cur_p_s)
                img_critic = self.crop_seg(img_critic,gt_bbox,gt_masks_t)
            else:
                img_c = img_t * cur_p_s
                img_critic = img_critic * gt_masks_t
                
            list_img.append(img_c)
            pre_gt = torch.tensor(float(img_metas[b]['pre_value']), dtype=torch.float).cuda()
            pre_value.append(pre_gt)

            list_img_critic.append(img_critic)

            #self.save_flg += 1
            pre_input = torch.stack(list_img, 0)###list_img for seg mask
            pre_value_gt = torch.stack(pre_value, 0)
            pre_value_gt = pre_value_gt.view(1, -1)
            critic_img = torch.stack(list_img_critic, 0)

            ### for critic
            ax = self.backbone1(critic_img)
            ax = ax[-1]
            ax = self.gap(ax)
            ax = ax.view(-1, ax.shape[1])
            ax = self.cfc(ax)

            ## for seg to predicted
            sp = self.backbone_pred(pre_input)
            sp = sp[-1]
            sp = self.gap_pred(sp)
            sp = sp.view(-1, sp.shape[1])
            sp = self.cfc_pred(sp)

            sp_loss = self.cls_loss(sp, ax)####预测与critic的loss
            #seg_gt_loss = self.cls_loss(sp, pre_value_gt) ##预测与gt的loss
            sp_loss += sp_loss
            #seg_gt_loss += seg_gt_loss
            # sp_loss /= num_pos
            # seg_gt_loss /= num_pos
            critc_pred.append(sp_loss)
            #pred_gt.append(seg_gt_loss)
        losses.update({'critic_pred': critc_pred})
        losses.update({'pred_gt': pred_gt})
        
        # l_cls = losses['loss_cls']
        # l_bbox = losses['loss_bbox']
        # l_seg = losses['loss_segm']
        # l_mask = losses['loss_mask']
        # for x in range(len(l_cls)):
        #     l_cls[x].requires_grad = True
        #     l_bbox[x].requires_grad = True
        #     l_seg[x].requires_grad = True
        #     l_mask[x].requires_grad = True
        # check NaN and Inf
        for loss_name in losses.keys():
            assert torch.isfinite(torch.stack(losses[loss_name]))\
                .all().item(), '{} becomes infinite or NaN!'\
                .format(loss_name)
        return losses

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # convert Bitmap mask or Polygon Mask to Tensor here
        gt_masks = [
            gt_mask.to_tensor(dtype=torch.uint8, device=img.device)
            for gt_mask in gt_masks
        ]
        if 0:####train eye images
            loss = self.train_eye(img,img_metas,gt_bboxes, gt_masks)
            return loss           
        if 0:####只训练预测网络
            loss = self.train_predict_net(img,img_metas,gt_bboxes, gt_masks)
            return loss
        if 0:###############训练分割网络
            losses,mask_pred,sampling_results,bbox_pred = self.train_seg(img,img_metas,gt_bboxes,gt_labels,gt_bboxes_ignore,gt_masks)
            return losses
        if 1:####train ac
            losses,mask_pred,sampling_results,bbox_pred = self.train_seg(img,img_metas,gt_bboxes,gt_labels,gt_bboxes_ignore,gt_masks)
            losses = self.train_ac(img,img_metas,gt_masks,mask_pred,sampling_results,losses,bbox_pred,gt_bboxes)
            return losses

    def test_critic_net(self, img, img_metas,gt_bboxes, gt_masks):
        bs, c, h, w = img.size()
        batch_imgs = []
        cls_target = []
        for batch in range(bs):           
            b_img = img[batch]
            b_mask = gt_masks[batch][0]
            if 1:
                gt_bboxes = gt_bboxes[0]
                gt_bbox = gt_bboxes[batch]
                b_img_mask = self.crop_seg(b_img,gt_bbox,b_mask)
            else:
                b_img_mask = b_img * b_mask
            

            
            batch_imgs.append(b_img_mask)
            cls_target.append(float(img_metas[batch]['pre_value'].strip()))


        batch_imgs = torch.stack(batch_imgs)
        x = self.backbone1(batch_imgs)
        x = x[-1]
        x = self.gap(x)
        fsize = x.shape[1]
        x = x.view(-1, fsize)
        x = self.cfc(x)
        
        

        self.predict_file.write(str(img_metas[0]['pre_value']) + ', ' + str(x.cpu()[0].item()) + '\n')
      
    def test_eye(self, img,img_metas,gt_masks):
        x = self.backbone1(img)
        x = x[-1]
        x = self.gap(x)
        x = x.view(-1, 32)
        x = self.cfc(x)
        self.predict_file.write(str(img_metas[0]['pre_value']) + ', ' + str(x.cpu()[0].item()) + '\n')

    def test_seg_ac(self,img,img_metas,gt_masks,rescale):
        feat = self.extract_feat(img)
        det_bboxes, det_labels, det_coeffs = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bbox, det_label, self.bbox_head.num_classes)
            for det_bbox, det_label in zip(det_bboxes, det_labels)
        ]

        segm_results,segm_pred = self.mask_head.simple_test(
            feat,
            det_bboxes,
            det_labels,
            det_coeffs,
            img_metas,
            rescale=rescale)

        bs, c, h, w = img.size()
        for b in range(bs):
            cur_mask_pred_feature = segm_pred[b]
            if not torch.is_tensor(cur_mask_pred_feature):
                return bbox_results, segm_results
            num_pos = cur_mask_pred_feature.shape[0]
            mask_pred = F.interpolate(
                cur_mask_pred_feature.unsqueeze(0), (550, 550),
                mode='bilinear',
                align_corners=False).squeeze(0) > 0.5          
            list_img = []
            pre_value = []
            max_arg = torch.argmax(det_bboxes[b][:,4])
            img_t = img[b]
            cur_p_s = mask_pred[max_arg]
            if 1:
                scale = img_metas[b]['scale_factor']
                device = img_t.device
                scale = torch.tensor(scale).to(device = device)
                bbox = det_bboxes[b][:,0:4] * scale
                img_c = self.crop_seg(img_t,bbox,cur_p_s)
            else:
                img_c = img_t * cur_p_s
            list_img.append(img_c)
            pre_gt = torch.tensor(float(img_metas[b]['pre_value']), dtype=torch.float).cuda()
            pre_value.append(pre_gt)
            # self.show_imgs(img_t,cur_p_s,gt_masks[b][0])
            self.save_flg += 1
            pre_input = torch.stack(list_img, 0).cuda()
            sp = self.backbone_pred(pre_input)
            sp = sp[-1]
            sp = self.gap_pred(sp)
            sp = sp.view(-1, sp.shape[1])
            sp = self.cfc_pred(sp)/num_pos
            self.predict_file.write(str(img_metas[0]['pre_value']) + ', ' + str(sp.cpu()[0].item()) + '\n')
        return bbox_results, segm_results

    def test_seg(self,img,img_metas,gt_masks,rescale):
        feat = self.extract_feat(img)
        det_bboxes, det_labels, det_coeffs = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bbox, det_label, self.bbox_head.num_classes)
            for det_bbox, det_label in zip(det_bboxes, det_labels)
        ]

        segm_results,segm_pred = self.mask_head.simple_test(
            feat,
            det_bboxes,
            det_labels,
            det_coeffs,
            img_metas,
            rescale=rescale)
        return bbox_results, segm_results

    def simple_test(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, rescale=False):
        """Test function without test-time augmentation."""
        start_time = time.time()
        gt_masks_t = gt_masks[0].cpu().numpy()
        if 0:####for critic network
            self.test_critic_net(img, img_metas,gt_bboxes, gt_masks)
            return []
        if 0:####for eye test_critic_netnetwork
            self.test_eye(img, img_metas,gt_masks)
            return []
        if 1:###seg_ac
            bbox_results, segm_results = self.test_seg_ac(img,img_metas,gt_masks,rescale)
            end_time = time.time()
            print('costtime %.6f\n' %(end_time-start_time))
            return list(zip(bbox_results, segm_results))
        if 0:  ###seg
            bbox_results, segm_results = self.test_seg(img,img_metas,gt_masks,rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError(
            'YOLACT does not support test-time augmentation')
