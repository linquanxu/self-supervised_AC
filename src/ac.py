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
class AC(SingleStageDetector):
    """Implementation of `AC <https://arxiv.org/abs/1904.02689>`_"""

    def __init__(self,
                 backbone,
                 backbone1,
                 backbone_pred,
                 neck,
                 bbox_head,
                 segm_head,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(AC, self).__init__(backbone, backbone1, backbone_pred, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)
        self.segm_head = build_head(segm_head)
        self.mask_head = build_head(mask_head)
        self.cls_loss = torch.nn.L1Loss()
        self.predict_file = open('predicted_final.txt', 'w')
        self.save_flg = 0
        self.show_img = 0
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

    def ts2Matrix(self, ts):
        matrix = np.zeros((self.m, self.n))
        T = len(ts)

        height = 1.0 / self.m
        width = T / self.n

        for idx in range(T):
            i = int((1 - ts[idx]) / height)
            if i == self.m:
                i -= 1

            t = idx + 1
            j = t / width
            if int(j) == round(j, 7):
                j = int(j) - 1
            else:
                j = int(j)

            matrix[i][j] += 1
        return matrix

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        feat = self.extract_feat(img)
        bbox_outs = self.bbox_head(feat)
        prototypes = self.mask_head.forward_dummy(feat[0])
        return (bbox_outs, prototypes)
    
    def train_predict_net(self, img, img_metas, gt_masks=None):
            bs, _, _, _ = img.size()
            batch_imgs = []
            cls_target = []
            for batch in range(bs):
                b_img = img[batch].cpu().numpy()
                b_mask = gt_masks[batch].cpu().numpy()
                b_img_mask = b_img * b_mask
                b_img_mask = torch.from_numpy(b_img_mask).cuda()
                batch_imgs.append(b_img_mask)
                cls_target.append(float(img_metas[batch]['pre_value'].strip()))
                if self.show_img:
                    cv_img = b_img.transpose(1, 2, 0)
                    cv_img = self.imdenormalize(cv_img)
                    cv_img = cv_img.transpose(2, 0, 1)
                    cv_img_mask = cv_img * b_mask
                    cv_img_mask = cv_img_mask.transpose(1, 2, 0)
                    gt_img = cv_img.transpose(1,2,0)
                    cv2.imwrite('save_img/' + str(self.save_flg) + '_seg.jpg', cv_img_mask.astype(np.uint8))
                    cv2.imwrite('save_img/' + str(self.save_flg) + '_ori.jpg', gt_img.astype(np.uint8))
                    self.save_flg += 1

            batch_imgs = torch.stack(batch_imgs)
            x = self.extract_feat(batch_imgs) ###for training critic.
            #x = self.extract_feat(img) ##for training eyes.
            cls_target = torch.tensor(cls_target, dtype=torch.float32).view(x.shape[0], 1).cuda()
            classifiy_loss = self.cls_loss(x, cls_target)
            a_losses = {}
            a_losses['cls_loss'] = classifiy_loss
            return a_losses
    
    def train_supervised_net(self, img, img_metas):
        x = self.extract_feat(img)
        x = x[-1]
        x = self.free_block(x)
        free_img_list = []
        cls_target = []
        for abc in range(img.size()[0]):
            x1 = x[abc]
            b_img = img[abc].cpu().numpy()
            free_f = F.interpolate(
                x1.unsqueeze(0), (550, 550),
                mode='bilinear',
                align_corners=False).squeeze(0) > 0.5
            free_f = free_f.cpu().numpy().astype(np.uint8)
            free_img = b_img * free_f
            free_img = torch.from_numpy(free_img).cuda()
            free_img_list.append(free_img)
            cls_target.append(float(img_metas[abc]['pre_value'].strip()))
        cls_target = torch.tensor(cls_target, dtype=torch.float32).view(x.shape[0], 1).cuda()
        free_img_input = torch.stack(free_img_list, 0)
        ax = self.backbone1(free_img_input)
        ax = ax[-1]
        ax = self.gap(ax)
        ax = ax.view(-1, 32)
        ax = self.cfc(ax)
        classifiy_loss = self.cls_loss(ax, cls_target)
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
        return losses,mask_pred,sampling_results

    def train_ac(self,img,img_metas,gt_masks,mask_pred,sampling_results,losses):
        bs, c, h, w = img.size()
        critc_pred = []
        pred_gt = []
        for b in range(bs):
            cur_p = mask_pred[b]
            num_pos = cur_p.shape[0]
            #pre_x = cur_mask_pred_feature[x]
            cur_p = F.interpolate(
                cur_p.unsqueeze(0), (550, 550),
                mode='bilinear',
                align_corners=False).squeeze(0) > 0.5
            cur_p = cur_p.cpu().numpy().astype(np.uint8)
            list_img = []
            pre_value = []
            list_img_critic = []
            ssid = 0
            if num_pos > 1:
                top1 = sampling_results[b].top1.cpu().numpy()
                pos_inds = sampling_results[b].pos_inds.cpu().numpy()
                for s in range(num_pos):
                    if pos_inds[s] == top1:
                        ssid = s

            ####find top 1
            te_img_list = []
            for p in range(1):
                img_t = img[b].cpu().numpy()
                cur_p_s = cur_p[ssid]
                img_c = img_t * cur_p_s

                img_c = torch.from_numpy(img_c).cuda()
                list_img.append(img_c)

                pre_gt = torch.tensor(float(img_metas[b]['pre_value']), dtype=torch.float).cuda()
                pre_value.append(pre_gt)

                ### for critic net
                img_critic = img[b].cpu().numpy()
                gt_masks_t = gt_masks[b].cpu().numpy()
                img_critic = img_critic * gt_masks_t
                img_critic = torch.from_numpy(img_critic).cuda()
                list_img_critic.append(img_critic)
           
            pre_input = torch.stack(list_img, 0)###list_img for seg mask
            pre_value_gt = torch.stack(pre_value, 0)
            pre_value_gt = pre_value_gt.view(1, -1)
            critic_img = torch.stack(list_img_critic, 0)

            ### for critic
            ax = self.backbone1(critic_img)
            ax = ax[-1]
            ax = self.gap(ax)
            ax = ax.view(-1, 32)
            ax = self.cfc(ax)


            ## for seg to predicted
            sp = self.backbone_pred(pre_input)
            sp = sp[-1]
            sp = self.gap_pred(sp)
            sp = sp.view(-1, 32)
            sp = self.cfc_pred(sp)

            sp_loss = self.cls_loss(sp, ax)####pre-value and critic
            sp_loss += sp_loss
            seg_gt_loss += seg_gt_loss
            sp_loss /= num_pos
            seg_gt_loss /= num_pos
            critc_pred.append(sp_loss)
            pred_gt.append(seg_gt_loss)
        losses.update({'critic_pred': critc_pred})
        losses.update({'pred_gt': pred_gt})

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
        losses,mask_pred,sampling_results = self.train_seg(img,img_metas,gt_bboxes,gt_labels,gt_bboxes_ignore,gt_masks)

        ## comput the smooothl1 loss
        losses = self.train_ac(img,img_metas,gt_masks,mask_pred,sampling_results,losses)
        return losses

    def test_critic_net(self, img, img_metas,gt_masks):
        bs, c, h, w = img.size()
        batch_imgs = []
        cls_target = []
        for batch in range(bs):
            b_img = img[batch].cpu().numpy()
            b_mask = gt_masks[batch].cpu()[0].permute(1, 0, 2)[0].numpy()
            b_img_mask = b_img * b_mask
            b_img_mask = torch.from_numpy(b_img_mask).cuda()
            batch_imgs.append(b_img_mask)
            cls_target.append(float(img_metas[batch]['pre_value'].strip()))


        batch_imgs = torch.stack(batch_imgs)
        x = self.extract_feat(batch_imgs)
        #x = self.extract_feat(img)
        self.predict_file.write(str(img_metas[0]['pre_value']) + ', ' + str(x.cpu()[0].item()) + '\n')
    
    def test_surpvised_net(self, img, img_metas,gt_masks):
        x = self.extract_feat(img)
        x = x[-1]
        x = self.free_block(x)
        free_img_list = []
        cls_target = []
        for abc in range(img.size()[0]):
            x1 = x[abc]
            b_img = img[abc].cpu().numpy()
            free_f = F.interpolate(
                x1.unsqueeze(0), (550, 550),
                mode='bilinear',
                align_corners=False).squeeze(0) > 0.5
            free_f = free_f.cpu().numpy().astype(np.uint8)
            free_img = b_img * free_f
            free_img = torch.from_numpy(free_img).cuda()
            free_img_list.append(free_img)
            cls_target.append(float(img_metas[abc]['pre_value'].strip()))

            #####vision for segm used self-surpervise
            cv_img = b_img.transpose(1, 2, 0)
            cv_img = self.imdenormalize(cv_img)
            cv_img = cv_img.transpose(2, 0, 1)
            cv_img = cv_img * free_f
            cv_img = cv_img.transpose(1, 2, 0)
            cv2.imwrite('save_img/_self_surpervise_' + str(self.save_flg) + '.jpg', cv_img.astype(np.uint8))
            self.save_flg += 1


        cls_target = torch.tensor(cls_target, dtype=torch.float32).view(x.shape[0], 1).cuda()
        free_img_input = torch.stack(free_img_list, 0)
        ax = self.backbone1(free_img_input)
        ax = ax[-1]
        ax = self.gap(ax)
        ax = ax.view(-1, 32)
        ax = self.cfc(ax)
        self.predict_file.write(str(img_metas[0]['pre_value']) + ', ' + str(ax.cpu()[0].item()) + '\n')

    def test_feature_engine(self, img, img_metas,gt_masks):
        img_t = img[0].cpu().numpy()
        gt_mask = gt_masks[0].cpu()[0].permute(1, 0, 2)[0].numpy()

        cv_img = img_t.transpose(1, 2, 0)
        cv_img = self.imdenormalize(cv_img)
        cv_img = cv_img.transpose(2, 0, 1)

        f_maks = cv_img * gt_mask
        fea_projects = []
        matrix_list = []
        te_img_list = []
        for f in range(len(f_maks)):
            p1 = []
            fp = f_maks[f]
            h, w = fp.shape
            for w1 in range(w):
                for h1 in range(h):
                    if not fp[h1][w1] == 0:
                        p1.append(fp[h1][w1] / 255.0)
            fea_projects.append(p1)
            m1 = self.ts2Matrix(p1)
            m1 = torch.from_numpy(m1).cuda()
            matrix_list.append(m1)
        te_img_list.append(torch.stack(matrix_list))
        te_img = torch.stack(te_img_list, 0).float()
        sp = self.backbone_pred(te_img)
        sp = sp[-1]
        sp = self.gap_pred(sp)
        sp = sp.view(-1, 32)
        sp = self.cfc_pred(sp)/1
        self.predict_file.write(str(img_metas[0]['pre_value']) + ', ' + str(sp.cpu()[0].item()) + '\n')

    def test_mask_fe(self, img, img_metas,gt_masks,img_t,cur_p_s,te_img_list):
        f_maks = img_t * cur_p_s
        fea_projects = []
        matrix_list = []
        for f in range(len(f_maks)):
            p1 = []
            fp = f_maks[f]
            h, w = fp.shape
            for w1 in range(w):
                for h1 in range(h):
                    if not fp[h1][w1] == 0:
                        p1.append(fp[h1][w1] / 255.0)
            fea_projects.append(p1)
            m1 = self.ts2Matrix(p1)
            m1 = torch.from_numpy(m1).cuda()
            matrix_list.append(m1)
        te_img_list.append(torch.stack(matrix_list))
        te_img = torch.stack(te_img_list, 0).float()
        return te_img

        
    def test_eye(self, img,img_metas,gt_masks):
        img_t = img.cpu().numpy()
        gt_mask = gt_masks[0].cpu()[0].permute(1, 0, 2)[0].numpy()
        img_c = img_t * gt_mask
        img_c = torch.from_numpy(img_c).cuda()
        feat = self.extract_feat(img_c)
        self.predict_file.write(str(img_metas[0]['pre_value']) + ', ' + str(feat.cpu()[0].item()) + '\n')


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
            num_pos = cur_mask_pred_feature.shape[0]
            mask_pred = F.interpolate(
                cur_mask_pred_feature.unsqueeze(0), (550, 550),
                mode='bilinear',
                align_corners=False).squeeze(0) > 0.5
            mask_pred = mask_pred.cpu().numpy().astype(np.uint8)
            list_img = []
            pre_value = []
            max_score = 0
            max_sindex = 0
            ####find top score
            if num_pos > 1:
                for pos in range(num_pos):
                    det = det_bboxes[0].cpu().numpy()
                    if det[pos][4] > max_score:
                        max_score = det[pos][4]
                        max_sindex = pos

            te_img_list = []
            for p in range(1):
                img_t = img[b].cpu().numpy()

                cur_p_s = mask_pred[max_sindex]
                img_c = img_t * cur_p_s
                img_c = torch.from_numpy(img_c).cuda()
                list_img.append(img_c)
                pre_gt = torch.tensor(float(img_metas[b]['pre_value']), dtype=torch.float).cuda()
                pre_value.append(pre_gt)
                self.save_flg += 1
            pre_input = torch.stack(list_img, 0).cuda()
            ## for seg to predicted
            sp = self.backbone_pred(pre_input)
            sp = sp[-1]
            sp = self.gap_pred(sp)
            sp = sp.view(-1, 32)
            sp = self.cfc_pred(sp)/num_pos
            self.predict_file.write(str(img_metas[0]['pre_value']) + ', ' + str(sp.cpu()[0].item()) + '\n')
            return bbox_results, segm_results

    def simple_test(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, rescale=False):
        """Test function without test-time augmentation."""
        bbox_results, segm_results = self.test_seg_ac(img,img_metas,gt_masks,rescale)
        return list(zip(bbox_results, segm_results))

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError(
            'AC does not support test-time augmentation')
