import h5py
import numpy as np
np.set_printoptions(threshold=np.inf)
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})

from bbox import bbox_overlaps


_PASCAL_CLASSES = ['airplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'table', 'dog', 'horse',
           'motorbike', 'person', 'plant',
           'sheep', 'sofa', 'train', 'monitor']

_COCO_CLASSES = [
          'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
          'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
          'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
          'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
          'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
          'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
          'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
          'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
          'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


CLASSES = _COCO_CLASSES

_REGION_MEAN_AND_STD = {
    'orig': (0.0, 1.0),
    # Flickr.
    'flickr-vgg-coco' : (0.1454045, 0.068037**0.5),
    'flickr-vgg-oidv2' : (0.1454045, 0.068037**0.5),
    'flickr-res-coco-200' : (0.2981893, 0.2709116**0.5),
    'flickr-vgg-pascal' : (0.1408328, 0.067438**0.5),
    'flickr-res-coco' : (0.203574, 0.1843775**0.5),
    'flickr-res-oidv2' : (0.203574, 0.1843775**0.5),
    'flickr-irv2-oidv2' : (0.410128, 0.581294),
    # ReferIt.
    'referit-vgg-pascal' : (0.15869775, 0.07185528**0.5),
    'referit-irv2-oidv2' : (0.410128, 0.581294),  # Based on flickr30k.
    'referit-vgg-coco' : (0.158470, 0.069272**0.5),
    'referit-res-coco' : (0.1984466, 0.20089**0.5),
    'referit-res-oidv2' : (0.1984466, 0.20089**0.5),
    'referit-vgg-oidv2' : (0.1454045, 0.068037**0.5),
}

class DatasetLoader:
    """ Loads batched region and phrase features."""
    def __init__(self, region_feat_path, phrase_feat_path, glove_path, split='train'):
        self.f_feats = {}
        print('Loading region features from', region_feat_path)
        self.f_feats['region'] = h5py.File(region_feat_path, 'r')
        print('Found %d region features.' % len(self.f_feats['region']))
        print('Loading phrase features from', phrase_feat_path)
        self.f_feats['phrase'] = h5py.File(phrase_feat_path, 'r')
        print('Found %d phrase features.' % len(self.f_feats['phrase']))
        self.glove = h5py.File(glove_path, 'r')
        print('Found %d tokens in glove.' % len(self.glove))
        self.split = split
        self.sent_im_ratio = 5 if 'flickr' in phrase_feat_path and 'merge' not in phrase_feat_path else 1
        print('Using %d sentence per image.' % self.sent_im_ratio)
        self.max_phrase_per_sentence = 16
        self.max_token_per_phrase = 10
        self.im_names = list(self.f_feats['region'].keys())
        print('Found %d images.' % len(self.im_names))
        self.example_inds = np.arange(len(self.im_names) * self.sent_im_ratio)
        # Shapes for variables returned.
        self.num_rp, self.full_dim = self.f_feats['region'][self.im_names[0]].shape
        print('Using %d region proposals per image.' % self.num_rp)
        arch = 'res'
        self.dim_r = 2048
        if 'vgg' in region_feat_path:
            self.dim_r = 4096
            arch = 'vgg'
        elif 'irv2' in region_feat_path:
            self.dim_r = 1536
            arch = 'irv2'
        det_ds = 'coco'
        if 'pascal' in region_feat_path:
            det_ds = 'pascal'
        elif 'oidv2' in region_feat_path:
            det_ds = 'oidv2'
        ds = 'flickr' if 'flickr' in region_feat_path else 'referit'
        print('Using region feature dimension %d.' % self.dim_r)
        self._region_mean, self._region_std = _REGION_MEAN_AND_STD['%s-%s-%s' % (ds, arch, det_ds)]
        print('Using region feature mean %f, std %f' % (self._region_mean, self._region_std)) 
        self.sample_k = 5
        self.shape = {'region_feats' : (1 * self.num_rp, self.dim_r),
                      'token_feats' : (1 * 1 * 16, 10, 300),
                      'alignment_mask' : (1 * 1, 16),
                      'lstm_mask' : (1 * 1 * 16, 10),
                      'alignment_gt' : (1 * 1, self.num_rp, 16)}

    def get_input_variables(self, batch_size=1, sample_size=None):
        input_vars = {}
        for name in self.shape:
            var_shape = np.array(self.shape[name])
            var_shape[0] = var_shape[0] * batch_size
            if name == 'image_sample_weights':
                var_shape = (None, None)
            dtype = np.float32 if not name.endswith('mask') else np.bool
            input_vars[name] = (dtype, var_shape)
        return input_vars

    def get_glove_embeds(self, sentences):
        S, P, T = sentences.shape
        # sentences s * 16 * 10
        embeds = np.zeros([S, P, T, 300], dtype=np.float32)
        lstm_mask = np.zeros([S, P, T], dtype=np.bool)
        for i in range(S):
            for j in range(P):
                if not sentences[i, j, 0]:  # If empty string, no more phrase.
                    break
                for k in range(T):
                    if not sentences[i, j, k]:  # If empty string, no more token.
                        break
                    if sentences[i, j, k] in self.glove:
                        embeds[i, j, k, :] = self.glove[sentences[i, j, k]]
                        lstm_mask[i, j, k] = True
        return embeds, lstm_mask

    def shuffle_inds(self):
        '''
        shuffle the indices in training (run this once per epoch)
        nop for testing and validation
        '''
        if self.split == 'train':
            np.random.shuffle(self.example_inds)

    def get_region_features(self, im_names):
        region_feat_b = np.zeros([len(im_names), self.num_rp, self.full_dim], dtype=np.float32)
        for i in range(len(im_names)):
            region_feat_raw = self.f_feats['region'][im_names[i]][:self.num_rp, :]
            # Add padding if less than NUM_R
            region_feat_b[i, :len(region_feat_raw), :] = region_feat_raw
        region_feat_b = np.reshape(region_feat_b, [-1, self.full_dim])  # 200n * (2048+x+4)
        # Normalize to zero-mean, unit-variance.
        normalized_region_feats = (region_feat_b[:, :self.dim_r] - self._region_mean) / self._region_std
        region_feat_b = np.c_[normalized_region_feats, region_feat_b[:, self.dim_r:]] 
        return region_feat_b

    def sample_items(self, sample_inds, sample_size):
        '''Return region-phrase features and region phrases from memory module.'''
        region_feats_b, phrase_feats_b = [], []
        lstm_mask_b, bbox_gt_b = [], []
        phrase_b = []
        for ind in sample_inds:  # ind is an index for sentence
            im_ind = ind // self.sent_im_ratio
            im_name = self.im_names[im_ind]
            # positive sentence sampling
            sent_index = []
            if sample_size > 1:
                sent_index = np.random.choice(
                            [i for i in range(self.sent_im_ratio) if i != (ind % self.sent_im_ratio)],
                            sample_size - 1, replace=False)
            sent_index = sorted(np.append(sent_index, ind % self.sent_im_ratio)) 
            region_feats_b.append(self.get_region_features([im_name]))
            phrase = self.f_feats['phrase'][im_name][sent_index, :, :]
            num_phrase = phrase.shape[1]
            phrase_index = np.arange(num_phrase)
            # For referit dataset, the number of phrase per sentence can vary and may exceed 16.
            if self.split == 'train' and  num_phrase > self.max_phrase_per_sentence:
                phrase_index = np.random.choice(num_phrase, self.max_phrase_per_sentence, replace=False)
            phrase = phrase[:, phrase_index, :]
            phrase = phrase.astype(np.str)  # Python3 compatibility.
            # For off-diag negative image mask.
            phrase_b.append(phrase)
            embed, lstm_mask = self.get_glove_embeds(phrase)
            phrase_feats_b.append(embed)
            lstm_mask_b.append(lstm_mask)
            # Select bbox. Index sent and phrase separately as numpy only allows one indexing at a time
            bbox = self.f_feats['phrase'][im_name + '_gt'][sent_index][:, phrase_index, :].astype(np.float32)
            bbox_gt_b.append(bbox)
        region_feats_b = np.concatenate(region_feats_b, axis=0)  # 200b * (2048+4)
        phrase_feats_b = np.concatenate(phrase_feats_b, axis=0)  # kb * 16 * 10 * 300
        lstm_mask_b = np.concatenate(lstm_mask_b, axis=0)        # kb * 16 * 10
        bbox_gt_b = np.concatenate(bbox_gt_b, axis=0)            # kb * 16 * 4
        phrase_b = np.concatenate(phrase_b, axis=0)              # kb * 16 * 10
        return (region_feats_b, phrase_feats_b, lstm_mask_b, bbox_gt_b, phrase_b)

    def get_groundtruth(self, gt_boxes, query_boxes, batch_size, sample_size):
        """Compute iou and return a mask with iou>0.5 region set to True."""
        iou = np.stack([bbox_overlaps(
                gt_boxes[i],
                query_boxes[i // sample_size * self.num_rp : i // sample_size * self.num_rp + self.num_rp, :])
                for i in range(batch_size * sample_size)])  # (b*s) * 16 * 200
        # In evaluation, count all regions with IoU > 0.5 as positive.
        alignment_gt = iou > 0.5
        return alignment_gt

    def get_batch(self, batch_index, batch_size, sample_size=1):
        input_values = {}
        start_ind = batch_index * batch_size
        end_ind = start_ind + batch_size
        sample_inds = self.example_inds[start_ind : end_ind]
        (region_feats, phrase_feats, lstm_mask, gt_boxes, phrases) = \
                self.sample_items(sample_inds, sample_size)
        # Split logits and coordinates from region features.
        normalized_region_feats = region_feats[:, :self.shape['region_feats'][-1]]
        input_values['region_feats'] = normalized_region_feats
        input_values['region_logits'] = region_feats[:, self.shape['region_feats'][-1] : -4]  # 200b * (20+1)
        query_boxes = region_feats[:, -4:]
        input_values['token_feats'] = phrase_feats.reshape([-1, 10, 300])
        # Mask for padding tokens.
        lstm_mask = lstm_mask.reshape([-1, 10])
        input_values['lstm_mask'] = lstm_mask
        # Bbox.
        input_values['query_boxes'] = query_boxes
        input_values['gt_boxes'] = gt_boxes
        input_values['phrases'] = phrases 
        # Groundtruth is used for computing grounding metrics in training.
        alignment_gt = self.get_groundtruth(gt_boxes, query_boxes, batch_size, sample_size)
        alignment_gt = alignment_gt.transpose([0, 2, 1])  # b * 200 * 16 
        input_values['alignment_gt'] = alignment_gt
        # Mask for padding phrases.
        if self.split == 'train':
            input_values['alignment_mask'] = np.any(
                    lstm_mask.reshape([batch_size, self.max_phrase_per_sentence, 10]), axis=-1)  # b * 16
        else:
            input_values['alignment_mask'] = gt_boxes[:, :, -1] > 0  # b * 16
        # For logging.
        input_values['image_names'] = np.array([self.im_names[i // self.sent_im_ratio] for i in sample_inds])
        return input_values

