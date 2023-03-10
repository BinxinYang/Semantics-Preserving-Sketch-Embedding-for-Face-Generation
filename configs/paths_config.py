dataset_paths = {
	'celeba_train': '/home/eeid/tennyson/dataset/sketch2face/train_B',
	'celeba_test': '/home/eeid/tennyson/dataset/sketch2face/test_B',
	'celeba_train_sketch': '/gpfs/home/eeid/tennyson/dataset/sketch2face/train_A_binary_edge_22',
	'celeba_test_sketch': '/gpfs/home/eeid/tennyson/dataset/sketch2face/test_A_binary_edge_22',
	'celeba_train_segmentation': '/gpfs/home/eeid/tennyson/dataset/sketch2face/train_A_semantic_edge_22',
	'celeba_test_segmentation': '/gpfs/home/eeid/tennyson/dataset/sketch2face/test_A_semantic_edge_22',
	'celeba_train_mask': '/home/eeid/tennyson/dataset/sketch2face/train_mask_unet',
	'celeba_test_mask': '/home/eeid/tennyson/dataset/sketch2face/test_mask_unet',
	'ffhq': '',
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'parsenet': 'pretrained_models/parsenet_unet.pth',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pt'
}