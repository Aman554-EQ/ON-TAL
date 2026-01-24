import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')
    parser.add_argument(
        '--segment_size',
        type=int,
        default=64),
    parser.add_argument(
        '--anchors',
        type=str,
        default='4,8,16,32,48,64'),
    parser.add_argument(
        '--seed', 
        default=52, 
        type=int,
        help='random seed for reproducibility')    
    
    # Overall Dataset settings
    parser.add_argument(
        '--num_of_class',
        type=int,
        default=21)   
    parser.add_argument(
        '--data_format',
        type=str,
        default="pickle")     
    parser.add_argument(
        '--data_rescale',
        default=False,
        action='store_true')  
    parser.add_argument(
        '--predefined_fps', 
        default=None, 
        type=float)    
    parser.add_argument(
        '--rgb_only',
        default=False,
        action='store_true')   
    parser.add_argument(
        '--video_anno',
        type=str,
        default="./data/thumos14_v2.json")
    parser.add_argument(
        '--video_feature_all_train',
        type=str,
        default="./data/thumos_all_feature_val_V3.pickle")
    parser.add_argument(
        '--video_feature_all_test',
        type=str,
        default="./data/thumos_all_feature_test_V3.pickle")
        
    #network
    parser.add_argument(
        '--feat_dim',
        type=int,
        default=4096)
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=1024)
    parser.add_argument(
        '--out_dim',
        type=int,
        default=21)
    parser.add_argument(
        '--enc_layer',
        type=int,
        default=3,
        help='Number of encoder layers')
    parser.add_argument(
        '--enc_head',
        type=int,
        default=8,
        help='Number of attention heads (Transformer only, kept for compatibility)')
    parser.add_argument(
        '--dec_layer',
        type=int,
        default=5,
        help='Number of decoder layers')
    parser.add_argument(
        '--dec_head',
        type=int,
        default=4,
        help='Number of attention heads (Transformer only, kept for compatibility)')
    
    # Mamba-specific hyperparameters
    parser.add_argument(
        '--mamba_state_dim',
        type=int,
        default=16,
        help='Mamba SSM state expansion dimension')
    parser.add_argument(
        '--mamba_conv_dim',
        type=int,
        default=4,
        help='Mamba local convolution kernel size')
    parser.add_argument(
        '--mamba_expand',
        type=int,
        default=2,
        help='Mamba block expansion factor')
        
    # Training settings
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64)
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--epoch',
        type=int,
        default=5)
    parser.add_argument(
        '--lr_step',
        type=int,
        default=3)
        
    # Post processing
    parser.add_argument(
        '--alpha',
        type=float,
        default=1)
    parser.add_argument(
        '--beta',
        type=float,
        default=1)
    parser.add_argument(
        '--pptype',
        type=str,
        default="net")
    parser.add_argument(
        '--pos_threshold',
        type=float,
        default=0.5)       
    parser.add_argument(
        '--sup_threshold',
        type=float,
        default=0.1)     
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1)     
    parser.add_argument(
        '--inference_subset',
        type=str,
        default="test")
    parser.add_argument(
        '--soft_nms',
        type=float,
        default=0.3)
    parser.add_argument(
        '--video_len_file',
        type=str,
        default="./output/video_len_{}.json")
    parser.add_argument(
        '--proposal_label_file',
        type=str,
        default="./output/proposal_label_{}.h5")
    parser.add_argument(
        '--suppress_label_file',
        type=str,
        default="./output/suppress_label_{}.h5")
    parser.add_argument(
        '--suppress_result_file',
        type=str,
        default="./output/suppress_result.h5")
    parser.add_argument(
        '--frame_result_file',
        type=str,
        default="./output/frame_result.h5")
    parser.add_argument(
        '--result_file',
        type=str,
        default="./output/result_proposal.json")
    parser.add_argument(
        '--wterm',
        type=bool,
        default=False)    
    
    args = parser.parse_args()

    return args
