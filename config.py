import argparse

def initParams():
    parser = argparse.ArgumentParser(description="Configuration for the project")

    parser.add_argument('--seed', type=int, help="Random number seed for reproducibility", default=688)

    # Data folder prepare
    parser.add_argument("--train_wav", type=str, help="Path to the training WAV files",
                        default="yourpath/codecfake_st/v2/train/")
    parser.add_argument("--train_label", type=str, help="Path to the training labels",
                        default="/data7/xyk/codecfake_st/v2/label/train.txt")     

    parser.add_argument("--dev_wav", type=str, help="Path to the development WAV files",
                        default="/data7/xyk/codecfake_st/v2/dev/")
    parser.add_argument("--dev_label", type=str, help="Path to the development labels",
                        default="/data7/xyk/codecfake_st/v2/label/dev.txt")     

    parser.add_argument("--eval_wav", type=str, help="Path to the evaluation WAV files",
                        default="/data7/xyk/codecfake_st/v2/eval_id/")
    parser.add_argument("--eval_label", type=str, help="Path to the evaluation labels",
                        default="/data7/xyk/codecfake_st/v2/label/eval_id.txt")  
    
    parser.add_argument("--SSL_path", type=str, help="Path to the SSL model",
                        default="/data3/xyk/huggingface/wav2vec2-xls-r-300m/") 

    parser.add_argument("-o", "--out_fold", type=str, help="Output folder for models", required=False, default='./models/try/')

    parser.add_argument("--id_source_mimi", type=str, help="Path to the NCSSD-mimi dataset", default="/data7/xyk/codecfake_st/4.NCSSD-mimi/")
    parser.add_argument("--id_source_snac", type=str, help="Path to the NCSSD-snac dataset", default="/data7/xyk/codecfake_st/4.NCSSD-snac/")
    parser.add_argument("--id_config", type=str, help="Path to the codec condition configuration", default="/data2/codecfake/codec_condition")
    parser.add_argument("--id_unseenreal_19_wav", type=str, help="Path to the ASVspoof2019 LA evaluation files", default='/data2/xyk/asv2019/LA/ASVspoof2019_LA_eval/flac')
    parser.add_argument("--id_unseenreal_19_label", type=str, help="Path to the ASVspoof2019 LA evaluation labels", default='/data2/xyk/asv2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')
    parser.add_argument("--id_unseenreal_ncssd", type=str, help="Path to the NCSSD unseen real dataset", default="yourpath/NCSSD")
    parser.add_argument("--id_unseenreal_itw", type=str, help="Path to the ITW WAV files", default='/data2/xyk/ITW/wav')
    parser.add_argument("--id_unseenreal_itw_label", type=str, help="Path to the ITW labels", default='/data2/xyk/ITW/meta.txt')
    
    parser.add_argument("--ood_wav", type=str, help="Path to the OOD evaluation WAV files", default="/data7/xyk/codecfake_st/v2/eval_ood/")
    parser.add_argument("--ood_label_7", type=str, help="Path to the OOD evaluation labels for set 7", default="/data7/xyk/codecfake_st/v2/label/eval_ood7.txt")
    parser.add_argument("--ood_label_8_11", type=str, help="Path to the OOD evaluation labels for sets 8-11", default="/data7/xyk/codecfake_st/v2/label/eval_ood8-11.txt")
    parser.add_argument("--ood_diff_config", type=str, help="Path to the OOD different configuration WAV files", default="/data7/xyk/codecfake_st/v2/eval_other/OOD_diff_config/wavtokenizer_small40/")
    parser.add_argument("--ood_diff_source", type=str, help="Path to the OOD different source dataset", default="/data7/xyk/codecfake_st/8.NCSSD_wavtokenizer/")
    
    parser.add_argument("--ALM_mini_omni", type=str, help="Path to the ALM mini-omni dataset", default="/data7/xyk/codecfake_st/9.ALM-test/mini-omni/A1-A2/")
    parser.add_argument("--ALM_moshi", type=str, help="Path to the ALM moshi dataset", default="/data7/xyk/codecfake_st/v2/eval_other/ALM/moshi/")
    parser.add_argument("--ALM_valle", type=str, help="Path to the ALM valle dataset", default="/data7/xyk/codecfake_st/v2/eval_other/ALM/valle/")
    parser.add_argument("--ALM_speechgpt", type=str, help="Path to the ALM speechgpt-gen dataset", default="/data7/xyk/codecfake_st/v2/eval_other/ALM/speechgpt-gen/")
    
    # Dataset prepare
    parser.add_argument('-m', '--model', help='Model architecture to use', default='W2VAASIST',
                        choices=['Rawaasist', 'W2VAASIST', 'mellcnn'])

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=24, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="Decay rate for learning rate")
    parser.add_argument('--interval', type=int, default=10, help="Interval (in epochs) to decay learning rate")

    parser.add_argument('--beta_1', type=float, default=0.9, help="Beta 1 parameter for Adam optimizer")
    parser.add_argument('--beta_2', type=float, default=0.999, help="Beta 2 parameter for Adam optimizer")
    parser.add_argument('--eps', type=float, default=1e-8, help="Epsilon parameter for Adam optimizer")
    parser.add_argument("--gpu", type=str, help="GPU index to use", default="7")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loading")

    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce"],
                        help="Loss function to use for basic training (ce: Cross Entropy, bce: Binary Cross Entropy)")
    parser.add_argument('--continue_training', action='store_true', help="Flag to continue training from a checkpoint")

    args = parser.parse_args()
    return args