import argparse, sys, os, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def str2bool(str):
    return True if str.lower() == 'true' else False

def transformer_opt(opt):
    opt = vars(opt)
    if opt['unamp']:
        opt['amp'] = False
    else:
        opt['amp'] = True
    del opt['yaml']
    del opt['weight']
    del opt['info']
    del opt['unamp']
    return opt

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml', type=str, default='ultralytics/models/v8-RGBT/yolov8-RGBT-share.yaml', help='model.yaml path')
    parser.add_argument('--weight', type=str, default='', help='pretrained model path')
    parser.add_argument('--cfg', type=str, default='hyp.yaml', help='hyperparameters path')
    parser.add_argument('--data', type=str, default='ultralytics/datasets/LLVIP_r20.yaml', help='data yaml path')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--unamp', action='store_true', help='Unuse Automatic Mixed Precision (AMP) training')
    parser.add_argument('--batch', type=int, default=16, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, default=640, help='size of input images as integer')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=2, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', type=str, default= 'runs/train/LLVIP-R20', help='save to project/name')
    parser.add_argument('--name', type=str, default='LLVIP-yolov8n-RGBT-share-no_pre', help='save to project/name')
    parser.add_argument('--resume', type=str, default='', help='resume training from last checkpoint')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'Adamax', 'NAdam', 'RAdam', 'AdamW', 'RMSProp', 'auto'], default='SGD', help='optimizer (auto -> ultralytics/yolo/engine/trainer.py in build_optimizer funciton.)')
    parser.add_argument('--close_mosaic', type=int, default=0, help='(int) disable mosaic augmentation for final epochs')
    parser.add_argument('--info', action="store_true", default=False, help='model info verbose')
    parser.add_argument('--use_simotm', type=str, choices=['Gray2BGR', 'SimOTM', 'SimOTMBBS','Gray','SimOTMSSS','Gray16bit','BGR','RGBT'], default='RGBT', help='simotm')
    parser.add_argument('--channels', type=int, default=4, help='input channels')
    parser.add_argument('--save', type=str2bool, default='True', help='save train checkpoints and predict results')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--deterministic', action="store_true", default=True, help='whether to enable deterministic mode')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--fraction', type=float, default=1.0, help='dataset fraction to train on (default is 1.0, all images in train set)')
    parser.add_argument('--profile', action='store_true', help='profile ONNX and TensorRT speeds during training for loggers')
    
    # Segmentation
    parser.add_argument('--overlap_mask', type=str2bool, default='True', help='masks should overlap during training (segment train only)')
    parser.add_argument('--mask_ratio', type=int, default=4, help='mask downsample ratio (segment train only)')

    # Classification
    parser.add_argument('--dropout', type=float, default=0.0, help='use dropout regularization (classify train only)')
    # parser.add_argument('--use_rir', action='store_true', default=False, help='RIR: random_interpolation_resize ')
    return parser.parse_known_args()[0]

class YOLOV8(YOLO):
    '''
    yaml:model.yaml path
    weigth:pretrained model path
    '''
    def __init__(self, yaml='ultralytics/models/v8/yolov8n.yaml', weight='', task=None) -> None:
        super().__init__(yaml, task)
        if weight:
            self.load(weight)
        
if __name__ == '__main__':
    opt = parse_opt()
    # print(opt.yaml)
    model = YOLOV8(yaml=opt.yaml, weight=opt.weight)
    if opt.info:
        model.info(detailed=True, verbose=True)
        model.profile(opt.imgsz)

        print('before fuse...')
        model.info(detailed=False, verbose=True)
        print('after fuse...')
        model.fuse()
    else:
        model.train(**transformer_opt(opt))