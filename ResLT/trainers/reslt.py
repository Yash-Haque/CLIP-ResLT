import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from dassl.evaluation import Classification

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


import time


"""
NOTES:
- CLASS ResLTAdapter: Adapter layer, comprises of ResLT adapter layer only.
- CLASS TextEncoder: Contains a very simple implementation of TextEncoder in CLIP. No modifications
- CLASS AttentionPool2d: [COMMENTED] Contains implementation of AttentionPool Layer proposed in "https://github.com/openai/CLIP/blob/main/clip/model.py"
- CLASS AverageMeter: Combines loss accuracy, implemented in the original ResLT framework. [NEED TO GET RID OF IT OR INTEGRATE COMPLETELY]
- CLASS CustomCLIP: Combines the ResLT adapter layer after the vision encoder model and before calculating the logits. 
- CLASS ResLT: MAIN TRAINER CLASS. Controls how the ResLT adapter is implemented and monitors the backpropagation of ResLTAdapter.

CURRENT PROBLEM: RuntimeError: The size of tensor a (799) must match the size of tensor b (1024) at non-singleton dimension 1. [383]
CURRENT OBJECTIVE: Create a stable/balanced ResLT Trainer for experiments.

NOTABLE ISSUES:
- Fixing the dimension mismatch between the target [16, 799] and the logit [16, 1024].
- Fixing the misalignment between number of classnames/classes in the CustomCLIP CLASS [266, 268] and the ResLT Trainer CLASS [346, 410].
"""


_tokenizer = _Tokenizer()


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.',
    'HERB': 'a photo of a {}.',
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model


class ResLTAdapter(nn.Module):
    def __init__(self, cfg, inplanes: int, use_final_block: bool = True,  
                 training: bool = True,
                 ):
        
        super(ResLTAdapter, self).__init__()
        self.inplanes = inplanes
        self.num_classes = 799
        self.expansion = 4
        self.gamma = 0.5
        self.is_dropout = True
        self.use_final_block = use_final_block
        self.training = training

        # ResLT Implementation
        self.finalBlock = nn.Sequential(
            nn.Conv2d(512 * self.expansion, 512 * self.expansion * 3, 1, bias=False),
            nn.BatchNorm2d(512 * self.expansion * 3),
            nn.ReLU(inplace = True),
        ) if self.use_final_block else None

        self.fc = nn.Linear(512 * self.expansion, 1024) # To Be Changed Later 
        if self.is_dropout:
            self.dropout_mark = True
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout_mark = False
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))


    def _forward_impl(self, x):
        batch_size = x.size(0)
        print(batch_size)
        if self.use_final_block and self.training:
            x = self.finalBlock(x)
            x = self.avgpool(x)
        else:
            x = self.avgpool(x)
            x = self.finalBlock(x)

        x = torch.flatten(x, 1)
        print(f"Shape of expanded feature map after flatten: {x.shape}")
        c = x.size(1) // 3
        bt = x.size(0)
        x1, x2, x3 = x[:,:c], x[:,c:c*2], x[:,c*2:c*3]
        print(f"Shapes of x1: {x1.shape}, x2: {x2.shape}, x3: {x3.shape}")
        print("Concatenating x1, x2, and x3!!!!")
        out = torch.cat((x1,x2,x3),dim=0)
        
        if self.dropout_mark:
            out = self.dropout(out)

        if self.training:
            print("Passing through FC")
            y = self.fc(out)
            print(f"FC output dim: {y.shape}")

        else:
            weight = self.fc.weight
            norm = torch.norm(weight, 2, 1, keepdim=True)
            weight = weight / torch.pow(norm, self.gamma)
            y = torch.mm(out, torch.t(weight))

        return y[:bt,:], y[bt:bt*2,:], y[bt*2:bt*3,:]
    
    def forward(self,x):
        return self._forward_impl(x)
    
class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.text_projection = nn.Linear(len(classnames), 799)
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        print(f"Prompt Dim: {prompts.shape}")
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        print("Updating loss")
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, cfg, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.cfg = cfg

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        open(self.cfg.OUTPUT_DIR+"/train.log","a+").write('\t'.join(entries)+"\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.model = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = ResLTAdapter(1024, 4).to(clip_model.dtype)
        self.classnames = classnames
        embed_dim = 64 * 32  # the ResNet feature dimension
        
            
    def forward(self, image):
        # image_features = self.image_encoder(image.type(self.dtype))
        # print(f"Img Dim before adapter: {image_features.shape}")
        print(f"Length of Classnames: {len(self.classnames)}")
        feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        # print(f"Image Dim before feature extractor: {image.type(self.dtype).shape}")
        feature_map = feature_extractor(image.type(self.dtype))
        # print(f"Number of classes: {len(self.classnames)}")
        # print(f"Feature Map Output: {feature_map.shape}")
        print("Passing Feature Map into ResLT Adapter!!!")
        feat_H, feat_M, feat_T = self.adapter(feature_map)
        print("Image Passed through RESLT Adapter!!!!")
        print(f"Feature Dimensions-> Head: {feat_H.shape} Med: {feat_M.shape}, Tail: {feat_T.shape}")

        # embed_dim = 64 * 32

        # print("Image Passed through RESLT Adapter!!!!")
        # print(f"Head Feature Dim after adapter: {feat_H.shape}")
        # print(f"Med Feature Dim after adapter: {feat_M.shape}")
        # print(f"Tail Feature Dim after adapter: {feat_T.shape}")

        # ratio = 0.2
        # image_features = ratio * x + (1 - ratio) * image_features

        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # print(f"Img Feature Dim: {image_features.shape}")
        text_features = self.text_encoder()
        
        feat_H = feat_H / feat_H.norm(dim=-1, keepdim=True)
        feat_M = feat_M / feat_M.norm(dim=-1, keepdim=True)
        feat_T = feat_T / feat_T.norm(dim=-1, keepdim=True)

        print(f"Head Feature after normalization: {feat_H}")
        print(f"Medium Feature after normalization: {feat_H}")
        print(f"Tail Feature after normalization: {feat_H}")

        print(f"Feature dim after normalization: Head: {feat_H.shape}, Medium: {feat_M.shape}, Tail: {feat_T.shape}")
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        print(f"Text Feature after normalization: {feat_H}")
        print(f"Text Feature Dim after output: {text_features.shape}")

        logit_scale = self.logit_scale.exp()
        logit_H = logit_scale * feat_H @ text_features.t()
        logit_M = logit_scale * feat_M @ text_features.t()
        logit_T = logit_scale * feat_T @ text_features.t()

        return logit_H, logit_M, logit_T


@TRAINER_REGISTRY.register()
class ResLT(TrainerX):
    """ ResLT-Adapter """

    """
    PROBLEM: Dimension mis-match in the "crossEntropy()" function. [383]

    TO-DO List:
    - Need to reposition the class attributes, such as criterion, softmax, class_numbers. [388]
    - Check the compatibility between labels and logits. [Ensure if the loss calculation is happening properly] [383]
    - Handle the explicit declaration of dataset split. []
    - Confirm whether the problem here is regarding multi-classification or embedding alignment.'
    - Convert self.beta to args.beta. [422]
    """

    def build_model(self):
        """
        NOTES:
        - Discrepancy between "self.dm.dataset.classnames" and "self.num_classes". 
        - Both are coming from Data Manager class. 
        - 
        """
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print(f"Length of Classnames: {len(classnames)}")
        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

        # Check and log gradients for ResLTAdapter
        print("Checking gradients for ResLTAdapter...")
        for name, param in self.model.adapter.named_parameters():
            if param.requires_grad:
                grad_norm = param.grad.norm().item() if param.grad is not None else None
                print(f"{name}: Gradient norm = {grad_norm}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        

        self.register_model('reslt_adapter', self.model.adapter, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)

    def crossEntropy(self, softmax, logit, label, weight, num_classes):
        print(f"Label: {label}")
        print(f"Label length: {len(label)}")
        print(f"Num Classes: {num_classes}")
        target = F.one_hot(label, num_classes=num_classes) # num_classes should be => "num_classes=num_classes"
        print("Inside the Cross Entropy Function [Line 379]!!!")
        print(f"Target Dimension: {target.shape}")
        print(f"Logit Dimension: {logit.shape}")
        print(f"Weight Dimension: {weight.shape}")
        loss = - (weight * (target * torch.log(softmax(logit)+1e-7)).sum(dim=1)).sum()
        return loss

    def forward_backward(self, batch):

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        F_losses = AverageMeter('F_Loss', ':.4e')
        I_losses = AverageMeter('I_Loss', ':.4e')

        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            self.cfg,
            len(batch),
            [batch_time, data_time, F_losses, I_losses, top1, top5],
            prefix="Epoch: [{}]".format(self.cfg.OPTIM.MAX_EPOCH))

        # Attributes
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss() 
        self.beta = 0.85
        end = time.time()
        
        image, label = self.parse_batch_train(batch)

        data_time.update(time.time() - end)
        
        
        print(f"Number of classnames: {self.num_classes}")
        logit_H, logit_M, _ = self.model(image)

        print(f"About to One-Hot Encode the labels")
        labelH=F.one_hot(label, num_classes=self.num_classes).sum(dim=1)
        labelM=F.one_hot(label, num_classes=self.num_classes)[:,:6600].sum(dim=1)

        print(f"One-Hot Encoded the labels!!!")
        I_loss = (self.crossEntropy(self.softmax, logit_H, label, labelH, self.num_classes) + self.crossEntropy(self.softmax, logit_M, label, labelM, self.num_classes)) / (labelH.sum() + labelM.sum()) 
        logit = logit_H + logit_M
        F_loss = self.criterion(logit, label)
        print(f"I Loss: {I_loss}")
        print(f"Fusion Loss: {F_loss}")
        loss= (1-self.beta) * F_loss + self.beta * I_loss
        print(f"Loss Calculated! Saving metrics----{loss}")

        # measure accuracy and record loss
        acc1, acc5 = compute_accuracy(logit, label, topk=(1, 5))
        print(f"Top 1 accuracy: {acc1}")
        print(f"Top 5 accuracy: {acc5}")
        F_losses.update(F_loss.detach().item(), image.size(0))
        I_losses.update(I_loss.detach().item(), image.size(0))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # self.model_backward_and_update(loss)
        print(f"Head Logit Shape: {logit_H.shape}")
        print(f"Medium Logit Shape: {logit_M.shape}")
        
        # compute gradient and do SGD step
        self.model_backward_and_update(loss)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        print(f"Passing the Loss Summary")
        loss_summary = {
            'loss': loss.item(),
            'acc': acc1
        }

        # if i % args.print_freq == 0:
        #     progress.display(i)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def validate(val_loader, model, cfg, lab2cname=None):
        evaluator = Classification(cfg, lab2cname)

        # Switch to evaluation mode
        model.eval()
        evaluator.reset()

        with torch.no_grad():
            start_time = time.time()
            
            for i, (images, target) in enumerate(val_loader):
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # Compute model outputs
                logitH, logitM, _ = model(images)
                output = logitH + logitM

                # Pass predictions and ground truth to the evaluator
                evaluator.process(output, target)

                if (i + 1) % cfg.TEST.PRINT_FREQ == 0:
                    elapsed = time.time() - start_time
                    print(f"Batch [{i+1}/{len(val_loader)}] - Elapsed Time: {elapsed:.2f}s")

        # Evaluate and print final metrics
        results = evaluator.evaluate()
        return results

    # def validate(self):
    #     batch_time = AverageMeter('Time', ':6.3f')
    #     losses = AverageMeter('Loss', ':.4e')
    #     top1 = AverageMeter('All_Acc@1', ':6.2f')
    #     top5 = AverageMeter('All_Acc@5', ':6.2f')

    #     progress = ProgressMeter(
    #         len(val_loader),
    #         [batch_time, losses, top1, top5],
    #         prefix='Test: ')

    #     # switch to evaluate mode
    #     model.eval()
    #     class_num=torch.zeros(8142).cuda()
    #     correct=torch.zeros(8142).cuda()
    #     with torch.no_grad():
    #         end = time.time()
    #         for i, (images, target) in enumerate(val_loader):
    #             if args.gpu is not None:
    #                 images = images.cuda(args.gpu, non_blocking=True)
    #             target = target.cuda(args.gpu, non_blocking=True)

    #             # compute output
    #             logitH, logitM, _ = model(images)
    #             output = logitH + logitM 
    #             loss = criterion(output, target)

    #             # measure accuracy and record loss
    #             acc1, acc5 = accuracy(output, target, topk=(1, 5))
    #             losses.update(loss.item(), images.size(0))
    #             top1.update(acc1[0], images.size(0))
    #             top5.update(acc5[0], images.size(0))
                
    #             # measure elapsed time
    #             batch_time.update(time.time() - end)
    #             end = time.time()

    #             _, predicted = output.max(1)
    #             target_one_hot = F.one_hot(target, args.num_classes)
    #             predict_one_hot = F.one_hot(predicted, args.num_classes)
    #             class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
    #             correct=correct + (target_one_hot + predict_one_hot==2).sum(dim=0).to(torch.float)

    #             if i % args.print_freq == 0:
    #                 progress.display(i)

    #         # TODO: this should also be done with the ProgressMeter
    #         acc_classes = correct / class_num
    #         head_acc = acc_classes[7300:].mean()
    #         medium_acc = acc_classes[3599:7300].mean()
    #         tail_acc = acc_classes[:3599].mean()
    #         open(args.root_model+"/train.log","a+").write((' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} HAcc {head_acc:.3f} MAcc {medium_acc:.3f} TAcc {tail_acc:.3f} \n').format(top1=top1, top5=top5, head_acc=head_acc, medium_acc=medium_acc, tail_acc=tail_acc))
    #     return top1.avg

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']
            
            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']
            
            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
