# Semantics-Preserving Sketch Embedding for Face Generation
![Teaser](figure/teaser_1.jpg)
### [Paper](https://arxiv.org/abs/2211.13015) 
<!-- <br> -->
[Binxin Yang](https://orcid.org/0000-0003-4110-1986), [Xuejin Chen](http://staff.ustc.edu.cn/~xjchen99/), Chaoqun Wang, Chi Zhang, Zihan Chen, [Xiaoyan Sun](http://staff.ustc.edu.cn/~xysun720/).
<!-- <br> -->

## Abstract
>With recent advances in image-to-image translation tasks, remarkable progress has been witnessed in generating face images from sketches. However, existing methods frequently fail to generate images with details that are semantically and geometrically consistent with the input sketch, especially when various decoration strokes are drawn. To address this issue, we introduce a novel $\mathcal{W}$ - $\mathcal{W^+}$ encoder architecture to take advantage of the high expressive power of $\mathcal{W^+}$ space and semantic controllability of $\mathcal{W}$ space. We introduce an explicit intermediate representation for sketch semantic embedding. With a semantic feature matching loss for effective semantic supervision, our sketch embedding precisely conveys the semantics in the input sketches to the synthesized images. Moreover, a novel sketch semantic interpretation approach is designed to automatically extract semantics from vectorized sketches. We conduct extensive experiments on both synthesized sketches and hand-drawn sketches, and the results demonstrate the superiority of our method over existing approaches on both semantics-preserving and generalization ability.
>

## Requirements
A suitable [conda](https://conda.io/) environment named `psp_env` can be created
and activated with:

```
conda env create -f psp_env.yaml
conda activate psp_env
```

## Pretrained Model
We provide the checkpoint ([Google Drive](https://drive.google.com/file/d/1jyoEqZXNfsz-MOlRSimjwG3q8GdeA4AZ/view?usp=share_link)) that is trained on [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ). By default, we assume that the pretrained model is downloaded and saved to the directory `checkpoints`.

## Testing

To sample from our model, you can use `scripts/inference.py`. For example, 
```
python scripts/inference.py \
--exp_dir=result \
--checkpoint_path=checkpoints/model.pt \
--data_path=examples/sketch \
--target_path=examples/appearance \
--test_batch_size=1 \
--couple_outputs \
--test_workers=1
```
or simply run:
```
sh test.sh
```
Visualization of inputs and output:

![](figure/1.png)
![](figure/2.png)
![](figure/3.png)
