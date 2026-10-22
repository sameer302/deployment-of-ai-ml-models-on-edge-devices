# Learning path: understanding FCN8-ResNet18 and its deployment

A self-contained reading plan, independent of the daily experiment work. Do the stages in order:
each builds on the one before and ends with a check on our own model. At about 1 hour a day it
takes roughly 4–5 weeks. Tick items off as they are done.

Files referred to below are in `segmentation/semantic/01_fcn/fcn8_resnet18/`. The slide numbers
refer to the experiments deck ("Edge Segmentation Experiments").

## Stage 1: Neural network basics (~5 h)
- [ ] Watch: 3Blue1Brown, "Neural networks" series, videos 1–4 (what a neural network is, gradient descent, backpropagation intuition and calculus)
- [ ] Read: Nielsen, *Neural Networks and Deep Learning* (free online), ch. 1–2
- [ ] Check: explain in your own words what a weight, an activation and a layer are, and what training changes

## Stage 2: Convolutional networks (~4 h)
- [ ] Read: CS231n notes, "Convolutional Neural Networks": conv layer, stride, padding, output-size formula, pooling, convolution as matrix multiplication (im2col)
- [ ] Check: by hand, compute the output shape, number of weights and MACs of our stem conv (7×7, 3→64 channels, stride 2, input 1024×1920); compare with row 0 of `architecture/layer_table.csv`
- [ ] Check: why does im2col copy each input value several times, and how big would the copy be for the stem conv? (≈ 276 MiB; see the "What memory does each fused layer need?" slide)

## Stage 3: Batch norm and ResNet (~3 h)
- [ ] Read: Ioffe & Szegedy, ICML 2015, §3 (what batch norm does; skim)
- [ ] Read: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016, §3.1–3.3 and Table 1 (18-layer column)
- [ ] Check: draw ResNet-18's stages with tensor shapes for a 1024×1920 input; why does each stage halve the resolution and double the channels?
- [ ] Check: what do the 1×1 downsample convs do (fused IDs 7, 12, 17), and why do the stride-2 convs (IDs 6, 11, 16) take about half the time of the other 3×3 convs?
- [ ] Notes to write: **residual skips**, i.e. why adding the block input makes deep networks trainable; identity vs projection shortcuts (BasicBlock slide)

## Stage 4: Semantic segmentation and FCN (~4 h)
- [ ] Read: Long et al., "Fully Convolutional Networks for Semantic Segmentation", CVPR 2015, §3 and §4.1–4.2 (FCN-32s/16s/8s, skip architecture)
- [ ] Read: Dumoulin & Visin, "A guide to convolution arithmetic for deep learning", ch. 4 (transposed convolution = learned upsampling)
- [ ] Learn: what mIoU measures (our baseline: 69.87% on Cityscapes val)
- [ ] Check: trace our FCN-8s head: which earlier feature maps does it reuse, and where does the ×8 Resize come in? Why does this keep extra tensors alive in fused IDs 21–27 (per-layer memory chart)?
- [ ] Notes to write: **FCN skips**, i.e. why adding layer2 / layer3 scores recovers spatial detail
- [ ] Notes to write (after stages 3 and 4): one-slide comparison of the two skip types. Residual skips: 8, short, inside the encoder, make training work. FCN skips: 2, encoder → decoder, recover detail

## Stage 5: Our model, hands-on (~2 h)
- [ ] Do: open our ONNX model (`models/`) in Netron next to `architecture/layer_table.csv` and the architecture diagram
- [ ] Check: find the 57 ONNX layers, then match them to the 30 fused layer IDs on the fusion map slide (e.g. which original layers make up ID 2?)

## Stage 6: Inference performance and memory (~6 h)
- [ ] Read: Williams, Waterman & Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", *Communications of the ACM*, 2009 (compute-bound vs memory-bound)
- [ ] Read: *Operating Systems: Three Easy Pieces* (free online), ch. 13 Address Spaces, ch. 14 Memory API, ch. 17 Free-Space Management (process memory, malloc/free, why freed memory may stay in the process; needed for the ONNX Runtime memory settings experiment)
- [ ] Read: Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks", CVPR 2016, §1–3 (Winograd: faster, but more memory)
- [ ] Read: Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018, §2 (how INT8 quantization works)
- [ ] Check: explain our roofline and time-breakdown slides, and why a conv kernel needs temporary buffers

## Optional, for depth
- [ ] Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd" (video)
- [ ] Goodfellow, Bengio & Courville, *Deep Learning*, ch. 6 (feedforward networks) and ch. 9 (convolutional networks)
- [ ] d2l.ai (*Dive into Deep Learning*), chapters on CNNs and ResNet
