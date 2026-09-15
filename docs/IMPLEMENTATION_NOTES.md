# Implementation notes and manuscript boundaries

This repository encodes every architecture, training and testing detail that is explicitly stated in the accepted manuscript. A few low-level values are not specified in the paper and therefore remain visible and configurable rather than being presented as manuscript facts.

## PPG sampling frequency

All PPG sources are segmented into non-overlapping 4-second clips and resampled to a common frequency. The default target is **30 Hz**, giving 120 samples per clip. This matches the 30 fps rPPG input and ensures that the PPG feature representation `F_p` and rPPG feature representation `F_r` both have length `L=120` for the feature-alignment loss.

## M-TCN downsampling layer

The paper states that each dilation block internally produces five `1 x 3` convolutional feature maps, contains a skip connection and downsampling layer, and passes a `K x L` map to the next block. It does not name the exact downsampling operator. `alive/mtcn.py` uses a `1 x 1` channel projection from five internal maps back to one `K x L` map, which preserves the dimensions described in the manuscript.

## MLP width and dropout probability

The paper shows MLP layers and dropout operations but does not report their numeric widths/probability. These are exposed in YAML:

- `model.mlp_hidden: [64]`
- `model.dropout: 0.01`

They can be changed without editing model code. Replace these defaults with the exact experiment values if a separate laboratory record is available.

## ROI landmark indices and block size

## ROI landmark indices and block size

The paper specifies forehead, cheek and chin regions, minimum enclosing rectangles, exclusion of eye and mouth areas, and division into non-overlapping blocks. The ROI block size used in the reported experiments was `7 × 7` pixels. 

## Checkpoints

No teacher or student checkpoint is distributed. The repository provides:

1. PPG preparation code;
2. teacher architecture and training code;
3. student architecture and teacher-guided training code;
4. student-only testing and quality-weighted fusion code.

## Teacher optimization and dataset mixing

The manuscript identifies the combined PPG sources used to pretrain `N_PPG`, but it does not report a separate teacher epoch count or a dataset-balancing ratio. `configs/teacher.yaml` therefore exposes the teacher epoch count and the loader samples prepared clips in proportion to their occurrence. The default subject split applies the manuscript's approximate 80%/10%/10% policy separately within each contributing dataset.
