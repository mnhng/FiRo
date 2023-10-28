Code for the following paper:
> [FiRo: Finite-context Indexing of Restricted Output Space for NLP Models Facing Noisy Input](https://arxiv.org/abs/2310.14110)

### Requirements

To setup the required environment

1. Install [*miniconda*](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
2. Run `conda env create -n firo -f environment.yml`

### Examples

The examples are in the `scripts` folder.

To train FiRo, refer to `scripts/firo` folder.

#### GLUE tasks

1. To evaluate baseline BERT performance on noisy input, refer to `scripts/0_base` folder.
2. To evaluate advesarially-finetuned BERT on noisy input, refer to `scripts/1_adv` folder.
3. To evaluate FiRo-assisted BERT on noisy input, refer to `scripts/2_firo` folder.

#### NER task

1. Refer to `scripts/3_ner` folder.

### Citation

If you find this code useful, please cite:

```
@inproceedings{nguyen2023firo,
    title={{FiRo: Finite-context Indexing of Restricted Output Space for NLP Models Facing Noisy Input}},
    author={Nguyen, Minh and Chen, Nancy F.},
    booktitle={Proceedings of AACL},
    year={2023}
}
```
