# Learn-Unlearn-Relearn-Noise-SLMs

### Abstract

With the growing need for efficient language models in resource-constrained environments, Small Language Models (SLMs) have emerged as compact and practical alternatives to Large Language Models (LLMs). While studies have explored noise handling in LLMs, little is known about how SLMs handle noise, a critical factor for their reliable real-world deployment. This study investigates the ability of SLMs with parameters between 1 and 3 billion to learn, retain, and subsequently eliminate different types of noise (word flip, character flip, transliteration, irrelevant content, and contradictory information). Four pre-trained SLMs (Olmo 1B, Qwen1.5 1.8B, Gemma1.1 2B, and Phi2 2.7B) were instruction-tuned on noise-free data and tested with in-context examples to assess noise learning. Subsequently, noise patterns were introduced in instruction tuning to assess their adaptability. The results revealed differences in how models handle noise, with smaller models like Olmo quickly adapting to noise patterns. Phi2's carefully curated, structured, and high-quality pretraining data enabled resistance to character level, transliteration, and counterfactual noise, while Gemma adapted successfully to transliteration noise through its multilingual pretraining. Subsequent clean data training effectively mitigated noise effects. These findings provide practical strategies for developing robust SLMs for real-world applications.

### Directory Structure

```plaintext

Learn-Unlearn-Relearn-Noise-SLMs/
├── assets/
│   ├── tokenizer.png                    # BPE tokenization of 'Science fiction' under different noises
├── counterfactuals/                     
│   ├── creating_counterfactuals.py      # script to create counterfactual examples
│   ├── identify_counterfactuals.py      # script to identify counterfactual samples
├── data/                                
│   ├── 50samples_train/                 # training data with 50 sample cases
│   ├── test/                            # testing dataset
│   └── train/                           # training dataset
├── gemma/                               # source, config, train, inference, and evaluation for Gemma
│   ├── configs/                         # configuration files for Gemma
│   ├── evaluation/                      # evaluation scripts for Gemma
│   ├── src/                             # source code specific to Gemma
│   ├── inference.py                     # inference script for Gemma model
│   └── train.py                         # training script for Gemma model
├── in_context/                          # in-context learning with LLMs and SLMs
│   ├── create_incontext_examples.py     # script to create in-context learning examples
│   ├── test_huggingface.py              # script to test models using Hugging Face
│   └── test_together.py                 # script to test models using Together API
├── olmo/                                # source, config, train, inference, and evaluation for Olmo
├── phi/                                 # source, config, train, inference, and evaluation for Phi
├── qwen/                                # source, config, train, inference, and evaluation for Qwen
├── transliteration/                                  
│   └── transliteration.py               # script for transliteration data generation
├── LICENSE                              # license file for the repository
└── README.md                            # README file with project overview and instructions

```

### Tokenization under different noise conditions

![Tokenization under different noise conditions for 'Science fiction'](assets/tokenizer.png)

### Citation

If you find our datasets and work beneficial, please cite our work:

```bibtex
@article{scaria2024can,
  title={Can Small Language Models Learn, Unlearn, and Retain Noise Patterns?},
  author={Scaria, Nicy and Kennedy, Silvester John Joseph and Subramani, Deepak},
  journal={arXiv preprint arXiv:2407.00996},
  year={2024}
}
```
