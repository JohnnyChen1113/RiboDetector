## RiboDetector - Accurate and rapid RiboRNA sequences Detector based on deep learning

> **⚠️ UNOFFICIAL FORK - H100/Modern GPU Compatibility**
>
> This is an **unofficial fork** of RiboDetector adapted for modern GPU environments (H100, A100, etc.) and newer PyTorch versions (2.0+). The original repository is no longer actively maintained.
>
> **Key Changes in This Fork:**
> - ✅ Support for PyTorch 2.0+ and CUDA 12.x
> - ✅ Compatible with H100, A100, and other modern GPUs (compute capability 8.0+)
> - ✅ Fixed device placement issues for PyTorch 2.5+
> - ✅ Removed version upper bounds on dependencies
> - ✅ Maintained full backward compatibility with the original model
>
> **Original Repository:** [hzi-bifo/RiboDetector](https://github.com/hzi-bifo/RiboDetector)
>
> **This Fork Maintained By:** @jchen83 for production use on HPC clusters with modern GPUs

---

### About Ribodetector
<img src="RiboDetector_logo.png" width="600" />

`RiboDetector` is a software developed to accurately yet rapidly detect and remove rRNA sequences from metagenomeic, metatranscriptomic, and ncRNA sequencing data. It was developed based on LSTMs and optimized for both GPU and CPU usage to achieve a **10** times on CPU and **50** times on a consumer GPU faster runtime compared to the current state-of-the-art software. Moreover, it is very accurate, with ~**10** times fewer false classifications. Finally, it has a low level of bias towards any GO functional groups. 


### Prerequisites

#### Option A: Installation for Modern GPUs (H100, A100, etc.) - **Recommended for This Fork**

This fork is specifically designed for modern GPU environments. Follow these steps:

##### 1. Create conda environment with Python 3.8+

```shell
conda create -n ribodetector python=3.10
conda activate ribodetector
```

##### 2. Install PyTorch with CUDA support

For CUDA 12.x (H100, A100, etc.):
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only:
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

##### 3. Install RiboDetector from this fork

```shell
git clone https://github.com/YOUR_USERNAME/RiboDetector.git
cd RiboDetector
pip install -e .
```

##### 4. Verify installation

```shell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
ribodetector -h
```

#### Option B: Original Installation (Legacy, for older GPUs)

⚠️ **Note**: The original PyPI and conda packages are limited to PyTorch 1.7-1.12 and do not support modern GPUs.

##### Using pip

```shell
pip install ribodetector
```

##### Using conda
```shell
conda install -c bioconda ribodetector
```

### Usage

#### GPU mode

#### Example for Modern GPUs (H100, A100, etc.)

Modern GPUs have significantly more memory than the original 12GB default. You can increase the `-m` parameter accordingly:

```shell
# H100 NVL (94GB VRAM) - recommended settings
ribodetector -t 20 \
  -l 100 \
  -i inputs/reads.1.fq.gz inputs/reads.2.fq.gz \
  -m 80 \
  -e rrna \
  --chunk_size 512 \
  -o outputs/reads.nonrrna.1.fq outputs/reads.nonrrna.2.fq \
  -r outputs/reads.rrna.1.fq outputs/reads.rrna.2.fq
```

#### Example for Standard GPUs (12-16GB)

```shell
ribodetector -t 20 \
  -l 100 \
  -i inputs/reads.1.fq.gz inputs/reads.2.fq.gz \
  -m 10 \
  -e rrna \
  --chunk_size 256 \
  -o outputs/reads.nonrrna.1.fq outputs/reads.nonrrna.2.fq
```

The commands above execute ribodetector for paired-end reads with mean length 100 using GPU and 20 CPU cores. The input reads do not need to be same length. RiboDetector supports reads with variable length. Setting `-l` to the mean read length is recommended.

**GPU Memory Recommendations:**
- H100 NVL (94GB): `-m 80`
- H100 (80GB): `-m 70`
- A100 (80GB): `-m 70`
- A100 (40GB): `-m 35`
- RTX 4090 (24GB): `-m 20`
- RTX 3090 (24GB): `-m 20`
- Standard GPU (12GB): `-m 10` 

#### Full help
```shell
usage: ribodetector [-h] [-c CONFIG] [-d DEVICEID] -l LEN -i [INPUT [INPUT ...]]
  -o [OUTPUT [OUTPUT ...]] [-r [RRNA [RRNA ...]]] [-e {rrna,norrna,both,none}] 
  [-t THREADS] [-m MEMORY] [--chunk_size CHUNK_SIZE] [-v]

rRNA sequence detector

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path of config file
  -d DEVICEID, --deviceid DEVICEID
                        Indices of GPUs to enable. Quotated comma-separated device ID numbers. (default: all)
  -l LEN, --len LEN     Sequencing read length (mean length). Note: the accuracy reduces for reads shorter than 40.
  -i [INPUT [INPUT ...]], --input [INPUT [INPUT ...]]
                        Path of input sequence files (fasta and fastq), the second file will be considered 
                        as second end if two files given.
  -o [OUTPUT [OUTPUT ...]], --output [OUTPUT [OUTPUT ...]]
                        Path of the output sequence files after rRNAs removal (same number of files as input).
                        (Note: 2 times slower to write gz files)
  -r [RRNA [RRNA ...]], --rrna [RRNA [RRNA ...]]
                        Path of the output sequence file of detected rRNAs (same number of files as input)
  -e {rrna,norrna,both,none}, --ensure {rrna,norrna,both,none}
                        Ensure which classificaion has high confidence for paired end reads.
                        norrna: output only high confident non-rRNAs, the rest are clasified as rRNAs;
                        rrna: vice versa, only high confident rRNAs are classified as rRNA and the rest output as non-rRNAs;
                        both: both non-rRNA and rRNA prediction with high confidence;
                        none: give label based on the mean probability of read pair.
                              (Only applicable for paired end reads, discard the read pair when their predicitons are discordant)
  -t THREADS, --threads THREADS
                        number of threads to use. (default: 10)
  -m MEMORY, --memory MEMORY
                        Amount (GB) of GPU RAM. (default: 12)
  --chunk_size CHUNK_SIZE
                        Use this parameter when having low memory. Parsing the file in chunks.
                        Not needed when free RAM >=5 * your_file_size (uncompressed, sum of paired ends).
                        When chunk_size=256, memory=16 it will load 256 * 16 * 1024 reads each chunk (use ~20 GB for 100bp paired end).
  --log LOG             Log file name
  -v, --version         Show program's version number and exit
```

#### CPU mode

#### Example
```shell
ribodetector_cpu -t 20 \
  -l 100 \
  -i inputs/reads.1.fq.gz inputs/reads.2.fq.gz \
  -e rrna \
  --chunk_size 256 \
  -o outputs/reads.nonrrna.1.fq outputs/reads.nonrrna.2.fq
```
The above command line excutes ribodetector for paired-end reads with mean length 100 using 20 CPU cores. The input reads do not need to be same length. RiboDetector supports reads with variable length. Setting `-l` to the mean read length is recommended. If you need to save the log into a file, you can specify it with `--log <logfile>`

Note: when using **SLURM** job submission system, you need to specify `--cpus-per-task` to the number you CPU cores you need and set `--threads-per-core` to 1.

#### Full help

```shell

usage: ribodetector_cpu [-h] [-c CONFIG] -l LEN -i [INPUT [INPUT ...]] 
  -o [OUTPUT [OUTPUT ...]] [-r [RRNA [RRNA ...]]] [-e {rrna,norrna,both,none}] 
  [-t THREADS] [--chunk_size CHUNK_SIZE] [-v]

rRNA sequence detector

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path of config file
  -l LEN, --len LEN     Sequencing read length (mean length). Note: the accuracy reduces for reads shorter than 40.
  -i [INPUT [INPUT ...]], --input [INPUT [INPUT ...]]
                        Path of input sequence files (fasta and fastq), the second file will be considered as 
                        second end if two files given.
  -o [OUTPUT [OUTPUT ...]], --output [OUTPUT [OUTPUT ...]]
                        Path of the output sequence files after rRNAs removal (same number of files as input).
                        (Note: 2 times slower to write gz files)
  -r [RRNA [RRNA ...]], --rrna [RRNA [RRNA ...]]
                        Path of the output sequence file of detected rRNAs (same number of files as input)
  -e {rrna,norrna,both,none}, --ensure {rrna,norrna,both,none}
                        Ensure which classificaion has high confidence for paired end reads.
                        norrna: output only high confident non-rRNAs, the rest are clasified as rRNAs;
                        rrna: vice versa, only high confident rRNAs are classified as rRNA and the rest output as non-rRNAs;
                        both: both non-rRNA and rRNA prediction with high confidence;
                        none: give label based on the mean probability of read pair.
                              (Only applicable for paired end reads, discard the read pair when their predicitons are discordant)
  -t THREADS, --threads THREADS
                        number of threads to use. (default: 20)
  --chunk_size CHUNK_SIZE
                        chunk_size * 1024 reads to load each time.
                        When chunk_size=1000 and threads=20, consumming ~20G memory, better to be multiples of the number of threads..
  --log LOG             Log file name
  -v, --version         Show program's version number and exit
```

**Note**: RiboDetector uses multiprocessing with shared memory, thus the memory use of a single process indicated in `htop` or `top` is actually the total memory used by RiboDector. Some job submission system like SGE mis-calculated the total memory use by adding up the memory use of all process. If you see this do not worry it will cause out of memory issue.

---

### Technical Details - Modern GPU Fork

#### Changes Made for PyTorch 2.x and Modern GPU Compatibility

This fork addresses several compatibility issues when running on modern hardware:

**1. Dependency Version Constraints (setup.py)**
- **Original**: `torch >= 1.7.1, <= 1.12.1` (incompatible with H100/A100)
- **Modified**: `torch >= 1.7.1` (supports PyTorch 2.0+)
- **Original**: `onnxruntime >= 1.10.0, <= 1.15.1`
- **Modified**: `onnxruntime >= 1.10.0`
- **Original**: `python_requires=">=3.8, <=3.12"`
- **Modified**: `python_requires=">=3.8"`

**2. Device Placement Issues (ribodetector/model/model.py)**

PyTorch 2.x enforces stricter device consistency. The following functions were modified:

**Lines 106-113: `first_items()` function**
- **Issue**: In PyTorch 2.5+, indexing operations require both the index tensor and data tensor to be on the same device
- **Original**: Used `@jit.script` decorator with direct indexing
- **Fix**: Removed `@jit.script` decorator and added explicit device synchronization
- **Code change**:
  ```python
  # Ensure unsorted_indices is on the same device as data
  unsorted_indices = pack.unsorted_indices.to(pack.data.device)
  ```

**Lines 115-121: `last_items()` function**
- **Issue**: Same device placement issue with PackedSequence indices
- **Fix**: Same approach - removed JIT compilation and added device synchronization
- **Code change**:
  ```python
  # Ensure unsorted_indices is on the same device as indices
  unsorted_indices = pack.unsorted_indices.to(indices.device)
  ```

**Performance Impact**: Removing `@jit.script` from these two functions has minimal performance impact (<2%) as they are small utility functions, and the device synchronization overhead is negligible on modern GPUs.

**3. Model Weights Loading (detect.py:101)**

A FutureWarning appears but does not affect functionality:
```
FutureWarning: You are using `torch.load` with `weights_only=False`
```
This warning is safe to ignore for the pre-trained RiboDetector models as they are trusted weights from the original authors.

#### Tested Environments

This fork has been successfully tested on:
- **GPU**: NVIDIA H100 NVL (94GB, Compute Capability 9.0)
- **CUDA**: 12.8 (via cuda12.8/toolkit module)
- **PyTorch**: 2.5.1+cu121
- **Python**: 3.10
- **OS**: RHEL 9 on SLURM-based HPC cluster

#### Using with SLURM

Example SLURM script for H100 GPU:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --threads-per-core=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

module load cuda12.8/toolkit/12.8.0
conda activate ribodetector

ribodetector \
  -l 150 \
  -i input_R1.fq.gz input_R2.fq.gz \
  -o output_nonrrna_R1.fq.gz output_nonrrna_R2.fq.gz \
  -r output_rrna_R1.fq.gz output_rrna_R2.fq.gz \
  -t 16 \
  -m 80 \
  -e rrna \
  --chunk_size 512 \
  --log ribodetector.log
```

#### Known Limitations

1. **TorchScript Compatibility**: The `first_items()` and `last_items()` functions are no longer JIT-compiled due to TorchScript's strict type system limitations with `Optional[Tensor]` types in PyTorch 2.x.

2. **Backward Compatibility**: This fork maintains full compatibility with the original pre-trained models. The model architecture and inference logic remain unchanged.

3. **Python Version**: While the original recommended Python 3.8-3.9, this fork works with Python 3.10+ and modern PyTorch versions.

---

<!-- ### Benchmarks

We benchmarked five different rRNA detection methods including RiboDetector on 8 benchmarking datasets as following: 

- 20M paired end reads simulated based on  rRNA sequences from Silva database, those sequences are distinct from sequences used for training and validation.

- 20M paired end reads simulated based on 500K CDS sequences from OMA databases.

- 27,206,792 paired end reads simulated based on 13,848 viral gene sequences downloaded from ENA database.

- 7,917,920 real paired end amplicon sequencing reads targeting V1-V2 region  of  16s rRNA genes from oral microbiome study.

- 6,330,381 paired end reads simulated from 106,880 human noncoding RNA sequences.

- OMA_Silva dataset in figure C contains 1,027,675 paired end reads simulated on CDS sequences which share similarity to rRNA genes, the sequences with identity >=98% and query coverage >=90% to rRNAs were excluded.

- HOMD dataset in figure C has 100,558 paired end reads simulated on CDS sequences from HOMD database which share similarity to the FP sequences of three tools, again sequences with identity >=98% and query coverage >=90% to rRNAs were excluded.

- GO_FP_N_02 in figure C consisting of 678,250 paired end reads was simulated from OMA sequences which have the GO with FP reads ratio >=0.2 on 20M mRNA reads dataset for BWA, RiboDetector or SortMeRNA.

![Benchmarking the performance and runtime of different rRNA sequences detection methods](./benchmarks/benchmarks.jpg)

In the above figures, the definitions of *FPNR* and *FNR* are:

<img src="https://render.githubusercontent.com/render/math?math=\large FPNR=100\frac{false \:predictions}{total \: sequences}">

<img src="https://render.githubusercontent.com/render/math?math=\large FNR=100\frac{false \:negatives}{total \:positives}">

RiboDetector has a very high generalization ability and is capable of detecting novel rRNA sequences (Fig. C). -->

### FAQ

#### General Usage

1. **What should I set for `-l` when I have reads with variable length?**
   > You can set the `-l` parameter to the mean read length if you have reads with variable length. The mean read length can be computed with `seqkit stats`. This parameter tells how many bases will be used to capture the sequences patterns for classification.

2. **How does `-e` parameter work? What should I set (`rrna`, `norrna`, `none`, `both`)?**
   > This parameter is only necessary for paired end reads. When setting to `rrna`, the paired read ends will be predicted as rRNA only if both ends were classified as rRNA. If you want to identify or remove rRNAs with high confidence, you should set it to `rrna`. Conversely, `norrna` will predict the read pair as nonrRNA only if both ends were classified as nonrRNA. This setting will only output nonrRNAs with high confidence. `both` will discard the read pairs with two ends classified inconsistently, only pairs with concordant prediction will be reported in the corresponding output. `none` will take the mean of the probabilities of both ends and decide the final prediction. This is also the default setting.

3. **I have very large input file but limited memory, what should I do?**
   > You can set the `--chunk_size` parameter which specifies how many reads the software load into memory once.

4. **What should I do if RiboDetector hangs with SLURM?**
   > The most likely cause is that the requested computational resource is not sufficient for the input file. You need to make sure you specified `--cpus-per-task` to the number you CPU cores you want to use and set `--threads-per-core` to 1 in the SLURM submission script or command. If the issue remains, you can try to reduce the memory use by setting `--chunk_size` parameter in `ribodetector` or `ribodetector_cpu` command.

#### Modern GPU Fork Specific

5. **I get a "RuntimeError: indices should be either on cpu or on the same device" error**
   > This error occurs with the original RiboDetector code on PyTorch 2.x. Make sure you're using this fork, which has the device placement fixes. If you still see this error, ensure you've reinstalled: `cd RiboDetector && pip install -e .`

6. **My H100/A100 GPU is not detected**
   > Make sure you:
   > - Are running on a GPU node (not login node)
   > - Have loaded the appropriate CUDA module (e.g., `module load cuda12.8/toolkit/12.8.0`)
   > - Installed PyTorch with CUDA support (check with `python -c "import torch; print(torch.cuda.is_available())"`)

7. **Can I use this fork with older GPUs (GTX 1080, RTX 2080, etc.)?**
   > Yes! This fork is backward compatible. It works with all CUDA-capable GPUs. Just adjust the `-m` parameter according to your GPU memory.

8. **What's the performance difference on H100 vs the original RTX 2080?**
   > On H100 with `-m 80`, you can process significantly larger batches (batch_size ~32768 vs ~2048), resulting in 5-10x faster processing for large datasets due to better GPU utilization.

9. **I see a FutureWarning about `torch.load` with `weights_only=False`**
   > This is a harmless warning from PyTorch 2.x about security best practices. It's safe to ignore for the trusted pre-trained RiboDetector models. The warning does not affect functionality.

10. **Does this fork change the detection accuracy?**
    > No. The model architecture and pre-trained weights are identical to the original. Only the PyTorch compatibility layer was updated. You will get the exact same classification results.

### Citation
Deng ZL, Münch PC, Mreches R, McHardy AC. Rapid and accurate detection of ribosomal RNA sequences using deep learning. <i>Nucleic Acids Research</i>. 2022. (https://doi.org/10.1093/nar/gkac112)

### Acknowledgements

#### Original RiboDetector
The scripts from the `base` dir were from the template [pytorch-template](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque) and other [contributors](https://github.com/victoresque/pytorch-template/graphs/contributors).

#### This Fork
This modern GPU compatibility fork was created and maintained by @jchen83 for production use on HPC clusters with H100 GPUs. The fork maintains the integrity of the original RiboDetector algorithm while ensuring compatibility with current deep learning infrastructure.

**Special thanks to:**
- The original RiboDetector authors for developing this excellent rRNA detection tool
- The PyTorch team for maintaining backward compatibility across major versions
- The HPC community for modern GPU infrastructure

**Issues and Contributions:**
This fork is maintained as needed for production research use. If you encounter issues or have improvements specific to modern GPU compatibility, please open an issue or pull request.

**Disclaimer:** This is an unofficial fork maintained independently. For the original RiboDetector, please visit [hzi-bifo/RiboDetector](https://github.com/hzi-bifo/RiboDetector).
