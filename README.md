# LossyFormer

This forms a suite of tools developed as part of the Spring 2026 ADLS module. It is aimed at optimising several transformer models for latency and throughput, whilst accepting minor, controlled accuracy drops. It is tested and working for BERT-Base, BERT-Tiny and RoBERTa.

## Usage

For example usage, look at `lf_tests/example.py`. This demonstrates how to instantiate the class and call `lossy.fit()` on transformer models.

```sh
uv sync
python lf_tests/usage.py # Executes the 
```





## Future Goals

- Develop and test the LossyFormer utility for the ViT model.
- Reflect on the GeLU to ReLU swap, which is expected to offer better latency and throughput, but for custom architectures built on FPGA or ASIC. Omitted due to GPU performance being unaffected.


**Contributors:**

- Dhyey Trivedi
- Hrishikesh Venkatesh
- Neil Radhu
- Utsav Goel