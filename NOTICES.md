# Third-Party Notices & Attribution

AQIVON OmniOCR incorporates algorithms, architectures, and concepts inspired by or adapted from open-source projects. In accordance with their respective open-source licenses, required notices and attributions are set forth below.

---

### 1. Baidu Unlimited-OCR
- **Project**: Unlimited-OCR & Unlimited-OCR-2
- **Origin**: Baidu Inc.
- **Repository**: https://github.com/baidu/Unlimited-OCR
- **License**: MIT License
- **Notice**:
```
MIT License

Copyright (c) 2024-2025 Baidu, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
- **AQIVON Integration**: Reference Sliding Window Attention (R-SWA) concept, multi-page page splitting, heading normalization, bounding-box coordinate tracking, and table structure reconstruction algorithms.

---

### 2. PaddleOCR
- **Origin**: Baidu PaddlePaddle Authors
- **Repository**: https://github.com/PaddlePaddle/PaddleOCR
- **License**: Apache License 2.0
- **Notice**:
```
Copyright (c) 2020-2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
- **AQIVON Integration**: DBNet text detection and SVTR/CRNN multilingual recognition adapter.

---

### 3. Microsoft TrOCR
- **Origin**: Microsoft Corporation
- **Repository**: https://github.com/microsoft/unilm/tree/master/trocr
- **License**: MIT / Apache License 2.0
- **Notice**:
```
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the Apache License, Version 2.0.
```
- **AQIVON Integration**: Transformer VisionEncoderDecoder handwritten text recognition line adapter.

---

### 4. docTR (Document Text Recognition)
- **Origin**: Mindee
- **Repository**: https://github.com/mindee/doctr
- **License**: Apache License 2.0
- **Notice**:
```
Copyright (C) 2021-2025, Mindee.
Licensed under the Apache License, Version 2.0.
```
- **AQIVON Integration**: End-to-end PyTorch document OCR adapter.

---

### 5. Surya OCR
- **Origin**: Vik Paruchuri
- **Repository**: https://github.com/VikParuchuri/surya
- **License**: GPL-3.0 / Commercial License
- **Notice**: Used via modular dynamic adapter.

---

### 6. MinerU
- **Origin**: OpenDataLab
- **Repository**: https://github.com/opendatalab/MinerU
- **License**: AGPL-3.0 / Apache License 2.0
- **Notice**: Used via modular layout adapter.

---

### 7. Google Tesseract OCR
- **Origin**: Google Inc. / Hewlett-Packard Company
- **Repository**: https://github.com/tesseract-ocr/tesseract
- **License**: Apache License 2.0
- **Notice**:
```
Copyright 2018 Google LLC
Licensed under the Apache License, Version 2.0.
```
- **AQIVON Integration**: Offline CPU fallback engine adapter.

