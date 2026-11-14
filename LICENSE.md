# Licence and data terms

## 1. Code in this repository

The code written for this project (everything in `src/`, `tests/` and `docs/`) is released under the MIT Licence.

```
MIT License

Copyright (c) 2026 Ayushi Agrawal and Pranjali Buddhe

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

## 2. Data used (not covered by the licence above)

| Item | Detail |
|---|---|
| Dataset | "Air Quality Data in India (2015 - 2020)" |
| Platform | Kaggle: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india |
| Uploader | Rohan Rao (Vopani) |
| File used | `city_day.csv` |
| Original source | Central Pollution Control Board (CPCB), Government of India, https://cpcb.nic.in/ |
| Access | Public and free, with a Kaggle login |
| Licence shown on the Kaggle page | CC0: Public Domain (checked on the dataset page on 14 November 2026) |
| Terms stated in the dataset description | The description states no reuse or attribution conditions. Under CC0 the data may be reused freely; we still credit CPCB and the uploader. The description also links the uploader's AQI tutorial: https://www.kaggle.com/rohanrao/calculating-aqi-air-quality-index |

### How the data is handled in this project

- **The raw data is not redistributed.** `city_day.csv` is not committed to this repository. The README explains how to download it from Kaggle.
- **Only derived summary outputs are published.** The website and `docs/data/` contain aggregate results (counts, precision, recall, per-city scores) and, for the timeline, model probabilities and yes/no labels. They do not contain the dataset's pollutant readings or AQI values.
- **Use is non-commercial and educational**, for a university course project.
- **Attribution:** data originates from CPCB monitoring stations and was compiled and uploaded to Kaggle by Rohan Rao. Anyone reusing this work should credit both and follow the terms on the Kaggle page.

### Known data caveats (stated for transparency)

- The `AQI` column was computed by the dataset uploader, not published by CPCB as an official index. It is not capped at 500 and contains implausible values.
- Readings come from a limited set of monitoring stations and do not represent whole cities.
- This project predicts; it does not explain causes, and it should not be used for real health or safety decisions.

## 3. Credentials

No passwords, API keys or `kaggle.json` are included in this repository.