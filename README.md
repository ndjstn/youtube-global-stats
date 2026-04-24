# Five Archetypes at the Top of YouTube

KMeans clustering on the Global YouTube Statistics 2023 dataset (995 top creators). At k=5 the segments map onto five recognisable archetypes: mega-scale creators, mainstream large channels, low-engagement large channels, music-video channels, and upload machines. The elbow supports k=5 and each cluster separates on a distinct feature combination.

## Key findings

| Cluster | Median subs | Median uploads | Views/sub | n | Label |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 36.5M | 1,331 | 592 | 135 | Mega-scale creators |
| 1 | 16.0M | 719 | 436 | 367 | Mainstream large |
| 2 | 14.9M | 461 | 133 | 158 | Low-engagement large |
| 3 | 20.7M | 12 | 445 | 114 | Music-video channels |
| 4 | 16.9M | 10,022 | 548 | 175 | Upload machines |

Music-video channels have the smallest median upload count (12) but the highest views per upload (668 million). Upload-machine channels have two orders of magnitude more uploads but a thousandth of the per-video virality. Cluster 2 is the low-engagement outlier where subscriber counts are large but views per subscriber collapse.

## What is in this repo

`src/run_analysis.py` runs the end-to-end pipeline. `notebooks/` has the narrative walk-through. `src/_palette.py` is the project's noir + signal-red creator-studio palette. `figures/` has country counts, category counts, subs-vs-views scatter, KMeans elbow, cluster scatter, cluster reveal animation, creator density per capita, and estimated category earnings.

`REPORT.md` is the long-form analysis.

## How to reproduce

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_analysis.py --data "data/Global YouTube Statistics.csv" --figures figures --outputs outputs
```

Download `Global YouTube Statistics.csv` from <https://www.kaggle.com/datasets/nelgiriyewithana/global-youtube-statistics-2023> and place at `data/`.

## Further reading

<https://ndjstn.github.io/posts/youtube-top995-five-archetypes/>.

## License

MIT.
