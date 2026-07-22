# Publishing to GitHub and minting a Zenodo DOI

This repo is initialised as a git repository locally. To make it citable in the
manuscript, push it to GitHub and connect Zenodo so a DOI is minted on release.

## 1. Push to the GitHub repository

The remote is https://github.com/YAOJ-bioin/peagent-analysis (already created).
From a shell with GitHub credentials configured:

```bash
cd peagent-analysis
git remote add origin https://github.com/YAOJ-bioin/peagent-analysis.git
git branch -M main
git push -u origin main
```

(Author metadata in `CITATION.cff`, `.zenodo.json` and `LICENSE` is already filled in.)

## 2. Connect Zenodo (one-time)

1. Sign in at https://zenodo.org with your GitHub account.
2. Go to **Settings → GitHub**, and flip the toggle **ON** for `peagent-analysis`.
   Zenodo now watches the repo for releases.

## 3. Cut a release to mint the DOI

On GitHub: **Releases → Draft a new release**, tag e.g. `v1.0.0`, publish.
Zenodo automatically archives the tagged snapshot and mints a DOI. The DOI badge
and a version-independent "concept DOI" appear on the Zenodo record.

## 4. Cite it

Add the Zenodo DOI to the manuscript (Data/Code Availability) and, optionally,
paste the DOI badge into `README.md`. Use the **concept DOI** to always resolve
to the latest version.
