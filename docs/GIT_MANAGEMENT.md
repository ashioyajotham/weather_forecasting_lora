# Git Repository Management Guide

## Handling Large Files in Weather Forecasting LoRA

---

## 📁 What's Ignored (.gitignore)

### Heavy Files (Excluded from Git)

**Model Files & Weights:**

- `models/` - All trained models
- `*.pth`, `*.pt`, `*.bin` - PyTorch weights
- `*.safetensors` - SafeTensors format
- `*.ckpt`, `*.h5` - Checkpoints
- `*.onnx` - Exported models
- `adapters/`, `lora_adapters/` - LoRA weights

**Training Artifacts:**

- `wandb/` - W&B run data
- `runs/` - TensorBoard logs
- `output/`, `outputs/` - Training outputs
- `checkpoints/` - Training checkpoints

**Large Datasets:**

- `data/raw/*.csv` - Raw weather data
- `data/processed/*.json` - Processed datasets
- `data/cache/` - Cached data
- `*.arrow`, `*.parquet` - Large data formats

**Caches:**

- `.cache/` - General cache
- `transformers_cache/` - HuggingFace cache
- `*.cache` - Various caches

---

## ✅ What's Tracked (Included in Git)

**Source Code:**

- `src/` - All Python source code
- `tests/` - Test suites
- `*.py` - Python scripts

**Configuration:**

- `config/` - YAML configurations
- `*.yaml`, `*.yml` - Config files
- `requirements.txt` - Dependencies

**Documentation:**

- `docs/` - Documentation files
- `*.md` - Markdown files
- `README.md`, `LICENSE` - Project docs

**Small Reference Data:**

- `data/samples/` - Sample data for testing
- `data/examples/` - Example datasets

---

## 🔧 Git LFS (Optional)

If you need to track some large files, use **Git Large File Storage (LFS)**.

### Setup Git LFS

```bash
# Install Git LFS (one-time)
git lfs install

# Track specific large files
git lfs track "models/*.pth"
git lfs track "data/reference/*.parquet"

# Commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"
```

### What's Configured for LFS (.gitattributes)

Already configured to use LFS if installed:

- Model files: `*.pth`, `*.pt`, `*.bin`, `*.safetensors`
- Large data: `*.parquet`, `*.arrow`, `*.feather`
- Archives: `*.zip`, `*.tar.gz`

---

## 📊 Current Repository Size Management

### Before Git Ignore Update

- ❌ Large model files could be accidentally committed
- ❌ Training artifacts would bloat repository
- ❌ Datasets would slow down clone/push

### After Git Ignore Update

- ✅ Only source code and configs tracked
- ✅ Repository stays lightweight (<100 MB)
- ✅ Fast clone and sync operations
- ✅ Clean git history

---

## 🚀 Best Practices

### 1. Model Versioning

**Don't:** Commit models to Git  
**Do:** Use W&B artifacts for model versioning

```python
# Models are automatically saved as W&B artifacts
wandb_logger.log_model_artifact(
    model_path="models/checkpoint-1000",
    artifact_name="weather-lora-v1",
    aliases=["latest", "production"]
)
```

### 2. Dataset Management

**Don't:** Commit large datasets to Git  
**Do:** Document data sources and provide download scripts

```bash
# Keep data collection script in Git
src/data/collect_sample_data.py  ✅

# Don't commit the actual data
data/processed/train.json  ❌ (ignored)
```

### 3. Sharing Models

#### Option A: W&B Artifacts (Recommended)

```bash
# Download model from W&B
wandb artifact get username/project/model:latest
```

#### Option B: External Storage

- Google Drive / OneDrive
- Hugging Face Hub
- AWS S3 / Azure Blob Storage

#### Option C: Git LFS (If needed)

```bash
# Only for critical reference models
git lfs track "models/baseline.pth"
```

---

## 🔍 Checking Repository Size

### Check Current Size

```bash
# See repository size
git count-objects -vH

# See largest files
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  sort -k3 -n -r | head -20
```

### Clean Up if Needed

```bash
# Remove file from Git history (if accidentally committed)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/large/file" \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

---

## 📋 Quick Reference

### Files Always Ignored

```text
models/              # Trained models
*.pth, *.pt         # PyTorch weights
data/processed/     # Large datasets
wandb/              # W&B artifacts
checkpoints/        # Training checkpoints
*.cache             # Cache files
```

### Files Always Tracked

```text
src/                # Source code
config/             # Configurations
docs/               # Documentation
tests/              # Test suites
requirements.txt    # Dependencies
*.py                # Python scripts
*.md                # Markdown docs
```

### Check Before Committing

```bash
# See what will be committed
git status

# See ignored files
git status --ignored

# Check file size
ls -lh path/to/file  # Linux/Mac
dir path\to\file     # Windows
```

---

## ⚠️ Important Notes

### Repository Size Limits

- GitHub: Soft limit 1 GB, hard limit 5 GB
- GitHub file limit: 100 MB (without LFS)
- Git LFS: 1 GB free storage per user

### What to Do If You Accidentally Commit Large Files

**1. Before Pushing:**

```bash
# Reset last commit
git reset HEAD~1

# Remove large file
git rm --cached path/to/large/file

# Add to .gitignore
echo "path/to/large/file" >> .gitignore

# Commit again
git add .
git commit -m "Remove large file"
```

**2. After Pushing:**

- Use BFG Repo-Cleaner or git filter-branch
- Contact repository admin if needed
- Consider using Git LFS for future large files

---

## 🎯 Summary

### Current Setup

- ✅ `.gitignore` updated with comprehensive exclusions
- ✅ `.gitattributes` configured for Git LFS (optional)
- ✅ Model versioning via W&B artifacts
- ✅ Repository optimized for source code only
- ✅ Large files properly excluded

### Benefits

- 🚀 Fast clone and sync operations
- 💾 Small repository size (<100 MB)
- 🔄 Clean git history
- 📊 Professional repository management
- ☁️ Models versioned in W&B (cloud-based)

### Next Steps

1. Review uncommitted large files: `git status --ignored`
2. Ensure models are tracked in W&B
3. Document data sources in README
4. Use W&B for model sharing and versioning

---

*Last Updated: October 12, 2025*  
*Author: Ashioya Jotham Victor*
